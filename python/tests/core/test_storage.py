# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.protobuf.timestamp_pb2 import Timestamp
import pyarrow.compute as pc
import pyarrow as pa
import pytest
from substrait.type_pb2 import NamedStruct, Type

from space.core.fs.parquet import write_parquet_file
from space.core.manifests import IndexManifestWriter
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as runtime
from space.core.storage import Storage
from space.core.utils.paths import _ENTRY_POINT_FILE

_SNAPSHOT_ID = 100
_SCHEMA = pa.schema(
    [pa.field("int64", pa.int64()),
     pa.field("string", pa.string())])


class TestStorage:

  @pytest.fixture
  def metadata(self):
    current_snapshot_id = _SNAPSHOT_ID
    now = Timestamp(seconds=123456)
    metadata = meta.StorageMetadata(create_time=now,
                                    last_update_time=now,
                                    current_snapshot_id=current_snapshot_id,
                                    type=meta.StorageMetadata.DATASET)
    metadata.snapshots[current_snapshot_id].CopyFrom(
        meta.Snapshot(snapshot_id=current_snapshot_id, create_time=now))

    # Add a previous snapshot for testing.
    previous_snapshot_id = 10
    metadata.snapshots[previous_snapshot_id].CopyFrom(
        meta.Snapshot(snapshot_id=previous_snapshot_id,
                      create_time=Timestamp(seconds=123000)))
    return metadata

  @pytest.fixture
  def storage(self, metadata):
    return Storage("location", metadata)

  def test_create_dir(self, storage, metadata):
    assert storage.metadata == metadata

  def test_snapshot(self, storage):
    # Test current snapshot ID.
    current_snapshot_id = _SNAPSHOT_ID
    assert storage.snapshot().snapshot_id == current_snapshot_id
    assert storage.snapshot(
        snapshot_id=current_snapshot_id).snapshot_id == current_snapshot_id

    # Test previous snapshot ID.
    assert storage.snapshot(snapshot_id=10).snapshot_id == 10

  def test_create_storage(self, tmp_path):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=_SCHEMA,
                             primary_keys=["int64"],
                             record_fields=["string"])

    entry_point_file = location / "metadata" / _ENTRY_POINT_FILE
    assert entry_point_file.exists()

    metadata = storage.metadata
    snapshot = storage.snapshot()
    assert metadata.current_snapshot_id == 0
    assert metadata.type == meta.StorageMetadata.DATASET
    assert snapshot.snapshot_id == 0
    assert (metadata.create_time == metadata.last_update_time ==
            snapshot.create_time)

    assert metadata.schema == meta.Schema(fields=NamedStruct(
        names=["int64", "string"],
        struct=Type.Struct(types=[
            Type(i64=Type.I64(type_variation_reference=0)),
            Type(string=Type.String(type_variation_reference=1))
        ])),
                                          primary_keys=["int64"],
                                          record_fields=["string"])

  def test_load_storage(self, tmp_path):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=_SCHEMA,
                             primary_keys=["int64"],
                             record_fields=[])

    loaded_storage = Storage.load(str(location))
    assert loaded_storage.metadata == storage.metadata

  def test_load_storage_file_not_found_should_fail(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      Storage.load(str(tmp_path / "dataset"))

  def test_commit(self, tmp_path):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=_SCHEMA,
                             primary_keys=["int64"],
                             record_fields=["string"])

    added_manifest_files = meta.ManifestFiles(
        index_manifest_files=["data/index_manifest0"],
        record_manifest_files=["data/record_manifest0"])
    added_storage_statistics = meta.StorageStatistics(
        num_rows=123,
        index_compressed_bytes=10,
        index_uncompressed_bytes=20,
        record_uncompressed_bytes=30)
    patch = runtime.Patch(addition=added_manifest_files,
                          storage_statistics_update=added_storage_statistics)
    storage.commit(patch)

    assert storage.snapshot(0) is not None
    new_snapshot = storage.snapshot(1)

    assert new_snapshot.manifest_files == added_manifest_files
    assert new_snapshot.storage_statistics == added_storage_statistics

    # Add more manifests
    added_storage_statistics2 = meta.StorageStatistics(
        num_rows=100,
        index_compressed_bytes=100,
        index_uncompressed_bytes=200,
        record_uncompressed_bytes=300)
    patch = runtime.Patch(addition=meta.ManifestFiles(
        index_manifest_files=["data/index_manifest1"],
        record_manifest_files=["data/record_manifest1"]),
                          storage_statistics_update=added_storage_statistics2)
    storage.commit(patch)

    new_snapshot = storage.snapshot(2)
    assert new_snapshot.manifest_files == meta.ManifestFiles(
        index_manifest_files=["data/index_manifest0", "data/index_manifest1"],
        record_manifest_files=[
            "data/record_manifest0", "data/record_manifest1"
        ])
    assert new_snapshot.storage_statistics == meta.StorageStatistics(
        num_rows=223,
        index_compressed_bytes=110,
        index_uncompressed_bytes=220,
        record_uncompressed_bytes=330)

    # Test deletion.
    patch = runtime.Patch(deletion=meta.ManifestFiles(
        index_manifest_files=["data/index_manifest0"]),
                          storage_statistics_update=meta.StorageStatistics(
                              num_rows=-123,
                              index_compressed_bytes=-10,
                              index_uncompressed_bytes=-20,
                              record_uncompressed_bytes=-30))
    storage.commit(patch)
    new_snapshot = storage.snapshot(3)
    assert new_snapshot.manifest_files.index_manifest_files == [
        "data/index_manifest1"
    ]
    assert new_snapshot.storage_statistics == added_storage_statistics2

  def test_data_files(self, tmp_path):
    location = tmp_path / "dataset"
    data_dir = location / "data"
    metadata_dir = location / "metadata"
    storage = Storage.create(location=str(location),
                             schema=_SCHEMA,
                             primary_keys=["int64"],
                             record_fields=[])
    schema = storage.physical_schema

    def create_index_manifest_writer():
      return IndexManifestWriter(metadata_dir=str(metadata_dir),
                                 schema=schema,
                                 primary_keys=["int64"])

    def commit_add_index_manifest(manifest_path: str):
      patch = runtime.Patch(addition=meta.ManifestFiles(
          index_manifest_files=[storage.short_path(manifest_path)]))
      storage.commit(patch)

    manifest_writer = create_index_manifest_writer()
    manifest_writer.write(
        "data/file0",
        write_parquet_file(str(data_dir / "file0"), schema, [
            pa.Table.from_pydict({
                "int64": [1, 2, 3],
                "string": ["a", "b", "c"]
            })
        ]))
    manifest_file = manifest_writer.finish()
    commit_add_index_manifest(manifest_file)
    manifests_dict1 = {1: storage.short_path(manifest_file)}

    index_file0 = runtime.DataFile(path="data/file0",
                                   manifest_file_id=1,
                                   storage_statistics=meta.StorageStatistics(
                                       num_rows=3,
                                       index_compressed_bytes=110,
                                       index_uncompressed_bytes=109))

    assert storage.data_files() == runtime.FileSet(
        index_files=[index_file0], index_manifest_files=manifests_dict1)

    # Write the 2nd data file, generate manifest, and commit.
    manifest_writer = create_index_manifest_writer()
    manifest_writer.write(
        "data/file1",
        write_parquet_file(str(data_dir / "file1"), schema, [
            pa.Table.from_pydict({
                "int64": [1000, 1000000],
                "string": ["abcedf", "ABCDEF"]
            })
        ]))
    manifest_file = manifest_writer.finish()
    commit_add_index_manifest(manifest_file)
    manifests_dict2 = manifests_dict1.copy()
    manifests_dict2[2] = storage.short_path(manifest_file)

    index_file1 = runtime.DataFile(path="data/file1",
                                   manifest_file_id=2,
                                   storage_statistics=meta.StorageStatistics(
                                       num_rows=2,
                                       index_compressed_bytes=104,
                                       index_uncompressed_bytes=100))

    assert storage.data_files() == runtime.FileSet(
        index_files=[index_file0, index_file1],
        index_manifest_files=manifests_dict2)

    # Test time travel data_files().
    assert storage.data_files(snapshot_id=0) == runtime.FileSet()
    assert storage.data_files(snapshot_id=1) == runtime.FileSet(
        index_files=[index_file0], index_manifest_files=manifests_dict1)

    # Test data_files() with filters.
    index_file1.manifest_file_id = 1
    assert storage.data_files(
        filter_=pc.field("int64") > 1000) == runtime.FileSet(
            index_files=[index_file1],
            index_manifest_files={1: manifests_dict2[2]})
