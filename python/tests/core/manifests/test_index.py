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

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from space.core.manifests import IndexManifestWriter
from space.core.manifests.index import read_index_manifests
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as runtime
from space.core.schema.arrow import field_metadata
from space.core.utils.parquet import write_parquet_file

_SCHEMA = pa.schema([
    pa.field("int64", pa.int64(), metadata=field_metadata(0)),
    pa.field("float64", pa.float64(), metadata=field_metadata(1)),
    pa.field("bool", pa.bool_(), metadata=field_metadata(2)),
    pa.field("string", pa.string(), metadata=field_metadata(3))
])


class TestIndexManifestWriter:

  def test_write_all_types_and_read(self, tmp_path):
    data_dir = tmp_path / "dataset" / "data"
    data_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True)

    schema = _SCHEMA
    manifest_writer = IndexManifestWriter(
        metadata_dir=str(metadata_dir),
        schema=schema,
        primary_keys=["int64", "float64", "bool", "string"])

    # TODO: the test should cover all types supported by column stats.
    manifest_writer.write(
        "data/file0",
        write_parquet_file(str(data_dir / "file0"), schema, [
            pa.Table.from_pydict({
                "int64": [1, 2, 3],
                "float64": [0.1, 0.2, 0.3],
                "bool": [True, False, False],
                "string": ["a", "b", "c"]
            }),
            pa.Table.from_pydict({
                "int64": [0, 10],
                "float64": [-0.1, 100.0],
                "bool": [False, False],
                "string": ["A", "z"]
            })
        ]))
    manifest_writer.write(
        "data/file1",
        write_parquet_file(str(data_dir / "file1"), schema, [
            pa.Table.from_pydict({
                "int64": [1000, 1000000],
                "float64": [-0.001, 0.001],
                "bool": [False, False],
                "string": ["abcedf", "ABCDEF"]
            })
        ]))

    manifest_path = manifest_writer.finish()

    assert manifest_path is not None
    assert pq.read_table(manifest_path).to_pydict() == {
        "_FILE": ["data/file0", "data/file1"],
        "_INDEX_COMPRESSED_BYTES": [645, 334],
        "_INDEX_UNCOMPRESSED_BYTES": [624, 320],
        "_NUM_ROWS": [5, 2],
        "_STATS_f0": [{
            "_MAX": 10,
            "_MIN": 0
        }, {
            "_MAX": 1000000,
            "_MIN": 1000
        }],
        "_STATS_f1": [{
            "_MAX": 100.0,
            "_MIN": -0.1
        }, {
            "_MAX": 0.001,
            "_MIN": -0.001
        }],
        "_STATS_f2": [{
            "_MAX": True,
            "_MIN": False
        }, {
            "_MAX": False,
            "_MIN": False
        }],
        "_STATS_f3": [{
            "_MAX": "z",
            "_MIN": "A"
        }, {
            "_MAX": "abcedf",
            "_MIN": "ABCDEF"
        }]
    }

    assert read_index_manifests(manifest_path) == runtime.FileSet(index_files=[
        runtime.DataFile(path="data/file0",
                         storage_statistics=meta.StorageStatistics(
                             num_rows=5,
                             index_compressed_bytes=645,
                             index_uncompressed_bytes=624)),
        runtime.DataFile(path="data/file1",
                         storage_statistics=meta.StorageStatistics(
                             num_rows=2,
                             index_compressed_bytes=334,
                             index_uncompressed_bytes=320))
    ])

    # Test index manifest filtering.
    # TODO: to move it to a separate test and add more test cases.
    filtered_manifests = read_index_manifests(
        manifest_path,
        pc.field("_STATS_f3", "_MIN") >= "ABCDEF")
    assert len(filtered_manifests.index_files) == 1
    assert filtered_manifests.index_files[0].path == "data/file1"

  def test_write_collet_stats_for_primary_keys_only(self, tmp_path):
    data_dir = tmp_path / "dataset" / "data"
    data_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True)

    schema = _SCHEMA
    manifest_writer = IndexManifestWriter(metadata_dir=str(metadata_dir),
                                          schema=schema,
                                          primary_keys=["int64"])

    file_path = str(data_dir / "file0")
    # TODO: the test should cover all types supported by column stats.
    manifest_writer.write(
        file_path,
        write_parquet_file(file_path, schema, [
            pa.Table.from_pydict({
                "int64": [1, 2, 3],
                "float64": [0.1, 0.2, 0.3],
                "bool": [True, False, False],
                "string": ["a", "b", "c"]
            })
        ]))

    manifest_path = manifest_writer.finish()

    data_dir_str = str(data_dir)
    assert manifest_path is not None
    assert pq.read_table(manifest_path).to_pydict() == {
        "_FILE": [f"{data_dir_str}/file0"],
        "_INDEX_COMPRESSED_BYTES": [110],
        "_INDEX_UNCOMPRESSED_BYTES": [109],
        "_NUM_ROWS": [3],
        "_STATS_f0": [{
            "_MAX": 3,
            "_MIN": 1
        }]
    }

  def test_empty_manifest_should_return_none(self, tmp_path):
    metadata_dir = tmp_path / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True)

    schema = _SCHEMA
    manifest_writer = IndexManifestWriter(metadata_dir=str(metadata_dir),
                                          schema=schema,
                                          primary_keys=["int64"])

    assert manifest_writer.finish() is None
