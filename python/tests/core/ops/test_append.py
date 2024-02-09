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

from typing import List
import pyarrow as pa
import pyarrow.parquet as pq

from space.core.ops.append import LocalAppendOp
from space.core.options import FileOptions
import space.core.proto.metadata_pb2 as meta
from space.core.storage import Storage

_default_file_options = FileOptions()


class TestLocalAppendOp:

  # TODO: to add tests using Arrow table input.

  def test_write_pydict_all_types(self, tmp_path, all_types_schema,
                                  all_types_input_data):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=all_types_schema,
                             primary_keys=["int64"],
                             record_fields=[])

    op = LocalAppendOp(str(location), storage.metadata, _default_file_options)
    for batch in all_types_input_data:
      op.write(batch)

    patch = op.finish()
    assert patch is not None

    index_manifests = []
    for f in patch.addition.index_manifest_files:
      index_manifests.append(pq.read_table(storage.full_path(f)))

    index_manifest = pa.concat_tables(index_manifests).to_pydict()
    assert "_FILE" in index_manifest

    assert index_manifest == {
        "_FILE": index_manifest["_FILE"],
        "_INDEX_COMPRESSED_BYTES": [114],
        "_INDEX_UNCOMPRESSED_BYTES": [126],
        "_NUM_ROWS": [5],
        "_STATS_f0": [{
            "_MAX": 10,
            "_MIN": 0
        }]
    }

    assert patch.storage_statistics_update == meta.StorageStatistics(
        num_rows=5, index_compressed_bytes=114, index_uncompressed_bytes=126)

  def test_write_pydict_with_record_fields(self, tmp_path, record_fields_schema,
                                           record_fields_input_data):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=record_fields_schema,
                             primary_keys=["int64"],
                             record_fields=["images", "objects"])

    op = LocalAppendOp(str(location), storage.metadata, _default_file_options)
    for batch in record_fields_input_data:
      op.write(batch)

    patch = op.finish()
    assert patch is not None

    # Validate index manifest files.
    index_manifest = self._read_manifests(
        storage, list(patch.addition.index_manifest_files))
    assert index_manifest == {
        "_FILE": index_manifest["_FILE"],
        "_INDEX_COMPRESSED_BYTES": [114],
        "_INDEX_UNCOMPRESSED_BYTES": [126],
        "_NUM_ROWS": [5],
        "_STATS_f0": [{
            "_MAX": 10,
            "_MIN": 0
        }]
    }

    # Validate record manifest files.
    record_manifest = self._read_manifests(
        storage, list(patch.addition.record_manifest_files))
    assert record_manifest == {
        "_FILE": record_manifest["_FILE"],
        "_FIELD_ID": [2, 3],
        "_NUM_ROWS": [5, 5],
        "_UNCOMPRESSED_BYTES": [55, 60]
    }

    # Data file exists.
    self._check_file_exists(location, index_manifest["_FILE"])
    self._check_file_exists(location, record_manifest["_FILE"])
    assert self._file_schema(
        location, index_manifest["_FILE"][0]) == storage.physical_schema

    # Validate statistics.
    assert patch.storage_statistics_update == meta.StorageStatistics(
        num_rows=5,
        index_compressed_bytes=114,
        index_uncompressed_bytes=126,
        record_uncompressed_bytes=115)

  def test_empty_op_return_none(self, tmp_path):
    location = tmp_path / "dataset"
    schema = pa.schema([pa.field("int64", pa.int64())])
    storage = Storage.create(location=str(location),
                             schema=schema,
                             primary_keys=["int64"],
                             record_fields=[])

    op = LocalAppendOp(str(location), storage.metadata, _default_file_options)
    assert op.finish() is None

  def _read_manifests(self, storage: Storage,
                      file_paths: List[str]) -> pa.Table:
    manifests = []
    for f in file_paths:
      manifests.append(pq.read_table(storage.full_path(f)))

    return pa.concat_tables(manifests).to_pydict()

  def _check_file_exists(self, location, file_paths: List[str]):
    for f in file_paths:
      assert (location / f).exists()

  def _file_schema(self, location, file_path: str) -> pa.Schema:
    return pq.read_schema(str(location / file_path))
