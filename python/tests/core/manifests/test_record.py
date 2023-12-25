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

import pyarrow.parquet as pq

from space.core.manifests import RecordManifestWriter
import space.core.proto.metadata_pb2 as meta


class TestRecordManifestWriter:

  def test_write(self, tmp_path):
    metadata_dir = tmp_path / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True)

    manifest_writer = RecordManifestWriter(metadata_dir=str(metadata_dir))

    manifest_writer.write(
        "data/file0.array_record", 0,
        meta.StorageStatistics(num_rows=123,
                               index_compressed_bytes=10,
                               index_uncompressed_bytes=20,
                               record_uncompressed_bytes=30))
    manifest_writer.write(
        "data/file1.array_record", 1,
        meta.StorageStatistics(num_rows=456,
                               index_compressed_bytes=10,
                               index_uncompressed_bytes=20,
                               record_uncompressed_bytes=100))

    manifest_path = manifest_writer.finish()

    assert manifest_path is not None
    assert pq.read_table(manifest_path).to_pydict() == {
        "_FILE": ["data/file0.array_record", "data/file1.array_record"],
        "_FIELD_ID": [0, 1],
        "_NUM_ROWS": [123, 456],
        "_UNCOMPRESSED_BYTES": [30, 100]
    }

  def test_empty_manifest_should_return_none(self, tmp_path):
    metadata_dir = tmp_path / "dataset" / "metadata"
    manifest_writer = RecordManifestWriter(metadata_dir=str(metadata_dir))

    assert manifest_writer.finish() is None
