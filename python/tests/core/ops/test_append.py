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
import pyarrow.parquet as pq

from space.core.ops import LocalAppendOp
import space.core.proto.metadata_pb2 as meta
from space.core.storage import Storage


class TestLocalAppendOp:

  def test_write_pydict_all_types(self, tmp_path):
    location = tmp_path / "dataset"
    schema = pa.schema([
        pa.field("int64", pa.int64()),
        pa.field("float64", pa.float64()),
        pa.field("bool", pa.bool_()),
        pa.field("string", pa.string())
    ])
    storage = Storage.create(location=str(location),
                             schema=schema,
                             primary_keys=["int64"])

    op = LocalAppendOp(str(location), storage.metadata)

    # TODO: the test should cover all types supported by column stats.
    op.write({
        "int64": [1, 2, 3],
        "float64": [0.1, 0.2, 0.3],
        "bool": [True, False, False],
        "string": ["a", "b", "c"]
    })
    op.write({
        "int64": [0, 10],
        "float64": [-0.1, 100.0],
        "bool": [False, False],
        "string": ["A", "z"]
    })

    patch = op.finish()
    assert patch is not None

    index_manifests = []
    for f in patch.added_index_manifest_files:
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

  def test_empty_op_return_none(self, tmp_path):
    location = tmp_path / "dataset"
    schema = pa.schema([pa.field("int64", pa.int64())])
    storage = Storage.create(location=str(location),
                             schema=schema,
                             primary_keys=["int64"])

    op = LocalAppendOp(str(location), storage.metadata)
    assert op.finish() is None
