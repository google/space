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

from space import Dataset
import space.core.proto.metadata_pb2 as meta
from space.core.fs.parquet import write_parquet_file


class TestLocalParquetLoadOp:

  def test_append_parquet(self, tmp_path):
    schema = pa.schema([
        pa.field("int64", pa.int64()),
        pa.field("float64", pa.float64()),
        pa.field("bool", pa.bool_()),
        pa.field("string", pa.string())
    ])
    ds = Dataset.create(str(tmp_path / "dataset"),
                        schema,
                        primary_keys=["int64"],
                        record_fields=[])

    input_data = [{
        "int64": [1, 2, 3],
        "float64": [0.1, 0.2, 0.3],
        "bool": [True, False, False],
        "string": ["a", "b", "c"]
    }, {
        "int64": [0, 10],
        "float64": [-0.1, 100.0],
        "bool": [False, False],
        "string": ["A", "z"]
    }]

    # Create dummy Parquet files.
    input_dir = tmp_path / "parquet"
    input_dir.mkdir(parents=True)
    write_parquet_file(str(input_dir / "file0.parquet"), schema,
                       [pa.Table.from_pydict(input_data[0])])
    write_parquet_file(str(input_dir / "file1.parquet"), schema,
                       [pa.Table.from_pydict(input_data[1])])

    runner = ds.local()
    response = runner.append_parquet(input_dir)
    assert response.storage_statistics_update == meta.StorageStatistics(
        num_rows=5,
        index_compressed_bytes=214,
        index_uncompressed_bytes=209,
        record_uncompressed_bytes=0)

    index_data = pa.concat_tables(
        (list(runner.read()))).combine_chunks().sort_by("int64")
    assert index_data == pa.concat_tables([
        pa.Table.from_pydict(d) for d in input_data
    ]).combine_chunks().sort_by("int64")
