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
#
"""Parquet file utilities."""

from typing import List

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_file(file_path: str, schema: pa.Schema,
                       data: List[pa.Table]) -> pq.FileMetaData:
  """Materialize a single Parquet file."""
  # TODO: currently assume this file is small, so always write a single file.
  writer = pq.ParquetWriter(file_path, schema)
  for batch in data:
    writer.write_table(batch)

  writer.close()
  return writer.writer.metadata
