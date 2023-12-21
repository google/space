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
"""Utilities for operation classes."""

import numpy as np
import pyarrow as pa

from space.core.schema import arrow
from space.core.proto import metadata_pb2 as meta


def update_index_storage_stats(
    base: meta.StorageStatistics,
    update: meta.StorageStatistics,
) -> None:
  """Update index storage statistics."""
  base.num_rows += update.num_rows
  base.index_compressed_bytes += update.index_compressed_bytes
  base.index_uncompressed_bytes += update.index_uncompressed_bytes


def update_record_stats_bytes(base: meta.StorageStatistics,
                              update: meta.StorageStatistics) -> None:
  """Update record storage statistics."""
  base.record_uncompressed_bytes += update.record_uncompressed_bytes


def address_column(file_path: str, start_row: int,
                   num_rows: int) -> pa.StructArray:
  """Construct an record address column by a file path and row ID range."""
  return pa.StructArray.from_arrays(
      [
          [file_path] * num_rows,  # type: ignore[arg-type]
          np.arange(start_row, start_row + num_rows, dtype=np.int32)
      ],
      fields=arrow.record_address_types())  # type: ignore[arg-type]
