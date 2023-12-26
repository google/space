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

from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from space.core.schema import arrow
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as runtime


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


def primary_key_filter(schema: meta.Schema, data: pa.Table) -> pc.Expression:
  """Return a filter that match the given primary keys in the input data."""
  columns = []
  for key in schema.primary_keys:
    columns.append(data.column(key).combine_chunks())

  filter_ = pc.scalar(False)
  for i_row in range(data.num_rows):
    for i_col, key in enumerate(schema.primary_keys):
      filter_ |= (pc.field(key) == columns[i_col][i_row])

  return filter_


def merge_patches(patches: List[runtime.Patch]) -> Optional[runtime.Patch]:
  """Merge multiple patches into one."""
  patch = runtime.Patch()
  stats_update = meta.StorageStatistics()

  empty = True
  for p in patches:
    if empty:
      empty = False

    # TODO: to manually merge patches when it gets more complex.
    patch.MergeFrom(p)

    # Update statistics.
    update_index_storage_stats(stats_update, p.storage_statistics_update)
    update_record_stats_bytes(stats_update, p.storage_statistics_update)

  if empty:
    return None

  patch.storage_statistics_update.CopyFrom(stats_update)
  return patch
