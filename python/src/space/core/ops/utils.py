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

from typing import List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from space.core.schema import arrow
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.utils import errors


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


def primary_key_filter(primary_keys: List[str],
                       data: pa.Table) -> Optional[pc.Expression]:
  """Return a filter that match the given primary keys in the input data.
  
  Raise an error if data contain duplicated primary keys.
  """
  columns = []
  for key in primary_keys:
    columns.append(data.column(key).combine_chunks())

  filter_ = None
  filter_strs: Set[str] = set()
  for i_row in range(data.num_rows):
    row_filter = None
    for i_col, key in enumerate(primary_keys):
      new_filter = pc.field(key) == columns[i_col][i_row]
      if row_filter is None:
        row_filter = new_filter
      else:
        row_filter &= new_filter

    # TODO: a simple method of detecting duplicated primary keys. To find a
    # more efficient method.
    filter_str = str(row_filter)
    if filter_str in filter_strs:
      raise errors.PrimaryKeyExistError(
          f"Found duplicated primary key: {filter_str}")

    filter_strs.add(filter_str)

    if filter_ is None:
      filter_ = row_filter
    else:
      filter_ |= row_filter

  return filter_


def merge_patches(patches: List[Optional[rt.Patch]]) -> Optional[rt.Patch]:
  """Merge multiple patches into one."""
  patch = rt.Patch()
  stats_update = meta.StorageStatistics()

  empty = True
  for p in patches:
    if p is None:
      continue

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
