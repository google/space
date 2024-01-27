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
"""Local read operation implementation."""

from __future__ import annotations
from abc import abstractmethod
from typing import Iterator, Dict, List, Tuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyroaring import BitMap  # type: ignore[import-not-found]

from space.core.options import ReadOptions
from space.core.fs.array_record import read_record_file
from space.core.ops.base import BaseOp
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.schema import arrow
from space.core.schema.constants import FILE_PATH_FIELD, ROW_ID_FIELD
from space.core.schema import utils as schema_utils
from space.core.utils.paths import StoragePathsMixin

_RECORD_KEY_FIELD = "__RECORD_KEY"


class BaseReadOp(BaseOp):
  """Abstract base read operation class."""

  @abstractmethod
  def __iter__(self) -> Iterator[pa.Table]:
    """Iterator of read data."""


class FileSetReadOp(BaseReadOp, StoragePathsMixin):
  """Read operation of a given file set running locally.
  
  It can be used as components of more complex operations and distributed
  read operation.

  Not thread safe.
  """

  # pylint: disable=too-many-arguments
  def __init__(self,
               location: str,
               metadata: meta.StorageMetadata,
               file_set: rt.FileSet,
               options: Optional[ReadOptions] = None):
    StoragePathsMixin.__init__(self, location)

    # TODO: to validate that filter_ does not contain record fields.

    self._metadata = metadata
    self._file_set = file_set

    # TODO: to validate options, e.g., fields are valid.
    self._options = options or ReadOptions()

    record_fields = set(self._metadata.schema.record_fields)
    self._physical_schema = arrow.arrow_schema(self._metadata.schema.fields,
                                               record_fields,
                                               physical=True)

    if self._options.fields is None:
      self._selected_fields = [f.name for f in self._physical_schema]
    else:
      self._selected_fields = self._options.fields

    self._index_fields, self._record_fields = arrow.classify_fields(
        self._physical_schema,
        record_fields,
        selected_fields=set(self._selected_fields))

    self._index_field_ids = set(schema_utils.field_ids(self._index_fields))

    self._record_fields_dict: Dict[int, schema_utils.Field] = {}
    for f in self._record_fields:
      self._record_fields_dict[f.field_id] = f

  def __iter__(self) -> Iterator[pa.Table]:
    for file in self._file_set.index_files:
      row_range_read = file.HasField("row_slice")

      # TODO: always loading the whole table is inefficient, to only load the
      # required row groups.
      index_data = pq.read_table(
          self.full_path(file.path),
          columns=self._selected_fields,
          filters=self._options.filter_)  # type: ignore[arg-type]

      if file.HasField("row_bitmap") and not file.row_bitmap.all_rows:
        index_data = index_data.filter(mask=_bitmap_mask(
            file.row_bitmap.roaring_bitmap, index_data.num_rows,
            file.row_slice if row_range_read else None))
      elif row_range_read:
        length = file.row_slice.end - file.row_slice.start
        index_data = index_data.slice(file.row_slice.start, length)

      if self._options.reference_read:
        yield index_data
        continue

      index_column_ids: List[int] = []
      record_columns: List[Tuple[int, pa.Field]] = []
      for column_id, field in enumerate(index_data.schema):
        field_id = arrow.field_id(field)
        if field_id in self._index_field_ids or field_id == arrow.NULL_FIELD_ID:
          index_column_ids.append(column_id)
        else:
          record_columns.append(
              (column_id,
               arrow.binary_field(self._record_fields_dict[field_id])))

      if row_range_read:
        # The batch size is already applied via row slice range.
        yield self._read_index_and_record(index_data, index_column_ids,
                                          record_columns)
      else:
        for batch in index_data.to_batches(
            max_chunksize=self._options.batch_size):
          yield self._read_index_and_record(pa.table(batch), index_column_ids,
                                            record_columns)

  def _read_index_and_record(
      self, index_data: pa.Table, index_column_ids: List[int],
      record_columns: List[Tuple[int, pa.Field]]) -> pa.Table:
    result_data = index_data.select(index_column_ids)  # type: ignore[arg-type]

    # Read record fields from addresses.
    for column_id, field in record_columns:
      result_data = result_data.append_column(
          field,
          read_record_column(
              self,
              index_data.select([column_id]),  # type: ignore[list-item]
              field.name))

    # TODO: to keep field order the same as schema.
    return result_data


def read_record_column(paths: StoragePathsMixin,
                       addresses: pa.Table,
                       field: Optional[str] = None) -> pa.BinaryArray:
  """Read rows in multiple ArrayRecord files specified in a record address
  table.
  
  Args:
    paths: provide `full_path` method.
    addresses: addresses of record rows to read.
    field: the field name to read, if `addresses` has field name prefix.
  """
  num_rows = addresses.num_rows
  # _RECORD_KEY_FIELD is the row index of addresses used for retrieving rows
  # after group by. It is not in the read result.
  addresses = addresses.flatten().append_column(
      _RECORD_KEY_FIELD, [np.arange(num_rows)])  # type: ignore[arg-type]

  if field is None:
    file_path_field = f"{FILE_PATH_FIELD}"
    row_id_field = f"{ROW_ID_FIELD}"
  else:
    file_path_field = f"{field}.{FILE_PATH_FIELD}"
    row_id_field = f"{field}.{ROW_ID_FIELD}"

  # Record row IDs and records key co-grouped by file path, for processing
  # one file at a time to minimize file reads.
  grouped_records = addresses.group_by(
      file_path_field).aggregate(  # type: ignore[arg-type]
          [(row_id_field, "list"), (_RECORD_KEY_FIELD, "list")])

  file_path_column = grouped_records.column(file_path_field).combine_chunks()
  row_ids_column = grouped_records.column(
      f"{row_id_field}_list").combine_chunks()
  record_keys_column = grouped_records.column(
      f"{_RECORD_KEY_FIELD}_list").combine_chunks()

  # TODO: to parallelize ArrayRecord file reads.
  record_values: List[List[bytes]] = []
  for file_path, row_ids in zip(file_path_column,
                                row_ids_column):  # type: ignore[call-overload]
    record_values.append(
        read_record_file(paths.full_path(file_path.as_py()), row_ids.as_py()))

  # Sort records by record_keys so the records can match indexes.
  sorted_values: List[bytes] = [None] * num_rows  # type: ignore[list-item]
  for values, keys in zip(record_values,
                          record_keys_column):  # type: ignore[call-overload]
    for value, key in zip(values, keys):
      sorted_values[key.as_py()] = value

  return pa.array(sorted_values, pa.binary())  # type: ignore[return-value]


def _bitmap_mask(serialized_bitmap: bytes, num_rows: int,
                 row_slice: Optional[rt.DataFile.Range]) -> List[bool]:
  bitmap = BitMap.deserialize(serialized_bitmap)

  mask = [False] * num_rows
  for row_id in bitmap.to_array():
    if row_slice is None or row_slice.start <= row_id < row_slice.end:
      mask[row_id] = True

  return mask
