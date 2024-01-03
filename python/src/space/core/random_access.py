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
"""Random access data source using Space storage."""

from __future__ import annotations
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pyarrow as pa
# pylint: disable=line-too-long
from tensorflow_datasets.core.utils import shard_utils  # type: ignore[import-untyped]
from tensorflow_datasets.core.utils.lazy_imports_utils import array_record_data_source as ards  # type: ignore[import-untyped]

from space.core.datasets import Dataset
from space.core.schema import constants
from space.core.storage import Storage
from space.core.utils import errors
from space.core.utils.paths import StoragePathsMixin
from space.core.serializers import FieldSerializer
from space.core.ops.read import read_record_column
from space.core.fs.array_record import read_record_file


class ArrayRecordDataSource(AbcSequence):
  """Data source built upon ArrayRecord data source implementation."""

  def __init__(self,
               file_instructions: List[shard_utils.FileInstruction],
               serializer: Optional[FieldSerializer] = None):
    self._data_source = ards.ArrayRecordDataSource(file_instructions)
    self._serializer = serializer

  def __len__(self) -> int:
    return len(self._data_source)

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def __getitem__(self, index: int) -> Any:  # type: ignore[override]
    if isinstance(index, list):
      return self.__getitems__(index)

    return self._maybe_deserialize(
        self._data_source[index])  # type: ignore[union-attr]

  def __getitems__(self, indexes: Sequence[int]) -> Sequence[Any]:
    records = self._data_source.__getitems__(indexes)
    if len(indexes) != len(records):
      raise IndexError(f"Requested {len(indexes)} records but got"
                       f" {len(records)} records. {indexes=}, {records=}")

    return [
        self._maybe_deserialize(record)  # type: ignore[union-attr]
        for record in records
    ]

  def _maybe_deserialize(self, record: bytes) -> Any:
    if self._serializer is None:
      return record

    return self._serializer.deserialize(record)  # type: ignore[union-attr]


class ArrowDataSource(AbcSequence):
  """Data source using addresses stored in Arrow table."""

  def __init__(self,
               storage: Storage,
               addresses: pa.Table,
               feature_fields: List[str],
               deserialize: bool = False):
    self._storage = storage
    self._addresses = addresses

    self._feature_fields = feature_fields

    self._serializer = None
    if deserialize:
      self._serializer = self._storage.serializer()

  def __len__(self) -> int:
    return self._addresses.num_rows

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def __getitem__(self, index: int) -> Any:  # type: ignore[override]
    if isinstance(index, list):
      return self.__getitems__(index)

    results: Dict[str, bytes] = {}
    for field_name in self._feature_fields:
      file_path = self._addresses[_file_path(field_name)][index].as_py()
      row_ids = [self._addresses[_row_id(field_name)][index].as_py()]
      records: List[bytes] = read_record_file(
          self._storage.full_path(file_path), row_ids)
      assert len(records) == 1
      results[field_name] = records[0]

    if self._serializer is None:
      return self._maybe_remove_dict(results)

    return self._maybe_remove_dict({
        f:
        self._serializer.field_serializer(f  # type: ignore[union-attr]
                                          ).deserialize(v)
        for f, v in results.items()
    })

  def __getitems__(self, indexes: Sequence[int]) -> Sequence[Any]:
    results: Dict[str, List[bytes]] = {}
    for field_name in self._feature_fields:
      file_paths = self._addresses[_file_path(field_name)].take(
          indexes)  # type: ignore[arg-type]
      row_ids = self._addresses[_row_id(field_name)].take(
          indexes)  # type: ignore[arg-type]
      addresses = pa.Table.from_arrays(
          [file_paths, row_ids],  # type: ignore[arg-type]
          [constants.FILE_PATH_FIELD, constants.ROW_ID_FIELD])

      records = read_record_column(self._storage, addresses).to_pylist()
      assert len(records) == addresses.num_rows
      results[field_name] = records

    return self._maybe_remove_dict(results if self._serializer is None else
                                   self._serializer.deserialize(results))

  def _maybe_remove_dict(self, values: Dict[str, Any]) -> Any:
    if len(self._feature_fields) == 1:
      return list(values.values())[0]

    return values


class RandomAccessDataSource(AbcSequence):
  """Random access data source using Space storage."""

  # pylint: disable=too-many-arguments
  def __init__(self,
               storage_or_location: Union[str, Storage],
               feature_fields: List[str],
               addresses: Optional[pa.Table] = None,
               deserialize: bool = False,
               use_array_record_data_source=True):
    """
    TODO: to support a filter on index fields.

    Args:
      storage: the Space dataset or materialized view storage.
      feature_fields: the record field containing data to read.
      addresses: a table of feature field addresses
      deserialize: deserialize output data if true, otherwise return bytes.
    """
    if isinstance(storage_or_location, str):
      storage = Storage.load(storage_or_location)
    else:
      storage = storage_or_location

    serializers: Dict[str, FieldSerializer] = {}
    if deserialize:
      for field_name in feature_fields:
        serializer = storage.serializer().field_serializer(field_name)
        if serializer is None:
          raise errors.UserInputError(
              f"Feature field {field_name} must be a record field")

        serializers[field_name] = serializer  # type: ignore[assignment]

    if addresses is None:
      addresses = self._read_addresses(storage, feature_fields)

    addresses = addresses.flatten()

    if use_array_record_data_source and len(feature_fields) == 1:
      field_name = feature_fields[0]
      file_paths = addresses.column(_file_path(field_name))
      row_ids = addresses.column(_row_id(field_name))
      file_instructions = build_file_instructions(
          storage, file_paths, row_ids)  # type: ignore[arg-type]
      self._data_source = ArrayRecordDataSource(
          file_instructions, serializers[field_name] if deserialize else None)
    else:
      self._data_source = ArrowDataSource(
          storage, addresses, feature_fields,
          deserialize)  # type: ignore[assignment]

    self._length = len(self._data_source)

  def __len__(self) -> int:
    return self._data_source.__len__()

  def __iter__(self) -> Iterator:
    return self._data_source.__iter__()

  def __getitem__(self, record_key: int) -> Any:  # type: ignore[override]
    return self._data_source.__getitem__(record_key)

  def __getitems__(self, record_keys: Sequence[int]) -> Sequence[Any]:
    return self._data_source.__getitems__(record_keys)

  def _read_addresses(self, storage: Storage,
                      feature_fields: List[str]) -> pa.Table:
    """Read index and references (record address) columns.
    
    TODO: to read index fields for supporting filters.
    """
    addresses = Dataset(storage).local().read_all(fields=feature_fields,
                                                  reference_read=True)

    if addresses is None:
      raise errors.UserInputError(f"Storage has no data: {storage.location}")

    return addresses


def build_file_instructions(
    paths: StoragePathsMixin, file_paths: pa.StringArray,
    row_ids: pa.Int32Array) -> List[shard_utils.FileInstruction]:
  """Convert record addresses to ArrayRecord file instructions.
  
  record_addresses must be fattened.
  """
  addresses = pa.Table.from_arrays(
      [file_paths, row_ids],  # type: ignore[arg-type]
      [constants.FILE_PATH_FIELD, constants.ROW_ID_FIELD])
  aggregated = addresses.group_by(constants.FILE_PATH_FIELD).aggregate([
      (constants.ROW_ID_FIELD, "list")
  ]).to_pydict()

  file_instructions = []
  for file_path, indexes in zip(aggregated[constants.FILE_PATH_FIELD],
                                aggregated[f"{constants.ROW_ID_FIELD}_list"]):
    full_file_path = paths.full_path(file_path)
    if not indexes:
      continue

    indexes.sort()

    previous_idx = indexes[0]
    start, end = indexes[0], indexes[-1]
    for idx in indexes:
      if idx not in (0, previous_idx + 1) or idx == end:
        if idx == end:
          previous_idx = idx

        # TODO: to populate FileInstruction.examples_in_shard.
        file_instructions.append(
            shard_utils.FileInstruction(filename=full_file_path,
                                        skip=start,
                                        take=previous_idx - start + 1,
                                        examples_in_shard=end + 1))
        start = idx
      previous_idx = idx

  return file_instructions


def _file_path(field: str) -> str:
  return f"{field}.{constants.FILE_PATH_FIELD}"


def _row_id(field: str) -> str:
  return f"{field}.{constants.ROW_ID_FIELD}"
