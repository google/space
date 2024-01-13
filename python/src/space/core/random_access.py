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
from space.core.schema.utils import file_path_field_name, row_id_field_name
from space.core.storage import Storage
from space.core.utils import errors
from space.core.utils.paths import StoragePathsMixin
from space.core.serializers import DictSerializer, FieldSerializer
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
               addresses: pa.Table,
               feature_fields: Dict[str, Storage],
               serializer: Optional[DictSerializer] = None):
    self._addresses = addresses
    self._feature_fields = feature_fields
    self._serializer = serializer

  def __len__(self) -> int:
    return self._addresses.num_rows

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def __getitem__(self, index: int) -> Any:  # type: ignore[override]
    if isinstance(index, list):
      return self.__getitems__(index)

    results: Dict[str, bytes] = {}
    for field, storage in self._feature_fields.items():
      file_path = self._addresses[file_path_field_name(field)][index].as_py()
      row_ids = [self._addresses[row_id_field_name(field)][index].as_py()]
      records: List[bytes] = read_record_file(storage.full_path(file_path),
                                              row_ids)
      assert len(records) == 1
      results[field] = records[0]

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
    for field, storage in self._feature_fields.items():
      file_paths = self._addresses[file_path_field_name(field)].take(
          indexes)  # type: ignore[arg-type]
      row_ids = self._addresses[row_id_field_name(field)].take(
          indexes)  # type: ignore[arg-type]
      addresses = pa.Table.from_arrays(
          [file_paths, row_ids],  # type: ignore[arg-type]
          [constants.FILE_PATH_FIELD, constants.ROW_ID_FIELD])

      records = read_record_column(storage, addresses).to_pylist()
      assert len(records) == addresses.num_rows
      results[field] = records

    return self._maybe_remove_dict(results if self._serializer is None else self
                                   ._serializer.deserialize(results))

  def _maybe_remove_dict(self, values: Dict[str, Any]) -> Any:
    if len(self._feature_fields) == 1:
      return list(values.values())[0]

    return values


class RandomAccessDataSource(AbcSequence):
  """Random access data source using Space storage."""

  # pylint: disable=too-many-arguments
  def __init__(self,
               feature_fields: Dict[str, Union[str, Storage]],
               addresses: Optional[pa.Table] = None,
               deserialize: bool = False,
               use_array_record_data_source=True):
    """
    TODO: to support a filter on index fields.

    Args:
      feature_fields: the record field containing data to read, in form of
        {<field>: <storage>}, where <storage> is the storage containing the
        field.
      addresses: a table of feature field addresses
      deserialize: deserialize output data if true, otherwise return bytes.
      use_array_record_data_source: use ArrayRecordDataSource if true, it only
        supports one feature field.
    """
    storages: Dict[str, Storage] = {}
    for field, storage_or_location in feature_fields.items():
      if isinstance(storage_or_location, str):
        s = Storage.load(storage_or_location)
        feature_fields[field] = s
        storages[storage_or_location] = s
      else:
        storages[storage_or_location.location] = storage_or_location

    serializers: Dict[str, FieldSerializer] = {}
    if deserialize:
      for field, s in feature_fields.items():  # type: ignore[assignment]
        serializer = s.serializer().field_serializer(field)
        if serializer is None:
          raise errors.UserInputError(
              f"Feature field {field} in {s.location} must be a record field")

        serializers[field] = serializer  # type: ignore[assignment]

    storage: Optional[Storage] = None
    if len(storages) == 1:
      storage = list(storages.values())[0]

    # Today the support of multiple storages is for external addresses
    # scenarios only, for example, addresses as the output of a JOIN.
    if addresses is None:
      if storage is None:
        raise errors.UserInputError(
            f"Expect exactly one storage, found {len(storages)}")

      addresses = self._read_addresses(storage, list(feature_fields.keys()))

    addresses = addresses.flatten()

    # Today ArrayRecordDataSource only support a single feature field.
    if use_array_record_data_source and len(feature_fields) == 1:
      assert storage is not None
      field = list(feature_fields.keys())[0]
      file_paths = addresses.column(file_path_field_name(field))
      row_ids = addresses.column(row_id_field_name(field))
      file_instructions = build_file_instructions(
          storage, file_paths, row_ids)  # type: ignore[arg-type]
      self._data_source = ArrayRecordDataSource(
          file_instructions, serializers[field] if deserialize else None)
    else:
      self._data_source = ArrowDataSource(
          addresses,
          feature_fields,  # type: ignore[arg-type]
          DictSerializer(serializers)
          if deserialize else None)  # type: ignore[assignment]

    self._length = len(self._data_source)

  def __len__(self) -> int:
    return self._data_source.__len__()

  def __iter__(self) -> Iterator:
    return self._data_source.__iter__()

  def __getitem__(self, record_key: int) -> Any:  # type: ignore[override]
    return self._data_source.__getitem__(record_key)

  def __getitems__(self, record_keys: Sequence[int]) -> Sequence[Any]:
    return self._data_source.__getitems__(record_keys)

  def _read_addresses(self, storage: Storage, fields: List[str]) -> pa.Table:
    """Read index and references (record address) columns.
    
    TODO: to read index fields for supporting filters.
    """
    addresses = Dataset(storage).local().read_all(fields=fields,
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

    # All records in the file, including deleted.
    num_records = indexes[-1] + 1
    start = indexes[0]
    for i, idx in enumerate(indexes):
      if i == len(indexes) - 1:
        file_instructions.append(
            _file_instruction(full_file_path, start, idx, num_records))
      elif indexes[i + 1] > idx + 1:
        file_instructions.append(
            _file_instruction(full_file_path, start, idx, num_records))
        start = indexes[i + 1]

  return file_instructions


def _file_instruction(file_path: str, start: int, end: int,
                      total_records) -> shard_utils.FileInstruction:
  return shard_utils.FileInstruction(filename=file_path,
                                     skip=start,
                                     take=end - start + 1,
                                     examples_in_shard=total_records)
