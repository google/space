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
"""TFDS random access data source using Space dataset."""

from collections.abc import Sequence as AbcSequence
from typing import Any, List, Sequence, Union

from absl import logging  # type: ignore[import-untyped]
import pyarrow as pa
# pylint: disable=line-too-long
from tensorflow_datasets.core.utils import shard_utils  # type: ignore[import-untyped]
from tensorflow_datasets.core.utils.lazy_imports_utils import array_record_data_source as ards  # type: ignore[import-untyped]

from space.core.datasets import Dataset
from space.core.schema import constants


class SpaceDataSource(AbcSequence):
  """TFDS random access data source using Space dataset."""

  def __init__(self, ds_or_location: Union[Dataset, str],
               feature_fields: List[str]):
    """
    TODO: to support a filder on index fields.
    
    Args:
      ds_or_location: a dataset object or location.
      feature_fields: the record field containing data to read.
    """
    # TODO: to auto infer feature_fields.
    if isinstance(ds_or_location, str):
      self._ds = Dataset.load(ds_or_location)
    else:
      self._ds = ds_or_location

    assert len(feature_fields) == 1, "Only support one feature field"
    self._feature_field: str = feature_fields[0]
    # TODO: to verify the feature field is a record field?

    self._serializer = self._ds.serializer().field_serializer(
        self._feature_field)
    # pylint: disable=line-too-long
    assert self._serializer is not None, f"Feature field {self._feature_field} must be a record field"

    self._data_source = ards.ArrayRecordDataSource(
        self._file_instructions(self._read_index_and_address()))
    self._length = len(self._data_source)

  def __len__(self) -> int:
    return self._length

  def __iter__(self):
    for i in range(self._length):
      yield self[i]

  def __getitem__(self, record_key: int) -> Any:  # type: ignore[override]
    if not isinstance(record_key, int):
      logging.error(
          "Calling ArrayRecordDataSource.__getitem__() with sequence "
          "of record keys (%s) is deprecated. Either pass a single "
          "integer or switch to __getitems__().",
          record_key,
      )

      return self.__getitems__(record_key)

    record = self._data_source[record_key]
    return self._serializer.deserialize(record)  # type: ignore[union-attr]

  def __getitems__(self, record_keys: Sequence[int]) -> Sequence[Any]:
    records = self._data_source.__getitems__(record_keys)
    if len(record_keys) != len(records):
      raise IndexError(f"Requested {len(record_keys)} records but got"
                       f" {len(records)} records."
                       f"{record_keys=}, {records=}")

    return [
        self._serializer.deserialize(record)  # type: ignore[union-attr]
        for record in records
    ]

  def _read_index_and_address(self) -> pa.Table:
    """Read index and record address columns.
    
    TODO: to read index for filters.
    """
    return self._ds.local().read_all(fields=[self._feature_field],
                                     reference_read=True)

  def _file_instructions(
      self, record_addresses: pa.Table) -> List[shard_utils.FileInstruction]:
    """Convert record addresses to ArrayRecord file instructions."""
    record_address_table = pa.Table.from_arrays(
        record_addresses.column(
            self._feature_field).flatten(),  # type: ignore[arg-type]
        [constants.FILE_PATH_FIELD, constants.ROW_ID_FIELD])
    aggregated = record_address_table.group_by(
        constants.FILE_PATH_FIELD).aggregate([(constants.ROW_ID_FIELD, "list")
                                              ]).to_pydict()

    file_instructions = []
    for file_path, indexes in zip(
        aggregated[constants.FILE_PATH_FIELD],
        aggregated[f"{constants.ROW_ID_FIELD}_list"]):
      full_file_path = self._ds.storage.full_path(file_path)  # pylint: disable=protected-access
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
