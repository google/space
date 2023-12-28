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
"""Local runner thats run data and metadata operations locally."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Union

from absl import logging  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.compute as pc

from space.core.ops.append import LocalAppendOp
from space.core.ops.base import InputData
from space.core.ops.change_data import ChangeType, read_change_data
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.insert import InsertOptions, LocalInsertOp
from space.core.ops.read import FileSetReadOp, ReadOptions
import space.core.proto.runtime_pb2 as runtime
from space.core.storage import Storage
from space.core.loaders.array_record import ArrayRecordIndexFn
from space.core.loaders.array_record import LocalArrayRecordLoadOp
from space.core.loaders.parquet import LocalParquetLoadOp


class BaseRunner(ABC):
  """Abstract base runner class."""

  def __init__(self, storage: Storage):
    self._storage = storage

  @abstractmethod
  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator."""

  def read_all(self,
               filter_: Optional[pc.Expression] = None,
               fields: Optional[List[str]] = None,
               snapshot_id: Optional[int] = None,
               reference_read: bool = False) -> pa.Table:
    """Read data from the dataset as an Arrow table."""
    return pa.concat_tables(
        list(self.read(filter_, fields, snapshot_id, reference_read)))

  @abstractmethod
  def append(self, data: InputData) -> runtime.JobResult:
    """Append data into the dataset."""

  @abstractmethod
  def append_from(self, source: Iterator[InputData]) -> runtime.JobResult:
    """Append data into the dataset from an iterator source."""

  @abstractmethod
  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> runtime.JobResult:
    """Append data from ArrayRecord files without copying data.
    
    TODO: to support a pattern of files to expand.

    Args:
      input_dir: the folder of ArrayRecord files.
      index_fn: a function that build index fields from each TFDS record.
    """

  @abstractmethod
  def append_parquet(self, input_dir: str) -> runtime.JobResult:
    """Append data from Parquet files without copying data.
    
    TODO: to support a pattern of files to expand.

    Args:
      input_dir: the folder of Parquet files.
    """

  def upsert(self, data: InputData) -> runtime.JobResult:
    """Upsert data into the dataset.
    
    Update existing data if primary key are found.
    """
    return self._insert(data, InsertOptions.Mode.UPSERT)

  def insert(self, data: InputData) -> runtime.JobResult:
    """Insert data into the dataset.
    
    Fail the operation if primary key are found.
    """
    return self._insert(data, InsertOptions.Mode.INSERT)

  @abstractmethod
  def _insert(self, data: InputData,
              mode: InsertOptions.Mode) -> runtime.JobResult:
    """Insert data into the dataset."""

  @abstractmethod
  def delete(self, filter_: pc.Expression) -> runtime.JobResult:
    """Delete data matching the filter from the dataset."""

  @abstractmethod
  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    """Read the change data between two versions.
    
    start_version is excluded; end_version is included. 
    """

  def _try_commit(self, patch: Optional[runtime.Patch]) -> runtime.JobResult:
    if patch is not None:
      self._storage.commit(patch)

    return _job_result(patch)


class LocalRunner(BaseRunner):
  """A runner that runs operations locally."""

  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    return iter(
        FileSetReadOp(
            self._storage.location, self._storage.metadata,
            self._storage.data_files(filter_, snapshot_id=snapshot_id),
            ReadOptions(filter_=filter_,
                        fields=fields,
                        reference_read=reference_read)))

  def append(self, data: InputData) -> runtime.JobResult:

    def make_iter():
      yield data

    return self.append_from(make_iter())

  def append_from(self, source: Iterator[InputData]) -> runtime.JobResult:
    op = LocalAppendOp(self._storage.location, self._storage.metadata)
    for data in source:
      op.write(data)

    return self._try_commit(op.finish())

  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> runtime.JobResult:
    op = LocalArrayRecordLoadOp(self._storage.location, self._storage.metadata,
                                input_dir, index_fn)
    return self._try_commit(op.write())

  def append_parquet(self, input_dir: str) -> runtime.JobResult:
    op = LocalParquetLoadOp(self._storage.location, self._storage.metadata,
                            input_dir)
    return self._try_commit(op.write())

  def _insert(self, data: InputData,
              mode: InsertOptions.Mode) -> runtime.JobResult:
    op = LocalInsertOp(self._storage, InsertOptions(mode=mode))
    return self._try_commit(op.write(data))

  def delete(self, filter_: pc.Expression) -> runtime.JobResult:
    ds = self._storage
    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         ds.data_files(filter_), filter_)
    return self._try_commit(op.delete())

  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    start_snapshot_id, end_snapshot_id = None, None

    if isinstance(start_version, int):
      start_snapshot_id = start_version

    if isinstance(end_version, int):
      end_snapshot_id = end_version

    if start_snapshot_id is None:
      raise RuntimeError(f"Start snapshot ID is invalid: {start_version}")

    if end_snapshot_id is None:
      raise RuntimeError(f"End snapshot ID is invalid: {end_version}")

    return read_change_data(self._storage, start_snapshot_id, end_snapshot_id)


def _job_result(patch: Optional[runtime.Patch]) -> runtime.JobResult:
  if patch is None:
    result = runtime.JobResult(state=runtime.JobResult.State.SKIPPED)
  else:
    # TODO: to catch failures and report failed state.
    result = runtime.JobResult(
        state=runtime.JobResult.State.SUCCEEDED,
        storage_statistics_update=patch.storage_statistics_update)

  logging.info(f"Job result:\n{result}")
  return result
