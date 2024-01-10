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
from functools import wraps
from typing import Callable, Iterator, List, Optional, Tuple, Union

from absl import logging  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.compute as pc

from space.core.jobs import JobResult
from space.core.loaders.array_record import ArrayRecordIndexFn
from space.core.loaders.array_record import LocalArrayRecordLoadOp
from space.core.loaders.parquet import LocalParquetLoadOp
from space.core.ops.append import LocalAppendOp
from space.core.ops.append import FileOptions
from space.core.ops.base import InputData, InputIteratorFn
from space.core.ops.change_data import ChangeType, read_change_data
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.insert import InsertOptions, LocalInsertOp
from space.core.ops.read import FileSetReadOp
from space.core.options import JoinOptions, ReadOptions
import space.core.proto.runtime_pb2 as rt
from space.core.storage import Storage, Version
from space.core.utils import errors


class BaseReadOnlyRunner(ABC):
  """Abstract base read-only runner class."""

  # pylint: disable=too-many-arguments
  @abstractmethod
  def read(
      self,
      filter_: Optional[pc.Expression] = None,
      fields: Optional[List[str]] = None,
      version: Optional[Version] = None,
      reference_read: bool = False,
      join_options: JoinOptions = JoinOptions()
  ) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator."""

  # pylint: disable=too-many-arguments
  def read_all(
      self,
      filter_: Optional[pc.Expression] = None,
      fields: Optional[List[str]] = None,
      version: Optional[Version] = None,
      reference_read: bool = False,
      join_options: JoinOptions = JoinOptions()
  ) -> Optional[pa.Table]:
    """Read data from the dataset as an Arrow table."""
    all_data = []
    for data in self.read(filter_, fields, version, reference_read,
                          join_options):
      if data.num_rows > 0:
        all_data.append(data)

    if not all_data:
      return None

    return pa.concat_tables(all_data)

  @abstractmethod
  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    """Read the change data between two versions.
    
    start_version is excluded; end_version is included. 
    """


class StorageMixin:
  """Provide storage commit utilities."""

  def __init__(self, storage: Storage):
    self._storage = storage

  @staticmethod
  def transactional(
      fn: Callable[..., Optional[rt.Patch]]) -> Callable[..., JobResult]:
    """A decorator that commits the result of a data operation."""

    @wraps(fn)
    def decorated(self, *args, **kwargs):
      try:
        with self._storage.transaction() as txn:  # pylint: disable=protected-access
          txn.commit(fn(self, *args, **kwargs))
          r = txn.result()
          logging.info(f"Job result:\n{r}")
          return r
      except (errors.SpaceRuntimeError, errors.UserInputError) as e:
        r = JobResult(JobResult.State.FAILED, None, repr(e))
        logging.warning(f"Job result:\n{r}")
        return r

    return decorated

  @staticmethod
  def reload(fn: Callable) -> Callable:
    """A decorator that reloads the storage before a data operation."""

    @wraps(fn)
    def decorated(self, *args, **kwargs):
      self._storage.reload()  # pylint: disable=protected-access
      return fn(self, *args, **kwargs)

    return decorated


class BaseReadWriteRunner(StorageMixin, BaseReadOnlyRunner):
  """Abstract base read and write runner class."""

  def __init__(self,
               storage: Storage,
               file_options: Optional[FileOptions] = None):
    StorageMixin.__init__(self, storage)
    self._file_options = FileOptions() if file_options is None else file_options

  @abstractmethod
  def append(self, data: InputData) -> JobResult:
    """Append data into the dataset."""

  @abstractmethod
  def append_from(
      self, source_fns: Union[InputIteratorFn,
                              List[InputIteratorFn]]) -> JobResult:
    """Append data into the dataset from an iterator source.
    
    source_fns contains a list of no args functions that return iterators. It
    accepts functions because iterators can not be pickled to run remotely. For
    example, `lambda: make_iter()` where `make_iter` returns an iterator.
    """

  @abstractmethod
  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> JobResult:
    """Append data from ArrayRecord files without copying data.
    
    TODO: to support a pattern of files to expand.

    Args:
      input_dir: the folder of ArrayRecord files.
      index_fn: a function that build index fields from each TFDS record.
    """

  @abstractmethod
  def append_parquet(self, input_dir: str) -> JobResult:
    """Append data from Parquet files without copying data.
    
    TODO: to support a pattern of files to expand.

    Args:
      input_dir: the folder of Parquet files.
    """

  def upsert(self, data: InputData) -> JobResult:
    """Upsert data into the dataset.
    
    Update existing data if primary key are found.
    """
    return self._insert(data, InsertOptions.Mode.UPSERT)

  def insert(self, data: InputData) -> JobResult:
    """Insert data into the dataset.
    
    Fail the operation if primary key are found.
    """
    return self._insert(data, InsertOptions.Mode.INSERT)

  @abstractmethod
  def _insert(self, data: InputData, mode: InsertOptions.Mode) -> JobResult:
    """Insert data into the dataset."""

  @abstractmethod
  def delete(self, filter_: pc.Expression) -> JobResult:
    """Delete data matching the filter from the dataset."""


class LocalRunner(BaseReadWriteRunner):
  """A runner that runs operations locally."""

  # pylint: disable=too-many-arguments
  @StorageMixin.reload
  def read(
      self,
      filter_: Optional[pc.Expression] = None,
      fields: Optional[List[str]] = None,
      version: Optional[Version] = None,
      reference_read: bool = False,
      join_options: JoinOptions = JoinOptions()
  ) -> Iterator[pa.Table]:
    snapshot_id = (None if version is None else
                   self._storage.version_to_snapshot_id(version))

    return iter(
        FileSetReadOp(
            self._storage.location, self._storage.metadata,
            self._storage.data_files(filter_, snapshot_id=snapshot_id),
            ReadOptions(filter_, fields, reference_read=reference_read)))

  @StorageMixin.reload
  def diff(self, start_version: Version,
           end_version: Version) -> Iterator[Tuple[ChangeType, pa.Table]]:
    return read_change_data(self._storage,
                            self._storage.version_to_snapshot_id(start_version),
                            self._storage.version_to_snapshot_id(end_version))

  @StorageMixin.transactional
  def append(self, data: InputData) -> Optional[rt.Patch]:
    op = LocalAppendOp(self._storage.location, self._storage.metadata,
                       self._file_options)
    op.write(data)
    return op.finish()

  @StorageMixin.transactional
  def append_from(
      self, source_fns: Union[InputIteratorFn, List[InputIteratorFn]]
  ) -> Optional[rt.Patch]:
    op = LocalAppendOp(self._storage.location, self._storage.metadata,
                       self._file_options)
    if not isinstance(source_fns, list):
      source_fns = [source_fns]

    for source_fn in source_fns:
      for data in source_fn():
        op.write(data)

    return op.finish()

  @StorageMixin.transactional
  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> Optional[rt.Patch]:
    op = LocalArrayRecordLoadOp(self._storage.location, self._storage.metadata,
                                input_dir, index_fn, self._file_options)
    return op.write()

  @StorageMixin.transactional
  def append_parquet(self, input_dir: str) -> Optional[rt.Patch]:
    op = LocalParquetLoadOp(self._storage.location, self._storage.metadata,
                            input_dir)
    return op.write()

  @StorageMixin.transactional
  def _insert(self, data: InputData,
              mode: InsertOptions.Mode) -> Optional[rt.Patch]:
    op = LocalInsertOp(self._storage, InsertOptions(mode=mode),
                       self._file_options)
    return op.write(data)

  @StorageMixin.transactional
  def delete(self, filter_: pc.Expression) -> Optional[rt.Patch]:
    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         self._storage.data_files(filter_), filter_,
                         self._file_options)
    return op.delete()
