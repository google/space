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
"""Ray runner implementations."""

from __future__ import annotations
import copy
from functools import partial
from typing import TYPE_CHECKING
from typing import Iterator, List, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import ray

from space.core.jobs import JobResult
from space.core.loaders.array_record import ArrayRecordIndexFn
from space.core.runners import BaseReadOnlyRunner, BaseReadWriteRunner
from space.core.runners import StorageMixin
from space.core.ops import utils
from space.core.ops.base import InputData, InputIteratorFn
from space.core.ops.change_data import ChangeData, ChangeType
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.insert import InsertOptions
from space.core.options import FileOptions, JoinOptions, ReadOptions
import space.core.proto.runtime_pb2 as rt
from space.core.utils import errors
from space.ray.ops.append import RayAppendOp
from space.ray.ops.change_data import read_change_data
from space.ray.ops.delete import RayDeleteOp
from space.ray.ops.insert import RayInsertOp
from space.ray.ops.utils import iter_batches, singleton_storage
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.datasets import Dataset
  from space.core.storage import Storage, Transaction, Version
  from space.core.views import MaterializedView, View


class RayReadOnlyRunner(BaseReadOnlyRunner):
  """A read-only Ray runner."""

  def __init__(self, view: View, ray_options: Optional[RayOptions]):
    self._view = view
    self._ray_options = ray_options or RayOptions()

  # pylint: disable=too-many-arguments
  def read(
      self,
      filter_: Optional[pc.Expression] = None,
      fields: Optional[List[str]] = None,
      version: Optional[Version] = None,
      reference_read: bool = False,
      batch_size: Optional[int] = None,
      join_options: JoinOptions = JoinOptions()
  ) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator.
    
    The view runner applies transforms on top of source dataset. It always
    transforms the whole dataset using Ray.

    Dataset itself is a special view without any transforms.
    """
    # Reload all sources because there are two sources for join.
    for ds in self._view.sources.values():
      ds.storage.reload()

    snapshot_id = (None if version is None else
                   self._source_storage.version_to_snapshot_id(version))
    read_options = ReadOptions(filter_, fields, snapshot_id, reference_read,
                               batch_size)

    return iter_batches(
        self._view._ray_dataset(self._ray_options, read_options, join_options))  # pylint: disable=protected-access

  def diff(self,
           start_version: Union[Version],
           end_version: Union[Version],
           batch_size: Optional[int] = None) -> Iterator[ChangeData]:
    for change in self.diff_ray(start_version, end_version, batch_size):
      assert isinstance(change.data, list)
      for ds in change.data:
        assert isinstance(ds, ray.data.Dataset)
        for data in iter_batches(ds):
          yield ChangeData(change.snapshot_id, change.type_, data)

  def diff_ray(self,
               start_version: Union[Version],
               end_version: Union[Version],
               batch_size: Optional[int] = None) -> Iterator[ChangeData]:
    """Return diff data in form of a list of Ray datasets."""
    self._source_storage.reload()
    source_changes = read_change_data(
        self._source_storage,
        self._source_storage.version_to_snapshot_id(start_version),
        self._source_storage.version_to_snapshot_id(end_version),
        self._ray_options, ReadOptions(batch_size=batch_size))

    for change in source_changes:
      if change.type_ == ChangeType.DELETE:
        yield change

      elif change.type_ == ChangeType.ADD:
        # Change data is a list of Ray datasets, because of parallel read
        # streams. It allows us to do parallel transforms here.
        assert isinstance(change.data, list)
        processed_data: List[ray.data.Dataset] = []
        for ds in change.data:
          assert isinstance(ds, ray.data.Dataset)
          processed_data.append(self._view.process_source(ds))

        yield ChangeData(change.snapshot_id, change.type_, processed_data)

      else:
        raise NotImplementedError(f"Change type {change.type_} not supported")

  @property
  def _source_storage(self) -> Storage:
    """Obtain the single storage of the source dataset, never write to it."""
    return singleton_storage(self._view)


class RayMaterializedViewRunner(RayReadOnlyRunner, StorageMixin):
  """Ray runner for materialized views."""

  def __init__(self, mv: MaterializedView, ray_options: Optional[RayOptions],
               file_options: Optional[FileOptions]):
    RayReadOnlyRunner.__init__(self, mv.view, ray_options)
    StorageMixin.__init__(self, mv.storage)
    self._file_options = file_options or FileOptions()
    self._ray_options = ray_options or RayOptions()

  # pylint: disable=too-many-arguments
  @StorageMixin.reload
  def read(
      self,
      filter_: Optional[pc.Expression] = None,
      fields: Optional[List[str]] = None,
      version: Optional[Version] = None,
      reference_read: bool = False,
      batch_size: Optional[int] = None,
      join_options: JoinOptions = JoinOptions()
  ) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator.
    
    Different from view's default runner (RayReadOnlyRunner), a runner of
    materialized view always reads from the storage to avoid transforming the
    source dataset, to save computation cost.

    The result may be stale, call refresh(...) to bring the MV up-to-date.

    To use RayReadOnlyRunner, use `runner = mv.view.ray()` instead of
    `mv.ray()`.
    """
    snapshot_id = (None if version is None else
                   self._storage.version_to_snapshot_id(version))
    read_options = ReadOptions(filter_, fields, snapshot_id, reference_read,
                               batch_size)
    return iter_batches(
        self._storage.ray_dataset(self._ray_options, read_options))

  def refresh(self,
              target_version: Optional[Version] = None,
              batch_size: Optional[int] = None) -> List[JobResult]:
    """Refresh the materialized view by synchronizing from source dataset.
    
    TODO: refreshing from a large source is slow, to save refresh state and
    resume from saved state, if any failure happens.
    """
    source_snapshot_id = self._source_storage.metadata.current_snapshot_id
    if target_version is None:
      end_snapshot_id = source_snapshot_id
    else:
      end_snapshot_id = self._source_storage.version_to_snapshot_id(
          target_version)
      if end_snapshot_id > source_snapshot_id:
        raise errors.VersionNotFoundError(
            f"Target snapshot ID {end_snapshot_id} higher than source dataset "
            f"version {source_snapshot_id}")

    start_snapshot_id = self._storage.metadata.current_snapshot_id

    job_results: List[JobResult] = []
    patches: List[Optional[rt.Patch]] = []
    previous_snapshot_id: Optional[int] = None

    txn = self._start_txn()
    for change in self.diff_ray(start_snapshot_id, end_snapshot_id, batch_size):
      assert isinstance(change.data, list)

      # Commit when changes from the same snapshot end.
      if (previous_snapshot_id is not None and
          change.snapshot_id != previous_snapshot_id):
        txn.commit(utils.merge_patches(patches))
        patches.clear()

        # Stop early when something is wrong.
        r = txn.result()
        if r.state == JobResult.State.FAILED:
          return job_results + [r]

        job_results.append(r)
        txn = self._start_txn()

      try:
        # TODO: to avoid creating a new delete/append op for each batch. The
        # output file size will be smaller than configured.
        if change.type_ == ChangeType.DELETE:
          patches.append(self._process_delete(change.data))
        elif change.type_ == ChangeType.ADD:
          patches.append(self._process_append(change.data))
        else:
          raise NotImplementedError(f"Change type {change.type_} not supported")
      except (errors.SpaceRuntimeError, errors.UserInputError) as e:
        r = JobResult(JobResult.State.FAILED, None, repr(e))
        return job_results + [r]

      previous_snapshot_id = change.snapshot_id

    if patches:
      txn.commit(utils.merge_patches(patches))
      job_results.append(txn.result())

    return job_results

  def _process_delete(self, data: List[ray.data.Dataset]) -> Optional[rt.Patch]:
    # Deletion does not use parallel read streams.
    assert len(data) == 1
    arrow_data = pa.concat_tables(iter_batches(
        data[0]))  # type: ignore[arg-type]

    filter_ = utils.primary_key_filter(self._storage.primary_keys, arrow_data)
    if filter_ is None:
      return None

    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         self._storage.data_files(filter_), filter_,
                         self._file_options)
    return op.delete()

  def _process_append(self, data: List[ray.data.Dataset]) -> Optional[rt.Patch]:
    return _append_from(self._storage,
                        [partial(iter_batches, ds) for ds in data],
                        self._ray_options, self._file_options)

  def _start_txn(self) -> Transaction:
    with self._storage.transaction() as txn:
      return txn


class RayReadWriterRunner(RayReadOnlyRunner, BaseReadWriteRunner):
  """Ray read write runner."""

  def __init__(self,
               dataset: Dataset,
               ray_options: Optional[RayOptions] = None,
               file_options: Optional[FileOptions] = None):
    RayReadOnlyRunner.__init__(self, dataset, ray_options)
    BaseReadWriteRunner.__init__(self, dataset.storage, file_options)
    self._ray_options = ray_options or RayOptions()

  @StorageMixin.transactional
  def append(self, data: InputData) -> Optional[rt.Patch]:
    op = RayAppendOp(self._storage.location, self._storage.metadata,
                     self._ray_options, self._file_options)
    op.write(data)
    return op.finish()

  @StorageMixin.transactional
  def append_from(
      self, source_fns: Union[InputIteratorFn, List[InputIteratorFn]]
  ) -> Optional[rt.Patch]:
    if not isinstance(source_fns, list):
      source_fns = [source_fns]

    ray_options = copy.deepcopy(self._ray_options)
    ray_options.max_parallelism = min(len(source_fns),
                                      ray_options.max_parallelism)

    return _append_from(self._storage, source_fns, ray_options,
                        self._file_options)

  @StorageMixin.transactional
  def append_array_record(self, pattern: str,
                          index_fn: ArrayRecordIndexFn) -> Optional[rt.Patch]:
    raise NotImplementedError(
        "append_array_record not supported yet in Ray runner")

  @StorageMixin.transactional
  def append_parquet(self, pattern: str) -> Optional[rt.Patch]:
    raise NotImplementedError("append_parquet not supported yet in Ray runner")

  @StorageMixin.transactional
  def _insert(self, data: InputData,
              mode: InsertOptions.Mode) -> Optional[rt.Patch]:
    op = RayInsertOp(self._storage, InsertOptions(mode=mode), self._ray_options,
                     self._file_options)
    return op.write(data)

  @StorageMixin.transactional
  def delete(self, filter_: pc.Expression) -> Optional[rt.Patch]:
    op = RayDeleteOp(self._storage, filter_, self._file_options)
    return op.delete()


def _append_from(storage: Storage, source_fns: Union[List[InputIteratorFn]],
                 ray_options: RayOptions,
                 file_options: FileOptions) -> Optional[rt.Patch]:
  op = RayAppendOp(storage.location, storage.metadata, ray_options,
                   file_options)
  op.write_from(source_fns)
  return op.finish()
