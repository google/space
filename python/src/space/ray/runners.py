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
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterator, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc

from space.core.loaders.array_record import ArrayRecordIndexFn
from space.core.runners import BaseReadOnlyRunner, BaseReadWriteRunner
from space.core.runners import StorageMixin
from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.ops.base import InputData
from space.core.ops.change_data import ChangeType, read_change_data
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.insert import InsertOptions
import space.core.proto.runtime_pb2 as rt
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray
from space.core.versions.utils import version_to_snapshot_id
from space.ray.ops.append import RayAppendOp
from space.ray.ops.delete import RayDeleteOp
from space.ray.ops.insert import RayInsertOp

if TYPE_CHECKING:
  from space.core.datasets import Dataset
  from space.core.views import MaterializedView, View


@dataclass
class RayOptions:
  """Options of Ray runners."""
  parallelism: int = 2


class RayReadOnlyRunner(BaseReadOnlyRunner):
  """A read-only Ray runner."""

  def __init__(self, view: View):
    self._view = view

  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator.
    
    The view runner applies transforms on top of source dataset. It always
    transforms the whole dataset using Ray.

    Dataset itself is a special view without any transforms.
    """
    self._source_storage.reload()
    for ref in self._view.ray_dataset(filter_, fields, snapshot_id,
                                      reference_read).to_arrow_refs():
      yield ray.get(ref)

  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    self._source_storage.reload()
    source_changes = read_change_data(self._source_storage,
                                      version_to_snapshot_id(start_version),
                                      version_to_snapshot_id(end_version))
    for change_type, data in source_changes:
      # TODO: skip processing the data for deletions; the caller is usually
      # only interested at deleted primary keys.
      processed_remote_data = self._view.process_source(data)
      processed_data = ray.get(processed_remote_data.to_arrow_refs())
      yield change_type, pa.concat_tables(processed_data)

  @property
  def _source_storage(self) -> Storage:
    """Obtain storage of the source dataset, never write to it."""
    sources = self._view.sources
    assert len(sources) == 1, "Views only support a single source dataset"
    return list(sources.values())[0].storage


class RayMaterializedViewRunner(RayReadOnlyRunner, StorageMixin):
  """Ray runner for materialized views."""

  def __init__(self, mv: MaterializedView):
    RayReadOnlyRunner.__init__(self, mv.view)
    StorageMixin.__init__(self, mv.storage)

  @StorageMixin.reload
  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator.
    
    Different from view's default runner (RayReadOnlyRunner), a runner of
    materialized view always reads from the storage to avoid transforming the
    source dataset, to save computation cost.

    The result may be stale, call refresh(...) to bring the MV up-to-date.

    To use RayReadOnlyRunner, use `runner = mv.view.ray()` instead of
    `mv.ray()`.
    """
    for ref in self._storage.ray_dataset(filter_, fields, snapshot_id,
                                         reference_read).to_arrow_refs():
      yield ray.get(ref)

  @StorageMixin.transactional
  def refresh(self,
              target_version: Optional[Union[int]] = None
              ) -> Optional[rt.Patch]:
    """Refresh the materialized view by synchronizing from source dataset."""
    source_snapshot_id = self._source_storage.metadata.current_snapshot_id
    if target_version is None:
      end_snapshot_id = source_snapshot_id
    else:
      end_snapshot_id = version_to_snapshot_id(target_version)
      if end_snapshot_id > source_snapshot_id:
        raise errors.SnapshotNotFoundError(
            f"Target snapshot ID {end_snapshot_id} higher than source dataset "
            "version {source_snapshot_id}")

    start_snapshot_id = self._storage.metadata.current_snapshot_id

    patches: List[Optional[rt.Patch]] = []
    for change_type, data in self.diff(start_snapshot_id, end_snapshot_id):
      # In the scope of changes from the same snapshot, must process DELETE
      # before ADD.
      if change_type == ChangeType.DELETE:
        patches.append(self._process_delete(data))
      elif change_type == ChangeType.ADD:
        patches.append(self._process_append(data))
      else:
        raise NotImplementedError(f"Change type {change_type} not supported")

    return utils.merge_patches(patches)

  def _process_delete(self, data: pa.Table) -> Optional[rt.Patch]:
    filter_ = utils.primary_key_filter(self._storage.primary_keys, data)
    if filter_ is None:
      return None

    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         self._storage.data_files(filter_), filter_)
    return op.delete()

  def _process_append(self, data: pa.Table) -> Optional[rt.Patch]:
    op = LocalAppendOp(self._storage.location, self._storage.metadata)
    op.write(data)
    return op.finish()


class RayReadWriterRunner(RayReadOnlyRunner, BaseReadWriteRunner):
  """Ray read write runner."""

  def __init__(self, dataset: Dataset, options: Optional[RayOptions] = None):
    RayReadOnlyRunner.__init__(self, dataset)
    BaseReadWriteRunner.__init__(self, dataset.storage)

    self._options = RayOptions() if options is None else options

  @StorageMixin.transactional
  def append(self, data: InputData) -> Optional[rt.Patch]:
    op = RayAppendOp(self._storage.location, self._storage.metadata,
                     self._options.parallelism)
    op.write(data)
    return op.finish()

  @StorageMixin.transactional
  def append_from(
      self, sources: Union[Iterator[InputData], List[Iterator[InputData]]]
  ) -> Optional[rt.Patch]:
    if not isinstance(sources, list):
      sources = [sources]

    op = RayAppendOp(self._storage.location, self._storage.metadata,
                     self._options.parallelism)
    op.write_from(sources)

    return op.finish()

  @StorageMixin.transactional
  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> Optional[rt.Patch]:
    raise NotImplementedError(
        "append_array_record not supported yet in Ray runner")

  @StorageMixin.transactional
  def append_parquet(self, input_dir: str) -> Optional[rt.Patch]:
    raise NotImplementedError("append_parquet not supported yet in Ray runner")

  @StorageMixin.transactional
  def _insert(self, data: InputData,
              mode: InsertOptions.Mode) -> Optional[rt.Patch]:
    op = RayInsertOp(self._storage, InsertOptions(mode=mode),
                     self._options.parallelism)
    return op.write(data)

  @StorageMixin.transactional
  def delete(self, filter_: pc.Expression) -> Optional[rt.Patch]:
    op = RayDeleteOp(self._storage, filter_)
    return op.delete()
