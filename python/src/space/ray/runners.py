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
from space.core.runners import StorageCommitMixin
from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.ops.base import InputData
from space.core.ops.change_data import ChangeType, read_change_data
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.insert import InsertOptions
import space.core.proto.runtime_pb2 as rt
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
    for ref in self._view.ray_dataset(filter_, fields, snapshot_id,
                                      reference_read).to_arrow_refs():
      yield ray.get(ref)

  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    source_changes = read_change_data(self._source().storage,
                                      version_to_snapshot_id(start_version),
                                      version_to_snapshot_id(end_version))
    for change_type, data in source_changes:
      # TODO: skip processing the data for deletions; the caller is usually
      # only interested at deleted primary keys.
      processed_remote_data = self._view.process_source(data)
      processed_data = ray.get(processed_remote_data.to_arrow_refs())
      yield change_type, pa.concat_tables(processed_data)

  def _source(self) -> Dataset:
    sources = self._view.sources
    assert len(sources) == 1, "Views only support a single source dataset"
    return list(sources.values())[0]


class RayMaterializedViewRunner(RayReadOnlyRunner, StorageCommitMixin):
  """Ray runner for materialized views."""

  def __init__(self, mv: MaterializedView):
    RayReadOnlyRunner.__init__(self, mv.view)
    StorageCommitMixin.__init__(self, mv.storage)

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

  def refresh(self,
              target_version: Optional[Union[int]] = None) -> rt.JobResult:
    """Refresh the materialized view by synchronizing from source dataset."""
    source_snapshot_id = self._source().storage.metadata.current_snapshot_id
    if target_version is None:
      end_snapshot_id = source_snapshot_id
    else:
      end_snapshot_id = version_to_snapshot_id(target_version)
      if end_snapshot_id > source_snapshot_id:
        raise RuntimeError(
            f"Target snapshot ID {end_snapshot_id} higher than source dataset "
            "version")

    start_snapshot_id = self._storage.metadata.current_snapshot_id

    patches: List[Optional[rt.Patch]] = []
    for change_type, data in self.diff(start_snapshot_id, end_snapshot_id):
      if change_type == ChangeType.DELETE:
        patches.append(self._process_delete(data))
      elif change_type == ChangeType.ADD:
        patches.append(self._process_append(data))
      else:
        raise NotImplementedError(f"Change type {change_type} not supported")

    return self._try_commit(utils.merge_patches(patches))

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

  def append(self, data: InputData) -> rt.JobResult:
    op = RayAppendOp(self._storage.location, self._storage.metadata,
                     self._options.parallelism)
    op.write(data)
    return self._try_commit(op.finish())

  def append_from(
      self, sources: Union[Iterator[InputData], List[Iterator[InputData]]]
  ) -> rt.JobResult:
    if not isinstance(sources, list):
      sources = [sources]

    op = RayAppendOp(self._storage.location, self._storage.metadata,
                     self._options.parallelism)
    op.write_from(sources)

    return self._try_commit(op.finish())

  def append_array_record(self, input_dir: str,
                          index_fn: ArrayRecordIndexFn) -> rt.JobResult:
    raise NotImplementedError(
        "append_array_record not supported yet in Ray runner")

  def append_parquet(self, input_dir: str) -> rt.JobResult:
    raise NotImplementedError("append_parquet not supported yet in Ray runner")

  def _insert(self, data: InputData, mode: InsertOptions.Mode) -> rt.JobResult:
    op = RayInsertOp(self._storage, InsertOptions(mode=mode),
                     self._options.parallelism)
    return self._try_commit(op.write(data))

  def delete(self, filter_: pc.Expression) -> rt.JobResult:
    op = RayDeleteOp(self._storage, filter_)
    return self._try_commit(op.delete())
