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
from typing import TYPE_CHECKING
from typing import Iterator, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc

from space.core.runners import BaseReadOnlyRunner, StorageCommitMixin
from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.ops.change_data import ChangeType, read_change_data
from space.core.ops.delete import FileSetDeleteOp
import space.core.proto.runtime_pb2 as runtime
from space.core.utils.lazy_imports_utils import ray
from space.core.versions.utils import version_to_snapshot_id

if TYPE_CHECKING:
  from space.core.views import MaterializedView, View


class RayReadOnlyRunner(BaseReadOnlyRunner):
  """A read-only Ray runner."""

  def __init__(self, view: View):
    self._view = view

  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    raise NotImplementedError()

  def diff(self, start_version: Union[int],
           end_version: Union[int]) -> Iterator[Tuple[ChangeType, pa.Table]]:
    sources = self._view.sources
    assert len(sources) == 1, "Views only support a single source dataset"
    ds = list(sources.values())[0]

    source_changes = read_change_data(ds.storage,
                                      version_to_snapshot_id(start_version),
                                      version_to_snapshot_id(end_version))
    for change_type, data in source_changes:
      # TODO: skip processing the data for deletions; the caller is usually
      # only interested at deleted primary keys.
      processed_ray_data = self._view.process_source(data)
      processed_data = ray.get(processed_ray_data.to_arrow_refs())
      yield change_type, pa.concat_tables(processed_data)


class RayMaterializedViewRunner(RayReadOnlyRunner, StorageCommitMixin):
  """Ray runner for materialized views."""

  def __init__(self, mv: MaterializedView):
    RayReadOnlyRunner.__init__(self, mv.view)
    StorageCommitMixin.__init__(self, mv.storage)

  def refresh(self, target_version: Union[int]) -> runtime.JobResult:
    """Refresh the materialized view by synchronizing from source dataset."""
    start_snapshot_id = self._storage.metadata.current_snapshot_id
    end_snapshot_id = version_to_snapshot_id(target_version)

    patches: List[Optional[runtime.Patch]] = []
    for change_type, data in self.diff(start_snapshot_id, end_snapshot_id):
      if change_type == ChangeType.DELETE:
        patches.append(self._process_delete(data))
      elif change_type == ChangeType.ADD:
        patches.append(self._process_append(data))
      else:
        raise NotImplementedError(f"Change type {change_type} not supported")

    return self._try_commit(utils.merge_patches(patches))

  def _process_delete(self, data: pa.Table) -> Optional[runtime.Patch]:
    filter_ = utils.primary_key_filter(self._storage.primary_keys, data)
    if filter_ is None:
      return None

    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         self._storage.data_files(filter_), filter_)
    return op.delete()

  def _process_append(self, data: pa.Table) -> Optional[runtime.Patch]:
    op = LocalAppendOp(self._storage.location, self._storage.metadata)
    op.write(data)
    return op.finish()
