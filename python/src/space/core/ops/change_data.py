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
"""Change data feed that computes delta between two snapshots."""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator, List, Union

import pyarrow as pa

from space.core.fs.factory import create_fs
from space.core.ops.read import FileSetReadOp
from space.core.options import ReadOptions
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.storage import Storage
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray
from space.core.utils.paths import StoragePathsMixin


class ChangeType(Enum):
  """Type of data changes."""
  # For added rows.
  ADD = 1

  # For deleted rows.
  DELETE = 2
  # TODO: to support UPDATE. UPDATE is currently described as an ADD after a
  # DELETE, on the same primary key in one snapshot change.


@dataclass
class ChangeData:
  """Information and data of a change."""
  # Snapshot ID that the change was committed to.
  snapshot_id: int

  # The change type.
  type_: ChangeType

  # The change data.
  data: Union[pa.Table, List["ray.data.Dataset"]]


def ordered_snapshot_ids(storage: Storage, start_snapshot_id: int,
                         end_snapshot_id: int) -> List[int]:
  """Return a list of ordered snapshot IDs between two snapshots.
  
  start_snapshot_id is excluded; end_snapshot_id is included.
  """
  if start_snapshot_id >= end_snapshot_id:
    raise errors.UserInputError(
        f"End snapshot ID {end_snapshot_id} should be higher than start "
        f"snapshot ID {start_snapshot_id}")

  all_snapshot_ids: List[int] = []
  current_snapshot = storage.snapshot(end_snapshot_id)
  while current_snapshot.snapshot_id >= start_snapshot_id:
    all_snapshot_ids.insert(0, current_snapshot.snapshot_id)
    if not current_snapshot.HasField("parent_snapshot_id"):
      break

    current_snapshot = storage.snapshot(current_snapshot.parent_snapshot_id)

  if start_snapshot_id != all_snapshot_ids[0]:
    raise errors.UserInputError(
        f"Start snapshot {start_snapshot_id} is not the ancestor of end "
        f"snapshot {end_snapshot_id}")

  return all_snapshot_ids[1:]


def read_change_data(storage: Storage, start_snapshot_id: int,
                     end_snapshot_id: int,
                     read_options: ReadOptions) -> Iterator[ChangeData]:
  """Read change data from a start to an end snapshot.
  
  start_snapshot_id is excluded; end_snapshot_id is included.
  """
  for snapshot_id in ordered_snapshot_ids(storage, start_snapshot_id,
                                          end_snapshot_id):
    yield from LocalChangeDataReadOp(storage, snapshot_id, read_options)


class LocalChangeDataReadOp(StoragePathsMixin):
  """Read changes of data from a given snapshot of a dataset."""

  def __init__(self, storage: Storage, snapshot_id: int,
               read_options: ReadOptions):
    StoragePathsMixin.__init__(self, storage.location)

    self._storage = storage
    self._metadata = self._storage.metadata
    self._snapshot_id = snapshot_id
    self._read_options = read_options

    self._pk_only_read_option = copy.deepcopy(read_options)
    self._pk_only_read_option.fields = self._storage.primary_keys

    if snapshot_id not in self._metadata.snapshots:
      raise errors.VersionNotFoundError(
          f"Change data read can't find snapshot ID {snapshot_id}")

    self._snapshot = self._metadata.snapshots[snapshot_id]
    self._change_log = _read_change_log_proto(
        self._storage.full_path(self._snapshot.change_log_file))

  def __iter__(self) -> Iterator[ChangeData]:
    # Must return deletion first, otherwise when the upstream re-apply
    # deletions and additions, it may delete newly added data.
    # TODO: to enforce this check upstream, or merge deletion+addition as a
    # update.
    for data in self._read_op(self._change_log.deleted_rows,
                              self._pk_only_read_option):
      yield ChangeData(self._snapshot_id, ChangeType.DELETE, data)

    for data in self._read_op(self._change_log.added_rows, self._read_options):
      yield ChangeData(self._snapshot_id, ChangeType.ADD, data)

  def _read_op(self, bitmaps: Iterable[meta.RowBitmap],
               read_options: ReadOptions) -> Iterator[pa.Table]:
    return iter(
        FileSetReadOp(self._storage.location,
                      self._metadata,
                      self._bitmaps_to_file_set(bitmaps),
                      options=read_options))

  @classmethod
  def _bitmaps_to_file_set(cls,
                           bitmaps: Iterable[meta.RowBitmap]) -> rt.FileSet:
    return rt.FileSet(
        index_files=[_bitmap_to_index_file(bitmap) for bitmap in bitmaps])


def _bitmap_to_index_file(bitmap: meta.RowBitmap) -> rt.DataFile:
  index_file = rt.DataFile(
      path=bitmap.file,
      storage_statistics=meta.StorageStatistics(num_rows=bitmap.num_rows))
  if not bitmap.all_rows:
    index_file.row_bitmap.CopyFrom(bitmap)

  return index_file


def _read_change_log_proto(file_path: str) -> meta.ChangeLog:
  fs = create_fs(file_path)
  return fs.read_proto(file_path, meta.ChangeLog())
