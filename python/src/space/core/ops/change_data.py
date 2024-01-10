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

from enum import Enum
from typing import Iterator, Tuple, List

import pyarrow as pa
from pyroaring import BitMap  # type: ignore[import-not-found]

from space.core.fs.base import BaseFileSystem
from space.core.fs.factory import create_fs
from space.core.ops.read import FileSetReadOp
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.storage import Storage
from space.core.utils import errors
from space.core.utils.paths import StoragePathsMixin


class ChangeType(Enum):
  """Type of data changes."""
  # For added rows.
  ADD = 1
  # For deleted rows.
  DELETE = 2
  # TODO: to support UPDATE. UPDATE is currently described as an ADD after a
  # DELETE, on the same primary key in one snapshot change.


def read_change_data(
    storage: Storage, start_snapshot_id: int,
    end_snapshot_id: int) -> Iterator[Tuple[ChangeType, pa.Table]]:
  """Read change data from a start to an end snapshot.
  
  start_snapshot_id is excluded; end_snapshot_id is included.
  """
  if start_snapshot_id > end_snapshot_id:
    raise errors.UserInputError(
        f"End snapshot ID {end_snapshot_id} should not be lower than start "
        f"snapshot ID {start_snapshot_id}")

  all_snapshot_ids = sorted(storage.snapshot_ids)
  all_snapshot_ids_set = set(all_snapshot_ids)

  if start_snapshot_id not in all_snapshot_ids_set:
    raise errors.VersionNotFoundError(
        f"Start snapshot ID not found: {start_snapshot_id}")

  if end_snapshot_id not in all_snapshot_ids_set:
    raise errors.VersionNotFoundError(
        f"Start snapshot ID not found: {end_snapshot_id}")

  for snapshot_id in all_snapshot_ids:
    if snapshot_id <= start_snapshot_id:
      continue

    if snapshot_id > end_snapshot_id:
      break

    for result in iter(_LocalChangeDataReadOp(storage, snapshot_id)):
      yield result


class _LocalChangeDataReadOp(StoragePathsMixin):
  """Read changes of data from a given snapshot of a dataset."""

  def __init__(self, storage: Storage, snapshot_id: int):
    StoragePathsMixin.__init__(self, storage.location)

    self._storage = storage
    self._metadata = self._storage.metadata

    if snapshot_id not in self._metadata.snapshots:
      raise errors.VersionNotFoundError(
          f"Change data read can't find snapshot ID {snapshot_id}")

    snapshot = self._metadata.snapshots[snapshot_id]

    fs = create_fs(self._location)
    change_log_file = self._storage.full_path(snapshot.change_log_file)
    self._change_log = _read_change_log_proto(fs, change_log_file)

  def __iter__(self) -> Iterator[Tuple[ChangeType, pa.Table]]:
    # TODO: must return deletion first, otherwise when the upstream re-apply
    # deletions and additions, it may delete newly added data.
    # TODO: to enforce this check upstream, or merge deletion+addition as a
    # update.
    for bitmap in self._change_log.deleted_rows:
      yield self._read_bitmap_rows(ChangeType.DELETE, bitmap)

    for bitmap in self._change_log.added_rows:
      yield self._read_bitmap_rows(ChangeType.ADD, bitmap)

  def _read_bitmap_rows(self, change_type: ChangeType,
                        bitmap: meta.RowBitmap) -> Tuple[ChangeType, pa.Table]:
    file_set = rt.FileSet(index_files=[rt.DataFile(path=bitmap.file)])
    read_op = FileSetReadOp(self._storage.location, self._metadata, file_set)

    data = pa.concat_tables(list(iter(read_op)))
    # TODO: to read index fields first, apply mask, then read record fields.
    if not bitmap.all_rows:
      data = data.filter(
          mask=_bitmap_mask(bitmap.roaring_bitmap, data.num_rows))

    return (change_type, data)


def _read_change_log_proto(fs: BaseFileSystem,
                           file_path: str) -> meta.ChangeLog:
  return fs.read_proto(file_path, meta.ChangeLog())


def _bitmap_mask(serialized_bitmap: bytes, num_rows: int) -> List[bool]:
  bitmap = BitMap.deserialize(serialized_bitmap)

  mask = [False] * num_rows
  for row_id in bitmap.to_array():
    mask[row_id] = True

  return mask
