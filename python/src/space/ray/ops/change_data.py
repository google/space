# Copyright 2024 Google LLC
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
"""Change data feed that computes delta between two snapshots by Ray."""

import math
from typing import Iterable, Iterator

import ray

from space.core.ops.change_data import (ChangeData, ChangeType,
                                        LocalChangeDataReadOp,
                                        ordered_snapshot_ids)
from space.core.options import ReadOptions
import space.core.proto.metadata_pb2 as meta
from space.core.storage import Storage
from space.ray import data_sources as ray_data_sources
from space.ray.options import RayOptions


def read_change_data(storage: Storage, start_snapshot_id: int,
                     end_snapshot_id: int, ray_options: RayOptions,
                     read_options: ReadOptions) -> Iterator[ChangeData]:
  """Read change data from a start to an end snapshot.
  
  start_snapshot_id is excluded; end_snapshot_id is included.
  """
  for snapshot_id in ordered_snapshot_ids(storage, start_snapshot_id,
                                          end_snapshot_id):
    yield from _RayChangeDataReadOp(storage, snapshot_id, ray_options,
                                    read_options)


class _RayChangeDataReadOp(LocalChangeDataReadOp):
  """Read changes of data from a given snapshot of a dataset."""

  def __init__(self, storage: Storage, snapshot_id: int,
               ray_options: RayOptions, read_options: ReadOptions):
    LocalChangeDataReadOp.__init__(self, storage, snapshot_id, read_options)
    self._ray_options = ray_options

  def __iter__(self) -> Iterator[ChangeData]:
    # Must return deletion first, otherwise when the upstream re-apply
    # deletions and additions, it may delete newly added data.
    # TODO: to enforce this check upstream, or merge deletion+addition as a
    # update.
    if self._change_log.deleted_rows:
      # Only read primary keys for deletions. The data to read is relatively
      # small. In addition, currently deletion has to aggregate primary keys
      # to delete (can't parallelize two sets of keys to delete). So we don't
      # spit it to parallel read streams.
      ds = self._ray_dataset(self._change_log.deleted_rows,
                             self._pk_only_read_option,
                             self._ray_options.max_parallelism)
      yield ChangeData(self._snapshot_id, ChangeType.DELETE, [ds])

    if self._change_log.added_rows:
      # Split added data into parallel read streams.
      num_files = len(self._change_log.added_rows)
      num_streams = self._ray_options.max_parallelism
      shard_size = math.ceil(num_files / num_streams)

      shards = []
      for i in range(num_streams):
        start = i * shard_size
        end = min((i + 1) * shard_size, num_files)
        shards.append(self._change_log.added_rows[start:end])

      # Parallelism 1 means one reader for each read stream.
      # There are `ray_options.max_parallelism` read streams.
      # TODO: to measure performance and adjust.
      yield ChangeData(self._snapshot_id, ChangeType.ADD, [
          self._ray_dataset(s, self._read_options, parallelism=1)
          for s in shards
      ])

  def _ray_dataset(self, bitmaps: Iterable[meta.RowBitmap],
                   read_options: ReadOptions,
                   parallelism: int) -> ray.data.Dataset:
    return ray.data.read_datasource(ray_data_sources.SpaceDataSource(),
                                    storage=self._storage,
                                    ray_options=self._ray_options,
                                    read_options=read_options,
                                    file_set=self._bitmaps_to_file_set(bitmaps),
                                    parallelism=parallelism)
