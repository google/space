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
"""Implement Ray data sources for Space datasets."""

from __future__ import annotations
from functools import partial
import math
from typing import (Any, Callable, Dict, Iterator, List, Optional,
                    TYPE_CHECKING)

from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, Reader, ReadTask
from ray.data.datasource.datasource import WriteResult
from ray.types import ObjectRef

from space.core.ops.read import FileSetReadOp
from space.core.options import ReadOptions
import space.core.proto.runtime_pb2 as rt
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.storage import Storage


class SpaceDataSource(Datasource):
  """A Ray data source for a Space dataset."""

  # pylint: disable=arguments-differ,too-many-arguments
  def create_reader(  # type: ignore[override]
      self, storage: Storage, ray_options: RayOptions,
      read_options: ReadOptions) -> Reader:
    return _SpaceDataSourceReader(storage, ray_options, read_options)

  def do_write(self, blocks: List[ObjectRef[Block]],
               metadata: List[BlockMetadata],
               ray_remote_args: Optional[Dict[str, Any]],
               location: str) -> List[ObjectRef[WriteResult]]:
    """Write a Ray dataset into Space datasets."""
    raise NotImplementedError("Write from a Ray dataset is not supported")

  def on_write_complete(  # type: ignore[override]
      self, write_results: List[WriteResult]) -> None:
    raise NotImplementedError("Write from a Ray dataset is not supported")


class _SpaceDataSourceReader(Reader):

  def __init__(self, storage: Storage, ray_options: RayOptions,
               read_options: ReadOptions):
    self._storage = storage
    self._ray_options = ray_options
    self._read_options = read_options

  def estimate_inmemory_data_size(self) -> Optional[int]:
    # TODO: to implement this method.
    return None

  # Create a list of `ReadTask`, one for each file group. Those tasks will be
  # executed in parallel.
  # Note: The `parallelism` which is supposed to indicate how many `ReadTask` to
  # return will have no effect here, since we map each query into a `ReadTask`.
  # TODO: to properly handle the error that returned list is empty.
  # TODO: to use parallelism when generating blocks.
  def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
    read_tasks: List[ReadTask] = []
    file_set = self._storage.data_files(self._read_options.filter_,
                                        self._read_options.snapshot_id)

    for index_file in file_set.index_files:
      num_rows = index_file.storage_statistics.num_rows

      if (self._ray_options.enable_index_file_row_range_block and
          self._read_options.batch_size):
        batch_size = self._read_options.batch_size
        num_blocks = math.ceil(num_rows / batch_size)

        for i in range(num_blocks):
          index_file_slice = rt.DataFile()
          index_file_slice.CopyFrom(index_file)

          rows = index_file_slice.selected_rows
          rows.start = i * batch_size
          rows.end = min((i + 1) * batch_size, num_rows)

          read_tasks.append(
              ReadTask(self._read_fn(index_file_slice),
                       _block_metadata(rows.end - rows.start)))
      else:
        read_tasks.append(
            ReadTask(self._read_fn(index_file), _block_metadata(num_rows)))

    return read_tasks

  def _read_fn(self, index_file: rt.DataFile) -> Callable[..., Iterator[Block]]:
    return partial(FileSetReadOp, self._storage.location,
                   self._storage.metadata, rt.FileSet(index_files=[index_file]),
                   self._read_options)  # type: ignore[return-value]


def _block_metadata(num_rows: int) -> BlockMetadata:
  """The metadata about the block that we know prior to actually executing the
  read task.
  """
  # TODO: to populate the storage values.
  return BlockMetadata(
      num_rows=num_rows,
      size_bytes=None,
      schema=None,
      input_files=None,
      exec_stats=None,
  )
