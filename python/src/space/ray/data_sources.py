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
from typing import Any, Dict, List, Optional

import pyarrow.compute as pc
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, Reader, ReadTask
from ray.data.datasource.datasource import WriteResult
from ray.types import ObjectRef

from space.core.storage import Storage
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as runtime
from space.core.ops.read import FileSetReadOp, ReadOptions


class SpaceDataSource(Datasource):
  """A Ray data source for a Space dataset."""

  # pylint: disable=arguments-differ,too-many-arguments
  def create_reader(  # type: ignore[override]
      self,
      storage: Storage,
      filter_: Optional[pc.Expression],
      fields: Optional[List[str]],
      snapshot_id: Optional[int],
      reference_read: bool = False) -> Reader:
    return _SpaceDataSourceReader(storage, filter_, fields, snapshot_id,
                                  reference_read)

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

  # pylint: disable=too-many-arguments
  def __init__(self,
               storage: Storage,
               filter_: Optional[pc.Expression],
               fields: Optional[List[str]],
               snapshot_id: Optional[int],
               reference_read: bool = False):
    self._storage = storage
    self._filter = filter_
    self._fields = fields
    self._snapshot_id = snapshot_id
    self._reference_read = reference_read

  def estimate_inmemory_data_size(self) -> Optional[int]:
    # TODO: to implement this method.
    return 1

  # Create a list of `ReadTask`, one for each file group. Those tasks will be
  # executed in parallel.
  # Note: The `parallelism` which is supposed to indicate how many `ReadTask` to
  # return will have no effect here, since we map each query into a `ReadTask`.
  def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
    read_tasks: List[ReadTask] = []
    file_set = self._storage.data_files(self._filter, self._snapshot_id)

    for index_file in file_set.index_files:
      task_file_set = runtime.FileSet(index_files=[index_file])

      # The metadata about the block that we know prior to actually executing
      # the read task.
      # TODO: to populate the storage values.
      block_metadata = BlockMetadata(
          num_rows=1,
          size_bytes=1,
          schema=None,
          input_files=None,
          exec_stats=None,
      )

      def _read_fn(location=self._storage.location,
                   metadata=self._storage.metadata,
                   file_set=task_file_set):
        return _read_file_set(
            location, metadata, file_set,
            ReadOptions(self._filter, self._fields, self._reference_read))

      read_tasks.append(ReadTask(_read_fn, block_metadata))

    return read_tasks


# TODO: A single index file (with record files) is a single block. To check
# whether row group granularity is needed.
def _read_file_set(location: str, metadata: meta.StorageMetadata,
                   file_set: runtime.FileSet,
                   read_options: ReadOptions) -> List[Block]:
  return list(FileSetReadOp(location, metadata, file_set, read_options))
