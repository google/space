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
"""Distributed append operation using Ray."""

from __future__ import annotations
from typing import List, Optional

import pyarrow as pa
import ray

from space.core.ops import utils
from space.core.ops.append import BaseAppendOp, LocalAppendOp
from space.core.ops.base import InputData, InputIteratorFn
from space.core.options import FileOptions
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.ray.options import RayOptions


class RayAppendOp(BaseAppendOp):
  """Ray append operation writing files distributedly."""

  # pylint: disable=too-many-arguments
  def __init__(self,
               location: str,
               metadata: meta.StorageMetadata,
               ray_options: RayOptions,
               file_options: FileOptions,
               record_address_input: bool = False):
    """
    Args:
      record_address_input: if true, input record fields are addresses.
    """
    self._ray_options = ray_options
    self._actors = [
        _AppendActor.remote(  # type: ignore[attr-defined] # pylint: disable=no-member
            location, metadata, file_options, record_address_input)
        for _ in range(self._ray_options.max_parallelism)
    ]

  def write(self, data: InputData) -> None:
    if not isinstance(data, pa.Table):
      data = pa.Table.from_pydict(data)

    num_shards = self._ray_options.max_parallelism

    shard_size = data.num_rows // num_shards
    if shard_size == 0:
      shard_size = 1

    responses = []
    offset = 0
    for i in range(num_shards):
      shard = data.slice(offset=offset, length=shard_size)
      responses.append(self._actors[i].write.remote(shard))

      offset += shard_size
      if offset >= data.num_rows:
        break

    if offset < data.num_rows:
      shard = data.slice(offset=offset)
      responses.append(self._actors[0].write.remote(shard))

    ray.get(responses)

  def write_from(self, source_fns: List[InputIteratorFn]) -> None:
    """Append data into the dataset from multiple iterator sources in
    parallel.
    """
    num_actors = len(self._actors)
    responses = []
    for i, source_fn in enumerate(source_fns):
      responses.append(self._actors[i %
                                    num_actors].write_from.remote(source_fn))

    ray.get(responses)

  def finish(self) -> Optional[rt.Patch]:
    patches = ray.get([actor.finish.remote() for actor in self._actors])
    return utils.merge_patches(patches)


@ray.remote
class _AppendActor:
  """A stateful Ray actor for appending data."""

  def __init__(self,
               location: str,
               metadata: meta.StorageMetadata,
               file_options: FileOptions,
               record_address_input: bool = False):
    self._op = LocalAppendOp(location, metadata, file_options,
                             record_address_input)

  def write_from(self, source_fn: InputIteratorFn) -> None:
    """Append data into the dataset from an iterator source."""
    for data in source_fn():
      self._op.write(data)

  def write(self, data: InputData) -> bool:
    """Append data into storage."""
    self._op.write(data)
    return True

  def finish(self) -> Optional[rt.Patch]:
    """Complete the append operation and return a metadata patch."""
    return self._op.finish()
