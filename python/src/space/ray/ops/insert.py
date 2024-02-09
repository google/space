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
"""Distributed insert operation using Ray."""

from typing import List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import ray

from space.ray.ops.append import RayAppendOp
from space.core.ops.insert import InsertOptions, LocalInsertOp
from space.core.ops.insert import filter_matched
from space.core.options import FileOptions
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.storage import Storage
from space.core.utils import errors
from space.ray.options import RayOptions


class RayInsertOp(LocalInsertOp):
  """Insert data to a dataset with distributed duplication check."""

  def __init__(self, storage: Storage, options: InsertOptions,
               ray_options: RayOptions, file_options: FileOptions):
    LocalInsertOp.__init__(self, storage, options, file_options)
    self._ray_options = ray_options

  def _check_duplication(self, data_files: rt.FileSet, filter_: pc.Expression):
    remote_duplicated_values = []
    for index_file in data_files.index_files:
      # pylint: disable=line-too-long
      remote_duplicated = _remote_filter_matched.options(  # type: ignore[attr-defined]
          num_returns=1).remote(self._storage.location, self._metadata,
                                rt.FileSet(index_files=[index_file]), filter_,
                                self._storage.primary_keys)
      remote_duplicated_values.append(remote_duplicated)

    for duplicated in ray.get(remote_duplicated_values):
      if duplicated:
        raise errors.PrimaryKeyExistError("Primary key to insert already exist")

  def _append(self, data: pa.Table, patches: List[Optional[rt.Patch]]) -> None:
    append_op = RayAppendOp(self._location, self._metadata, self._ray_options,
                            self._file_options)
    append_op.write(data)
    patches.append(append_op.finish())


@ray.remote
def _remote_filter_matched(location: str, metadata: meta.StorageMetadata,
                           data_files: rt.FileSet, pk_filter: pc.Expression,
                           primary_keys: List[str]) -> bool:
  return filter_matched(location, metadata, data_files, pk_filter, primary_keys)
