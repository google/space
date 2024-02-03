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
"""Distributed delete operation using Ray."""

from __future__ import annotations
from typing import Optional

import pyarrow.compute as pc
import ray

from space.core.ops import utils
from space.core.ops.delete import BaseDeleteOp, FileSetDeleteOp
from space.core.options import FileOptions
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.storage import Storage
from space.core.utils.paths import StoragePathsMixin


class RayDeleteOp(BaseDeleteOp, StoragePathsMixin):
  """Ray delete operation processing files distributedly."""

  def __init__(self, storage: Storage, filter_: pc.Expression,
               file_options: FileOptions):
    StoragePathsMixin.__init__(self, storage.location)

    self._storage = storage
    self._filter = filter_
    self._file_options = file_options

  def delete(self) -> Optional[rt.Patch]:
    """Delete data matching the filter from the dataset."""
    metadata = self._storage.metadata
    matched_file_set = self._storage.data_files(self._filter)

    remote_delete_patches = []
    for index_file in matched_file_set.index_files:
      # Deletion only needs index file information (no record file information).
      file_set = rt.FileSet(
          index_files=[index_file],
          # TODO: attach all manifest files here, to select related manifests.
          index_manifest_files=matched_file_set.index_manifest_files)

      result = _delete.options(  # type: ignore[attr-defined]
          num_returns=1).remote(self._storage.location, metadata, file_set,
                                self._filter, self._file_options)
      remote_delete_patches.append(result)

    patches = ray.get(remote_delete_patches)
    return utils.merge_patches(patches)


@ray.remote
def _delete(location: str, metadata: meta.StorageMetadata, file_set: rt.FileSet,
            filter_: pc.Expression,
            file_options: FileOptions) -> Optional[rt.Patch]:
  return FileSetDeleteOp(location, metadata, file_set, filter_,
                         file_options).delete()
