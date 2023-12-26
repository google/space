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
"""Local insert operation implementation."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pyarrow as pa

from space.core.ops.append import BaseAppendOp, LocalAppendOp
from space.core.ops.delete import BaseDeleteOp, FileSetDeleteOp
from space.core.ops.read import FileSetReadOp, ReadOptions
from space.core.ops import utils
from space.core.ops.base import BaseOp, InputData
import space.core.proto.runtime_pb2 as runtime
from space.core.storage import Storage
from space.core.utils.paths import StoragePaths


@dataclass
class InsertOptions:
  """Options of inserting data."""

  class Mode(Enum):
    """Mode of insert operation."""
    # Fail if duplicated primary key is found.
    INSERT = 1
    # Update the existing row if duplicated primary key is found.
    UPSERT = 2

  # The insert mode.
  mode: Mode = Mode.INSERT


class BaseInsertOp(BaseOp):
  """Abstract base insert operation class."""

  def write(self, data: InputData) -> Optional[runtime.Patch]:
    """Insert data into storage."""


class LocalInsertOp(BaseInsertOp, StoragePaths):
  '''Append data to a dataset.'''

  def __init__(self, location: str, storage: Storage, options: InsertOptions):
    StoragePaths.__init__(self, location)

    self._storage = storage
    self._metadata = self._storage.metadata

    self._options = InsertOptions() if options is None else options

  def write(self, data: InputData) -> Optional[runtime.Patch]:
    if not isinstance(data, pa.Table):
      data = pa.Table.from_pydict(data)

    return self._write_arrow(data)

  def _write_arrow(self, data: pa.Table) -> Optional[runtime.Patch]:
    if data.num_rows == 0:
      return None

    pk_filter = utils.primary_key_filter(
        list(self._metadata.schema.primary_keys), data)
    assert pk_filter is not None

    data_files = self._storage.data_files(pk_filter)

    mode = self._options.mode
    patches: List[runtime.Patch] = []
    if data_files.index_files:
      if mode == InsertOptions.Mode.INSERT:
        read_op = FileSetReadOp(
            self._location, self._metadata, data_files,
            ReadOptions(filter_=pk_filter, fields=self._storage.primary_keys))

        for batch in iter(read_op):
          if batch.num_rows > 0:
            # TODO: to customize the error and converted it to JobResult failed
            # status.
            raise RuntimeError('Primary key to insert already exist')
      elif mode == InsertOptions.Mode.UPSERT:
        _try_delete_data(
            FileSetDeleteOp(self._location, self._metadata, data_files,
                            pk_filter), patches)
      else:
        raise RuntimeError(f"Insert mode {mode} not supported")

    _try_append_data(LocalAppendOp(self._location, self._metadata), data,
                     patches)
    return utils.merge_patches(patches)


def _try_delete_data(op: BaseDeleteOp, patches: List[runtime.Patch]) -> None:
  patch = op.delete()
  if patch is not None:
    patches.append(patch)


def _try_append_data(op: BaseAppendOp, data: pa.Table,
                     patches: List[runtime.Patch]) -> None:
  op.write(data)
  patch = op.finish()
  if patch is not None:
    patches.append(patch)
