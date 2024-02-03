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
import pyarrow.compute as pc

from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.ops.base import BaseOp, InputData
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.read import FileSetReadOp
from space.core.options import FileOptions, ReadOptions
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.storage import Storage
from space.core.utils import errors
from space.core.utils.paths import StoragePathsMixin


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

  def write(self, data: InputData) -> Optional[rt.Patch]:
    """Insert data into storage."""


class LocalInsertOp(BaseInsertOp, StoragePathsMixin):
  """Insert data to a dataset."""

  def __init__(self, storage: Storage, options: InsertOptions,
               file_options: FileOptions):
    StoragePathsMixin.__init__(self, storage.location)

    self._storage = storage
    self._metadata = self._storage.metadata

    self._options = options or InsertOptions()
    self._file_options = file_options

  def write(self, data: InputData) -> Optional[rt.Patch]:
    if not isinstance(data, pa.Table):
      data = pa.Table.from_pydict(data)

    return self._write_arrow(data)

  def _write_arrow(self, data: pa.Table) -> Optional[rt.Patch]:
    if data.num_rows == 0:
      return None

    filter_ = utils.primary_key_filter(self._storage.primary_keys, data)
    assert filter_ is not None

    data_files = self._storage.data_files(filter_)

    mode = self._options.mode
    patches: List[Optional[rt.Patch]] = []
    if data_files.index_files:
      if mode == InsertOptions.Mode.INSERT:
        self._check_duplication(data_files, filter_)
      elif mode == InsertOptions.Mode.UPSERT:
        self._delete(filter_, data_files, patches)
      else:
        raise errors.SpaceRuntimeError(f"Insert mode {mode} not supported")

    self._append(data, patches)
    return utils.merge_patches(patches)

  def _check_duplication(self, data_files: rt.FileSet, filter_: pc.Expression):
    if filter_matched(self._location, self._metadata, data_files, filter_,
                      self._storage.primary_keys):
      raise errors.SpaceRuntimeError("Primary key to insert already exist")

  def _delete(self, filter_: pc.Expression, data_files: rt.FileSet,
              patches: List[Optional[rt.Patch]]) -> None:
    delete_op = FileSetDeleteOp(self._location, self._metadata, data_files,
                                filter_, self._file_options)
    patches.append(delete_op.delete())

  def _append(self, data: pa.Table, patches: List[Optional[rt.Patch]]) -> None:
    append_op = LocalAppendOp(self._location, self._metadata,
                              self._file_options)
    append_op.write(data)
    patches.append(append_op.finish())


def filter_matched(location: str, metadata: meta.StorageMetadata,
                   data_files: rt.FileSet, filter_: pc.Expression,
                   primary_keys: List[str]) -> bool:
  """Return True if there are data matching the provided filter."""
  op = FileSetReadOp(location,
                     metadata,
                     data_files,
                     options=ReadOptions(filter_=filter_, fields=primary_keys))

  for data in iter(op):
    if data.num_rows > 0:
      # TODO: to customize the error and converted it to JobResult failed
      # status.
      return True

  return False
