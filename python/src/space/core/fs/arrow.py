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
"""Arrow file system implementation."""

from abc import abstractmethod

from google.protobuf import text_format
from pyarrow import fs

from space.core.fs.base import BaseFileSystem, ProtoT
from space.core.utils import errors
from space.core.utils.protos import proto_to_text
from space.core.utils.uuids import random_id


class ArrowFileSystem(BaseFileSystem):
  """Abstract Arrow file system."""

  def __init__(self):
    super().__init__()
    self._fs = self.create_fs()

  @abstractmethod
  def create_fs(self) -> fs.FileSystem:
    """Create a new underlying Arrow file system."""

  def create_dir(self, dir_path: str) -> None:
    self._fs.create_dir(dir_path)

  def write_proto(self,
                  file_path: str,
                  msg: ProtoT,
                  fail_if_exists: bool = False) -> None:
    # TODO: this check is not atomic; a more file system specific implement is
    # needed.
    if fail_if_exists and self._fs.get_file_info(
        file_path).type != fs.FileType.NotFound:
      raise errors.FileExistError(f"File {file_path} already exists")

    tmp_file_path = f"{file_path}.{random_id()}.tmp"

    with self._fs.open_output_stream(tmp_file_path) as f:
      f.write(proto_to_text(msg))

    self._fs.move(tmp_file_path, file_path)

  def read_proto(self, file_path: str, empty_msg: ProtoT) -> ProtoT:
    with self._fs.open_input_file(file_path) as f:
      result = text_format.Parse(f.readall(), empty_msg)
    return result


class ArrowLocalFileSystem(ArrowFileSystem):
  """Arrow local file system implementation."""

  def create_fs(self) -> fs.FileSystem:
    return fs.LocalFileSystem()
