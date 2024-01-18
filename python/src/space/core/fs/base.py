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
"""Abstract base file system."""

from abc import ABC, abstractmethod
from typing import TypeVar

from google.protobuf import message

ProtoT = TypeVar("ProtoT", bound=message.Message)


class BaseFileSystem(ABC):
  """Abstract file system."""

  @abstractmethod
  def create_dir(self, dir_path: str) -> None:
    """Create a new directory."""

  @abstractmethod
  def write_proto(self,
                  file_path: str,
                  msg: ProtoT,
                  fail_if_exists: bool = False) -> None:
    """Write a proto message in text format to a file.
    
    Args:
      file_path: full path of the file to write to
      msg: the proto message to write
      fail_if_exists: if true, fail when the file already exists; otherwise
        truncate the file
    """

  @abstractmethod
  def read_proto(self, file_path: str, empty_msg: ProtoT) -> ProtoT:
    """Read a proto message in text format from a file."""
