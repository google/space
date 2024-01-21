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
"""Define a custom Arrow type for files."""

from __future__ import annotations
import json
from os import path
from typing import Union

from google.protobuf import json_format
import pyarrow as pa

import space.core.proto.metadata_pb2 as meta
from space.core.utils.constants import UTF_8


class File(pa.ExtensionType):
  """A custom Arrow type representing data in a standalone file.
  
  TODO: several features to add, e.g., auto read file content, write a new file
  at data write time, serializer/deserializer.
  """

  EXTENSION_NAME = "space.file"

  def __init__(self, directory: str = ""):
    """
    Args:
      directory: a directory to add as a prefix of file paths
    """
    # TODO: managed is not supported yet.
    self._file_type = meta.FileType(directory=directory)
    pa.ExtensionType.__init__(self, pa.string(), self.EXTENSION_NAME)

  def __arrow_ext_serialize__(self) -> bytes:
    return json.dumps(json_format.MessageToJson(self._file_type)).encode(UTF_8)

  @classmethod
  def __arrow_ext_deserialize__(
      cls,
      storage_type: pa.DataType,  # pylint: disable=unused-argument
      serialized: Union[bytes, str]) -> File:
    if isinstance(serialized, bytes):
      serialized = serialized.decode(UTF_8)

    file_type = json_format.Parse(json.loads(serialized), meta.FileType())

    return File(directory=file_type.directory)

  def full_path(self, file_path: str) -> str:
    """Return the full path of file."""
    return path.join(self._file_type.directory, file_path)
