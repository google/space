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

import pyarrow as pa

import space.core.proto.metadata_pb2 as meta
from space.core.schema.types import File
from space.core.utils.constants import UTF_8


class TestFile:

  def test_arrow_ext_serialize_deserialize(self):
    file_type = File(directory="test_folder")
    serialized = file_type.__arrow_ext_serialize__()
    assert file_type.__arrow_ext_serialize__().decode(
        UTF_8) == '"{\\n  \\"directory\\": \\"test_folder\\"\\n}"'

    # Bytes input.
    deserialized_file_type = File.__arrow_ext_deserialize__(
        storage_type=None, serialized=serialized)
    assert deserialized_file_type._file_type == meta.FileType(
        directory="test_folder")

  def test_full_path(self):
    file_type = File(directory="")
    assert file_type.full_path("") == ""
    assert file_type.full_path("123") == "123"

    file_type = File(directory="test_folder")
    assert file_type.full_path("") == "test_folder/"
    assert file_type.full_path("123") == "test_folder/123"
