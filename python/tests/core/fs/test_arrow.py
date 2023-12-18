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

import pytest

from space.core.fs.arrow import ArrowLocalFileSystem
import space.core.proto.metadata_pb2 as meta


class TestArrowLocalFileSystem:

  @pytest.fixture
  def fs(self):
    return ArrowLocalFileSystem()

  def test_create_dir(self, tmp_path, fs):
    dir_path = tmp_path / "test_create_dir"
    fs.create_dir(str(dir_path))
    assert dir_path.exists()

  def _read_proto(self, fs, file_path):
    read_msg = meta.StorageMetadata()
    fs.read_proto(file_path, read_msg)
    return read_msg

  def test_write_read_proto(self, tmp_path, fs):
    dir_path = tmp_path / "test_write_read_proto"
    fs.create_dir(str(dir_path))

    file_path = str(dir_path / "output.txtpb")
    write_msg = meta.StorageMetadata(current_snapshot_id=100)
    fs.write_proto(file_path, write_msg)
    assert dir_path.exists()

    assert self._read_proto(fs, file_path) == write_msg

  def test_overwrite_proto_file(self, tmp_path, fs):
    dir_path = tmp_path / "test_write_read_proto"
    fs.create_dir(str(dir_path))

    file_path = str(dir_path / "output.txtpb")
    write_msg = meta.StorageMetadata(current_snapshot_id=100)
    fs.write_proto(file_path, write_msg)
    assert self._read_proto(fs, file_path).current_snapshot_id == 100

    write_msg = meta.StorageMetadata(current_snapshot_id=200)
    fs.write_proto(file_path, write_msg)
    assert self._read_proto(fs, file_path).current_snapshot_id == 200
