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

from google.protobuf.timestamp_pb2 import Timestamp
import pytest
from pathlib import Path

from space.core.fs.arrow import ArrowLocalFileSystem
import space.core.proto.metadata_pb2 as meta
from space.core.storage import Storage
from space.core.utils.paths import _ENTRY_POINT_FILE


class TestStorage:

  _LOCATION = "location"
  _SNAPSHOT_ID = 100

  @pytest.fixture
  def metadata(self):
    current_snapshot_id = self._SNAPSHOT_ID
    now = Timestamp(seconds=123456)
    metadata = meta.StorageMetadata(create_time=now,
                                    last_update_time=now,
                                    current_snapshot_id=current_snapshot_id,
                                    type=meta.StorageMetadata.DATASET)
    metadata.snapshots[current_snapshot_id].CopyFrom(
        meta.Snapshot(snapshot_id=current_snapshot_id, create_time=now))

    # Add a previous snapshot for testing.
    previous_snapshot_id = 10
    metadata.snapshots[previous_snapshot_id].CopyFrom(
        meta.Snapshot(snapshot_id=previous_snapshot_id,
                      create_time=Timestamp(seconds=123000)))
    return metadata

  @pytest.fixture
  def storage(self, metadata):
    return Storage(self._LOCATION, metadata)

  def test_create_dir(self, storage, metadata):
    assert storage.metadata == metadata

  def test_snapshot(self, storage):
    # Test current snapshot ID.
    current_snapshot_id = self._SNAPSHOT_ID
    assert storage.snapshot().snapshot_id == current_snapshot_id
    assert storage.snapshot(
        snapshot_id=current_snapshot_id).snapshot_id == current_snapshot_id

    # Test previous snapshot ID.
    assert storage.snapshot(snapshot_id=10).snapshot_id == 10

  def test_create_storage(self, tmp_path):
    dir_path = tmp_path / "test_create_storage" / "dataset"
    storage = Storage.create(location=str(dir_path), logical_schema=None)

    entry_point_file = dir_path / "metadata" / _ENTRY_POINT_FILE
    assert entry_point_file.exists()

    metadata = storage.metadata
    snapshot = storage.snapshot()
    assert metadata.current_snapshot_id == 0
    assert metadata.type == meta.StorageMetadata.DATASET
    assert snapshot.snapshot_id == 0
    assert metadata.create_time == metadata.last_update_time == snapshot.create_time

  def test_load_storage(self, tmp_path):
    dir_path = tmp_path / "test_create_storage" / "dataset"
    storage = Storage.create(location=str(dir_path), logical_schema=None)

    loaded_storage = Storage.load(str(dir_path))
    assert loaded_storage.metadata == storage.metadata

  def test_load_storage_file_not_found_should_fail(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      Storage.load(str(tmp_path / "test_create_storage" / "dataset"))
