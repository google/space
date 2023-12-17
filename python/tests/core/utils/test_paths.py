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

from mock import patch
import pytest

from space.core.utils import paths

_UUID_PATH = "space.core.utils.paths.uuid_"


def _mocked_uuid() -> str:
  return "<uuid>"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_new_index_file_path(mock_uuid):
  assert paths.new_index_file_path("data") == "data/index_<uuid>.parquet"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_new_record_file_path(mock_uuid):
  assert paths.new_record_file_path("data",
                                    "field") == "data/field_<uuid>.arrowrecord"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_new_index_manifest_path(mock_uuid):
  assert paths.new_index_manifest_path(
      "metadata") == "metadata/index_manifest_<uuid>.parquet"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_new_record_manifest_path(mock_uuid):
  assert paths.new_record_manifest_path(
      "metadata") == "metadata/record_manifest_<uuid>.parquet"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_data_dir(mock_uuid):
  assert paths.data_dir("location") == "location/data"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_metadata_dir(mock_uuid):
  assert paths.metadata_dir("location") == "location/metadata"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_entry_point_path(mock_uuid):
  assert paths.entry_point_path(
      "location") == "location/metadata/entrypoint.txtpb"


@patch(_UUID_PATH, side_effect=_mocked_uuid)
def test_new_metadata_path(mock_uuid):
  assert paths.new_metadata_path(
      "metadata") == "metadata/metadata_<uuid>.txtpb"


class TestStoragePaths:

  _LOCATION = "location"

  @pytest.fixture
  def storage_paths(self):
    return paths.StoragePaths(self._LOCATION)

  def test_data_dir(self, storage_paths):
    assert storage_paths.data_dir == f"{self._LOCATION}/data"

  def test_short_path(self, storage_paths):
    assert storage_paths.short_path(
        f"{self._LOCATION}/metadata/file.parquet") == "metadata/file.parquet"

  def test_full_path(self, storage_paths):
    assert storage_paths.full_path(
        "data/file.parquet") == f"{self._LOCATION}/data/file.parquet"

  def test_new_metadata_path(self, storage_paths):
    with patch(_UUID_PATH, side_effect=_mocked_uuid):
      assert storage_paths.new_metadata_path(
      ) == f"{self._LOCATION}/metadata/metadata_<uuid>.txtpb"
