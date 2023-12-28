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
"""Utility methods for file paths."""

from os import path

from space.core.utils.uuids import uuid_

# Folders of storage metadata.
_ENTRY_POINT_FILE = "entrypoint.txtpb"
_DATA_DIR = "data"
_METADATA_DIR = "metadata"
_CHANGE_DATA_DIR = "changes"
# Folder of user defined functions for materialized views.
UDF_DIR = 'udfs'


def new_index_file_path(data_dir_: str):
  """Return a random index file path in a given data directory.."""
  return path.join(data_dir_, f"index_{uuid_()}.parquet")


def new_record_file_path(data_dir_: str, field_name: str):
  """Return a random record file path in a given data directory.."""
  return path.join(data_dir_, f"{field_name}_{uuid_()}.array_record")


def new_index_manifest_path(metadata_dir_: str):
  """Return a random index manifest file path in a given metadata directory."""
  return path.join(metadata_dir_, f"index_manifest_{uuid_()}.parquet")


def new_record_manifest_path(metadata_dir_: str):
  """Return a random record manifest file path in a given metadata directory."""
  return path.join(metadata_dir_, f"record_manifest_{uuid_()}.parquet")


def data_dir(location: str) -> str:
  """Return the data directory path in a given location."""
  return path.join(location, _DATA_DIR)


def metadata_dir(location: str) -> str:
  """Return the metadata directory path in a given location."""
  return path.join(location, _METADATA_DIR)


def entry_point_path(location: str) -> str:
  """Return the static entry point file path in a given location."""
  return path.join(location, _METADATA_DIR, _ENTRY_POINT_FILE)


def new_metadata_path(metadata_dir_: str) -> str:
  """Return a random metadata file path in a given metadata directory."""
  return path.join(metadata_dir_, f"metadata_{uuid_()}.txtpb")


class StoragePathsMixin:
  """Provides util methods for file and directory paths."""

  def __init__(self, location: str):
    self._location = location

    self._data_dir = data_dir(self._location)
    self._metadata_dir = metadata_dir(self._location)
    self._change_data_dir = path.join(self._metadata_dir, _CHANGE_DATA_DIR)
    self._entry_point_file = entry_point_path(self._location)

  @property
  def location(self) -> str:
    """Return the storage base folder location."""
    return self._location

  @property
  def data_dir(self) -> str:
    """Return the data directory."""
    return self._data_dir

  @property
  def metadata_dir(self) -> str:
    """Return the metadata directory."""
    return self._metadata_dir

  def short_path(self, full_path: str) -> str:
    """Return the short relative path from a full path."""
    return path.relpath(full_path, self._location)

  def full_path(self, short_path: str) -> str:
    """Return the full path from a full or short path."""
    return path.join(self._location, short_path)

  def new_metadata_path(self) -> str:
    """Return a random metadata file path."""
    return new_metadata_path(self._metadata_dir)

  def new_change_log_path(self) -> str:
    """Return a random change log file path."""
    return path.join(self._change_data_dir, f"change_{uuid_()}.txtpb")
