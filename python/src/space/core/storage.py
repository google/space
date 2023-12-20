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
"""Underlying storage implementation for datasets."""

from __future__ import annotations
from os import path
from typing import List, Optional

import pyarrow as pa

from space.core.schema import FieldIdManager
from space.core.schema import substrait as substrait_schema
from space.core.fs.factory import create_fs
import space.core.proto.metadata_pb2 as meta
from space.core.utils import paths
from space.core.utils.protos import proto_now

# Initial snapshot ID.
_INIT_SNAPSHOT_ID = 0


class Storage(paths.StoragePaths):
  """Storage manages data files by metadata using the Space format."""

  def __init__(self, location: str, metadata: meta.StorageMetadata):
    super().__init__(location)
    self._metadata = metadata
    self._fs = create_fs(location)

  def _initialize(self, metadata_path: str) -> None:
    """Initialize a new storage by creating folders and files."""
    self._fs.create_dir(self._data_dir)
    self._fs.create_dir(self._metadata_dir)
    self._write_metadata(metadata_path, self._metadata)

  def _write_metadata(
      self,
      metadata_path: str,
      metadata: meta.StorageMetadata,
  ) -> None:
    """Persist a StorageMetadata to files."""
    self._fs.write_proto(metadata_path, metadata)
    self._fs.write_proto(
        self._entry_point_file,
        meta.EntryPoint(metadata_file=self.short_path(metadata_path)))

  @property
  def metadata(self) -> meta.StorageMetadata:
    """Return the storage metadata."""
    return self._metadata

  def snapshot(self, snapshot_id: Optional[int] = None) -> meta.Snapshot:
    """Return the snapshot specified by a snapshot ID, or current snapshot ID
    if not specified.
    """
    if snapshot_id is None:
      snapshot_id = self._metadata.current_snapshot_id

    if snapshot_id in self._metadata.snapshots:
      return self._metadata.snapshots[snapshot_id]

    raise RuntimeError(f"Snapshot {snapshot_id} is not found")

  @classmethod
  def create(
      cls,
      location: str,
      schema: pa.Schema,
      primary_keys: List[str],
  ) -> Storage:  # pylint: disable=unused-argument
    """Create a new empty storage.
    
    Args:
      location: the directory path to the storage.
      schema: the schema of the storage.
      primary_keys: un-enforced primary keys.
    """
    # TODO: to verify that location is an empty directory.

    field_id_mgr = FieldIdManager()
    schema = field_id_mgr.assign_field_ids(schema)

    now = proto_now()
    # TODO: to convert Arrow schema to Substrait schema.
    metadata = meta.StorageMetadata(
        create_time=now,
        last_update_time=now,
        schema=meta.Schema(fields=substrait_schema.substrait_fields(schema),
                           primary_keys=primary_keys),
        current_snapshot_id=_INIT_SNAPSHOT_ID,
        type=meta.StorageMetadata.DATASET)

    new_metadata_path = paths.new_metadata_path(paths.metadata_dir(location))

    snapshot = meta.Snapshot(snapshot_id=_INIT_SNAPSHOT_ID, create_time=now)
    metadata.snapshots[metadata.current_snapshot_id].CopyFrom(snapshot)

    storage = Storage(location, metadata)
    storage._initialize(new_metadata_path)
    return storage

  @classmethod
  def load(cls, location: str) -> Storage:
    """Load an existing storage from the given location."""
    fs = create_fs(location)
    entry_point = fs.read_proto(paths.entry_point_path(location),
                                meta.EntryPoint())
    metadata = fs.read_proto(path.join(location, entry_point.metadata_file),
                             meta.StorageMetadata())
    return Storage(location, metadata)
