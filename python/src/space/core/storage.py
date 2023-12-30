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
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc

from space.core.fs.factory import create_fs
from space.core.manifests.falsifiable_filters import build_manifest_filter
from space.core.manifests.index import read_index_manifests
from space.core.ops import utils as ops_utils
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.schema import FieldIdManager
from space.core.schema import arrow
from space.core.schema import substrait as substrait_schema
from space.core.utils import paths
from space.core.utils.protos import proto_now

# Initial snapshot ID.
_INIT_SNAPSHOT_ID = 0


class Storage(paths.StoragePathsMixin):
  """Storage manages data files by metadata using the Space format."""

  def __init__(self, location: str, metadata: meta.StorageMetadata):
    super().__init__(location)
    self._metadata = metadata
    self._fs = create_fs(location)

    record_fields = set(self._metadata.schema.record_fields)
    self._logical_schema = arrow.arrow_schema(self._metadata.schema.fields,
                                              record_fields,
                                              physical=False)
    self._physical_schema = arrow.logical_to_physical_schema(
        self._logical_schema, record_fields)

    self._field_name_ids: Dict[str, int] = arrow.field_name_to_id_dict(
        self._physical_schema)

    self._primary_keys = set(self._metadata.schema.primary_keys)

  @property
  def metadata(self) -> meta.StorageMetadata:
    """Return the storage metadata."""
    return self._metadata

  @property
  def primary_keys(self) -> List[str]:
    """Return the storage primary keys."""
    return list(self._metadata.schema.primary_keys)

  @property
  def record_fields(self) -> List[str]:
    """Return record field names."""
    return list(self._metadata.schema.record_fields)

  @property
  def logical_schema(self) -> pa.Schema:
    """Return the user specified schema."""
    return self._logical_schema

  @property
  def physical_schema(self) -> pa.Schema:
    """Return the physcal schema that uses reference for record fields."""
    return self._physical_schema

  def snapshot(self, snapshot_id: Optional[int] = None) -> meta.Snapshot:
    """Return the snapshot specified by a snapshot ID, or current snapshot ID
    if not specified.
    """
    if snapshot_id is None:
      snapshot_id = self._metadata.current_snapshot_id

    if snapshot_id in self._metadata.snapshots:
      return self._metadata.snapshots[snapshot_id]

    raise RuntimeError(f"Snapshot {snapshot_id} is not found")

  # pylint: disable=too-many-arguments
  @classmethod
  def create(
      cls,
      location: str,
      schema: pa.Schema,
      primary_keys: List[str],
      record_fields: List[str],
      logical_plan: Optional[meta.LogicalPlan] = None
  ) -> Storage:  # pylint: disable=unused-argument
    """Create a new empty storage.

    Args:
      location: the directory path to the storage.
      schema: the schema of the storage.
      primary_keys: un-enforced primary keys.
      record_fields: fields stored in row format (ArrayRecord).
      logical_plan: logical plan of materialized view.
    """
    # TODO: to verify that location is an empty directory.
    # TODO: to verify primary key fields and record_fields (and types) are
    # valid.
    # TODO: to auto infer record_fields.

    field_id_mgr = FieldIdManager()
    schema = field_id_mgr.assign_field_ids(schema)

    now = proto_now()
    # TODO: to convert Arrow schema to Substrait schema.
    metadata = meta.StorageMetadata(
        create_time=now,
        last_update_time=now,
        schema=meta.Schema(
            fields=substrait_schema.substrait_fields(schema),
            primary_keys=primary_keys,
            # TODO: to optionally auto infer record fields.
            record_fields=record_fields),
        current_snapshot_id=_INIT_SNAPSHOT_ID,
        type=meta.StorageMetadata.DATASET)

    if logical_plan is not None:
      metadata.type = meta.StorageMetadata.MATERIALIZED_VIEW
      metadata.logical_plan.CopyFrom(logical_plan)

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

  def commit(self, patch: rt.Patch) -> None:
    """Commit changes to the storage.

    TODO: only support a single writer; to ensure atomicity in commit by
    concurrent writers.

    Args:
      patch: a patch describing changes made to the storage.
    """
    new_metadata = meta.StorageMetadata()
    new_metadata.CopyFrom(self._metadata)

    new_snapshot_id = self._next_snapshot_id()
    new_metadata.current_snapshot_id = new_snapshot_id
    current_snapshot = self.snapshot()

    new_metadata_path = self.new_metadata_path()

    snapshot = meta.Snapshot(
        snapshot_id=new_snapshot_id,
        create_time=proto_now(),
        manifest_files=current_snapshot.manifest_files,
        storage_statistics=current_snapshot.storage_statistics)
    _patch_manifests(snapshot.manifest_files, patch)

    if patch.HasField('change_log'):
      change_log_file = self.new_change_log_path()
      self._fs.write_proto(change_log_file, patch.change_log)
      snapshot.change_log_file = self.short_path(change_log_file)

    # Update storage statistics.
    ops_utils.update_index_storage_stats(snapshot.storage_statistics,
                                         patch.storage_statistics_update)
    ops_utils.update_record_stats_bytes(snapshot.storage_statistics,
                                        patch.storage_statistics_update)

    # Set new snapshot and materialize new metadata.
    new_metadata.snapshots[new_snapshot_id].CopyFrom(snapshot)
    self._write_metadata(new_metadata_path, new_metadata)
    self._metadata = new_metadata

  def data_files(self,
                 filter_: Optional[pc.Expression] = None,
                 snapshot_id: Optional[int] = None) -> rt.FileSet:
    """Return the data files and the manifest files containing them.
    
    Args:
      filter_: a filter on the data.
      snapshot_id: read a specified snapshot instead of the current.
    """
    manifest_files = self.snapshot(snapshot_id).manifest_files
    result = rt.FileSet()

    # A temporily assigned identifier for tracking manifest files.
    # Start from 1 to detect unassigned values 0 that is default.
    manifest_file_id = 1

    # Construct falsifiable filter to prune manifest rows.
    manifest_filter = None
    if filter_ is not None:
      manifest_filter = build_manifest_filter(self._physical_schema,
                                              self._primary_keys,
                                              self._field_name_ids, filter_)

    for manifest_file in manifest_files.index_manifest_files:
      result_per_manifest = read_index_manifests(self.full_path(manifest_file),
                                                 manifest_file_id,
                                                 manifest_filter)
      if not result_per_manifest.index_files:
        continue

      result.index_manifest_files[manifest_file_id] = manifest_file
      manifest_file_id += 1
      result.index_files.extend(result_per_manifest.index_files)

    return result

  @property
  def snapshot_ids(self) -> List[int]:
    """A list of all alive snapshot IDs in the dataset."""
    return list(self._metadata.snapshots)

  def _initialize(self, metadata_path: str) -> None:
    """Initialize a new storage by creating folders and files."""
    self._fs.create_dir(self._data_dir)
    self._fs.create_dir(self._metadata_dir)
    self._fs.create_dir(self._change_data_dir)
    self._write_metadata(metadata_path, self._metadata)

  def _next_snapshot_id(self) -> int:
    return self._metadata.current_snapshot_id + 1

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


def _patch_manifests(manifest_files: meta.ManifestFiles, patch: rt.Patch):
  """Apply changes in a patch to manifest files for a commit."""
  # Process deleted index manifest files.
  deleted_manifests = set(patch.deletion.index_manifest_files)
  for i in range(len(manifest_files.index_manifest_files) - 1, -1, -1):
    if manifest_files.index_manifest_files[i] in deleted_manifests:
      del manifest_files.index_manifest_files[i]

  # Process added manifest files.
  for f in patch.addition.index_manifest_files:
    manifest_files.index_manifest_files.append(f)

  for f in patch.addition.record_manifest_files:
    manifest_files.record_manifest_files.append(f)
