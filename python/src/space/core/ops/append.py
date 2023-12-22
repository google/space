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
"""Local append operation implementation."""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from space.core.manifests.index import IndexManifestWriter
from space.core.ops import utils
from space.core.ops.base import BaseOp, InputData
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as runtime
from space.core.schema.arrow import arrow_schema
from space.core.utils import paths
from space.core.utils.paths import StoragePaths

# TODO: to obtain the values from user provided options.
# Thresholds for writing Parquet files. The sizes are uncompressed bytes.
_MAX_ARRAY_RECORD_BYTES = 100 * 1024 * 1024
_MAX_PARQUET_BYTES = 1 * 1024 * 1024
_MAX_ROW_GROUP_BYTES = 100 * 1024


class BaseAppendOp(BaseOp):
  """Abstract base Append operation class."""

  @abstractmethod
  def write(self, data: InputData) -> None:
    """Append data into storage."""

  @abstractmethod
  def finish(self) -> Optional[runtime.Patch]:
    """Complete the append operation and return a metadata patch."""


@dataclass
class _IndexWriterInfo:
  """Contain information of index file writer."""
  writer: pq.ParquetWriter
  file_path: str


class LocalAppendOp(BaseAppendOp, StoragePaths):
  """Append operation running locally.
  
  It can be used as components of more complex operations and distributed
  append operation.

  Not thread safe.
  """

  def __init__(self, location: str, metadata: meta.StorageMetadata):
    StoragePaths.__init__(self, location)

    self._metadata = metadata
    self._schema = arrow_schema(self._metadata.schema.fields)

    # Data file writers.
    self._index_writer_info: Optional[_IndexWriterInfo] = None

    # Local runtime caches.
    self._cached_index_data: Optional[pa.Table] = None
    self._cached_index_file_bytes = 0

    # Manifest file writers.
    self._index_manifest_writer = IndexManifestWriter(
        self._metadata_dir, self._schema,
        self._metadata.schema.primary_keys)  # type: ignore[arg-type]

    self._patch = runtime.Patch()

  def write(self, data: InputData) -> None:
    if not isinstance(data, pa.Table):
      data = pa.Table.from_pydict(data)

    self._append_arrow(data)

  def finish(self) -> Optional[runtime.Patch]:
    """Complete the append operation and return a metadata patch.

    Returns:
      A patch to the storage or None if no actual storage modification happens.
    """
    # Flush all cached index data.
    if self._cached_index_data is not None:
      self._maybe_create_index_writer()
      assert self._index_writer_info is not None
      self._index_writer_info.writer.write_table(self._cached_index_data)

    if self._index_writer_info is not None:
      self._finish_index_writer()

    index_manifest_full_path = self._index_manifest_writer.finish()
    if index_manifest_full_path is not None:
      self._patch.added_index_manifest_files.append(
          self.short_path(index_manifest_full_path))

    if self._patch.storage_statistics_update.num_rows == 0:
      return None

    return self._patch

  def _append_arrow(self, data: pa.Table) -> None:
    # TODO: to verify the schema of input data.
    if data.num_rows == 0:
      return

    # TODO: index data is a subset of fields of data when we consider record
    # fields.
    index_data = data
    self._maybe_create_index_writer()

    self._cached_index_file_bytes += index_data.nbytes

    if self._cached_index_data is None:
      self._cached_index_data = index_data
    else:
      self._cached_index_data = pa.concat_tables(
          [self._cached_index_data, index_data])

    # _cached_index_data is written as a new row group.
    if self._cached_index_data.nbytes > _MAX_ROW_GROUP_BYTES:
      assert self._index_writer_info is not None
      self._index_writer_info.writer.write_table(self._cached_index_data)
      self._cached_index_data = None

      # Materialize the index file.
      if self._cached_index_file_bytes > _MAX_PARQUET_BYTES:
        self._finish_index_writer()

  def _maybe_create_index_writer(self) -> None:
    """Create a new index file writer if needed."""
    if self._index_writer_info is None:
      full_file_path = paths.new_index_file_path(self._data_dir)
      writer = pq.ParquetWriter(full_file_path, self._schema)
      self._index_writer_info = _IndexWriterInfo(
          writer, self.short_path(full_file_path))

  def _finish_index_writer(self) -> None:
    """Materialize a new index file, update metadata and stats."""
    if self._index_writer_info is None:
      return

    self._index_writer_info.writer.close()

    # Update metadata in manifest files.
    stats = self._index_manifest_writer.write(
        self._index_writer_info.file_path,
        self._index_writer_info.writer.writer.metadata)
    utils.update_index_storage_statistics(
        base=self._patch.storage_statistics_update, update=stats)

    self._index_writer_info = None
    self._cached_index_file_bytes = 0
