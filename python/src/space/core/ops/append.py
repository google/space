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
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from space.core.manifests import IndexManifestWriter
from space.core.manifests import RecordManifestWriter
from space.core.ops import utils
from space.core.ops.base import BaseOp, InputData
from space.core.options import FileOptions
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.schema import arrow
from space.core.schema import utils as schema_utils
from space.core.utils import paths
from space.core.utils.lazy_imports_utils import array_record_module as ar
from space.core.utils.paths import StoragePathsMixin


class BaseAppendOp(BaseOp):
  """Abstract base append operation class."""

  @abstractmethod
  def write(self, data: InputData) -> None:
    """Append data into storage."""

  @abstractmethod
  def finish(self) -> Optional[rt.Patch]:
    """Complete the append operation and return a metadata patch."""


@dataclass
class _IndexWriterInfo:
  """Information of index file writer."""
  writer: pq.ParquetWriter
  file_path: str


@dataclass
class _RecordWriterInfo:
  """Information of record file writer."""
  writer: ar.ArrayRecordWriter
  file_path: str
  file_id: int = 0
  next_row_id: int = 0
  storage_statistics: meta.StorageStatistics = dataclass_field(
      default_factory=meta.StorageStatistics)


class LocalAppendOp(BaseAppendOp, StoragePathsMixin):
  """Append operation running locally.
  
  It can be used as components of more complex operations and distributed
  append operation.

  Not thread safe.
  """

  def __init__(self,
               location: str,
               metadata: meta.StorageMetadata,
               file_options: FileOptions,
               record_address_input: bool = False):
    """
    Args:
      record_address_input: if true, input record fields are addresses.
    """
    StoragePathsMixin.__init__(self, location)
    self._parquet_options = file_options.parquet_options
    self._array_record_options = file_options.array_record_options

    self._metadata = metadata
    record_fields = set(self._metadata.schema.record_fields)
    self._physical_schema = arrow.arrow_schema(self._metadata.schema.fields,
                                               record_fields,
                                               physical=True)
    self._index_fields, self._record_fields = arrow.classify_fields(
        self._physical_schema, record_fields, selected_fields=None)

    self._record_address_input = record_address_input

    # Data file writers.
    self._index_writer_info: Optional[_IndexWriterInfo] = None

    # Key is field name.
    self._record_writers: Dict[str, _RecordWriterInfo] = {}

    self._cached_index_data: Optional[pa.Table] = None
    self._cached_index_file_bytes = 0

    # Manifest file writers.
    self._index_manifest_writer = IndexManifestWriter(
        self._metadata_dir, self._physical_schema,
        self._metadata.schema.primary_keys)  # type: ignore[arg-type]
    self._record_manifest_writer = RecordManifestWriter(self._metadata_dir)

    self._patch = rt.Patch()

  def write(self, data: InputData) -> None:
    if not isinstance(data, pa.Table):
      data = pa.Table.from_pydict(data)

    self._write_arrow(data)

  def finish(self) -> Optional[rt.Patch]:
    """Complete the append operation and return a metadata patch.

    Returns:
      A patch to the storage or None if no actual storage modification happens.
    """
    # Flush all cached record data.
    for f in self._record_fields:
      if f.name in self._record_writers:
        self._finish_record_writer(f, self._record_writers[f.name])

    # Flush all cached index data.
    if self._cached_index_data is not None:
      self._maybe_create_index_writer()
      assert self._index_writer_info is not None
      self._index_writer_info.writer.write_table(self._cached_index_data)

    if self._index_writer_info is not None:
      self._finish_index_writer()

    # Write manifest files.
    index_manifest_full_path = self._index_manifest_writer.finish()
    if index_manifest_full_path is not None:
      self._patch.addition.index_manifest_files.append(
          self.short_path(index_manifest_full_path))

    record_manifest_path = self._record_manifest_writer.finish()
    if record_manifest_path:
      self._patch.addition.record_manifest_files.append(
          self.short_path(record_manifest_path))

    if self._patch.storage_statistics_update.num_rows == 0:
      return None

    return self._patch

  def append_index_manifest(self, index_manifest_table: pa.Table) -> None:
    """Append external index manifest data."""
    return self._index_manifest_writer.write_arrow(index_manifest_table)

  def _write_arrow(self, data: pa.Table) -> None:
    """Write an Arrow table into the storage.

    The Parquet writer uses storage physical schema, it checks `data`'s schema
    is compatible at write time.
    """
    if data.num_rows == 0:
      return

    # TODO: index data is a subset of fields of data when we consider record
    # fields.
    index_data = data
    self._maybe_create_index_writer()

    index_data = data.select(schema_utils.field_names(self._index_fields))

    # Write record fields into files.
    # TODO: to parallelize it.
    # TODO: to preserve the field order in schema.
    if self._record_address_input:
      for f in self._record_fields:
        index_data = index_data.append_column(f.name, data.column(
            f.name))  # type: ignore[arg-type]
    else:
      record_addresses = [
          self._write_record_column(f, data.column(f.name))
          for f in self._record_fields
      ]

      for field_name, address_column in record_addresses:
        # TODO: the field/column added must have field ID.
        index_data = index_data.append_column(field_name, address_column)

    # Write index fields into files.
    self._cached_index_file_bytes += index_data.nbytes

    if self._cached_index_data is None:
      self._cached_index_data = index_data
    else:
      self._cached_index_data = pa.concat_tables(
          [self._cached_index_data, index_data])

    # _cached_index_data is written as a new row group.
    if (self._cached_index_data.nbytes
        > self._parquet_options.max_uncompressed_row_group_bytes):
      assert self._index_writer_info is not None
      self._index_writer_info.writer.write_table(self._cached_index_data)
      self._cached_index_data = None

      # Materialize the index file.
      if (self._cached_index_file_bytes
          > self._parquet_options.max_uncompressed_file_bytes):
        self._finish_index_writer()

  def _maybe_create_index_writer(self) -> None:
    """Create a new index file writer if needed."""
    if self._index_writer_info is None:
      full_file_path = paths.new_index_file_path(self._data_dir)
      writer = pq.ParquetWriter(full_file_path, self._physical_schema)
      self._index_writer_info = _IndexWriterInfo(
          writer, self.short_path(full_file_path))

  def _finish_index_writer(self) -> None:
    """Materialize a new index file (Parquet), update metadata and stats."""
    if self._index_writer_info is None:
      return

    self._index_writer_info.writer.close()

    # Update metadata in manifest files.
    file_path = self._index_writer_info.file_path
    stats = self._index_manifest_writer.write(
        file_path, self._index_writer_info.writer.writer.metadata)
    utils.update_index_storage_stats(base=self._patch.storage_statistics_update,
                                     update=stats)

    self._patch.change_log.added_rows.append(
        meta.RowBitmap(file=file_path, all_rows=True, num_rows=stats.num_rows))

    self._index_writer_info = None
    self._cached_index_file_bytes = 0

  def _write_record_column(
      self, field: schema_utils.Field,
      column: pa.ChunkedArray) -> Tuple[str, pa.StructArray]:
    """Write record field into files.

    Returns:
      A tuple (field_name, address_column).
    """
    field_name = field.name

    # TODO: this section needs to be locked when supporting threaded execution.
    if field_name in self._record_writers:
      writer_info = self._record_writers[field_name]
    else:
      file_path = paths.new_record_file_path(self._data_dir, field_name)
      writer = ar.ArrayRecordWriter(file_path,
                                    options=self._array_record_options.options)
      writer_info = _RecordWriterInfo(writer, self.short_path(file_path))
      self._record_writers[field_name] = writer_info

    num_rows = column.length()
    writer_info.storage_statistics.num_rows += num_rows
    writer_info.storage_statistics.record_uncompressed_bytes += column.nbytes

    for chunk in column.chunks:
      for v in chunk:
        writer_info.writer.write(v.as_py())

    # Generate record address field values to return.
    next_row_id = writer_info.next_row_id + num_rows
    address_column = utils.address_column(writer_info.file_path,
                                          writer_info.next_row_id, num_rows)
    writer_info.next_row_id = next_row_id

    # Materialize the file when size is over threshold.
    if (writer_info.storage_statistics.record_uncompressed_bytes
        > self._array_record_options.max_uncompressed_file_bytes):
      self._finish_record_writer(field, writer_info)

    return field_name, address_column

  def _finish_record_writer(self, field: schema_utils.Field,
                            writer_info: _RecordWriterInfo) -> None:
    """Materialize a new record file (ArrayRecord), update metadata and
    stats.
    """
    writer_info.writer.close()
    self._record_manifest_writer.write(writer_info.file_path, field.field_id,
                                       writer_info.storage_statistics)
    utils.update_record_stats_bytes(self._patch.storage_statistics_update,
                                    writer_info.storage_statistics)

    del self._record_writers[field.name]
