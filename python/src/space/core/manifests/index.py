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
"""Index manifest files writer and reader implementation."""

import typing
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from space.core.fs.parquet import write_parquet_file
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt
from space.core.schema import constants
from space.core.schema import utils as schema_utils
from space.core.schema.arrow import field_id, field_id_to_column_id_dict
from space.core.utils import paths


def _stats_subfields(type_: pa.DataType) -> List[pa.Field]:
  """Column stats struct field sub-fields."""
  return [
      pa.field(constants.MIN_FIELD, type_),
      pa.field(constants.MAX_FIELD, type_)
  ]


def _manifest_schema(
    schema: pa.Schema,
    primary_keys: List[str]) -> Tuple[pa.Schema, List[Tuple[int, pa.DataType]]]:
  """Build the index manifest file schema, based on storage schema."""
  primary_keys_ = set(primary_keys)

  fields = [(constants.FILE_PATH_FIELD, pa.utf8()),
            (constants.NUM_ROWS_FIELD, pa.int64()),
            (constants.INDEX_COMPRESSED_BYTES_FIELD, pa.int64()),
            (constants.INDEX_UNCOMPRESSED_BYTES_FIELD, pa.int64())]

  # Fields to collect Parquet column statistics: [(field_id, type), ...].
  stats_fields: List[Tuple[int, pa.DataType]] = []

  for f in schema:
    if f.name not in primary_keys_:
      continue

    field_id_ = field_id(f)
    fields.append((schema_utils.stats_field_name(field_id_),
                   pa.struct(_stats_subfields(f.type))))
    stats_fields.append((field_id_, f.type))

  return pa.schema(fields), stats_fields  # type: ignore[arg-type]


class _FieldStats:
  """A local cache for merging stats from row groups."""

  def __init__(self, type_: pa.DataType):
    self._fields = _stats_subfields(type_)

    self._min: Any = None
    self._max: Any = None

    # _min and _max cache the stats of one file; after finalizing they are
    # moved to _min_list and _max_list that cache stats of all files.
    self._min_list: List[Any] = []
    self._max_list: List[Any] = []

  def merge(self, stats: pq.Statistics) -> None:
    """Merge the cached stats with a new stats."""
    if self._min is None or stats.min < self._min:
      self._min = stats.min

    if self._max is None or stats.max > self._max:
      self._max = stats.max

  def finish(self) -> None:
    """Finialize the stats for the current file."""
    self._min_list.append(self._min)
    self._max_list.append(self._max)

    self._min = None
    self._max = None

  def to_arrow(self) -> pa.Array:
    """Return the Arrow format of the stats."""
    return pa.StructArray.from_arrays(
        [self._min_list, self._max_list],  # type: ignore[arg-type]
        fields=self._fields)


class IndexManifestWriter:
  """Writer of index manifest files."""

  def __init__(self, metadata_dir: str, schema: pa.Schema,
               primary_keys: List[str]):
    """
    Args:
      metadata_dir: full directory path of metadata folder.
      schema: the storage schema, either logical or physical schema works
        because only primary key fields are used for column stats.
      primary_keys: un-enforced storage primary keys; index fields only.
    """
    self._metadata_dir = metadata_dir
    self._manifest_schema, self._stats_fields = _manifest_schema(
        schema, primary_keys)

    # Cache manifest file field values in memory.
    self._file_paths: List[str] = []
    self._num_rows: List[int] = []
    self._index_compressed_bytes: List[int] = []
    self._index_uncompressed_bytes: List[int] = []

    # Caches for column stats.
    self._stats_column_ids: List[int] = []
    # Key is column ID.
    self._field_stats_dict: Dict[int, _FieldStats] = {}

    column_id_dict = field_id_to_column_id_dict(schema)
    for field_id_, type_ in self._stats_fields:
      column_id = column_id_dict[field_id_]
      self._stats_column_ids.append(column_id)
      self._field_stats_dict[column_id] = _FieldStats(type_)

    self._cached_manifest_data: List[pa.Table] = []

  @property
  def manifest_schema(self) -> pa.Schema:
    """The schema of index manifest files."""
    return self._manifest_schema

  def write(self, file_path: str,
            parquet_metadata: pq.FileMetaData) -> meta.StorageStatistics:
    """Write a new manifest row.
    
    Args:
      file_path: a relative file path of the index file.
      parquet_metadata: the Parquet metadata of the index file.

    Returns:
      The storage statistics of the index file.
    """
    self._file_paths.append(file_path)
    self._num_rows.append(parquet_metadata.num_rows)

    index_compressed_bytes = 0
    index_uncompressed_bytes = 0

    for rg in range(parquet_metadata.num_row_groups):
      rg_metadata = parquet_metadata.row_group(rg)
      for column_id, field_stats in self._field_stats_dict.items():
        column_stats = rg_metadata.column(column_id)
        field_stats.merge(column_stats.statistics)
        index_compressed_bytes += column_stats.total_compressed_size
        index_uncompressed_bytes += column_stats.total_uncompressed_size

    for field_stats in self._field_stats_dict.values():
      field_stats.finish()

    self._index_compressed_bytes.append(index_compressed_bytes)
    self._index_uncompressed_bytes.append(index_uncompressed_bytes)

    return meta.StorageStatistics(
        num_rows=parquet_metadata.num_rows,
        index_compressed_bytes=index_compressed_bytes,
        index_uncompressed_bytes=index_uncompressed_bytes)

  def write_arrow(self, manifest_data: pa.Table) -> None:
    """Write manifest rows in Arrow format."""
    self._cached_manifest_data.append(manifest_data)

  def finish(self) -> Optional[str]:
    """Materialize the manifest file and return the file path."""
    # Convert cached manifest data to Arrow.
    if self._file_paths:
      arrays = [
          self._file_paths, self._num_rows, self._index_compressed_bytes,
          self._index_uncompressed_bytes
      ]

      for column_id in self._stats_column_ids:
        arrays.append(self._field_stats_dict[column_id].to_arrow())

      self._cached_manifest_data.append(
          pa.Table.from_arrays(
              arrays=arrays,
              schema=self._manifest_schema))  # type: ignore[call-arg]

    if not self._cached_manifest_data:
      return None

    manifest_data = pa.concat_tables(self._cached_manifest_data)
    if manifest_data.num_rows == 0:
      return None

    file_path = paths.new_index_manifest_path(self._metadata_dir)
    write_parquet_file(file_path, self._manifest_schema, [manifest_data])
    return file_path


@dataclass
class _IndexManifests:
  """Represent index manfiests read from a Parquet file."""
  file_path: pa.StringArray
  num_rows: pa.Int64Array
  index_compressed_bytes: pa.Int64Array
  index_uncompressed_bytes: pa.Int64Array


def read_index_manifests(manifest_path: str,
                         manifest_file_id: int,
                         filter_: Optional[pc.Expression] = None) -> rt.FileSet:
  """Read an index manifest file.
  
  Args:
    manifest_path: full file path of the manifest file.
    manifest_file_id: a temporary manifest file ID assigned by the caller.
    filter_: a filter on the index manifest rows.

  Returns:
    A file set of data files in the manifest file.
  """
  table = pq.read_table(manifest_path,
                        filters=filter_)  # type: ignore[arg-type]

  manifests = _index_manifests(table)

  file_set = rt.FileSet()
  for i in range(table.num_rows):
    # TODO: to populate record_uncompressed_bytes by sampling record manifest
    # files.
    stats = meta.StorageStatistics(
        num_rows=manifests.num_rows[i].as_py(),
        index_compressed_bytes=manifests.index_compressed_bytes[i].as_py(),
        index_uncompressed_bytes=manifests.index_uncompressed_bytes[i].as_py())
    file = rt.DataFile(path=manifests.file_path[i].as_py(),
                       manifest_file_id=manifest_file_id,
                       storage_statistics=stats)
    file_set.index_files.append(file)

  return file_set


@typing.no_type_check
def _index_manifests(table: pa.Table) -> _IndexManifests:
  return _IndexManifests(
      file_path=table.column(constants.FILE_PATH_FIELD).combine_chunks(),
      num_rows=table.column(constants.NUM_ROWS_FIELD).combine_chunks(),
      index_compressed_bytes=table.column(
          constants.INDEX_COMPRESSED_BYTES_FIELD).combine_chunks(),
      index_uncompressed_bytes=table.column(
          constants.INDEX_UNCOMPRESSED_BYTES_FIELD).combine_chunks())
