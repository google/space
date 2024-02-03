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
"""Local delete operation implementation."""

from abc import abstractmethod
from typing import List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pyroaring import BitMap  # type: ignore[import-not-found]

from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.ops.base import BaseOp
from space.core.options import FileOptions
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.utils import errors
from space.core.utils.paths import StoragePathsMixin
from space.core.schema import constants

# A temporary row ID field used for generating deletion row bitmap.
_ROW_ID_FIELD = "__ROW_ID"


class BaseDeleteOp(BaseOp):
  """Abstract base delete operation class.
  
  The deletion only applies to index files. The rows in record files are
  cleaned up by a separate garbage collection operation.
  """

  @abstractmethod
  def delete(self) -> Optional[rt.Patch]:
    """Delete data matching the filter from the storage.
    
    TODO: a class is not needed for the current single thread implementation.
    To revisit the interface.
    """


class FileSetDeleteOp(BaseDeleteOp, StoragePathsMixin):
  """Delete operation of a given file set running locally.
  
  It can be used as components of more complex operations and distributed
  delete operation.

  Not thread safe.
  """

  # pylint: disable=too-many-arguments
  def __init__(self, location: str, metadata: meta.StorageMetadata,
               file_set: rt.FileSet, filter_: pc.Expression,
               file_options: FileOptions):
    StoragePathsMixin.__init__(self, location)

    if not _validate_files(file_set):
      raise errors.SpaceRuntimeError(
          f"Invalid input file set for delete op:\n{file_set}")

    self._file_set = file_set
    # Rows not matching not filter will be reinserted.
    self._reinsert_filter = ~filter_

    self._append_op = LocalAppendOp(location,
                                    metadata,
                                    file_options,
                                    record_address_input=True)

  def delete(self) -> Optional[rt.Patch]:
    # The index files and manifests deleted, to remove them from index
    # manifests.
    patch = rt.Patch()
    deleted_files: List[str] = []
    deleted_manifest_ids: Set[int] = set()

    deleted_rows = 0
    stats_before_delete = meta.StorageStatistics()
    for file in self._file_set.index_files:
      utils.update_index_storage_stats(stats_before_delete,
                                       file.storage_statistics)

      # TODO: this can be down at row group level.
      index_data = pq.read_table(self.full_path(file.path))
      index_data = index_data.append_column(
          _ROW_ID_FIELD,
          pa.array(np.arange(0, index_data.num_rows,
                             dtype=np.int32)))  # type: ignore[arg-type]
      index_data = index_data.filter(mask=self._reinsert_filter)

      # No row is deleted. No need to re-insert rows.
      if index_data.num_rows == file.storage_statistics.num_rows:
        continue

      # Collect statistics.
      deleted_rows += (file.storage_statistics.num_rows - index_data.num_rows)
      all_deleted = index_data.num_rows == 0

      # Compute deleted row bitmap for change log.
      patch.change_log.deleted_rows.append(
          _build_bitmap(file, index_data, all_deleted))
      index_data = index_data.drop(_ROW_ID_FIELD)  # type: ignore[attr-defined]

      # Record deleted files and manifests information.
      deleted_files.append(file.path)
      deleted_manifest_ids.add(file.manifest_file_id)

      # Write reinsert file for survived rows.
      if all_deleted:
        continue

      # Re-insert survived rows.
      self._append_op.write(index_data)

    if deleted_rows == 0:
      return None

    # Carry over unmodified index files in reinserted manifests.
    deleted_manifest_files: List[str] = []
    for manifest_id in deleted_manifest_ids:
      if manifest_id not in self._file_set.index_manifest_files:
        raise errors.SpaceRuntimeError(
            f"Index manifest ID {manifest_id} not found in file set")

      deleted_manifest_files.append(
          self._file_set.index_manifest_files[manifest_id])

    # Carry over survivor files to the new manifest data.
    # Survivor stands for unmodified files.
    survivor_files_filter = ~_build_file_path_filter(constants.FILE_PATH_FIELD,
                                                     deleted_files)
    survivor_index_manifests = pq.ParquetDataset(
        [self.full_path(f) for f in deleted_manifest_files],
        filters=survivor_files_filter).read()
    if survivor_index_manifests.num_rows > 0:
      self._append_op.append_index_manifest(survivor_index_manifests)

    reinsert_patch = self._append_op.finish()

    # Populate the patch for the delete.
    if reinsert_patch is not None:
      patch.addition.index_manifest_files.extend(
          reinsert_patch.addition.index_manifest_files)

    for f in deleted_manifest_files:
      patch.deletion.index_manifest_files.append(f)

    # Compute storage statistics update.
    survivor_stats = _read_index_statistics(survivor_index_manifests)
    reinsert_stats = (reinsert_patch.storage_statistics_update if reinsert_patch
                      is not None else meta.StorageStatistics())

    deleted_compressed_bytes = (reinsert_stats.index_compressed_bytes +
                                survivor_stats.index_compressed_bytes
                               ) - stats_before_delete.index_compressed_bytes
    deleted_uncompressed_bytes = (
        reinsert_stats.index_uncompressed_bytes +
        survivor_stats.index_uncompressed_bytes
    ) - stats_before_delete.index_uncompressed_bytes

    patch.storage_statistics_update.CopyFrom(
        meta.StorageStatistics(
            num_rows=-deleted_rows,
            index_compressed_bytes=deleted_compressed_bytes,
            index_uncompressed_bytes=deleted_uncompressed_bytes))
    return patch


def _build_file_path_filter(field_name: str, file_paths: List[str]):
  filter_ = pc.scalar(False)
  for f in file_paths:
    filter_ |= (pc.field(field_name) == pc.scalar(f))
  return filter_


def _read_index_statistics(manifest_data: pa.Table) -> meta.StorageStatistics:
  """Read storage statistics of unmodified index files from manifests."""

  def sum_bytes(field_name: str) -> int:
    return sum(manifest_data.column(
        field_name).combine_chunks().to_pylist())  # type: ignore[arg-type]

  return meta.StorageStatistics(index_compressed_bytes=sum_bytes(
      constants.INDEX_COMPRESSED_BYTES_FIELD),
                                index_uncompressed_bytes=sum_bytes(
                                    constants.INDEX_UNCOMPRESSED_BYTES_FIELD))


def _validate_files(file_set: rt.FileSet) -> bool:
  """Return false if the file set does not contain sufficient information for
  deletion.
  """
  for f in file_set.index_files:
    if (not f.path or f.storage_statistics.num_rows == 0 or
        f.manifest_file_id == 0):
      return False

  return len(file_set.index_manifest_files) > 0


def _build_bitmap(file: rt.DataFile, index_data: pa.Table,
                  all_deleted: bool) -> meta.RowBitmap:
  row_bitmap = meta.RowBitmap(file=file.path,
                              num_rows=file.storage_statistics.num_rows)
  if all_deleted:
    row_bitmap.all_rows = True
  else:
    deleted_bitmaps = BitMap()
    deleted_bitmaps.add_range(0, file.storage_statistics.num_rows)
    survivor_bitmaps = BitMap(index_data.column(_ROW_ID_FIELD).to_numpy())
    deleted_bitmaps.difference_update(survivor_bitmaps)
    row_bitmap.roaring_bitmap = deleted_bitmaps.serialize()

  return row_bitmap
