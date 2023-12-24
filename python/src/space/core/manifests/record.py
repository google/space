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
"""Record manifest files writer and reader implementation."""

from typing import List, Optional

import pyarrow as pa

from space.core.fs.parquet import write_parquet_file
import space.core.proto.metadata_pb2 as meta
from space.core.utils import paths
from space.core.schema import constants


def _manifest_schema() -> pa.Schema:
  fields = [(constants.FILE_PATH_FIELD, pa.utf8()),
            (constants.FIELD_ID_FIELD, pa.int32()),
            (constants.NUM_ROWS_FIELD, pa.int64()),
            (constants.UNCOMPRESSED_BYTES_FIELD, pa.int64())]
  return pa.schema(fields)  # type: ignore[arg-type]


class RecordManifestWriter:
  """Writer of record manifest files."""

  def __init__(self, metadata_dir: str):
    self._metadata_dir = metadata_dir
    self._manifest_schema = _manifest_schema()

    self._file_paths: List[str] = []
    self._field_ids: List[int] = []
    self._num_rows: List[int] = []
    self._uncompressed_bytes: List[int] = []

  def write(self, file_path: str, field_id: int,
            storage_statistics: meta.StorageStatistics) -> None:
    """Write a new manifest row.
    
    Args:
      file_path: a relative file path of the index file.
      field_id: the field ID of the associated field for this ArrayRecord file.
      storage_statistics: storage statistics of the file.
    """
    self._file_paths.append(file_path)
    self._field_ids.append(field_id)
    self._num_rows.append(storage_statistics.num_rows)
    self._uncompressed_bytes.append(
        storage_statistics.record_uncompressed_bytes)

  def finish(self) -> Optional[str]:
    """Materialize the manifest file and return the file path."""
    if not self._file_paths:
      return None

    arrays = [
        self._file_paths, self._field_ids, self._num_rows,
        self._uncompressed_bytes
    ]
    manifest_data = pa.Table.from_arrays(
        arrays,  # type: ignore[arg-type]
        schema=self._manifest_schema)  # type: ignore[call-arg]

    file_path = paths.new_record_manifest_path(self._metadata_dir)
    write_parquet_file(file_path, self._manifest_schema, [manifest_data])
    return file_path
