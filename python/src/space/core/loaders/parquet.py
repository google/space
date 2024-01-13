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
"""Load Parquet files into Space datasets."""

from typing import Optional
import glob

import pyarrow.parquet as pq

from space.core.manifests import IndexManifestWriter
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.ops import utils
from space.core.schema import arrow
from space.core.utils.paths import StoragePathsMixin


class LocalParquetLoadOp(StoragePathsMixin):
  """Load ArrayRecord files into Space without copying data."""

  def __init__(self, location: str, metadata: meta.StorageMetadata,
               pattern: str):
    """
    Args:
      pattern: file path pattern of the input Parquet files, e.g.,
        "/directory/*.parquet"
    """
    StoragePathsMixin.__init__(self, location)

    self._metadata = metadata

    assert len(self._metadata.schema.record_fields) == 0
    self._physical_schema = arrow.arrow_schema(self._metadata.schema.fields,
                                               set(),
                                               physical=True)
    self._input_files = glob.glob(pattern)

  def write(self) -> Optional[rt.Patch]:
    """Write metadata files to load Parquet files to Space dataset."""
    index_manifest_writer = IndexManifestWriter(
        self._metadata_dir, self._physical_schema,
        self._metadata.schema.primary_keys)  # type: ignore[arg-type]
    patch = rt.Patch()

    for f in self._input_files:
      stats = _write_index_manifest(index_manifest_writer, f)
      utils.update_index_storage_stats(base=patch.storage_statistics_update,
                                       update=stats)

    index_manifest_full_path = index_manifest_writer.finish()
    if index_manifest_full_path is not None:
      patch.addition.index_manifest_files.append(
          self.short_path(index_manifest_full_path))

    return patch


def _write_index_manifest(manifest_writer: IndexManifestWriter,
                          file_path: str) -> meta.StorageStatistics:
  # TODO: to verify that file schemas are compatible with dataset.
  metadata = pq.read_metadata(file_path)
  return manifest_writer.write(file_path, metadata)
