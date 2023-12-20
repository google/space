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
"""Utilities for operation classes."""

from space.core.proto import metadata_pb2 as meta


def update_index_storage_statistics(
    base: meta.StorageStatistics,
    update: meta.StorageStatistics,
) -> None:
  """Update index storage statistics."""
  base.num_rows += base.num_rows
  base.index_compressed_bytes += update.index_compressed_bytes
  base.index_uncompressed_bytes += update.index_uncompressed_bytes
