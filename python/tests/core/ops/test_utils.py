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

from space.core.ops import utils
from space.core.proto import metadata_pb2 as meta


def test_update_index_storage_stats_positive():
  base = meta.StorageStatistics(num_rows=100,
                                index_compressed_bytes=200,
                                index_uncompressed_bytes=300)

  utils.update_index_storage_stats(
      base,
      meta.StorageStatistics(num_rows=10,
                             index_compressed_bytes=20,
                             index_uncompressed_bytes=30))
  assert base == meta.StorageStatistics(num_rows=110,
                                        index_compressed_bytes=220,
                                        index_uncompressed_bytes=330)


def test_update_index_storage_stats_negative():
  base = meta.StorageStatistics(num_rows=100,
                                index_compressed_bytes=200,
                                index_uncompressed_bytes=300)

  utils.update_index_storage_stats(
      base,
      meta.StorageStatistics(num_rows=-10,
                             index_compressed_bytes=-20,
                             index_uncompressed_bytes=-30))
  assert base == meta.StorageStatistics(num_rows=90,
                                        index_compressed_bytes=180,
                                        index_uncompressed_bytes=270)
