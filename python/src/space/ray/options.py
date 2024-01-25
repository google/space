# Copyright 2024 Google LLC
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
"""Options of Space Ray lib."""

from dataclasses import dataclass


@dataclass
class RayOptions:
  """Options of Ray runners."""
  # The max parallelism of computing resources to use in a Ray cluster.
  max_parallelism: int = 8

  # Enable using a row range of an index file as a Ray data block, in the Ray
  # datasource.
  #
  # When disabled, the minimal Ray block is data from one index file and
  # the records it references. Read batch size is achieved by repartition the
  # dataset. For an index Parquet file with 1 million rows, loading the block
  # needs to read all 1 million records, which is too expensive.
  #
  # If enabled, a Ray block size is capped by the provided read batch size.
  # The cost is possible duplicated read of index files. It should be disabled
  # when most data are stored in index files.
  enable_row_range_block: bool = True
