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
"""Options of Space core lib."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import pyarrow.compute as pc


@dataclass
class ReadOptions:
  """Options of reading data."""
  # Filters on index fields.
  filter_: Optional[pc.Expression] = None

  # When specified, only read the given fields instead of all fields.
  fields: Optional[List[str]] = None

  # The snapshot to read.
  snapshot_id: Optional[int] = None

  # If true, read the references (e.g., address) of read record fields instead
  # of values.
  reference_read: bool = False

  # The max number of rows per batch in read result.
  #
  # `None` will not enforce batch size, data will be read at the step of row
  # groups. For large row group size, the cost of loading all record fields may
  # be expensive and slow, choose a proper batch size will help.
  #
  # Too small batch size causes too many Ray blocks in Ray runner and will have
  # negative impact on performance.
  #
  # TODO: currently a batch can be smaller than batch_size (e.g., at boundary
  # of row groups), to enforce size to be equal to batch_size.
  batch_size: Optional[int] = 16


@dataclass
class Range:
  """A range of a field."""
  # Always inclusive.
  min_: Any

  # Default exclusive.
  max_: Any

  # Max is inclusive when true.
  include_max: bool = False


@dataclass
class JoinOptions:
  """Options of joining data."""
  # Partition the join key range into multiple ranges for parallel processing.
  partition_fn: Optional[Callable[[Range], List[Range]]] = None

  # TODO: to support ReadOptions for left and right views, e.g., filter_,
  # snapshot_id
  # TODO: to support join type in PyArrow, only `inner` is supported now.
