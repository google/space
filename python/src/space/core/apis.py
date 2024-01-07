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
"""APIs of Space core lib."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


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
