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
"""Abstract base operation."""

from __future__ import annotations
from abc import ABC
from typing import Any, Callable, Dict, Iterator, Union
from typing_extensions import TypeAlias

import pyarrow as pa

# Input data can be either nested Py dict or Arrow table.
InputData: TypeAlias = Union[Dict[str, Any], pa.Table]

# A no args function that returns an iterator.
InputIteratorFn: TypeAlias = Callable[[], Iterator[InputData]]


class BaseOp(ABC):
  """Abstract base operation class."""
