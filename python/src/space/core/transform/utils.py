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
"""Utilities for transforms."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

import pyarrow.compute as pc

from space.core.datasets import Dataset
from space.core.utils.lazy_imports_utils import ray

if TYPE_CHECKING:
  from space.core.views import View


def ray_dataset(view: View, filter_: Optional[pc.Expression],
                fields: Optional[List[str]], snapshot_id: Optional[int],
                reference_read: bool) -> ray.Dataset:
  """A wrapper for creating Ray dataset for datasets and views."""
  if isinstance(view, Dataset):
    # Push input_fields down to the dataset to read less data.
    return view.ray_dataset(filter_, fields, snapshot_id, reference_read)

  return view.ray_dataset(filter_, None, snapshot_id,
                          reference_read).select_columns(fields)
