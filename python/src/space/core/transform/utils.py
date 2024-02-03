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
from typing import TYPE_CHECKING

from space.core.datasets import Dataset
from space.core.options import JoinOptions, ReadOptions
from space.core.utils.lazy_imports_utils import ray
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.views import View


def ray_dataset(view: View, ray_options: RayOptions,
                read_options: ReadOptions) -> ray.data.Dataset:
  """A wrapper for creating Ray dataset for datasets and views."""
  empty_join_options = JoinOptions()

  if isinstance(view, Dataset):
    # Push input_fields down to the dataset to read less data.
    return view._ray_dataset(ray_options, read_options, empty_join_options)  # pylint: disable=protected-access

  # For non-dataset views, fields can't be pushed down to storage.
  fields = read_options.fields
  read_options.fields = None
  ds = view._ray_dataset(ray_options, read_options, empty_join_options)  # pylint: disable=protected-access

  return ds if fields is None else ds.select_columns(fields)
