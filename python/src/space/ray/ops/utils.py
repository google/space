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
"""Utilities for Ray operations."""

from __future__ import annotations
from typing import TYPE_CHECKING

from space.core.storage import Storage
from space.core.utils import errors

if TYPE_CHECKING:
  from space.core.views import View


def singleton_storage(view: View) -> Storage:
  """Return the singleton source storage for views with only one source
  dataset.
  """
  if len(view.sources) == 0:
    raise errors.UserInputError("Source of view not found")

  if len(view.sources) != 1:
    raise errors.UserInputError("Joining results of joins is not supported")

  return list(view.sources.values())[0].storage
