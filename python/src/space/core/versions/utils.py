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
"""Utilities for version management."""

from typing import Union


def version_to_snapshot_id(version: Union[int]) -> int:
  """Convert a version to a snapshot ID."""
  snapshot_id = None

  if isinstance(version, int):
    snapshot_id = version

  if snapshot_id is None:
    raise RuntimeError(f"Invalid version: {version}")

  return snapshot_id
