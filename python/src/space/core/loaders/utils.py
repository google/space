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
"""Utilities for loaders."""

import os
from typing import List, Optional


def list_files(directory: str,
               substr: Optional[str] = None,
               suffix: Optional[str] = None) -> List[str]:
  """List files in a directory."""
  files: List[str] = []
  for f in os.listdir(directory):
    full_path = os.path.join(directory, f)
    if (os.path.isfile(full_path) and (substr is None or substr in f)
        and (suffix is None or f.endswith(suffix))):
      files.append(full_path)

  return files
