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
"""Jobs of Space operations."""

from typing import Optional

from dataclasses import dataclass
from enum import Enum

import space.core.proto.metadata_pb2 as meta


@dataclass
class JobResult:
  """The result of a job."""

  class State(Enum):
    """The job state."""
    # The job has suceeded.
    SUCCEEDED = 1
    # The job has failed.
    FAILED = 2
    # The job is a no-op.
    SKIPPED = 3

  # The job state.
  state: State

  # The update to storage statistics as the result of the job.
  storage_statistics_update: Optional[meta.StorageStatistics] = None

  # Error message if the job failed.
  error_message: Optional[str] = None
