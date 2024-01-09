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
"""Define errors thrown by Space runtime."""


class UserInputError(ValueError):
  """Errors caused by invalid user input."""


class SnapshotNotFoundError(UserInputError):
  """The snapshot is not found in metadata."""


class PrimaryKeyExistError(UserInputError):
  """Errors caused by duplicated primary keys."""

class SnapshotReferenceAlreadyExistError(UserInputError):
  """Errors caused by duplicated reference."""

class SnapshotReferenceNotFoundError(UserInputError):
  """Errors caused by reference not found."""

class SpaceRuntimeError(RuntimeError):
  """Basic class of errors thrown from Space runtime."""


class TransactionError(SpaceRuntimeError):
  """Errors from Space transaction commit."""


class LogicalPlanError(SpaceRuntimeError):
  """Errors from parsing logical plan."""
