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
"""Space dataset is the interface to interact with underlying storage."""

from __future__ import annotations
from typing import List

import pyarrow as pa

from space.core.runners import LocalRunner
from space.core.serializers.base import DictSerializer
from space.core.storage import Storage


class Dataset:
  """Dataset is the interface to interact with Space storage."""

  def __init__(self, storage: Storage):
    self._storage = storage

  @classmethod
  def create(cls, location: str, schema: pa.Schema, primary_keys: List[str],
             record_fields: List[str]) -> Dataset:
    """Create a new empty dataset.
    
    Args:
      location: the directory path to the storage.
      schema: the schema of the storage.
      primary_keys: un-enforced primary keys.
      record_fields: fields stored in row format (ArrayRecord).
    """
    return Dataset(
        Storage.create(location, schema, primary_keys, record_fields))

  @classmethod
  def load(cls, location: str) -> Dataset:
    """Load an existing dataset from the given location."""
    return Dataset(Storage.load(location))

  @property
  def schema(self) -> pa.Schema:
    """Return the dataset schema."""
    return self._storage.logical_schema

  def serializer(self) -> DictSerializer:
    """Return a serializer (deserializer) for the dataset."""
    return DictSerializer(self.schema)

  def local(self) -> LocalRunner:
    """Get a runner that runs operations locally."""
    return LocalRunner(self._storage)

  def index_files(self) -> List[str]:
    """A list of full path of index files."""
    data_files = self._storage.data_files()
    return [self._storage.full_path(f.path) for f in data_files.index_files]

  @property
  def snapshot_ids(self) -> List[int]:
    """A list of all alive snapshot IDs in the dataset."""
    return self._storage.snapshot_ids
