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
from typing import Dict, List, Optional

import pyarrow as pa
from substrait.algebra_pb2 import ReadRel, Rel

from space.core.ops.utils import FileOptions
from space.core.options import JoinOptions, ReadOptions
from space.core.runners import LocalRunner
from space.core.storage import Storage
from space.core.transform.plans import LogicalPlanBuilder
from space.core.utils.lazy_imports_utils import ray
from space.core.views import View
from space.ray.runners import RayOptions, RayReadWriterRunner


class Dataset(View):
  """Dataset is the interface to interact with Space storage."""

  def __init__(self, storage: Storage):
    self._storage = storage

  @property
  def storage(self) -> Storage:
    """Return storage of the dataset."""
    return self._storage

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

  @property
  def primary_keys(self) -> List[str]:
    return self._storage.primary_keys

  @property
  def record_fields(self) -> List[str]:
    return self._storage.record_fields

  def add_tag(self, tag: str, snapshot_id: Optional[int] = None):
    """Add yag to a snapshot."""
    self._storage.add_tag(tag, snapshot_id)

  def remove_tag(self, tag: str):
    """Remove tag from a snapshot."""
    self._storage.remove_tag(tag)

  def local(self, file_options: Optional[FileOptions] = None) -> LocalRunner:
    """Get a runner that runs operations locally."""
    return LocalRunner(self._storage, file_options)

  def index_files(self) -> List[str]:
    """A list of full path of index files."""
    data_files = self._storage.data_files()
    return [self._storage.full_path(f.path) for f in data_files.index_files]

  @property
  def snapshot_ids(self) -> List[int]:
    """A list of all alive snapshot IDs in the dataset."""
    return self._storage.snapshot_ids

  @property
  def sources(self) -> Dict[str, Dataset]:
    return {self._storage.location: self}

  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    # TODO: using location as table name is a limitation, because the location
    # could be mapped from Cloud Storage. The solution is external catalog
    # service integration, and using a unique identifier registered in the
    # catalog instead.
    location = self._storage.location
    return Rel(read=ReadRel(named_table=ReadRel.NamedTable(names=[location]),
                            base_schema=self._storage.metadata.schema.fields))

  def process_source(self, data: pa.Table) -> ray.Dataset:
    # Dataset is the source, there is no transform, so simply return the data.
    return ray.data.from_arrow(data)

  def ray_dataset(
      self,
      read_options: ReadOptions = ReadOptions(),
      join_options: JoinOptions = JoinOptions()
  ) -> ray.Dataset:
    """Return a Ray dataset for a Space dataset."""
    return self._storage.ray_dataset(read_options)

  def ray(self,
          file_options: Optional[FileOptions] = None,
          ray_options: Optional[RayOptions] = None) -> RayReadWriterRunner:
    """Get a Ray runner."""
    return RayReadWriterRunner(self, file_options, ray_options)
