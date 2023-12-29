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
"""Views (materialized views) are transforms applied to datasets."""

from __future__ import annotations
from abc import ABC, abstractmethod
from os import path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import pyarrow as pa
from substrait.algebra_pb2 import Rel

from space.core.fs.factory import create_fs
import space.core.proto.metadata_pb2 as meta
from space.core.schema import FieldIdManager
from space.core.storage import Storage
from space.core.utils.paths import UDF_DIR, metadata_dir
from space.core.utils.plans import LogicalPlanBuilder, UserDefinedFn

if TYPE_CHECKING:
  from space.core.datasets import Dataset


class View(ABC):
  """A view is a dataset, or a transform applied to a dataset, or a transform
  applied to another view.
  """

  @property
  @abstractmethod
  def schema(self) -> pa.Schema:
    """Return the view schema."""

  @property
  @abstractmethod
  def primary_keys(self) -> List[str]:
    """Return the primary keys."""

  @property
  @abstractmethod
  def record_fields(self) -> List[str]:
    """Return the record field names."""

  @property
  @abstractmethod
  def sources(self) -> Dict[str, Dataset]:
    """Return the datasets in its upstream views.
    
    Key is dataset location as the identifier.
    """

  @abstractmethod
  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    """Obtain the logical plan relation of the view.
    
    The relation describes a dataset or a transform in the Substrait format. 
    """

  def materialize(self, location: str) -> MaterializedView:
    """Materialize a view to files in the Space storage format.
    
    Args:
      location: the folder location of the materialized view files.
    """
    plan_builder = LogicalPlanBuilder()
    rel = self.to_relation(plan_builder)
    logical_plan = meta.LogicalPlan(logical_plan=plan_builder.build(rel))
    return MaterializedView.create(location, self, logical_plan,
                                   plan_builder.udfs)

  # pylint: disable=too-many-arguments
  def map_batches(self,
                  fn: Callable,
                  input_fields: List[str],
                  output_schema: pa.Schema,
                  output_record_fields: List[str],
                  batch_size: int = -1) -> View:
    """Transform batches of data by a user defined function.

    Args:
      fn: a user defined function on batches.
      input_fields: the fields to read from the input view.
      output_schema: the output schema.
      batch_size: the number of rows per batch.
    """
    # Assign field IDs to the output schema.
    field_id_mgr = FieldIdManager(next_field_id=0)
    output_schema = field_id_mgr.assign_field_ids(output_schema)

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform import MapTransform
    return MapTransform(
        UserDefinedFn(fn, output_schema, output_record_fields, batch_size),
        self, input_fields)

  def filter(self,
             fn: Callable,
             input_fields: Optional[List[str]] = None) -> View:
    """Filter rows by the provided user defined function.
    
    Args:
      fn: a user defined function on batches.
      input_fields: the fields to read from the input view.
    """
    if input_fields is None:
      input_fields = []

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform import FilterTransform
    return FilterTransform(UserDefinedFn(fn, self.schema, self.record_fields),
                           self, input_fields)


class MaterializedView:
  """A view materialized as a Space storage.
  
  When the source datasets are modified, refreshing the materialized view
  keeps the view up-to-date by reading the changes in the sources, processing
  the changes by the transforms in logical plan, and writing the results into
  materialized view storage.
  """

  def __init__(self, storage: Storage, view: View):
    self._storage = storage
    self._view = view

  @property
  def storage(self) -> Storage:
    """Return storage of the materialized view."""
    return self._storage

  @property
  def view(self) -> View:
    """Return view of the materialized view."""
    return self._view

  @classmethod
  def create(cls, location: str, view: View, logical_plan: meta.LogicalPlan,
             udfs: Dict[str, UserDefinedFn]) -> MaterializedView:
    """Create a new materialized view."""
    udf_dir = path.join(metadata_dir(location), UDF_DIR)
    create_fs(location).create_dir(udf_dir)

    for name, udf in udfs.items():
      full_path = path.join(udf_dir, f"{name}.pkl")
      udf.dump(full_path)
      logical_plan.udfs[name] = path.relpath(full_path, location)

    storage = Storage.create(location, view.schema, view.primary_keys,
                             view.record_fields, logical_plan)
    return MaterializedView(storage, view)

  @classmethod
  def load(cls, location: str) -> MaterializedView:
    """Load a materialized view from files."""
    storage = Storage.load(location)
    metadata = storage.metadata
    plan = metadata.logical_plan.logical_plan

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform import load_view
    view = load_view(storage.location, metadata, plan)
    return MaterializedView(storage, view)
