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
from space.core.options import FileOptions, JoinOptions, ReadOptions
import space.core.proto.metadata_pb2 as meta
from space.core.schema import FieldIdManager
from space.core.storage import Storage
from space.core.transform.plans import LogicalPlanBuilder, UserDefinedFn
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray, ray_runners  # pylint: disable=unused-import
from space.core.utils.paths import UDF_DIR, metadata_dir
from space.core.runners import LocalRunner
from space.core.schema.utils import validate_logical_schema
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.datasets import Dataset


class View(ABC):
  """A view is a dataset, or a transform applied to a dataset, or a transform
  applied to another view.

  Non-dataset views must use Ray runner instead of local runner, because the
  transforms are implemented with Ray dataset transforms.

  TODO: auto assign a view name for representing transforms.
  """

  @property
  @abstractmethod
  def schema(self) -> pa.Schema:
    """Return the view schema.
    
    For materialized view, the schema is used to create storage. The writer
    checks view's read data is compatible with the schema at write time.

    TODO: it is not checked if view is not materialized, to enforce this check.
    """

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

  @abstractmethod
  def process_source(self, data: ray.data.Dataset) -> ray.data.Dataset:
    """Process input data using the transform defined by the view."""

  def ray_dataset(
      self,
      ray_options: Optional[RayOptions] = None,
      read_options: Optional[ReadOptions] = None,
      join_options: Optional[JoinOptions] = None) -> ray.data.Dataset:
    """Return a Ray dataset for a Space view."""
    return self._ray_dataset(ray_options or RayOptions(), read_options or
                             ReadOptions(), join_options or JoinOptions())

  @abstractmethod
  def _ray_dataset(self, ray_options: RayOptions, read_options: ReadOptions,
                   join_options: JoinOptions) -> ray.data.Dataset:
    """Return a Ray dataset for a Space view. Internal implementation."""

  def ray(
      self,
      ray_options: Optional[RayOptions] = None
  ) -> ray_runners.RayReadOnlyRunner:
    """Return a Ray runner for the view."""
    return ray_runners.RayReadOnlyRunner(self, ray_options)

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
                  output_schema: pa.Schema,
                  input_fields: Optional[List[str]] = None,
                  output_record_fields: Optional[List[str]] = None,
                  batch_size: Optional[int] = None) -> View:
    """Transform batches of data by a user defined function.

    Args:
      fn: a user defined function on batches.
      input_fields: the fields to read from the input view, default to all
        fields.
      output_schema: the output schema.
      output_record_fields: record fields in the output, default to empty.
      batch_size: the number of rows per fn input batch.
    """
    # Assign field IDs to the output schema.
    field_id_mgr = FieldIdManager(next_field_id=0)
    output_schema = field_id_mgr.assign_field_ids(output_schema)

    input_fields = self._default_or_validate_input_fields(
        self.schema, input_fields)

    if output_record_fields is None:
      output_record_fields = []

    validate_logical_schema(output_schema, self.primary_keys,
                            output_record_fields)

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform.udfs import MapTransform
    return MapTransform(
        UserDefinedFn(fn, output_schema, output_record_fields, batch_size),
        self, input_fields)

  def filter(self,
             fn: Callable,
             input_fields: Optional[List[str]] = None) -> View:
    """Filter rows by the provided user defined function.

    TODO: this filter is not applied to the deleted rows returned by diff(), it
    thus returns more rows than expected. It does not affect correctness when
    syncing the deletion to target MV, because the additional rows don't exist.
    
    Args:
      fn: a user defined function on batches.
      input_fields: the fields to read from the input view, default to all
        fields.
    """
    input_fields = self._default_or_validate_input_fields(
        self.schema, input_fields)

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform.udfs import FilterTransform
    return FilterTransform(UserDefinedFn(fn, self.schema, self.record_fields),
                           self, input_fields)

  def join(self,
           right: View,
           keys: List[str],
           left_fields: Optional[List[str]] = None,
           right_fields: Optional[List[str]] = None,
           left_reference_read: bool = False,
           right_reference_read: bool = False) -> View:
    """Join two views.
    
    Args:
      keys: join keys, must be primary keys of both left and right.
    """
    left = self
    if len(keys) != 1:
      raise errors.UserInputError("Support exactly one join key")

    join_key = keys[0]
    if not (join_key in left.primary_keys and join_key in right.primary_keys):
      raise errors.UserInputError("Join key must be primary key on both sides")

    def _sanitize_fields(field_names: List[str]) -> None:
      nonlocal join_key
      if not field_names or field_names == [join_key]:
        raise errors.UserInputError(
            "Join requires reading at least one non-join key")

      if join_key not in field_names:
        field_names.append(join_key)

    if left_fields is not None:
      _sanitize_fields(left_fields)

    if right_fields is not None:
      _sanitize_fields(right_fields)

    # pylint: disable=cyclic-import,import-outside-toplevel
    from space.core.transform.join import JoinTransform
    from space.ray.ops.join import JoinInput
    return JoinTransform(join_keys=keys,
                         left=JoinInput(left, left_fields, left_reference_read),
                         right=JoinInput(right, right_fields,
                                         right_reference_read))

  def _default_or_validate_input_fields(
      self, input_schema: pa.Schema,
      input_fields: Optional[List[str]]) -> List[str]:
    if input_fields is None:
      return self.schema.names

    input_fields_sets = set(input_fields)

    if not set(self.primary_keys).issubset(input_fields_sets):
      raise errors.UserInputError(
          f"Input fields {input_fields} must contain all primary keys "
          f"{self.primary_keys}")

    if not input_fields_sets.issubset(set(input_schema.names)):
      raise errors.UserInputError(
          f"Input fields {input_fields} must be a subtset of input schema: "
          f"{input_schema}")

    return input_fields


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
  def dataset(self) -> Dataset:
    """Return storage of the materialized view as a dataset."""
    # pylint: disable=import-outside-toplevel
    from space.core.datasets import Dataset
    return Dataset(self._storage)

  @property
  def view(self) -> View:
    """Return view of the materialized view."""
    return self._view

  def ray(
      self,
      ray_options: Optional[RayOptions] = None,
      file_options: Optional[FileOptions] = None
  ) -> ray_runners.RayMaterializedViewRunner:
    """Return a Ray runner for the materialized view."""
    return ray_runners.RayMaterializedViewRunner(self, ray_options,
                                                 file_options)

  def local(self, file_options: Optional[FileOptions] = None) -> LocalRunner:
    """Get a runner that runs operations locally.
    
    TODO: should use a read-only local runner.
    """
    return LocalRunner(self._storage, file_options)

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
    return load_materialized_view(Storage.load(location))


def load_materialized_view(storage: Storage) -> MaterializedView:
  """Load a materialized view from a storage."""
  metadata = storage.metadata
  plan = metadata.logical_plan.logical_plan

  # pylint: disable=cyclic-import,import-outside-toplevel
  from space.core.transform.udfs import load_view
  view = load_view(storage.location, metadata, plan)
  return MaterializedView(storage, view)
