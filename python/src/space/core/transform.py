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
"""Classes for transforming datasets and """

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from os import path
from typing import Dict, List, Tuple, Optional

import pyarrow as pa
import pyarrow.compute as pc
from substrait.algebra_pb2 import Expression
from substrait.algebra_pb2 import FilterRel
from substrait.algebra_pb2 import FunctionArgument
from substrait.algebra_pb2 import ProjectRel
from substrait.algebra_pb2 import Rel
from substrait.extensions.extensions_pb2 import SimpleExtensionDeclaration
from substrait.extensions.extensions_pb2 import SimpleExtensionURI
from substrait.plan_pb2 import Plan
from substrait.type_pb2 import Type

from space.core.datasets import Dataset
import space.core.proto.metadata_pb2 as meta
from space.core.schema import arrow
from space.core.utils.lazy_imports_utils import ray
from space.core.utils.plans import SIMPLE_UDF_URI
from space.core.utils.plans import LogicalPlanBuilder, UserDefinedFn
from space.core.views import View


@dataclass
class BaseUdfTransform(View):
  """Base class for transforms containing a single user defined function."""

  # The UDF of this transform.
  udf: UserDefinedFn
  # The input view to apply the UDF to.
  input_: View
  # The fields to read from the input view.
  # TODO: to push input field selection down to the input view, for reading
  # less data from the sources.
  input_fields: List[str]

  @property
  def primary_keys(self) -> List[str]:
    return self.input_.primary_keys

  @property
  def sources(self) -> Dict[str, Dataset]:
    return self.input_.sources

  def _add_udf(self, builder: LogicalPlanBuilder) -> int:
    """Add the UDF to the logical plan.
    
    Returns:
      The function anchor as the function_reference of this UDF.
    """
    # TODO: to support user provided UDF name.
    new_udf_name = builder.new_udf_name()
    builder.add_udf(new_udf_name, self.udf)

    extension_uri_anchor = builder.next_ext_uri_anchor()
    fn_anchor = builder.next_function_anchor()

    builder.append_ext_uri(
        SimpleExtensionURI(extension_uri_anchor=extension_uri_anchor,
                           uri=SIMPLE_UDF_URI))
    builder.append_ext(
        SimpleExtensionDeclaration(
            extension_function=SimpleExtensionDeclaration.ExtensionFunction(
                extension_uri_reference=extension_uri_anchor,
                function_anchor=fn_anchor,
                name=new_udf_name)))

    return fn_anchor

  def _arguments(self) -> List[FunctionArgument]:
    """Returns a list of UDF input args.
    
    Fields are represented by field IDs.
    """
    if not self.input_fields:
      input_fields = self.input_.schema.names
    else:
      input_fields = self.input_fields

    field_id_dict = arrow.field_name_to_id_dict(self.input_.schema)
    return [_fn_arg(field_id_dict[name]) for name in input_fields]

  def process_source(self, data: pa.Table) -> ray.Dataset:
    return self._transform(self.input_.process_source(data))

  def ray_dataset(self,
                  filter_: Optional[pc.Expression] = None,
                  fields: Optional[List[str]] = None,
                  snapshot_id: Optional[int] = None,
                  reference_read: bool = False) -> ray.Dataset:
    return self._transform(
        self.input_.ray_dataset(filter_, fields, snapshot_id, reference_read))

  @abstractmethod
  def _transform(self, ds: ray.Dataset) -> ray.Dataset:
    """Transform a Ray dataset using the UDF."""


class MapTransform(BaseUdfTransform):
  """Map a view by a user defined function."""

  @property
  def schema(self) -> pa.Schema:
    return self.udf.output_schema

  @property
  def record_fields(self) -> List[str]:
    return self.udf.output_record_fields

  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    input_rel = self.input_.to_relation(builder)
    fn_anchor = self._add_udf(builder)
    # NOTE: output_type is unset because it is a single field type per
    # project expression in Substrait protocol, but a schema is needed here.
    # The output types are recorded in view schema.
    # TODO: to populate output_type as a Type.Struct.
    project_expr = Expression(scalar_function=Expression.ScalarFunction(
        function_reference=fn_anchor, arguments=self._arguments()))
    return Rel(project=ProjectRel(input=input_rel, expressions=[project_expr]))

  @classmethod
  def from_relation(cls, location: str, metadata: meta.StorageMetadata,
                    rel: Rel, plan: _CompactPlan) -> MapTransform:
    """Build a MapTransform from logical plan relation."""
    return MapTransform(*_load_udf(location, metadata, rel.project.
                                   expressions[0], rel.project.input, plan))

  def _transform(self, ds: ray.Dataset) -> ray.Dataset:
    batch_size = self.udf.batch_size if self.udf.batch_size >= 0 else "default"
    return ds.map_batches(self.udf.fn, batch_size=batch_size)


@dataclass
class FilterTransform(BaseUdfTransform):
  """Filter a view by a user defined function."""

  @property
  def schema(self) -> pa.Schema:
    return self.input_.schema

  @property
  def record_fields(self) -> List[str]:
    return self.input_.record_fields

  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    input_rel = self.input_.to_relation(builder)
    fn_anchor = self._add_udf(builder)
    condition_expr = Expression(
        scalar_function=Expression.ScalarFunction(function_reference=fn_anchor,
                                                  arguments=self._arguments(),
                                                  output_type=Type(
                                                      bool=Type.Boolean())))
    return Rel(filter=FilterRel(input=input_rel, condition=condition_expr))

  @classmethod
  def from_relation(cls, location: str, metadata: meta.StorageMetadata,
                    rel: Rel, plan: _CompactPlan) -> FilterTransform:
    """Build a FilterTransform from logical plan relation."""
    return FilterTransform(*_load_udf(location, metadata, rel.filter.condition,
                                      rel.filter.input, plan))

  def _transform(self, ds: ray.Dataset) -> ray.Dataset:
    return ds.filter(self.udf.fn)


@dataclass
class _CompactPlan:
  """A helper class storing information from a Substrait plan in a read
  friendly format.
  """

  # Key is function_anchor.
  ext_fn_dict: Dict[int, SimpleExtensionDeclaration]
  # Key is extension_uri_anchor.
  ext_uri_dict: Dict[int, SimpleExtensionURI]

  @classmethod
  def from_plan(cls, plan: Plan) -> _CompactPlan:
    """Build a _CompactPlan from a plan."""
    ext_fn_dict = {}
    for ext in plan.extensions:
      ext_fn_dict[ext.extension_function.function_anchor] = ext

    ext_uri_dict = {}
    for uri in plan.extension_uris:
      ext_uri_dict[uri.extension_uri_anchor] = uri

    return _CompactPlan(ext_fn_dict, ext_uri_dict)


def _load_udf(location: str, metadata: meta.StorageMetadata,
              expression: Expression, input_rel: Rel,
              plan: _CompactPlan) -> Tuple[UserDefinedFn, View, List[str]]:
  """Load UDF information for building a transform from a relation.
  
  Returns:
    A tuple of: (1) UDF class, (2) the input view, (3) input argument field
    names.
  """
  scalar_fn = expression.scalar_function
  fn_extension = plan.ext_fn_dict[scalar_fn.function_reference]

  # Sanity check.
  if plan.ext_uri_dict[fn_extension.extension_function.
                       extension_uri_reference].uri != SIMPLE_UDF_URI:
    raise RuntimeError("Only UDF is supported in logical plan extension URIs")

  # Load the UDF from file.
  pickle_path = metadata.logical_plan.udfs[
      fn_extension.extension_function.name]
  udf = UserDefinedFn.load(path.join(location, pickle_path))

  # Build the input view and input argument field names.
  input_ = load_view_(location, metadata, input_rel, plan)
  field_name_dict = arrow.field_id_to_name_dict(input_.schema)
  input_fields = [
      field_name_dict[arg.value.selection.direct_reference.struct_field.field]
      for arg in scalar_fn.arguments
  ]

  return udf, input_, input_fields


def load_view(location: str, metadata: meta.StorageMetadata,
              plan: Plan) -> View:
  """Build a view from logical plan relation."""
  rel = plan.relations[0].root.input
  return load_view_(location, metadata, rel, _CompactPlan.from_plan(plan))


def load_view_(location: str, metadata: meta.StorageMetadata, rel: Rel,
               plan: _CompactPlan) -> View:
  """Build a view from logical plan relation."""
  if rel.HasField("read"):
    return Dataset.load(rel.read.named_table.names[0])
  elif rel.HasField("project"):
    return MapTransform.from_relation(location, metadata, rel, plan)
  elif rel.HasField("filter"):
    return FilterTransform.from_relation(location, metadata, rel, plan)

  raise RuntimeError(f"Substrait relation not supported: {rel}")


def _fn_arg(field_id: int) -> FunctionArgument:
  """Return a Substrait function argument for a field ID.
  
  NOTE: StructField.field in Substrait was for the position of a field in the
  field name list in depth first order. Its meaning is replaced by field ID
  here. It does not affect any functionality of fetching a field from an
  integer. To revisit the design in future.
  """
  return FunctionArgument(value=Expression(selection=Expression.FieldReference(
      direct_reference=Expression.ReferenceSegment(
          struct_field=Expression.ReferenceSegment.StructField(
              field=field_id)))))
