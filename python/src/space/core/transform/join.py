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
"""Classes for transforming datasets using join."""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, TYPE_CHECKING

import pyarrow as pa
from substrait.algebra_pb2 import Rel

from space.core.options import JoinOptions, ReadOptions
from space.core.schema.arrow import record_address_types
from space.core.transform.plans import LogicalPlanBuilder
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray
from space.core.views import View
from space.ray.ops.join import JoinInput, RayJoinOp
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.datasets import Dataset


@dataclass
class JoinTransform(View):
  """Transform that joins two views/datasets."""

  # Join keys must be parts of primary keys.
  join_keys: List[str]
  # The input views/datasets of the join.
  left: JoinInput
  right: JoinInput

  output_schema: pa.Schema = dataclass_field(init=False)

  def __post_init__(self):
    self.output_schema = self._output_schema()

  @property
  def primary_keys(self) -> List[str]:
    return self.join_keys

  @property
  def sources(self) -> Dict[str, Dataset]:
    return {**self.left.view.sources, **self.right.view.sources}

  @property
  def schema(self) -> pa.Schema:
    return self.output_schema

  def _output_schema(self) -> pa.Schema:
    assert len(self.join_keys) == 1
    join_key = self.join_keys[0]
    record_fields = set(self.record_fields)

    def _fields(input_: JoinInput) -> List[pa.Field]:
      nonlocal join_key, record_fields
      results = []
      for f in (input_.fields or input_.view.schema.names):
        if f == join_key:
          continue

        if input_.reference_read and f in record_fields:
          results.append(pa.field(f, pa.struct(
              record_address_types())))  # type: ignore[arg-type]
        else:
          results.append(input_.view.schema.field(
              f).remove_metadata())  # type: ignore[arg-type]

      return results

    # TODO: to handle reference read. If true, use the address field schema.
    try:
      left_fields = _fields(self.left)
      right_fields = _fields(self.right)

      # TODO: to check field names that are the same in left and right; add a
      # validation first, and then support field rename.
      return pa.schema([
          self.left.view.schema.field(
              join_key).remove_metadata()  # type: ignore[arg-type]
      ] + left_fields + right_fields)
    except KeyError as e:
      raise errors.UserInputError(repr(e))

  @property
  def record_fields(self) -> List[str]:
    # TODO: For now just inherit record fields from input, to allow updating.
    left_record_fields = set(self.left.view.record_fields).intersection(
        set(self.left.fields or self.left.view.schema.names))
    right_record_fields = set(self.right.view.record_fields).intersection(
        set(self.right.fields or self.right.view.schema.names))
    return list(left_record_fields) + list(right_record_fields)

  def process_source(self, data: ray.data.Dataset) -> ray.data.Dataset:
    raise NotImplementedError("Processing change data in join is not supported")

  def _ray_dataset(self, ray_options: RayOptions, read_options: ReadOptions,
                   join_options: JoinOptions) -> ray.data.Dataset:
    # TODO: to use paralelism specified by ray_options. Today parallelism is
    # controlled by join_options.partition_fn.
    if read_options.fields is not None:
      raise errors.UserInputError(
          "`fields` is not supported for join, use `left_fields` and"
          " `right_fields` of join instead")

    if read_options.reference_read:
      # TODO: need such options for both left and right, will be supported
      # after refactoring the arguments.
      raise errors.UserInputError("`reference_read` is not supported for join")

    return RayJoinOp(self.left, self.right, self.join_keys, self.schema,
                     join_options, ray_options).ray_dataset()

  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    raise NotImplementedError("Materialized view of join is not supported")
