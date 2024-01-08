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
from typing import Dict, List, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
from substrait.algebra_pb2 import Rel

from space.core.apis import JoinOptions
from space.core.transform.plans import LogicalPlanBuilder
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray
from space.core.views import View
from space.ray.ops.join import RayJoinOp

if TYPE_CHECKING:
  from space.core.datasets import Dataset


@dataclass
class JoinTransform(View):
  """Transform that joins two views/datasets."""

  # The input views/datasets of the join.
  left: View
  right: View
  # The fields to read from the input view or dataset.
  left_fields: Optional[List[str]]
  right_fields: Optional[List[str]]
  # Join keys must be parts of primary keys.
  join_keys: List[str]
  join_options: JoinOptions
  output_schema: pa.Schema = dataclass_field(init=False)

  def __post_init__(self):
    self.output_schema = self._output_schema()

  @property
  def primary_keys(self) -> List[str]:
    return self.join_keys

  @property
  def sources(self) -> Dict[str, Dataset]:
    return {**self.left.sources, **self.right.sources}

  @property
  def schema(self) -> pa.Schema:
    return self.output_schema

  def _output_schema(self) -> pa.Schema:
    assert len(self.join_keys) == 1
    join_key = self.join_keys[0]

    def _fields(view: View,
                field_names: Optional[List[str]]) -> List[pa.Field]:
      nonlocal join_key
      return [
          view.schema.field(f).remove_metadata()  # type: ignore[arg-type]
          for f in (field_names or view.schema.names) if f != join_key
      ]

    # TODO: to handle reference read. If true, use the address field schema.
    try:
      left_fields_ = _fields(self.left, self.left_fields)
      right_fields_ = _fields(self.right, self.right_fields)

      # TODO: to check field names that are the same in left and right; add a
      # validation first, and then support field rename.
      return pa.schema([
          self.left.schema.field(
              join_key).remove_metadata()  # type: ignore[arg-type]
      ] + left_fields_ + right_fields_)
    except KeyError as e:
      raise errors.UserInputError(repr(e))

  @property
  def record_fields(self) -> List[str]:
    # TODO: For now just inherit record fields from input, to allow updating.
    left_record_fields = set(self.left.record_fields).intersection(
        set(self.left_fields or self.left.schema.names))
    right_record_fields = set(self.right.record_fields).intersection(
        set(self.right_fields or self.right.schema.names))
    return list(left_record_fields) + list(right_record_fields)

  def process_source(self, data: pa.Table) -> ray.Dataset:
    raise NotImplementedError(
        "Processing change data in join is not supported")

  def ray_dataset(self,
                  filter_: Optional[pc.Expression] = None,
                  fields: Optional[List[str]] = None,
                  snapshot_id: Optional[int] = None,
                  reference_read: bool = False) -> ray.Dataset:
    if fields is not None:
      raise errors.UserInputError(
          "`fields` is not supported for join, use `left_fields` and"
          " `right_fields` of join instead")

    if reference_read:
      # TODO: need such options for both left and right, will be supported
      # after refactoring the arguments.
      raise errors.UserInputError("`reference_read` is not supported for join")

    return RayJoinOp(self.left, self.left_fields, self.right,
                     self.right_fields, self.join_keys, self.schema,
                     self.join_options).ray_dataset()

  def to_relation(self, builder: LogicalPlanBuilder) -> Rel:
    raise NotImplementedError("Materialized view of join is not supported")
