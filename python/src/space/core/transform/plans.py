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
"""Plans for view/dataset transforms."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cloudpickle  # type: ignore[import-untyped]
import pyarrow as pa
from space.core.utils.uuids import random_id

from space.core.utils import errors
from substrait.algebra_pb2 import Rel, RelRoot
from substrait.extensions.extensions_pb2 import SimpleExtensionDeclaration
from substrait.extensions.extensions_pb2 import SimpleExtensionURI
from substrait.plan_pb2 import Plan, PlanRel

# Substrait URI representing user defined functions.
# When constructing a materialized view from logical plan, the UDF is loaded
# from a pickle file path in the storage metadata's UDF registry.
SIMPLE_UDF_URI = "urn:space:substrait_simple_extension_function"


@dataclass
class UserDefinedFn:
  """A user defined function in the logical plan.
  
  The class object is persisted in the storage metadata's UDF registry.
  """
  # A callable provided by users. The requirement on signature varies depending
  # on the transform type.
  fn: Callable
  # The output schema after applying fn on the input view.
  output_schema: pa.Schema
  # The record fields in the output schema.
  output_record_fields: List[str]
  # If reading the input view by batches, number of rows per input batch.
  batch_size: Optional[int] = None

  # TODO: file operations need to be through the FileSystem interface.

  @classmethod
  def load(cls, file_path: str) -> UserDefinedFn:
    """Load a UDF from a file."""
    with open(file_path, "rb") as f:
      udf = cloudpickle.load(f)

    return udf

  def dump(self, file_path: str) -> None:
    """Dump UDF into a file."""
    with open(file_path, 'wb') as f:
      cloudpickle.dump(self, f)


class LogicalPlanBuilder:
  """A builder of logical plan in the Substrait format."""

  def __init__(self):
    self._plan = Plan()
    self._udfs: Dict[str, UserDefinedFn] = {}

    self._extension_uri_anchor = 1
    self._function_anchor = 1

  def next_ext_uri_anchor(self) -> int:
    """Return the next extension URI anchor."""
    result = self._extension_uri_anchor
    self._extension_uri_anchor += 1
    return result

  def next_function_anchor(self) -> int:
    """Return the next function anchor."""
    result = self._function_anchor
    self._function_anchor += 1
    return result

  def append_ext_uri(self, uri: SimpleExtensionURI) -> None:
    """Append an extension URI in the plan."""
    self._plan.extension_uris.append(uri)

  def append_ext(self, ext: SimpleExtensionDeclaration) -> None:
    """Append an extension in the plan."""
    self._plan.extensions.append(ext)

  def build(self, relation: Rel) -> Plan:
    """Build the plan."""
    self._plan.relations.append(PlanRel(root=RelRoot(input=relation)))
    return self._plan

  def add_udf(self, name: str, fn: UserDefinedFn) -> None:
    """Add a new user defined function to the plan."""
    self._udfs[name] = fn

  def new_udf_name(self) -> str:
    """Return a random UDF name, unique in the plan scope."""
    retry_count = 0
    while retry_count < 10:
      retry_count += 1
      name = f"udf_{random_id()}"
      if name not in self._udfs:
        return name

    raise errors.SpaceRuntimeError("Failed to generate an unused UDF name")

  @property
  def udfs(self) -> Dict[str, UserDefinedFn]:
    """Return user defined functions in the plan."""
    return self._udfs
