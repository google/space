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
"""Utilities for schema field IDs."""

from typing import List, Optional
import pyarrow as pa

from space.core.schema import arrow

# The start value of field IDs.
_START_FIELD_ID = 0


class FieldIdManager:
  """Assign field IDs to schema fields using Depth First Search.

  Rules for nested fields:
  - For a list field with ID i, its element is assigned i+1.
  - For a struct field with ID i, its fields are assigned starting from i+1.

  Not thread safe.
  """

  def __init__(self, next_field_id: Optional[int] = None):
    if next_field_id is None:
      self._next_field_id = _START_FIELD_ID
    else:
      assert next_field_id >= _START_FIELD_ID
      self._next_field_id = next_field_id

  def assign_field_ids(self, schema: pa.Schema) -> pa.Schema:
    """Return a new schema with field IDs assigned."""
    return pa.schema(self._assign_field_ids(list(schema)))

  def _assign_field_ids(self, fields: List[pa.Field]) -> List[pa.Field]:
    return [self._assign_field_id(f) for f in fields]

  def _assign_field_id(self, field: pa.Field) -> pa.Field:
    metadata = arrow.field_metadata(self._next_field_id)
    self._next_field_id += 1

    name, type_ = field.name, field.type

    if pa.types.is_list(type_):
      return pa.field(
          name,
          pa.list_(self._assign_field_id(
              type_.value_field)),  # type: ignore[attr-defined]
          metadata=metadata)

    if pa.types.is_struct(type_):
      struct_type = pa.struct(
          self._assign_field_ids(
              [type_.field(i) for i in range(type_.num_fields)]))
      return pa.field(name, struct_type, metadata=metadata)

    # TODO: to support more types, e.g., fixed_size_list, map.

    return field.with_metadata(metadata)  # type: ignore[arg-type]
