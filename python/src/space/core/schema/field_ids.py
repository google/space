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

_INIT_FIELD_ID = 0


# pylint: disable=too-few-public-methods
class FieldIdManager:
  """Assign field IDs to schema fields."""

  def __init__(self, next_field_id: Optional[int] = None):
    self._next_field_id = (next_field_id
                           if next_field_id is not None else _INIT_FIELD_ID)

  def _assign_field_id(self, field: pa.Field) -> pa.Field:
    this_field_id = self._next_field_id
    metadata = arrow.field_metadata(this_field_id)
    self._next_field_id += 1

    name = field.name
    type_ = field.type
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

    return field.with_metadata(metadata)

  def _assign_field_ids(self, fields: List[pa.Field]) -> List[pa.Field]:
    return [self._assign_field_id(f) for f in fields]

  def assign_field_ids(self, schema: pa.Schema) -> pa.Schema:
    """Assign field IDs to schema fields."""
    return pa.schema(
        self._assign_field_ids([schema.field(i) for i in range(len(schema))]))
