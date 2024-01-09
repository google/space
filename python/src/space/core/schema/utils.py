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
"""Common utilities for schemas."""

from dataclasses import dataclass
from typing import List

import pyarrow as pa

from space.core.schema import constants
from space.core.schema.types import TfFeatures
from space.core.utils import errors


@dataclass
class Field:
  """Information of a field."""
  name: str
  field_id: int


def field_names(fields: List[Field]) -> List[str]:
  """Extract field names from a list of fields."""
  return list(map(lambda f: f.name, fields))


def field_ids(fields: List[Field]) -> List[int]:
  """Extract field IDs from a list of fields."""
  return list(map(lambda f: f.field_id, fields))


def stats_field_name(field_id_: int) -> str:
  """Column stats struct field name.
  
  It uses field ID instead of name. Manifest file has all Parquet files and it
  is not tied with one Parquet schema, we can't do table field name to file
  field name projection. Using field ID ensures that we can always uniquely
  identifies a field.
  """
  return f"{constants.STATS_FIELD}_f{field_id_}"


def file_path_field_name(field: str) -> str:
  """File path field name in flatten addresses."""
  return f"{field}.{constants.FILE_PATH_FIELD}"


def row_id_field_name(field: str) -> str:
  """Row ID field name in flatten addresses."""
  return f"{field}.{constants.ROW_ID_FIELD}"


def validate_logical_schema(schema: pa.Schema, primary_keys: List[str],
                            record_fields: List[str]) -> None:
  """Validate the logical schema of a Space storage."""
  if not primary_keys:
    raise errors.UserInputError("Must specify at least one primary key")

  all_fields = set(schema.names)

  for name in primary_keys:
    if name not in all_fields:
      raise errors.UserInputError(f"Primary key {name} not found in schema")

    field = schema.field(name)  # type: ignore[arg-type]
    if pa.types.is_list(field.type) or pa.types.is_struct(
        field.type) or isinstance(field.type, TfFeatures):
      raise errors.UserInputError(f"Primary key type not supported: {field}")

  all_primary_keys = set(primary_keys)
  for name in record_fields:
    if name not in all_fields:
      raise errors.UserInputError(f"Record field {name} not found in schema")

    if name in all_primary_keys:
      raise errors.UserInputError(
          f"Record field {name} cannot be a primary key")

    field = schema.field(name)  # type: ignore[arg-type]
    if not (pa.types.is_binary(field.type)
            or isinstance(field.type, TfFeatures)):
      raise errors.UserInputError(f"Record field type not supported: {field}")
