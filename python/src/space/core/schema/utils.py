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

from space.core.schema import constants


@dataclass
class Field:
  """Information of a field."""
  name: str
  field_id: int


def field_names(fields: List[Field]) -> List[str]:
  """Extract field names from a list of fields."""
  return list(map(lambda f: f.name, fields))


def stats_field_name(field_id_: int) -> str:
  """Column stats struct field name.
  
  It uses field ID instead of name. Manifest file has all Parquet files and it
  is not tied with one Parquet schema, we can't do table field name to file
  field name projection. Using field ID ensures that we can always uniquely
  identifies a field.
  """
  return f"{constants.STATS_FIELD}_f{field_id_}"
