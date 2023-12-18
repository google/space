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
"""Utilities for schemas in the Arrow format."""

from typing import Dict

import pyarrow as pa

_PARQUET_FIELD_ID_KEY = "PARQUET:field_id"


def field_metadata(field_id_: int) -> Dict[str, str]:
  """Return Arrow field metadata for a field."""
  return {_PARQUET_FIELD_ID_KEY: str(field_id_)}


def field_id(field: pa.Field) -> int:
  """Return field ID of an Arrow field."""
  return int(field.metadata[_PARQUET_FIELD_ID_KEY])
