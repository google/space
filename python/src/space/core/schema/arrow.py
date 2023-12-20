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

from dataclasses import dataclass
from typing import Dict, List, Optional

import pyarrow as pa
from substrait.type_pb2 import NamedStruct, Type

from space.core.utils.constants import UTF_8
from space.core.schema.constants import TF_FEATURES_TYPE
from space.core.schema.types import TfFeatures

_PARQUET_FIELD_ID_KEY = b"PARQUET:field_id"


def field_metadata(field_id_: int) -> Dict[bytes, bytes]:
  """Return Arrow field metadata for a field."""
  return {_PARQUET_FIELD_ID_KEY: str(field_id_).encode(UTF_8)}


def field_id(field: pa.Field) -> int:
  """Return field ID of an Arrow field."""
  return int(field.metadata[_PARQUET_FIELD_ID_KEY])


@dataclass
class _NamesVisitor:
  names: List[str]
  idx: int = 0

  def next(self) -> str:
    """Return the next name."""
    name = self.names[self.idx]
    self.idx += 1
    return name


def arrow_schema(fields: NamedStruct) -> pa.Schema:
  """Return Arrow schema from Substrait fields.
  
  Args:
    fields: schema fields in the Substrait format.
    physical: if true, return the physical schema. Physical schema matches with
      the underlying index (Parquet) file schema. Record fields are stored by
      their references, e.g., row position in ArrayRecord file.
  """
  return pa.schema(
      _arrow_fields(
          _NamesVisitor(fields.names),  # type: ignore[arg-type]
          fields.struct.types))  # type: ignore[arg-type]


def _arrow_fields(names_visitor: _NamesVisitor,
                  types: List[Type]) -> List[pa.Field]:
  fields: List[pa.Field] = []

  for type_ in types:
    name = names_visitor.next()
    arrow_field = pa.field(name,
                           _arrow_type(type_, names_visitor),
                           metadata=field_metadata(_substrait_field_id(type_)))
    fields.append(arrow_field)

  return fields


def _substrait_field_id(type_: Type) -> int:
  return getattr(type_, type_.WhichOneof(
      "kind")).type_variation_reference  # type: ignore[arg-type]


# pylint: disable=too-many-return-statements
def _arrow_type(type_: Type,
                names_visitor: Optional[_NamesVisitor] = None) -> pa.DataType:
  """Return the Arrow type for a Substrait type."""
  # TODO: to support more types in Substrait, e.g., fixed_size_list, map.
  if type_.HasField("bool"):
    return pa.bool_()
  if type_.HasField("i32"):
    return pa.int32()
  if type_.HasField("i64"):
    return pa.int64()
  if type_.HasField("fp32"):
    return pa.float32()
  if type_.HasField("fp64"):
    return pa.float64()
  if type_.HasField("string"):
    return pa.string()
  if type_.HasField("binary"):
    return pa.binary()
  if type_.HasField("list"):
    return pa.list_(_arrow_type(type_.list.type, names_visitor))
  if type_.HasField("struct"):
    assert names_visitor is not None
    subfields = []
    for t in type_.struct.types:
      subfields.append(
          pa.field(names_visitor.next(), _arrow_type(t, names_visitor)))
    return pa.struct(subfields)
  if type_.HasField("user_defined"):
    return _user_defined_arrow_type(type_)

  raise TypeError(f"Unsupported Substrait type: {type_}")


def _user_defined_arrow_type(type_: Type) -> pa.ExtensionType:
  type_name = type_.user_defined.type_parameters[0].string
  serialized = type_.user_defined.type_parameters[1].string

  if type_name == TF_FEATURES_TYPE:
    return TfFeatures.__arrow_ext_deserialize__(
        None, serialized)  # type: ignore[arg-type]

  raise TypeError(f"Unsupported Substrait user defined type: {type_}")


def field_name_to_id_dict(schema: pa.Schema) -> Dict[str, int]:
  """Return a dict with field name as key and field ID as value."""
  return {f.name: field_id(f) for f in schema}


def field_id_to_column_id_dict(schema: pa.Schema) -> Dict[int, int]:
  """Return a dict with field ID as key and column ID as value."""
  field_id_dict = field_name_to_id_dict(schema)
  return {
      field_id_dict[name]: column_id
      for column_id, name in enumerate(schema.names)
  }
