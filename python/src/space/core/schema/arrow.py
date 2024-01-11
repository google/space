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
from typing import Dict, List, Optional, Set, Tuple

import pyarrow as pa
from substrait.type_pb2 import NamedStruct, Type

from space.core.schema import constants
from space.core.schema.types import File, TfFeatures
from space.core.schema import utils
from space.core.utils.constants import UTF_8

# A special field ID representing unassigned field ID.
# Used for external Parquet files not created by Space that don't have field
# ID. Schema evolution is limitted for datasets containing such files.
NULL_FIELD_ID = -1

_PARQUET_FIELD_ID_KEY = b"PARQUET:field_id"


def field_metadata(field_id_: int) -> Dict[bytes, bytes]:
  """Return Arrow field metadata for a field."""
  return {_PARQUET_FIELD_ID_KEY: str(field_id_).encode(UTF_8)}


def field_id(field: pa.Field) -> int:
  """Return field ID of an Arrow field."""
  if field.metadata is None or _PARQUET_FIELD_ID_KEY not in field.metadata:
    return NULL_FIELD_ID

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


def arrow_schema(fields: NamedStruct, record_fields: Set[str],
                 physical: bool) -> pa.Schema:
  """Return Arrow schema from Substrait fields.
  
  Args:
    fields: schema fields in the Substrait format.
    record_fields: a set of record field names.
    physical: if true, return the physical schema. Physical schema matches with
      the underlying index (Parquet) file schema. Record fields are stored by
      their references, e.g., row position in ArrayRecord file.
  """
  return pa.schema(
      _arrow_fields(
          _NamesVisitor(fields.names),  # type: ignore[arg-type]
          fields.struct.types,  # type: ignore[arg-type]
          record_fields,
          physical))


def _arrow_fields(names_visitor: _NamesVisitor, types: List[Type],
                  record_fields: Set[str], physical: bool) -> List[pa.Field]:
  fields: List[pa.Field] = []

  for type_ in types:
    name = names_visitor.next()

    if physical and name in record_fields:
      arrow_type: pa.DataType = pa.struct(
          record_address_types())  # type: ignore[arg-type]
    else:
      arrow_type = _arrow_type(type_, physical, names_visitor)

    fields.append(
        pa.field(name,
                 arrow_type,
                 metadata=field_metadata(_substrait_field_id(type_))))

  return fields


def _substrait_field_id(type_: Type) -> int:
  return getattr(type_, type_.WhichOneof(
      "kind")).type_variation_reference  # type: ignore[arg-type]


# pylint: disable=too-many-return-statements
def _arrow_type(type_: Type,
                physical: bool,
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
    return pa.list_(_arrow_type(type_.list.type, physical, names_visitor))
  if type_.HasField("struct"):
    assert names_visitor is not None
    subfields = []
    for t in type_.struct.types:
      subfields.append(
          pa.field(names_visitor.next(), _arrow_type(t, physical,
                                                     names_visitor)))
    return pa.struct(subfields)
  if type_.HasField("user_defined"):
    return _user_defined_arrow_type(type_, physical)

  raise TypeError(f"Unsupported Substrait type: {type_}")


def _user_defined_arrow_type(type_: Type, physical: bool) -> pa.DataType:
  type_name = type_.user_defined.type_parameters[0].string
  serialized = type_.user_defined.type_parameters[1].string

  if type_name == TfFeatures.EXTENSION_NAME:
    # Physical type has been handled when checking record fields.
    return TfFeatures.__arrow_ext_deserialize__(
        None, serialized)  # type: ignore[arg-type]

  if type_name == File.EXTENSION_NAME:
    if physical:
      return pa.string()

    return File.__arrow_ext_deserialize__(
        None,  # type: ignore[arg-type]
        serialized)

  raise TypeError(f"Unsupported Substrait user defined type: {type_}")


def field_name_to_id_dict(schema: pa.Schema) -> Dict[str, int]:
  """Return a dict with field name as key and field ID as value."""
  return {f.name: field_id(f) for f in schema}


def field_id_to_name_dict(schema: pa.Schema) -> Dict[int, str]:
  """Return a dict with field ID as key and field name as value."""
  return {field_id(f): f.name for f in schema}


def field_id_to_column_id_dict(schema: pa.Schema) -> Dict[int, int]:
  """Return a dict with field ID as key and column ID as value."""
  field_id_dict = field_name_to_id_dict(schema)
  return {
      field_id_dict[name]: column_id
      for column_id, name in enumerate(schema.names)
  }


def classify_fields(
    schema: pa.Schema,
    record_fields: Set[str],
    selected_fields: Optional[Set[str]] = None
) -> Tuple[List[utils.Field], List[utils.Field]]:
  """Classify fields into indexes and records.
  
  Args:
    schema: storage logical or physical schema.
    record_fields: names of record fields.
    selected_fields: selected fields to be accessed.

  Returns:
    A tuple (index_fields, record_fields).
  """
  index_fields: List[utils.Field] = []
  record_fields_: List[utils.Field] = []

  for f in schema:
    if selected_fields is not None and f.name not in selected_fields:
      continue

    field = utils.Field(f.name, field_id(f))
    if f.name in record_fields:
      record_fields_.append(field)
    else:
      index_fields.append(field)

  return index_fields, record_fields_


def record_address_types() -> List[Tuple[str, pa.DataType]]:
  """Returns Arrow fields of record addresses."""
  return [(constants.FILE_PATH_FIELD, pa.string()),
          (constants.ROW_ID_FIELD, pa.int32())]


def binary_field(field: utils.Field) -> pa.Field:
  """Return a binary Arrow field for the given field."""
  return _set_field_type(field, pa.binary())


def _set_field_type(field: utils.Field, type_: pa.DataType) -> pa.Field:
  return pa.field(field.name, type_, metadata=field_metadata(field.field_id))


def logical_to_physical_schema(logical_schema: pa.Schema,
                               record_fields: Set[str]) -> pa.Schema:
  """Convert a logical schema to a physical schema."""
  fields: List[pa.Field] = []
  for f in logical_schema:
    if f.name in record_fields:
      fields.append(
          pa.field(
              f.name,
              pa.struct(record_address_types()),  # type: ignore[arg-type]
              metadata=f.metadata))
    else:
      fields.append(f)

  return pa.schema(fields)
