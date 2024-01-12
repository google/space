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
"""Utilities for schemas in the Substrait format."""

from __future__ import annotations
from typing import Any, List

import pyarrow as pa
from substrait.type_pb2 import NamedStruct, Type

import space.core.schema.arrow as arrow_schema
from space.core.schema.types import File, TfFeatures


def substrait_fields(schema: pa.Schema) -> NamedStruct:
  """Convert Arrow schema to equivalent Substrait fields.

  According to the Substrait spec, traverse schema fields in the Depth First
  Search order. The field names are persisted in `mutable_names` in the same
  order.
  """
  mutable_names: List[str] = []
  types = _substrait_fields(list(schema), mutable_names)
  return NamedStruct(names=mutable_names, struct=Type.Struct(types=types))


def _substrait_fields(fields: List[pa.Field],
                      mutable_names: List[str]) -> List[Type]:
  return [_substrait_field(f, mutable_names) for f in fields]


# pylint: disable=too-many-branches
def _substrait_field(field: pa.Field,
                     mutable_names: List[str],
                     is_list_item=False) -> Type:
  if not is_list_item:
    mutable_names.append(field.name)

  type_ = Type()
  field_id = arrow_schema.field_id(field)

  # TODO: to support more types in Substrait, e.g., fixed_size_list, map.
  if pa.types.is_int64(field.type):
    _set_field_id(type_.i64, field_id)
  elif pa.types.is_int32(field.type):
    _set_field_id(type_.i32, field_id)
  elif pa.types.is_string(field.type):
    _set_field_id(type_.string, field_id)
  elif pa.types.is_binary(field.type):
    _set_field_id(type_.binary, field_id)
  elif pa.types.is_boolean(field.type):
    _set_field_id(type_.bool, field_id)
  elif pa.types.is_float64(field.type):
    _set_field_id(type_.fp64, field_id)
  elif pa.types.is_float32(field.type):
    _set_field_id(type_.fp32, field_id)
  elif pa.types.is_list(field.type):
    _set_field_id(type_.list, field_id)
    type_.list.type.CopyFrom(
        _substrait_field(
            field.type.value_field,  # type: ignore[attr-defined]
            mutable_names,
            is_list_item=True))
  elif pa.types.is_struct(field.type):
    _set_field_id(type_.struct, field_id)
    subfields = list(field.type)  # type: ignore[call-overload]
    type_.struct.types.extend(_substrait_fields(subfields, mutable_names))
  elif isinstance(field.type, TfFeatures):
    # TfFeatures is persisted in Substrait as a user defined type, with
    # parameters [TF_FEATURES_TYPE, __arrow_ext_serialize__()].
    _set_field_id(type_.user_defined, field_id)
    type_.user_defined.type_parameters.extend([
        Type.Parameter(string=TfFeatures.EXTENSION_NAME),
        _serialized_ext_type(field.type)
    ])
  elif isinstance(field.type, File):
    # File is persisted in Substrait as a user defined type, with
    # parameters [FILE_TYPE, __arrow_ext_serialize__()].
    _set_field_id(type_.user_defined, field_id)
    type_.user_defined.type_parameters.extend([
        Type.Parameter(string=File.EXTENSION_NAME),
        _serialized_ext_type(field.type)
    ])
  else:
    raise TypeError(f"Type {field.type} of field {field.name} is not supported")

  return type_


def _set_field_id(msg: Any, field_id: int) -> None:
  msg.type_variation_reference = field_id


def _serialized_ext_type(type_: pa.ExtensionType) -> Type.Parameter:
  return Type.Parameter(
      string=type_.__arrow_ext_serialize__())  # type: ignore[arg-type]
