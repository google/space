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
from typing import List

import pyarrow as pa

from substrait.type_pb2 import NamedStruct, Type

import space.core.schema.arrow as arrow_schema
from space.core.schema.types import TfFeatures
from space.core.utils import constants

# Substrait type name of Arrow custom type TfFeatures.
TF_FEATURES_TYPE = "TF_FEATURES"


def substrait_fields(schema: pa.Schema) -> NamedStruct:
  """Convert Arrow schema to equivalent Substrait fields."""
  mutable_names: List[str] = []
  types = _substrait_fields([schema.field(i) for i in range(len(schema))],
                            mutable_names)
  return NamedStruct(names=mutable_names, struct=Type.Struct(types=types))


def _substrait_fields(fields: List[pa.Field],
                      mutable_names: List[str]) -> List[Type]:
  """Convert a list of Arrow fields to Substrait types, and record field names.
  """
  return [_substrait_field(f, mutable_names) for f in fields]


def _substrait_field(field: pa.Field,
                     mutable_names: List[str],
                     skip_name=False) -> Type:
  """Convert an Arrow fields to a Substrait type, and record its field name."""
  if not skip_name:
    mutable_names.append(field.name)

  type_ = Type()
  field_id = arrow_schema.field_id(field)

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
            skip_name=True))
    # TODO: to support fixed_size_list in substrait.
  elif pa.types.is_struct(field.type):
    _set_field_id(type_.struct, field_id)
    subfields = [field.type.field(i) for i in range(field.type.num_fields)]
    type_.struct.types.extend(_substrait_fields(subfields, mutable_names))
  elif isinstance(field.type, TfFeatures):
    _set_field_id(type_.user_defined, field_id)
    type_.user_defined.type_parameters.extend([
        Type.Parameter(string=TF_FEATURES_TYPE),
        Type.Parameter(string=field.type.__arrow_ext_serialize__().decode(
            constants.UTF_8))
    ])
  else:
    raise ValueError(f"Type is not supported: {field.type}")

  return type_


def _set_field_id(msg, field_id: int) -> None:
  msg.type_variation_reference = field_id
