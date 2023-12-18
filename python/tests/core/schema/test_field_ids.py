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

import pyarrow as pa
import pytest

from space.core.schema import FieldIdManager
from space.core.schema.arrow import field_metadata


class TestFieldIdManager:

  # pylint: disable=too-many-statements
  def test_assign_field_ids(self):
    field_id_mgr = FieldIdManager(next_field_id=123)
    schema = field_id_mgr.assign_field_ids(
        pa.schema([
            pa.field("int64", pa.int64()),
            pa.field("list", pa.list_(pa.string())),
            pa.field(
                "struct",
                pa.struct([
                    pa.field("binary", pa.binary()),
                    pa.field("bool", pa.bool_())
                ])),
            pa.field("list_struct",
                     pa.list_(pa.struct([
                         pa.field("int32", pa.int32()),
                     ]))),
            pa.field("struct_list",
                     pa.struct([pa.field("list", pa.list_(pa.float64()))])),
        ]))

    field = schema.field(0)
    assert field.name == "int64"
    assert pa.types.is_int64(field.type)
    assert field.metadata == field_metadata(123)

    # For list.
    list_field = schema.field(1)
    assert list_field.name == "list"
    assert pa.types.is_list(list_field.type)
    assert list_field.metadata == field_metadata(124)

    field = list_field.type.value_field
    assert field.name == "item"  # list value field name is auto-assigned.
    assert pa.types.is_string(field.type)
    assert field.metadata == field_metadata(125)

    # For struct.
    struct_field = schema.field(2)
    assert struct_field.name == "struct"
    assert pa.types.is_struct(struct_field.type)
    assert struct_field.metadata == field_metadata(126)

    field = struct_field.type.field(0)
    assert field.name == "binary"
    assert pa.types.is_binary(field.type)
    assert field.metadata == field_metadata(127)

    field = struct_field.type.field(1)
    assert field.name == "bool"
    assert pa.types.is_boolean(field.type)
    assert field.metadata == field_metadata(128)

    # For list_struct.
    list_struct_field = schema.field(3)
    assert list_struct_field.name == "list_struct"
    assert pa.types.is_list(list_struct_field.type)
    assert list_struct_field.metadata == field_metadata(129)

    struct_field = list_struct_field.type.value_field
    assert struct_field.name == "item"
    assert pa.types.is_struct(struct_field.type)
    assert struct_field.metadata == field_metadata(130)

    field = struct_field.type.field(0)
    assert field.name == "int32"
    assert pa.types.is_int32(field.type)
    assert field.metadata == field_metadata(131)

    # For struct_list.
    struct_list_field = schema.field(4)
    assert struct_list_field.name == "struct_list"
    assert pa.types.is_struct(struct_list_field.type)
    assert struct_list_field.metadata == field_metadata(132)

    list_field = struct_list_field.type.field(0)
    assert list_field.name == "list"
    assert pa.types.is_list(list_field.type)
    assert list_field.metadata == field_metadata(133)

    field = list_field.type.value_field
    assert field.name == "item"
    assert pa.types.is_float64(field.type)
    assert field.metadata == field_metadata(134)

  def test_assign_field_ids_next_id_unset(self):
    field_id_mgr = FieldIdManager()
    schema = field_id_mgr.assign_field_ids(
        pa.schema(
            [pa.field("int64", pa.int64()),
             pa.field("float32", pa.float32())]))

    field = schema.field(0)
    assert field.name == "int64"
    assert pa.types.is_int64(field.type)
    assert field.metadata == field_metadata(0)

    field = schema.field(1)
    assert field.name == "float32"
    assert pa.types.is_float32(field.type)
    assert field.metadata == field_metadata(1)

  def test_assign_field_ids_invalid_next_id(self):
    with pytest.raises(AssertionError):
      FieldIdManager(-1)

  def test_assign_field_ids_nested_lists(self):
    field_id_mgr = FieldIdManager(next_field_id=567)
    schema = field_id_mgr.assign_field_ids(
        pa.schema([
            pa.field("list_list",
                     pa.list_(pa.list_(pa.field("float64", pa.float64()))))
        ]))

    field = schema.field(0)
    assert field.name == "list_list"
    assert pa.types.is_list(field.type)
    assert field.metadata == field_metadata(567)

    field = field.type.value_field
    assert field.name == "item"
    assert pa.types.is_list(field.type)
    assert field.metadata == field_metadata(568)

    field = field.type.value_field
    assert field.name == "float64"
    assert pa.types.is_float64(field.type)
    assert field.metadata == field_metadata(569)

  def test_assign_field_ids_list_value_field_with_name(self):
    field_id_mgr = FieldIdManager(next_field_id=0)
    schema = field_id_mgr.assign_field_ids(
        pa.schema(
            [pa.field("list", pa.list_(pa.field("float64", pa.float64())))]))

    field = schema.field(0)
    assert field.name == "list"
    assert pa.types.is_list(field.type)
    assert field.metadata == field_metadata(0)

    field = field.type.value_field
    assert field.name == "float64"  # Use user provided value field name.
    assert pa.types.is_float64(field.type)
    assert field.metadata == field_metadata(1)
