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
from substrait.type_pb2 import NamedStruct, Type

from space.core.schema.arrow import field_metadata
from space.core.schema.substrait import substrait_fields


def test_substrait_fields():
  # The value field of list is a field instead of a type to populate field ID.
  schema = pa.schema([
      pa.field("float32", pa.float32(), metadata=field_metadata(100)),
      pa.field("list",
               pa.list_(
                   pa.field("int32", pa.int32(),
                            metadata=field_metadata(110))),
               metadata=field_metadata(120)),
      pa.field("struct",
               pa.struct([
                   pa.field("int64", pa.int64(), metadata=field_metadata(130)),
                   pa.field("float64",
                            pa.float64(),
                            metadata=field_metadata(140))
               ]),
               metadata=field_metadata(150)),
      pa.field("list_struct",
               pa.list_(
                   pa.field("struct",
                            pa.struct([
                                pa.field("bool",
                                         pa.bool_(),
                                         metadata=field_metadata(200))
                            ]),
                            metadata=field_metadata(210))),
               metadata=field_metadata(220)),
      pa.field("struct_list",
               pa.struct([
                   pa.field("list",
                            pa.list_(
                                pa.field("string",
                                         pa.string(),
                                         metadata=field_metadata(230))),
                            metadata=field_metadata(240)),
                   pa.field("binary",
                            pa.binary(),
                            metadata=field_metadata(250))
               ]),
               metadata=field_metadata(260))
  ])

  assert substrait_fields(schema) == NamedStruct(
      names=[
          "float32", "list", "struct", "int64", "float64", "list_struct",
          "bool", "struct_list", "list", "binary"
      ],
      struct=Type.Struct(types=[
          Type(fp32=Type.FP32(type_variation_reference=100)),
          Type(list=Type.List(type=Type(i32=Type.I32(
              type_variation_reference=110)),
                              type_variation_reference=120)),
          Type(struct=Type.Struct(types=[
              Type(i64=Type.I64(type_variation_reference=130)),
              Type(fp64=Type.FP64(type_variation_reference=140))
          ],
                                  type_variation_reference=150)),
          Type(list=Type.List(type=Type(struct=Type.Struct(
              types=[Type(bool=Type.Boolean(type_variation_reference=200))],
              type_variation_reference=210)),
                              type_variation_reference=220)),
          Type(struct=Type.Struct(types=[
              Type(list=Type.List(type=Type(string=Type.String(
                  type_variation_reference=230)),
                                  type_variation_reference=240)),
              Type(binary=Type.Binary(type_variation_reference=250))
          ],
                                  type_variation_reference=260))
      ]))
