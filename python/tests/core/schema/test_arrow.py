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

from space.core.schema import arrow


def test_field_metadata():
  assert arrow.field_metadata(123) == {b"PARQUET:field_id": b"123"}


def test_field_id():
  assert arrow.field_id(
      pa.field("name", pa.int64(), metadata={b"PARQUET:field_id":
                                             b"123"})) == 123


def test_arrow_schema(sample_substrait_fields, sample_arrow_schema):
  assert sample_arrow_schema == arrow.arrow_schema(sample_substrait_fields)


def test_field_name_to_id_dict(sample_arrow_schema):
  assert arrow.field_name_to_id_dict(sample_arrow_schema) == {
      "float32": 100,
      "list": 120,
      "struct": 150,
      "list_struct": 220,
      "struct_list": 260
  }


def test_field_id_to_column_id_dict(sample_arrow_schema):
  assert arrow.field_id_to_column_id_dict(sample_arrow_schema) == {
      100: 0,
      120: 1,
      150: 2,
      220: 3,
      260: 4
  }


def test_classify_fields(sample_arrow_schema):
  index_fields, record_fields = arrow.classify_fields(sample_arrow_schema,
                                                      ["float32", "list"])

  assert index_fields == [
      arrow.Field("struct", 150),
      arrow.Field("list_struct", 220),
      arrow.Field("struct_list", 260)
  ]
  assert record_fields == [
      arrow.Field("float32", 100),
      arrow.Field("list", 120)
  ]


def test_classify_fields_with_selected_fields(sample_arrow_schema):
  index_fields, record_fields = arrow.classify_fields(sample_arrow_schema,
                                                      ["float32", "list"],
                                                      ["list", "struct"])

  assert index_fields == [arrow.Field("struct", 150)]
  assert record_fields == [arrow.Field("list", 120)]


def test_field_names():
  assert arrow.field_names([
      arrow.Field("struct", 150),
      arrow.Field("list_struct", 220),
      arrow.Field("struct_list", 260)
  ]) == ["struct", "list_struct", "struct_list"]
