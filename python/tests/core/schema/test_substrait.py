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

from space.core.schema.substrait import substrait_fields


def test_substrait_fields(sample_arrow_schema, sample_substrait_fields):
  assert substrait_fields(sample_arrow_schema) == sample_substrait_fields


def test_substrait_fields_tf_features(tf_features_arrow_schema,
                                      tf_features_substrait_fields):
  assert substrait_fields(
      tf_features_arrow_schema) == tf_features_substrait_fields


def test_substrait_fields_file(file_arrow_schema, file_substrait_fields):
  assert substrait_fields(file_arrow_schema) == file_substrait_fields
