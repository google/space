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

import numpy as np
import pyarrow as pa
import pytest
from tensorflow_datasets import features  # type: ignore[import-untyped]

from space.core.schema.types import TfFeatures


# TODO: the test should cover all types supported by column stats.
@pytest.fixture
def all_types_schema():
  return pa.schema([
      pa.field("int64", pa.int64()),
      pa.field("float64", pa.float64()),
      pa.field("bool", pa.bool_()),
      pa.field("string", pa.string())
  ])


@pytest.fixture
def all_types_input_data():
  return [{
      "int64": [1, 2, 3],
      "float64": [0.1, 0.2, 0.3],
      "bool": [True, False, False],
      "string": ["a", "b", "c"]
  }, {
      "int64": [0, 10],
      "float64": [-0.1, 100.0],
      "bool": [False, False],
      "string": ["A", "z"]
  }]


@pytest.fixture
def record_fields_schema():
  tf_features_images = features.FeaturesDict(
      {"images": features.Image(shape=(None, None, 3), dtype=np.uint8)})
  tf_features_objects = features.FeaturesDict({
      "objects":
      features.Sequence({
          "bbox": features.BBoxFeature(),
          "id": np.int64
      }),
  })

  return pa.schema([
      pa.field("int64", pa.int64()),
      pa.field("string", pa.string()),
      pa.field("images", TfFeatures(tf_features_images)),
      pa.field("objects", TfFeatures(tf_features_objects))
  ])


@pytest.fixture
def record_fields_input_data():
  return [{
      "int64": [1, 2, 3],
      "string": ["a", "b", "c"],
      "images": [b"images0", b"images1", b"images2"],
      "objects": [b"objects0", b"objects1", b"objects2"]
  }, {
      "int64": [0, 10],
      "string": ["A", "z"],
      "images": [b"images3", b"images4"],
      "objects": [b"objects3", b"objects4"]
  }]
