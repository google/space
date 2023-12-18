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

import json
import numpy as np
import pytest
import tensorflow_datasets as tfds  # type: ignore[import-untyped]
from tensorflow_datasets import features as f  # type: ignore[import-untyped]

from space.core.schema.types import TfFeatures
from space.core.utils.constants import UTF_8


class TestTfFeatures:

  @pytest.fixture
  def tf_features(self):
    features_dict = f.FeaturesDict({
        "objects":
        f.Sequence({
            "bbox": f.BBoxFeature(),
            "id": np.int64
        }),
    })
    return TfFeatures(features_dict)

  @pytest.fixture
  def sample_objects(self):
    return {
        "objects": [{
            "bbox":
            tfds.features.BBox(ymin=0.3, xmin=0.8, ymax=0.5, xmax=1.0),
            "id":
            123
        }]
    }

  def test_arrow_ext_serialize_deserialize(self, tf_features, sample_objects):
    serialized = tf_features.__arrow_ext_serialize__()
    features_dict = json.loads(serialized.decode(UTF_8))
    assert features_dict[
        "type"] == "tensorflow_datasets.core.features.features_dict.FeaturesDict"  # pylint: disable=line-too-long
    assert "sequence" in features_dict["content"]["features"]["objects"]

    tf_features = TfFeatures.__arrow_ext_deserialize__(storage_type=None,
                                                       serialized=serialized)
    assert len(tf_features.serialize(sample_objects)) > 0

  def test_serialize_deserialize(self, tf_features, sample_objects):
    value_bytes = tf_features.serialize(sample_objects)
    assert len(value_bytes) > 0

    objects = tf_features.deserialize(value_bytes)["objects"]
    np.testing.assert_array_equal(
        objects["bbox"], np.array([[0.3, 0.8, 0.5, 1.]], dtype=np.float32))
    np.testing.assert_array_equal(objects["id"], np.array([123]))
