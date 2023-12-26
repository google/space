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

from typing import List

import numpy as np
from numpy.testing import assert_equal
import pyarrow as pa
import pytest
from tensorflow_datasets import features as f  # type: ignore[import-untyped]

from space import Dataset, TfFeatures
import space.core.proto.metadata_pb2 as meta
from space.core.utils.lazy_imports_utils import array_record_module as ar
from space.core.utils.uuids import uuid_

from space.tf.data_sources import SpaceDataSource


class TestSpaceDataSource:

  @pytest.fixture
  def tf_features(self):
    features_dict = f.FeaturesDict({
        "image_id":
        np.int64,
        "objects":
        f.Sequence({"bbox": f.BBoxFeature()}),
    })
    return TfFeatures(features_dict)

  def test_read_space_data_source(self, tmp_path, tf_features):
    schema = pa.schema([("id", pa.int64()), ("features", tf_features)])
    ds = Dataset.create(str(tmp_path / "dataset"),
                        schema,
                        primary_keys=["id"],
                        record_fields=["features"])

    # TODO: to test more records per file.
    input_data = [{
        "id": [123],
        "features": [{
            "image_id": 123,
            "objects": {
                "bbox": np.array([[0.3, 0.8, 0.5, 1.0]], np.float32)
            }
        }]
    }, {
        "id": [456, 789],
        "features": [{
            "image_id": 456,
            "objects": {
                "bbox":
                np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                         np.float32)
            }
        }, {
            "image_id": 789,
            "objects": {
                "bbox":
                np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                         np.float32)
            }
        }]
    }]

    runner = ds.local()
    serializer = ds.serializer()
    for data in input_data:
      runner.append(serializer.serialize(data))

    data_source = SpaceDataSource(ds, ["features"])
    assert_equal(data_source[0], input_data[0]["features"][0])
    assert_equal(data_source[1], input_data[1]["features"][0])
