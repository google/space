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
import pyarrow as pa
import pytest
from tensorflow_datasets import features as f  # type: ignore[import-untyped]

from space import Dataset, TfFeatures
import space.core.proto.metadata_pb2 as meta
from space.core.utils.lazy_imports_utils import array_record_module as ar
from space.core.utils.uuids import uuid_


class TestLocalArrayRecordLoadOp:

  @pytest.fixture
  def tf_features(self):
    features_dict = f.FeaturesDict({
        "image_id": np.int64,
        "objects": f.Sequence({"bbox": f.BBoxFeature()}),
    })
    return TfFeatures(features_dict)

  def test_append_array_record(self, tmp_path, tf_features):
    schema = pa.schema([("id", pa.int64()), ("num_objects", pa.int64()),
                        ("features", tf_features)])
    ds = Dataset.create(str(tmp_path / "dataset"),
                        schema,
                        primary_keys=["id"],
                        record_fields=["features"])

    features_data = [{
        "image_id": 123,
        "objects": {
            "bbox": np.array([[0.3, 0.8, 0.5, 1.0]], np.float32)
        }
    }, {
        "image_id": 456,
        "objects": {
            "bbox":
                np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                         np.float32)
        }
    }]

    # Create dummy ArrayRecord files.
    input_dir = tmp_path / "array_record"
    input_dir.mkdir(parents=True)
    _write_array_record_files(input_dir,
                              [tf_features.serialize(r) for r in features_data])

    def index_fn(record):
      assert len(record['features']) == 1
      features = record['features'][0]
      return {
          "id": features["image_id"],
          'num_objects': features["objects"]["bbox"].shape[0]
      }

    runner = ds.local()
    response = runner.append_array_record(f"{input_dir}/*.array_record",
                                          index_fn)
    assert response.storage_statistics_update == meta.StorageStatistics(
        num_rows=2,
        index_compressed_bytes=104,
        index_uncompressed_bytes=100,
        record_uncompressed_bytes=135)

    index_data = runner.read_all().select(["id", "num_objects"])
    assert index_data == pa.Table.from_pydict({
        "id": [123, 456],
        "num_objects": [1, 2]
    })


def _write_array_record_files(input_dir, records: List[bytes]):
  file_path = f"{uuid_()}.array_record"
  writer = ar.ArrayRecordWriter(str(input_dir / file_path), options="")
  for r in records:
    writer.write(r)

  writer.close()
