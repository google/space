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

import random

import numpy as np
from numpy.testing import assert_equal
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from tensorflow_datasets import features as f  # type: ignore[import-untyped]

from space import Dataset, TfFeatures
from space.core.random_access import RandomAccessDataSource
from space.core.utils.constants import UTF_8


class TestRandomAccessDataSource:

  @pytest.fixture
  def tf_features(self):
    features_dict = f.FeaturesDict({
        "image_id":
        np.int64,
        "objects":
        f.Sequence({"bbox": f.BBoxFeature()}),
    })
    return TfFeatures(features_dict)

  @pytest.mark.parametrize(
      "feature_fields, test_serializer,test_array_record_data_source,"
      "test_external_addresses,test_single_index",
      [
          # Test single feature field, read muliple indexes.
          (["features"], True, True, False, False),
          (["features"], True, False, False, False),
          (["features"], False, True, False, False),
          (["features"], False, False, False, False),
          # Test single feature field, read single index.
          (["features"], False, True, False, True),
          (["features"], False, False, False, True),
          # Test two feature fields, read muliple indexes.
          (["f0", "f1"], True, False, False, False),
          (["f0", "f1"], False, False, False, False),
          (["f0", "f1"], True, False, True, False),
          (["f0", "f1"], False, False, True, False),
          # Test two feature fields, read single index.
          (["f0", "f1"], False, False, False, True),
          (["f0", "f1"], False, False, True, True)
      ])
  def test_read_space_data_source(self, tmp_path, tf_features, feature_fields,
                                  test_serializer,
                                  test_array_record_data_source,
                                  test_external_addresses, test_single_index):
    fields = [("id", pa.int64())]
    for f in feature_fields:
      fields.append((f, tf_features))

    schema = pa.schema(fields)

    location = str(tmp_path / "dataset")
    ds = Dataset.create(location,
                        schema,
                        primary_keys=["id"],
                        record_fields=feature_fields)

    batch_sizes = [1, 2, 5, 10, 100, 1, 10, 20, 1]
    input_data = _generate_data(batch_sizes, feature_fields, test_serializer)

    runner = ds.local()
    serializer = ds.storage.serializer()
    for data in input_data:
      runner.append(serializer.serialize(data) if test_serializer else data)

    addresses = None
    if test_external_addresses:
      addresses = runner.read_all(filter_=pc.field("id") > 50,
                                  fields=feature_fields,
                                  reference_read=True)
      assert addresses is not None
      addresses = addresses.flatten()

    data_source = RandomAccessDataSource(
        location,
        feature_fields,
        addresses,
        deserialize=test_serializer,
        use_array_record_data_source=test_array_record_data_source)
    assert len(data_source) == sum(
        batch_sizes) if addresses is None else addresses.num_rows

    indexes = list(range(len(data_source)))
    random.shuffle(indexes)

    if test_single_index and len(feature_fields) > 1:
      results = [data_source[i] for i in indexes]
      if len(feature_fields) > 1:
        results = pa.Table.from_pylist(results).to_pydict()
    else:
      results = data_source[indexes]

    input_data_rows = _read_batches(input_data, feature_fields,
                                    test_external_addresses)
    if len(feature_fields) > 1:
      expected = {
          f: [input_data_rows[i][f] for i in indexes]
          for f in feature_fields
      }
    else:
      f = feature_fields[0]
      expected = [input_data_rows[i][f] for i in indexes]

    def _sort(results, expected):
      if test_serializer:
        key_fn = lambda v: v["image_id"]  # pylint: disable=unnecessary-lambda-assignment
        results.sort(key=key_fn)
        expected.sort(key=key_fn)
      else:
        results.sort()
        expected.sort()

    if len(feature_fields) > 1:
      for f in feature_fields:
        _sort(results[f], expected[f])
    else:
      _sort(results, expected)

    assert_equal(results, expected)


def _generate_data(batch_sizes, feature_fields, test_serializer):
  result = []
  next_batch_index = 0
  for batch_size in batch_sizes:
    ids = list(range(next_batch_index, next_batch_index + batch_size))

    batch = {"id": ids}
    for f in feature_fields:
      batch[f] = _generate_features(ids, test_serializer)

    result.append(batch)
    next_batch_index += batch_size

  return result


def _read_batches(data, feature_fields, test_external_addresses):
  result = []
  for batch in data:
    for i in range(len(batch["id"])):
      if test_external_addresses and batch["id"][i] <= 50:
        continue

      row = {"id": batch["id"][i]}
      for f in feature_fields:
        row[f] = batch[f][i]

      result.append(row)

  return result


def _generate_features(ids, test_serializer):
  if test_serializer:
    return [{
        "image_id": id_,
        "objects": {
            "bbox": np.array([[0.1, 0.1, 0.1, 0.1]], np.float32)
        }
    } for id_ in ids]

  return [f"bytes_{id_}".encode(UTF_8) for id_ in ids]
