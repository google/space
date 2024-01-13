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
        "image_id": np.int64,
        "objects": f.Sequence({"bbox": f.BBoxFeature()}),
    })
    return TfFeatures(features_dict)

  @pytest.mark.parametrize(
      "feature_fields,test_serializer,test_array_record_data_source,"
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
  def test_data_source(self, tmp_path, tf_features, feature_fields,
                       test_serializer, test_array_record_data_source,
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
    _write_data(ds, runner, input_data, test_serializer)
    runner.delete(pc.field("id") == 60)  # Test modified dataset

    addresses = None
    if test_external_addresses:
      addresses = runner.read_all(filter_=pc.field("id") > 50,
                                  fields=feature_fields,
                                  reference_read=True)
      assert addresses is not None
      addresses = addresses.flatten()

    data_source = RandomAccessDataSource(
        {f: location for f in feature_fields},
        addresses,
        deserialize=test_serializer,
        use_array_record_data_source=test_array_record_data_source)
    assert len(data_source) == sum(
        batch_sizes) - 1 if addresses is None else addresses.num_rows

    indexes = list(range(len(data_source)))
    random.shuffle(indexes)

    if test_single_index and len(feature_fields) > 1:
      results = [data_source[i] for i in indexes]
      if len(feature_fields) > 1:
        results = pa.Table.from_pylist(results).to_pydict()
    else:
      results = data_source[indexes]

    filter_ = lambda a: a != 60  # pylint: disable=unnecessary-lambda-assignment
    if test_external_addresses:
      filter_ = lambda a: a > 50 and a != 60  # pylint: disable=unnecessary-lambda-assignment

    input_data_rows = _read_batches(input_data, feature_fields, filter_)
    if len(feature_fields) > 1:
      expected = {
          f: [input_data_rows[i][f] for i in indexes] for f in feature_fields
      }
    else:
      f = feature_fields[0]
      expected = [input_data_rows[i][f] for i in indexes]

    def _sort(data):
      if test_serializer:
        data.sort(key=lambda v: v["image_id"])
      else:
        data.sort()

    def _multi_field_sort(data):
      key_fn = None
      if test_serializer:
        key_fn = lambda v: v[0]["image_id"]  # pylint: disable=unnecessary-lambda-assignment

      data["f0"], data["f1"] = zip(
          *sorted(zip(data["f0"], data["f1"]), key=key_fn))

    if len(feature_fields) > 1:
      _multi_field_sort(results)
      _multi_field_sort(expected)
    else:
      _sort(results)
      _sort(expected)

    assert_equal(results, expected)

  @pytest.mark.parametrize(
      "test_serializer,test_single_index,test_storage_input",
      [(False, False, False), (True, False, False), (False, True, False),
       (False, True, True)])
  def test_multiple_storages(self, tmp_path, tf_features, test_serializer,
                             test_single_index, test_storage_input):
    # Create the 1st dataset.
    batch_sizes0 = [1, 2, 3]
    location0 = str(tmp_path / "ds0")
    ds0, addresses0, input_data0 = _create_and_write_dataset(
        location0, "feature0", batch_sizes0, tf_features, test_serializer)

    # Create the 2nd dataset.
    batch_sizes1 = [3, 2, 1]
    location1 = str(tmp_path / "ds1")
    ds1, addresses1, input_data1 = _create_and_write_dataset(
        location1, "feature1", batch_sizes1, tf_features, test_serializer)

    assert sum(batch_sizes0) == sum(batch_sizes1)

    # Join two dataset's addresses.
    joined_addresses = addresses0.join(addresses1, keys="id")

    data_source = RandomAccessDataSource(
        {
            "feature0": ds0.storage if test_storage_input else location0,
            "feature1": ds1.storage if test_storage_input else location1,
        },
        joined_addresses,
        deserialize=test_serializer,
        use_array_record_data_source=False)
    assert len(data_source) == addresses0.num_rows

    indexes = list(range(len(data_source)))
    random.shuffle(indexes)

    if test_single_index:
      results = [data_source[i] for i in indexes]
      results = pa.Table.from_pylist(results).to_pydict()
    else:
      results = data_source[indexes]

    input_data_rows0 = _read_batches(input_data0, ["feature0"], None)
    input_data_rows1 = _read_batches(input_data1, ["feature1"], None)
    expected = {
        "feature0": [input_data_rows0[i]["feature0"] for i in indexes],
        "feature1": [input_data_rows1[i]["feature1"] for i in indexes]
    }

    def _sort(data):
      key_fn = None
      if test_serializer:
        key_fn = lambda v: v[0]["image_id"]  # pylint: disable=unnecessary-lambda-assignment

      data["feature0"], data["feature1"] = zip(
          *sorted(zip(data["feature0"], data["feature1"]), key=key_fn))

    _sort(results)
    _sort(expected)
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


def _read_batches(data, feature_fields, filter_):
  result = []
  for batch in data:
    for i in range(len(batch["id"])):
      if filter_ is not None and not filter_(batch["id"][i]):
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


def _write_data(ds, runner, input_data, test_serializer):
  serializer = ds.storage.serializer()
  for data in input_data:
    runner.append(serializer.serialize(data) if test_serializer else data)


def _create_and_write_dataset(location, feature_field, batch_sizes, tf_features,
                              test_serializer):
  feature_fields = [feature_field]
  schema = pa.schema([("id", pa.int64()), (feature_field, tf_features)])
  ds = Dataset.create(location,
                      schema,
                      primary_keys=["id"],
                      record_fields=feature_fields)

  input_data = _generate_data(batch_sizes, feature_fields, test_serializer)

  runner = ds.local()
  _write_data(ds, runner, input_data, test_serializer)

  addresses = runner.read_all(fields=["id"] + feature_fields,
                              reference_read=True)
  assert addresses is not None
  return ds, addresses.flatten(), input_data
