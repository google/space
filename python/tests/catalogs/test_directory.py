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

import os
from typing import Dict

import numpy as np
import pyarrow as pa
import pytest

from space import DatasetInfo, DirCatalog, RayOptions
from space.core.utils import errors


# A sample UDF for testing.
def _sample_map_udf(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  batch["float64"] = batch["float64"] + 1
  return batch


class TestDirectoryCatalog:

  def test_dataset_crud(self, tmp_path):
    schema = pa.schema([("f", pa.int64())])
    pks = ["f"]
    records = []

    location = str(tmp_path / "cat")
    cat = DirCatalog(location)

    with pytest.raises(FileNotFoundError):
      cat.datasets()

    os.mkdir(location)
    assert not cat.datasets()

    ds1 = cat.create_dataset("ds1", schema, pks, records)
    ds1_data = {"f": [1, 2, 3]}
    ds1.local().append(ds1_data)

    ds1_loaded = cat.dataset("ds1")
    assert ds1_loaded.local().read_all().to_pydict() == ds1_data

    ds1_info = DatasetInfo("ds1", ds1_loaded.storage.location)
    assert cat.datasets() == [ds1_info]

    ds2 = cat.create_dataset("ds2", schema, pks, records)

    key_fn = lambda ds: ds.location  # pylint: disable=unnecessary-lambda-assignment
    assert sorted(cat.datasets(), key=key_fn) == sorted(
        [ds1_info, DatasetInfo("ds2", ds2.storage.location)], key=key_fn)

    with pytest.raises(errors.StorageExistError) as excinfo:
      cat.create_dataset("ds2", schema, pks, records)

    assert "already exists" in str(excinfo.value)

    with pytest.raises(errors.StorageNotFoundError) as excinfo:
      cat.dataset("ds_not_exist")

    assert "Failed to open local file" in str(excinfo.value)

  def test_materialized_view_crud(self, tmp_path):
    schema = pa.schema([("f", pa.int64()), ("float64", pa.float64())])
    pks = ["f"]
    records = []

    location = str(tmp_path / "cat")
    cat = DirCatalog(location)

    ds = cat.create_dataset("ds", schema, pks, records)
    view = ds.map_batches(fn=_sample_map_udf,
                          input_fields=["f", "float64"],
                          output_schema=schema,
                          output_record_fields=[])

    mv1 = cat.materialize("mv1", view)

    ds.local().append({"f": [1, 2, 3], "float64": [0.1, 0.2, 0.3]})
    mv1.ray(RayOptions(max_parallelism=1)).refresh()
    expected_data = {"f": [1, 2, 3], "float64": [1.1, 1.2, 1.3]}
    assert mv1.local().read_all().to_pydict() == expected_data

    mv1_loaded = cat.dataset("mv1")
    assert mv1_loaded.local().read_all().to_pydict() == expected_data

    with pytest.raises(errors.StorageExistError):
      cat.materialize("mv1", view)

    with pytest.raises(errors.StorageExistError):
      cat.materialize("ds", view)

    key_fn = lambda ds: ds.location  # pylint: disable=unnecessary-lambda-assignment
    assert sorted(cat.datasets(), key=key_fn) == sorted([
        DatasetInfo("ds", ds.storage.location),
        DatasetInfo("mv1", mv1.storage.location)
    ],
                                                        key=key_fn)
