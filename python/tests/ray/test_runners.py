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

from typing import Any, Dict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import ray

from space import Dataset
from space.core.ops.change_data import ChangeType


def setup_module():
  ray.init(ignore_reinit_error=True, num_cpus=1)


@pytest.fixture
def sample_schema():
  return pa.schema([
      pa.field("int64", pa.int64()),
      pa.field("float64", pa.float64()),
      pa.field("binary", pa.binary())
  ])


@pytest.fixture
def sample_dataset(tmp_path, sample_schema):
  ds = Dataset.create(str(tmp_path / "dataset"),
                      sample_schema,
                      primary_keys=["int64"],
                      record_fields=["binary"])
  return ds


class TestRayReadOnlyRunner:

  def test_diff_map_batches(self, sample_dataset):
    # A sample UDF for testing.
    def _sample_map_udf(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
      batch["float64"] = batch["float64"] + 1
      del batch["binary"]
      return batch

    view_schema = pa.schema(
        [pa.field("int64", pa.int64()),
         pa.field("float64", pa.float64())])
    view = sample_dataset.map_batches(fn=_sample_map_udf,
                                      input_fields=["int64", "binary"],
                                      output_schema=view_schema,
                                      output_record_fields=["binary"])

    ds_runner = sample_dataset.local()
    view_runner = view.ray()

    # Test append.
    ds_runner.append({
        "int64": [1, 2, 3],
        "float64": [0.1, 0.2, 0.3],
        "binary": [b"b1", b"b2", b"b3"]
    })

    expected_change0 = (ChangeType.ADD,
                        pa.Table.from_pydict({
                            "int64": [1, 2, 3],
                            "float64": [1.1, 1.2, 1.3],
                        }))
    assert list(view_runner.diff(0, 1)) == [expected_change0]

    # Test deletion.
    ds_runner.delete(pc.field("int64") == 2)
    expected_change1 = (ChangeType.DELETE,
                        pa.Table.from_pydict({
                            "int64": [2],
                            "float64": [1.2]
                        }))
    assert list(view_runner.diff(1, 2)) == [expected_change1]

    # Test several changes.
    assert list(view_runner.diff(0, 2)) == [expected_change0, expected_change1]

  def test_diff_filter(self, sample_dataset):
    # A sample UDF for testing.
    def _sample_filter_udf(row: Dict[str, Any]) -> Dict[str, Any]:
      return row["float64"] > 0.1

    view = sample_dataset.filter(fn=_sample_filter_udf,
                                 input_fields=["int64", "float64"])

    ds_runner = sample_dataset.local()
    view_runner = view.ray()

    # Test append.
    ds_runner.append({
        "int64": [1, 2, 3],
        "float64": [0.1, 0.2, 0.3],
        "binary": [b"b1", b"b2", b"b3"]
    })

    expected_change0 = (ChangeType.ADD,
                        pa.Table.from_pydict({
                            "int64": [2, 3],
                            "float64": [0.2, 0.3],
                            "binary": [b"b2", b"b3"]
                        }))
    assert list(view_runner.diff(0, 1)) == [expected_change0]

    # Test deletion.
    ds_runner.delete(pc.field("int64") == 2)
    expected_change1 = (ChangeType.DELETE,
                        pa.Table.from_pydict({
                            "int64": [2],
                            "float64": [0.2],
                            "binary": [b"b2"]
                        }))
    assert list(view_runner.diff(1, 2)) == [expected_change1]

    # Test several changes.
    assert list(view_runner.diff(0, 2)) == [expected_change0, expected_change1]
