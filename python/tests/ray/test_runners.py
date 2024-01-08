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

from __future__ import annotations
from typing import Any, Dict, List, Iterable

import numpy as np
from numpy.testing import assert_equal
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import ray

from space import Dataset, Range
from space.core.jobs import JobResult
from space.core.ops.change_data import ChangeType
from space.core.utils import errors
from space.core.utils.uuids import random_id


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
  ds = Dataset.create(str(tmp_path / f"dataset_{random_id()}"),
                      sample_schema,
                      primary_keys=["int64"],
                      record_fields=["binary"])
  return ds


def _sample_partition_fn(range_: Range) -> List[Range]:
  assert range_ == Range(0, 99, True)

  return [
      Range(0, 10, False),
      Range(10, 50, False),
      Range(50, 99, False),
      Range(99, 99, True)
  ]


class TestRayReadWriteRunner:

  def test_write_read_dataset(self, sample_dataset):
    runner = sample_dataset.ray()

    # Test append.
    input_data0 = generate_data([1, 2, 3])
    runner.append(input_data0)

    assert_equal(runner.read_all().sort_by("int64"),
                 input_data0.sort_by("int64"))

    input_data1 = generate_data([4, 5])
    input_data2 = generate_data([6, 7])
    input_data3 = generate_data([8])
    input_data4 = generate_data([9, 10, 11])

    runner.append_from([
        lambda: iter([input_data1, input_data2]), lambda: iter([input_data3]),
        lambda: iter([input_data4])
    ])

    assert_equal(
        runner.read_all().sort_by("int64"),
        pa.concat_tables(
            [input_data0, input_data1, input_data2, input_data3,
             input_data4]).sort_by("int64"))

    # Test insert.
    result = runner.insert(generate_data([7, 12]))
    assert result.state == JobResult.State.FAILED
    assert "Primary key to insert already exist" in result.error_message

    runner.upsert(generate_data([7, 12]))
    assert_equal(
        runner.read_all().sort_by("int64"),
        pa.concat_tables([
            input_data0, input_data1, input_data2, input_data3, input_data4,
            generate_data([12])
        ]).sort_by("int64"))

    # Test delete.
    runner.delete(pc.field("int64") < 10)
    assert_equal(
        runner.read_all().sort_by("int64"),
        pa.concat_tables([generate_data([10, 11, 12])]).sort_by("int64"))

    # Test reading views.
    # A sample UDF for testing.
    def _sample_map_udf(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
      batch["float64"] = batch["float64"] + 1
      return batch

    view = sample_dataset.map_batches(fn=_sample_map_udf,
                                      output_schema=sample_dataset.schema,
                                      output_record_fields=["binary"])
    view_runner = view.ray()
    assert_equal(
        view_runner.read_all().sort_by("int64"),
        pa.concat_tables([
            pa.Table.from_pydict({
                "int64": [10, 11, 12],
                "float64": [v / 10 + 1 for v in [10, 11, 12]],
                "binary": [f"b{v}".encode("utf-8") for v in [10, 11, 12]]
            })
        ]).sort_by("int64"))

  def test_diff_map_batches(self, tmp_path, sample_dataset):
    # A sample UDF for testing.
    def _sample_map_udf(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
      batch["float64"] = batch["float64"] + 1
      return batch

    view_schema = pa.schema(
        [pa.field("int64", pa.int64()),
         pa.field("float64", pa.float64())])
    view = sample_dataset.map_batches(fn=_sample_map_udf,
                                      input_fields=["int64", "float64"],
                                      output_schema=view_schema)
    mv = view.materialize(str(tmp_path / "mv"))

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

    # Test materialized views.
    ray_runner = mv.ray()
    local_runner = mv.local()

    ray_runner.refresh(1)
    assert local_runner.read_all() == expected_change0[1]
    assert ray_runner.read_all() == expected_change0[1]

    ray_runner.refresh()
    assert local_runner.read_all() == pa.Table.from_pydict({
        "int64": [1, 3],
        "float64": [1.1, 1.3],
    })
    assert ray_runner.read_all() == pa.Table.from_pydict({
        "int64": [1, 3],
        "float64": [1.1, 1.3],
    })

    with pytest.raises(
        errors.SnapshotNotFoundError,
        match=r".*Target snapshot ID 3 higher than source dataset version 2.*"
    ):
      ray_runner.refresh(3)

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
                            "float64": [0.2, 0.3]
                        }))
    assert list(view_runner.diff(0, 1)) == [expected_change0]

    # Test deletion.
    ds_runner.delete(pc.field("int64") == 2)
    expected_change1 = (ChangeType.DELETE,
                        pa.Table.from_pydict({
                            "int64": [2],
                            "float64": [0.2]
                        }))
    assert list(view_runner.diff(1, 2)) == [expected_change1]

    # Test several changes.
    assert list(view_runner.diff(0, 2)) == [expected_change0, expected_change1]

  @pytest.mark.parametrize("left_fields,right_fields,partition_fn",
                           [(None, None, None),
                            (None, None, _sample_partition_fn),
                            (["float64", "binary"], ["string"], None),
                            (["float64"], ["string"], None)])
  def test_join(self, tmp_path, sample_dataset, left_fields, right_fields,
                partition_fn):
    ds1 = sample_dataset
    ds1.local().append(generate_data(range(50)))
    ds1.local().append(generate_data(range(50, 100)))

    ds2 = Dataset.create(str(tmp_path / f"dataset_{random_id()}"),
                         pa.schema([
                             pa.field("int64", pa.int64()),
                             pa.field("string", pa.string())
                         ]),
                         primary_keys=["int64"],
                         record_fields=[])

    def generate_data2(values: Iterable[int]) -> pa.Table:
      return pa.Table.from_pydict({
          "int64": values,
          "string": [f"s{v}" for v in values]
      })

    ds2.local().append(generate_data2(range(-10, 5)))
    ds2.local().append(generate_data2(range(40, 60)))
    ds2.local().append(generate_data2(range(90, 105)))

    view = ds1.join(ds2,
                    keys=["int64"],
                    left_fields=left_fields,
                    right_fields=right_fields,
                    partition_fn=partition_fn)

    assert view.schema == pa.schema([
        pa.field("int64", pa.int64()), *[
            ds1.schema.field(f).remove_metadata()
            for f in left_fields or ds1.schema.names if f != "int64"
        ], *[
            ds2.schema.field(f).remove_metadata()
            for f in right_fields or ds2.schema.names if f != "int64"
        ]
    ])
    assert view.record_fields == (["binary"]
                                  if "binary" in view.schema.names else [])

    def generate_expected(values: Iterable[int]) -> pa.Table:
      return pa.Table.from_pydict({
          "int64":
          values,
          "float64": [v / 10 for v in values],
          "binary": [f"b{v}".encode("utf-8") for v in values],
          "string": [f"s{v}" for v in values]
      })

    expected = generate_expected(
        list(range(0, 5)) + list(range(40, 60)) + list(range(90, 100)))
    assert view.ray().read_all() == expected.select(view.schema.names)

  @pytest.mark.parametrize("left_fields,right_fields",
                           [(["float64"], []), (["int64"], ["string"])])
  def test_join_input_validation_non_join_key(self, tmp_path, sample_dataset,
                                              left_fields, right_fields):
    ds1 = sample_dataset
    ds2 = Dataset.create(str(tmp_path / f"dataset_{random_id()}"),
                         pa.schema([
                             pa.field("int64", pa.int64()),
                             pa.field("string", pa.string())
                         ]),
                         primary_keys=["int64"],
                         record_fields=[])
    with pytest.raises(
        errors.UserInputError,
        match=r".*Join requires reading at least one non-join key.*"):
      ds1.join(ds2,
               keys=["int64"],
               left_fields=left_fields,
               right_fields=right_fields)

  def test_join_input_validation(self, tmp_path, sample_dataset):
    ds1 = sample_dataset
    ds2 = Dataset.create(str(tmp_path / f"dataset_{random_id()}"),
                         pa.schema([
                             pa.field("int64", pa.int64()),
                             pa.field("string", pa.string())
                         ]),
                         primary_keys=["int64"],
                         record_fields=[])

    with pytest.raises(errors.UserInputError,
                       match=r".*Support exactly one join key.*"):
      ds1.join(ds2,
               keys=["int64", "another_key"],
               left_fields=["float64"],
               right_fields=["string"])

    with pytest.raises(
        errors.UserInputError,
        match=r".*Join key must be primary key on both sides.*"):
      ds1.join(ds2,
               keys=["string"],
               left_fields=["float64"],
               right_fields=["string"])


def generate_data(values: Iterable[int]) -> pa.Table:
  return pa.Table.from_pydict({
      "int64":
      values,
      "float64": [v / 10 for v in values],
      "binary": [f"b{v}".encode("utf-8") for v in values]
  })
