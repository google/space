# Copyright 2024 Google LLC
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

import pyarrow as pa
import pyarrow.compute as pc

import pytest

from space.core.datasets import Dataset
from space.core.ops.change_data import (ChangeData, ChangeType,
                                        ordered_snapshot_ids)
from space.core.utils import errors
from space.core.utils.uuids import random_id


def test_read_change_data(tmp_path, all_types_schema, all_types_input_data):
  location = tmp_path / "dataset"
  ds = Dataset.create(location=str(location),
                      schema=all_types_schema,
                      primary_keys=["int64"],
                      record_fields=[])

  # Validate ADD changes.
  runner = ds.local()
  runner.append_from(lambda: iter(all_types_input_data))

  changes = list(runner.diff(0, 1))
  assert len(changes) == 1
  expected_change0 = ChangeData(ds.storage.metadata.current_snapshot_id,
                                ChangeType.ADD, runner.read_all())
  assert changes[0] == expected_change0

  # Validate DELETE changes.
  runner.delete((pc.field("string") == "a") | (pc.field("string") == "A"))
  changes = list(runner.diff(1, 2))
  assert len(changes) == 1
  expected_change1 = ChangeData(
      ds.storage.metadata.current_snapshot_id, ChangeType.DELETE,
      pa.Table.from_pydict({
          "int64": [1, 0],
          "float64": [0.1, -0.1],
          "bool": [True, False],
          "string": ["a", "A"]
      }))
  assert changes[0] == expected_change1

  # Validate Upsert operation's changes.
  upsert_data = {
      "int64": [2, 3, 4, 5],
      "float64": [0.1, -0.1, 0.4, 0.5],
      "bool": [True, False, True, False],
      "string": ["a", "A", "4", "5"]
  }
  runner.upsert(upsert_data)
  changes = list(runner.diff(2, 3))
  assert len(changes) == 2
  expected_change2 = ChangeData(
      ds.storage.metadata.current_snapshot_id, ChangeType.DELETE,
      pa.Table.from_pydict({
          "int64": [2, 3],
          "float64": [0.2, 0.3],
          "bool": [False, False],
          "string": ["b", "c"]
      }))
  expected_change3 = ChangeData(ds.storage.metadata.current_snapshot_id,
                                ChangeType.ADD,
                                pa.Table.from_pydict(upsert_data))
  assert changes == [expected_change2, expected_change3]

  # Validate diff with several snapshot in-between
  changes = list(runner.diff(0, 3))
  assert len(changes) == 4
  assert changes == [
      expected_change0, expected_change1, expected_change2, expected_change3
  ]


def test_ordered_snapshot_ids(tmp_path):
  schema = pa.schema([
      pa.field("int64", pa.int64()),
      pa.field("float64", pa.float64()),
      pa.field("binary", pa.binary())
  ])
  ds = Dataset.create(str(tmp_path / f"dataset_{random_id()}"),
                      schema,
                      primary_keys=["int64"],
                      record_fields=["binary"])

  runner = ds.local()
  runner.append({"int64": [1], "float64": [0.1], "binary": [b"b1"]})
  runner.append({"int64": [2], "float64": [0.2], "binary": [b"b2"]})
  runner.append({"int64": [3], "float64": [0.3], "binary": [b"b3"]})

  with pytest.raises(
      errors.UserInputError,
      match=r".*End snapshot ID 0 should be higher than start snapshot ID 0.*"):
    ordered_snapshot_ids(ds.storage, 0, 0)

  assert ordered_snapshot_ids(ds.storage, 0, 1) == [1]
  assert ordered_snapshot_ids(ds.storage, 0, 3) == [1, 2, 3]
