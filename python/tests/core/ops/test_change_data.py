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

import pyarrow as pa
import pyarrow.compute as pc

from space import Dataset
from space.core.ops.change_data import ChangeType, read_change_data


def test_read_change_data(tmp_path, all_types_schema, all_types_input_data):
  location = tmp_path / "dataset"
  ds = Dataset.create(location=str(location),
                      schema=all_types_schema,
                      primary_keys=["int64"],
                      record_fields=[])

  # Validate ADD changes.
  def make_iter():
    for d in all_types_input_data:
      yield d

  runner = ds.local()
  runner.append_from(make_iter())

  changes = list(runner.diff(0, 1))
  assert len(changes) == 1
  expected_change0 = (ChangeType.ADD, runner.read_all())
  assert changes[0] == expected_change0

  # Validate DELETE changes.
  runner.delete((pc.field("string") == "a") | (pc.field("string") == "A"))
  changes = list(runner.diff(1, 2))
  assert len(changes) == 1
  expected_change1 = (ChangeType.DELETE,
                      pa.Table.from_pydict({
                          "int64": [1, 0],
                          "float64": [0.1, -0.1],
                          "bool": [True, False],
                          "string": ["a", "A"]
                      }))
  assert changes[0] == expected_change1

  changes = list(runner.diff(0, 2))
  assert len(changes) == 2
  assert changes == [expected_change0, expected_change1]
