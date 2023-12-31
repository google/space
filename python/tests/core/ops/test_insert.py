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
from space.core.jobs import JobResult


class TestLocalInsertOp:

  # TODO: to add tests using Arrow table input.
  def test_insert_and_upsert(self, tmp_path, all_types_schema,
                             all_types_input_data):
    location = tmp_path / "dataset"
    ds = Dataset.create(location=str(location),
                        schema=all_types_schema,
                        primary_keys=["int64"],
                        record_fields=[])

    runner = ds.local()
    runner.append_from(iter(all_types_input_data))

    # Test insert.
    result = runner.insert({
        "int64": [3, 4],
        "float64": [0.3, 0.4],
        "bool": [False, False],
        "string": ["d", "e"]
    })
    assert result.state == JobResult.State.FAILED
    assert "Primary key to insert already exist" in result.error_message

    input_data = {
        "int64": [4, 5],
        "float64": [0.4, 0.5],
        "bool": [False, False],
        "string": ["e", "f"]
    }
    runner.insert(input_data)

    filter_ = (pc.field("int64") == 4) | (pc.field("int64") == 5)
    assert runner.read_all(filter_=filter_) == pa.Table.from_pydict(input_data)

    # Test upsert.
    input_data = {
        "int64": [4, 5, 6],
        "float64": [1.4, 1.5, 1.6],
        "bool": [True, True, True],
        "string": ["e", "f", "g"]
    }
    runner.upsert(input_data)

    filter_ |= (pc.field("int64") == 6)
    assert runner.read_all(filter_=filter_) == pa.Table.from_pydict(input_data)
