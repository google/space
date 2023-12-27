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

from space import Dataset
from space.huggingface.load import load_space_dataset


def test_load_space_dataset(tmp_path):
  schema = pa.schema([("int64", pa.int64()), ("string", pa.string())])
  location = str(tmp_path / "dataset")
  ds = Dataset.create(location,
                      schema,
                      primary_keys=["int64"],
                      record_fields=[])

  input_data = [{
      "int64": [1, 2, 3],
      "string": ["a", "b", "c"]
  }, {
      "int64": [0, 10],
      "string": ["A", "z"]
  }]

  runner = ds.local()
  for data in input_data:
    runner.append(data)

  huggingface_ds = load_space_dataset(location)
  assert huggingface_ds["train"].data == runner.read_all()
