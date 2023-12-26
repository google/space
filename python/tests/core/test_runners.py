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

from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tensorflow_datasets import features as f

from space import Dataset, LocalRunner, TfFeatures


class TestLocalRunner:

  def test_data_mutation_and_read(self, tmp_path):
    simple_tf_features_dict = f.FeaturesDict({
        "image":
        f.Image(shape=(None, None, 3), dtype=np.uint8),
    })
    schema = pa.schema([("id", pa.int64()), ("name", pa.string()),
                        ("feat1", TfFeatures(simple_tf_features_dict)),
                        ("feat2", TfFeatures(simple_tf_features_dict))])
    ds = Dataset.create(str(tmp_path / "dataset"),
                        schema,
                        primary_keys=["id"],
                        record_fields=["feat1", "feat2"])
    local_runner = ds.local()

    id_batches = [range(0, 50), range(50, 90)]
    data_batches = [_generate_data(id_batch) for id_batch in id_batches]

    for data_batch in data_batches:
      local_runner.append(data_batch)

    filter_ = (pc.field("id") >= 49) & (pc.field("id") <= 50)
    assert _read_pyarrow(local_runner, filter_) == _generate_data([49, 50])
    assert _read_pyarrow(local_runner).num_rows == 90

    local_runner.delete(pc.field("id") >= 10)
    assert _read_pyarrow(local_runner) == _generate_data(range(10))


def _read_pyarrow(runner: LocalRunner,
                  filter_: Optional[pc.Expression] = None) -> pa.Table:
  return runner.read_all(filter_)


def _generate_data(ids: Iterable[int]) -> pa.Table:
  return pa.Table.from_pydict({
      "id":
      ids,
      "name": [f"name_{i}" for i in ids],
      "feat1": [bytes(f"feat1_{i}", "utf-8") for i in ids],
      "feat2": [bytes(f"feat2_{i}", "utf-8") for i in ids],
  })
