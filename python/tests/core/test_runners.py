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

import threading
from typing import Iterable, Optional

import pytest

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tensorflow_datasets import features as f

from space import Dataset, LocalRunner, TfFeatures
from space.core.jobs import JobResult
from space.core.ops.change_data import ChangeType


class TestLocalRunner:

  @pytest.fixture
  def sample_dataset(self, tmp_path):
    simple_tf_features_dict = f.FeaturesDict({
        "image":
        f.Image(shape=(None, None, 3), dtype=np.uint8),
    })
    schema = pa.schema([("id", pa.int64()), ("name", pa.string()),
                        ("feat1", TfFeatures(simple_tf_features_dict)),
                        ("feat2", TfFeatures(simple_tf_features_dict))])
    return Dataset.create(str(tmp_path / "dataset"),
                          schema,
                          primary_keys=["id"],
                          record_fields=["feat1", "feat2"])

  def test_data_mutation_and_read(self, sample_dataset):
    local_runner = sample_dataset.local()

    id_batches = [range(0, 50), range(50, 90)]
    data_batches = [_generate_data(id_batch) for id_batch in id_batches]

    for data_batch in data_batches:
      local_runner.append(data_batch)

    filter_ = (pc.field("id") >= 49) & (pc.field("id") <= 50)
    assert _read_pyarrow(local_runner, filter_) == _generate_data([49, 50])
    assert _read_pyarrow(local_runner).num_rows == 90

    local_runner.delete(pc.field("id") >= 10)
    assert _read_pyarrow(local_runner) == _generate_data(range(10))

    storage = sample_dataset.storage
    index_manifest = pa.concat_tables(storage.index_manifest()).to_pydict()
    assert index_manifest["_NUM_ROWS"] == [10]

    # Deletion does not change rows in record files.
    record_manifest = pa.concat_tables(storage.record_manifest()).to_pydict()
    assert record_manifest["_NUM_ROWS"] == [50, 50, 40, 40]

  def test_append_empty_data_should_skip_commit(self, sample_dataset):
    local_runner = sample_dataset.local()

    assert local_runner.append({
        "id": [],
        "name": [],
        "feat1": [],
        "feat2": []
    }).state == JobResult.State.SKIPPED

  # pylint: disable=consider-using-with
  def test_conflict_commit_should_fail(self, sample_dataset):
    local_runner = sample_dataset.local()
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock1.acquire()
    lock2.acquire()

    sample_data = _generate_data([1, 2])

    def make_iter():
      yield sample_data
      lock2.release()
      lock1.acquire()
      yield sample_data
      lock1.release()

    job_result = [None]

    def append_data():
      job_result[0] = local_runner.append_from(make_iter)

    t = threading.Thread(target=append_data)
    t.start()

    lock2.acquire()
    local_runner.append(sample_data)
    lock2.release()
    lock1.release()
    t.join()

    assert job_result[0].state == JobResult.State.FAILED
    assert "has been modified" in job_result[0].error_message

  def test_read_and_write_should_reload_storage(self, sample_dataset):
    ds1 = sample_dataset
    ds2 = Dataset.load(ds1.storage.location)
    local_runner1 = ds1.local()
    local_runner2 = ds2.local()

    sample_data1 = _generate_data([1, 2])
    local_runner1.append(sample_data1)
    assert local_runner2.read_all() == sample_data1

    sample_data2 = _generate_data([3, 4])
    local_runner1.append(sample_data2)
    assert list(local_runner2.diff(0, 2)) == [(ChangeType.ADD, sample_data1),
                                              (ChangeType.ADD, sample_data2)]

    sample_data3 = _generate_data([5])
    sample_data4 = _generate_data([6])
    local_runner1.append(sample_data3)
    local_runner2.append(sample_data4)

    assert local_runner1.read_all() == pa.concat_tables(
        [sample_data1, sample_data2, sample_data3, sample_data4])


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
