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

from typing import Any, Callable, Dict

from google.protobuf import text_format
import numpy as np
import pyarrow as pa
import pytest
from substrait.plan_pb2 import Plan

from space.core.datasets import Dataset
import space.core.proto.metadata_pb2 as meta
from space.core.views import MaterializedView, View


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


class TestLogicalPlan:

  def test_map_batches_to_relation(self, tmp_path, sample_dataset,
                                   sample_map_batch_plan):
    # A sample UDF for testing.
    def _sample_map_udf(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
      batch["float64"] += 1
      return batch

    view = sample_dataset.map_batches(fn=_sample_map_udf,
                                      input_fields=["int64", "binary"],
                                      output_schema=sample_dataset.schema,
                                      output_record_fields=["binary"])
    mv = view.materialize(str(tmp_path / "mv"))

    def named_table_names(plan):
      return plan.relations[0].root.input.project.input.read.named_table.names

    expected_plan = text_format.Parse(sample_map_batch_plan, Plan())
    _verify_logical_plan_and_view(view, mv, sample_dataset, expected_plan,
                                  named_table_names)

  def test_filter_to_relation(self, tmp_path, sample_dataset,
                              sample_filter_plan):
    # A sample UDF for testing.
    def _sample_filter_udf(row: Dict[str, Any]) -> Dict[str, Any]:
      return row["float64"] > 100

    view = sample_dataset.filter(fn=_sample_filter_udf,
                                 input_fields=["int64", "float64"])
    mv = view.materialize(str(tmp_path / "mv"))

    def named_table_names(plan):
      return plan.relations[0].root.input.filter.input.read.named_table.names

    expected_plan = text_format.Parse(sample_filter_plan, Plan())
    _verify_logical_plan_and_view(view, mv, sample_dataset, expected_plan,
                                  named_table_names)


def _verify_logical_plan_and_view(view: View, mv: MaterializedView,
                                  sample_dataset: Dataset, expected_plan: Plan,
                                  named_table_names: Callable):
  # Test logical plan.
  logical_plan = mv.storage.metadata.logical_plan

  def extension_function(plan):
    return plan.extensions[0].extension_function

  named_table_names(expected_plan)[0] = named_table_names(
      logical_plan.logical_plan)[0]
  extension_function(expected_plan).name = extension_function(
      logical_plan.logical_plan).name

  assert logical_plan.logical_plan == expected_plan
  _verify_single_udf(logical_plan)

  # Test loading materialized view.
  loaded_mv = MaterializedView.load(mv.storage.location)
  assert loaded_mv.storage.metadata == mv.storage.metadata

  # Test views.
  _verify_views(view, mv.view, loaded_mv.view, sample_dataset)


def _verify_single_udf(logical_plan: meta.LogicalPlan):
  assert len(logical_plan.udfs) == 1
  udf_name = list(logical_plan.udfs.keys())[0]
  assert logical_plan.udfs[udf_name] == f"metadata/udfs/{udf_name}.pkl"


def _verify_views(a: View, b: View, c: View, d: Dataset):
  assert a.schema == b.schema == c.schema == d.schema
  assert a.primary_keys == b.primary_keys == c.primary_keys == d.primary_keys
  assert (a.record_fields == b.record_fields == c.record_fields ==
          d.record_fields)
  assert (a.sources.keys() == b.sources.keys() == c.sources.keys() ==
          d.sources.keys())
