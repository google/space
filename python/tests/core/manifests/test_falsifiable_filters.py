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
import pytest

from space.core.manifests import falsifiable_filters as ff


@pytest.mark.parametrize(
    "filter_,expected_falsifiable_filter",
    [
        ((pc.field("a") < 10) | (pc.field("b") > 1),
         (pc.field("_STATS_f0", "_MIN") >= 10) &
         (pc.field("_STATS_f1", "_MAX") <= 1)),
        ((pc.field("a") > 10) & (pc.field("b") == 1),
         (pc.field("_STATS_f0", "_MAX") <= 10) |
         ((pc.field("_STATS_f1", "_MIN") > 1) |
          (pc.field("_STATS_f1", "_MAX") < 1))),
        ((pc.field("a") != 10), (pc.field("_STATS_f0", "_MIN") == 10) &
         (pc.field("_STATS_f0", "_MAX") == 10)),
        # Only primary keys are used.
        ((pc.field("a") < 10) | (pc.field("c") > "a"),
         (pc.field("_STATS_f0", "_MIN") >= 10) & False),
        # Corner cases.
        (pc.scalar(False), ~pc.scalar(False)),
        ((pc.scalar(False) | (pc.field("a") <= 10)),
         (~pc.scalar(False) & (pc.field("_STATS_f0", "_MIN") > 10))),
        (~(pc.field("a") >= 10), ~(pc.field("_STATS_f0", "_MAX") < 10)),
        (pc.field("a") > pc.field("a"), pc.field("_STATS_f0", "_MAX")
         <= pc.field("_STATS_f0", "_MIN")),
        (pc.scalar(1) < pc.scalar(2), pc.scalar(1) >= pc.scalar(2))
    ])
def test_build_manifest_filter(filter_, expected_falsifiable_filter):
  arrow_schema = pa.schema([("a", pa.int64()), ("b", pa.float64()),
                            ("c", pa.string())])
  field_name_ids = {"a": 0, "b": 1, "c": 2}

  manifest_filter = ff.build_manifest_filter(arrow_schema, set(["a", "b"]),
                                             field_name_ids, filter_)
  assert str(manifest_filter) == str(~expected_falsifiable_filter)


@pytest.mark.parametrize("filter_", [pc.field("a")])
def test_build_manifest_filter_not_supported_return_none(filter_):
  arrow_schema = pa.schema([("a", pa.int64()), ("b", pa.float64())])
  field_name_ids = {"a": 0, "b": 1}

  assert ff.build_manifest_filter(arrow_schema, set(["a", "b"]),
                                  field_name_ids, filter_) is None
