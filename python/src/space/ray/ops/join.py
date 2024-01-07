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
#
"""Distributed join operation using Ray."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from space.core.apis import JoinOptions, Range
from space.core.schema import arrow
from space.core.schema import constants
from space.core.schema import utils as schema_utils
from space.core.storage import Storage
import space.core.transform.utils as transform_utils
from space.core.utils import errors
from space.core.utils.lazy_imports_utils import ray
from space.ray.ops.utils import singleton_storage

if TYPE_CHECKING:
  from space.core.views import View


class RayJoinOp:
  """Join operation running on Ray."""

  # pylint: disable=too-many-arguments
  def __init__(self, left: View, left_fields: Optional[List[str]], right: View,
               right_fields: Optional[List[str]], join_keys: List[str],
               schema: pa.Schema, options: JoinOptions):
    assert len(join_keys) == 1

    self._left, self._left_fields = left, left_fields
    self._right, self._right_fields = right, right_fields
    self._options = options
    self._schema = schema

    self._join_key = join_keys[0]

  def ray_dataset(self) -> ray.Dataset:
    """Return join result as a Ray dataset."""
    left_range = _join_key_range(singleton_storage(self._left), self._join_key,
                                 self._left.schema)
    right_range = _join_key_range(singleton_storage(self._right),
                                  self._join_key, self._right.schema)

    ranges: List[Range] = []
    if not (left_range is None or right_range is None):
      join_range = Range(min_=max(left_range.min_, right_range.min_),
                         max_=min(left_range.max_, right_range.max_),
                         include_max=True)
      ranges = ([join_range] if self._options.partition_fn is None else
                self._options.partition_fn(join_range))

    results = []
    for range_ in ranges:
      filter_ = _range_to_filter(self._join_key, range_)
      left_ds = transform_utils.ray_dataset(self._left,
                                            filter_,
                                            self._left_fields,
                                            snapshot_id=None,
                                            reference_read=False)
      right_ds = transform_utils.ray_dataset(self._right,
                                             filter_,
                                             self._right_fields,
                                             snapshot_id=None,
                                             reference_read=False)
      results.append(
          _join.options(num_returns=1).remote(  # type: ignore[attr-defined]
              left_ds, right_ds, self._join_key, self._schema.names))

    return ray.data.from_arrow_refs(results)


@ray.remote
def _join(left: ray.Dataset, right: ray.Dataset, join_key: str,
          output_fields: List[str]) -> Optional[pa.Table]:
  left_data, right_data = _read_all(left), _read_all(right)
  if left_data is None or right_data is None:
    return None

  # output_fields is used for re-ordering fields.
  # TODO: to make join_type an option.
  return left_data.join(right_data, keys=join_key,
                        join_type="inner").select(output_fields)


def _read_all(ds: ray.Dataset) -> Optional[pa.Table]:
  results = []
  for ref in ds.to_arrow_refs():
    data = ray.get(ref)
    if data is None or data.num_rows == 0:
      continue

    results.append(data)

  if not results:
    return None

  return pa.concat_tables(results)


def _join_key_range(source_storage: Storage, field_name: str,
                    schema: pa.Schema) -> Optional[Range]:
  field_name_ids = arrow.field_name_to_id_dict(schema)
  if field_name not in field_name_ids:
    raise errors.UserInputError(
        f"Join key {field_name} not found in schema: {schema}")

  field_id = field_name_ids[field_name]

  stats_field = schema_utils.stats_field_name(field_id)
  min_, max_ = None, None
  for manifest in source_storage.index_manifest():
    stats = manifest.column(stats_field).combine_chunks()

    batch_min = pc.min(  # type: ignore[attr-defined] # pylint: disable=no-member
        stats.field(constants.MIN_FIELD)).as_py()  # type: ignore[arg-type]
    min_ = batch_min if min_ is None else min(  # type: ignore[call-overload]
        min_, batch_min)

    batch_max = pc.max(  # type: ignore[attr-defined] # pylint: disable=no-member
        stats.field(constants.MAX_FIELD)).as_py()  # type: ignore[arg-type]
    max_ = batch_max if max_ is None else max(  # type: ignore[call-overload]
        max_, batch_max)

  if min_ is None and max_ is None:
    return None

  assert min_ is not None and max_ is not None
  return Range(min_=min_, max_=max_, include_max=True)


def _range_to_filter(field_name: str, range_: Range) -> pc.Expression:
  field = pc.field(field_name)
  if range_.include_max:
    return (field >= range_.min_) & (field <= range_.max_)

  return (field >= range_.min_) & (field < range_.max_)
