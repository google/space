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
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import ray

from space.core.options import JoinOptions, Range, ReadOptions
from space.core.schema import arrow
from space.core.schema import constants
from space.core.schema.utils import (file_path_field_name, stats_field_name,
                                     row_id_field_name)
import space.core.transform.utils as transform_utils
from space.core.utils import errors
from space.ray.ops.utils import iter_batches, singleton_storage
from space.ray.options import RayOptions

if TYPE_CHECKING:
  from space.core.views import View


@dataclass
class JoinInput:
  """A helper wraper of join arguments."""
  view: View
  # Fields to read from the view.
  fields: Optional[List[str]]
  # If true, read references (addresses) for record fields.
  reference_read: bool


class RayJoinOp:
  """Join operation running on Ray."""

  # pylint: disable=too-many-arguments
  def __init__(self, left: JoinInput, right: JoinInput, join_keys: List[str],
               schema: pa.Schema, join_options: JoinOptions,
               ray_options: RayOptions):
    assert len(join_keys) == 1
    self._join_key = join_keys[0]

    self._left, self._right = left, right
    self._schema = schema
    self._join_options = join_options
    self._ray_options = ray_options

    self._left_record_fields = _selected_record_fields(left)
    self._right_record_fields = _selected_record_fields(right)

  def ray_dataset(self) -> ray.data.Dataset:
    """Return join result as a Ray dataset."""
    left_range = _join_key_range(self._left, self._join_key)
    right_range = _join_key_range(self._right, self._join_key)

    ranges: List[Range] = []
    if not (left_range is None or right_range is None):
      join_range = Range(min_=max(left_range.min_, right_range.min_),
                         max_=min(left_range.max_, right_range.max_),
                         include_max=True)
      ranges = ([join_range] if self._join_options.partition_fn is None else
                self._join_options.partition_fn(join_range))

    results = []
    for range_ in ranges:
      filter_ = _range_to_filter(self._join_key, range_)
      left_ds = transform_utils.ray_dataset(
          self._left.view, self._ray_options,
          ReadOptions(filter_,
                      self._left.fields,
                      reference_read=self._left.reference_read))
      right_ds = transform_utils.ray_dataset(
          self._right.view, self._ray_options,
          ReadOptions(filter_,
                      self._right.fields,
                      reference_read=self._right.reference_read))
      results.append(
          _join.options(num_returns=1).remote(  # type: ignore[attr-defined]
              _JoinInputInternal(left_ds, self._left.reference_read,
                                 self._left_record_fields),
              _JoinInputInternal(right_ds, self._right.reference_read,
                                 self._right_record_fields), self._join_key,
              self._schema.names))

    return ray.data.from_arrow_refs(results)


@dataclass
class _JoinInputInternal:
  ds: ray.data.Dataset
  reference_read: bool
  record_fields: List[str]


@ray.remote
def _join(left: _JoinInputInternal, right: _JoinInputInternal, join_key: str,
          output_fields: List[str]) -> Optional[pa.Table]:
  left_data, right_data = _read_all(left.ds), _read_all(right.ds)
  if left_data is None or right_data is None:
    return None

  # PyArrow does not support joining struct, so tables are flattened.
  # TODO: we only flatten/fold record addresses field. If any other fields are
  # struct or list, then join won't work. It is a PyArrow limitation.
  if left.reference_read:
    left_data = left_data.flatten()

  if right.reference_read:
    right_data = right_data.flatten()

  # TODO: to make join_type an user provided option.
  result = left_data.join(right_data, keys=join_key, join_type="inner")

  if left.reference_read:
    result = _fold_addresses(result, left.record_fields)

  if right.reference_read:
    result = _fold_addresses(result, right.record_fields)

  # output_fields is used for re-ordering fields.
  return result.select(output_fields)


def _read_all(ds: ray.data.Dataset) -> Optional[pa.Table]:
  results = list(iter_batches(ds))
  if not results:
    return None

  return pa.concat_tables(results)


def _join_key_range(input_: JoinInput, field_name: str) -> Optional[Range]:
  schema = input_.view.schema
  field_name_ids = arrow.field_name_to_id_dict(schema)
  if field_name not in field_name_ids:
    raise errors.UserInputError(
        f"Join key {field_name} not found in schema: {schema}")

  field_id = field_name_ids[field_name]

  stats_field = stats_field_name(field_id)
  min_, max_ = None, None
  for manifest in singleton_storage(input_.view).index_manifest():
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


def _selected_record_fields(input_: JoinInput) -> List[str]:
  if input_.fields is None:
    return input_.view.record_fields

  return list(set(input_.view.record_fields).intersection(set(input_.fields)))


def _fold_addresses(data: pa.Table, record_fields: List[str]) -> pa.Table:
  for f in record_fields:
    file_path_field = file_path_field_name(f)
    row_id_field = row_id_field_name(f)

    file_column = data.column(file_path_field)
    row_column = data.column(row_id_field)

    data = data.drop(file_path_field)  # type: ignore[attr-defined]
    data = data.drop(row_id_field)  # type: ignore[attr-defined]

    arrays = [file_column.combine_chunks(), row_column.combine_chunks()]
    address_column = pa.StructArray.from_arrays(
        arrays,  # type: ignore[arg-type]
        names=[constants.FILE_PATH_FIELD, constants.ROW_ID_FIELD])
    data = data.append_column(f, address_column)

  return data
