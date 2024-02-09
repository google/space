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
"""Options of Space core lib."""

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, List, Optional

import pyarrow.compute as pc

# Default number of rows per batch in read result.
DEFAULT_READ_BATCH_SIZE = 16


@dataclass
class ReadOptions:
  """Options of reading data."""
  # Filters on index fields.
  filter_: Optional[pc.Expression] = None

  # When specified, only read the given fields instead of all fields.
  fields: Optional[List[str]] = None

  # The snapshot to read.
  # TODO: to change it to version.
  snapshot_id: Optional[int] = None

  # If true, read the references (e.g., address) of read record fields instead
  # of values.
  reference_read: bool = False

  # The max number of rows per batch in read result.
  #
  # `None` will not enforce batch size, data will be read at the step of row
  # groups. For large row group size, the cost of loading all record fields may
  # be expensive and slow, choose a proper batch size will help.
  #
  # Too small batch size causes too many Ray blocks in Ray runner and will have
  # negative impact on performance.
  #
  # TODO: currently a batch can be smaller than batch_size (e.g., at boundary
  # of row groups), to enforce size to be equal to batch_size.
  batch_size: Optional[int] = None

  def __post_init__(self):
    self.batch_size = self.batch_size or DEFAULT_READ_BATCH_SIZE


@dataclass
class ParquetWriterOptions:
  """Options of Parquet file writer."""
  # Max uncompressed bytes per row group.
  max_uncompressed_row_group_bytes: int = 100 * 1024

  # Max uncompressed bytes per file.
  max_uncompressed_file_bytes: int = 1 * 1024 * 1024


# pylint: disable=line-too-long
@dataclass
class ArrayRecordOptions:
  """Options of ArrayRecord file writer."""
  # Max uncompressed bytes per file.
  max_uncompressed_file_bytes: int = 100 * 1024 * 1024

  # ArrayRecord lib options.
  #
  # See https://github.com/google/array_record/blob/2ac1d904f6be31e5aa2f09549774af65d84bff5a/cpp/array_record_writer.h#L83
  # Default group size 1 maximizes random read performance.
  # It matches the options of TFDS:
  # https://github.com/tensorflow/datasets/blob/92ebd18102b62cf85557ba4b905c970203d8914d/tensorflow_datasets/core/sequential_writer.py#L108
  #
  # A larger group size improves read throughput from Cloud Storage, because
  # each RPC reads a larger chunk of data, which performs better on Cloud
  # Storage.
  options: str = "group_size:1"


@dataclass
class FileOptions:
  """Options of file IO."""
  # Parquet file options.
  parquet_options: ParquetWriterOptions = dataclass_field(
      default_factory=ParquetWriterOptions)

  # ArrayRecord file options.
  array_record_options: ArrayRecordOptions = dataclass_field(
      default_factory=ArrayRecordOptions)


@dataclass
class Range:
  """A range of a field."""
  # Always inclusive.
  min_: Any

  # Default exclusive.
  max_: Any

  # Max is inclusive when true.
  include_max: bool = False


@dataclass
class JoinOptions:
  """Options of joining data."""
  # Partition the join key range into multiple ranges for parallel processing.
  partition_fn: Optional[Callable[[Range], List[Range]]] = None

  # TODO: to support ReadOptions for left and right views, e.g., filter_,
  # snapshot_id
  # TODO: to support join type in PyArrow, only `inner` is supported now.
