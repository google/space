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
"""Load ArrayRecord files into Space datasets."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import glob

import pyarrow as pa
from typing_extensions import TypeAlias

from space.core.fs.array_record import read_record_file
from space.core.proto import metadata_pb2 as meta
from space.core.proto import runtime_pb2 as rt
from space.core.ops import utils
from space.core.ops.append import LocalAppendOp
from space.core.options import FileOptions
from space.core.schema import arrow
from space.core.serializers import DictSerializer
from space.core.utils.paths import StoragePathsMixin

ArrayRecordIndexFn: TypeAlias = Callable[[Dict[str, Any]], Dict[str, Any]]


class LocalArrayRecordLoadOp(StoragePathsMixin):
  """Load ArrayRecord files into Space without copying data."""

  # pylint: disable=too-many-arguments
  def __init__(self, location: str, metadata: meta.StorageMetadata,
               pattern: str, index_fn: ArrayRecordIndexFn,
               file_options: FileOptions):
    """
    Args:
      pattern: file path pattern of the input ArrayRecord files, e.g.,
        "/directory/*.array_record"
    """
    StoragePathsMixin.__init__(self, location)
    self._file_options = file_options

    self._metadata = metadata
    self._index_fn = index_fn

    record_fields = set(self._metadata.schema.record_fields)
    logical_schema = arrow.arrow_schema(self._metadata.schema.fields,
                                        record_fields,
                                        physical=False)
    self._physical_schema = arrow.logical_to_physical_schema(
        logical_schema, record_fields)

    _, self._record_fields = arrow.classify_fields(self._physical_schema,
                                                   record_fields,
                                                   selected_fields=None)

    assert len(self._record_fields) == 1, "Support only one record field"
    self._record_field = self._record_fields[0]

    self._serializer = DictSerializer.create(logical_schema)
    self._input_files = glob.glob(pattern)

  def write(self) -> Optional[rt.Patch]:
    """Write index files to load ArrayRecord files to Space dataset."""
    append_op = LocalAppendOp(self._location,
                              self._metadata,
                              self._file_options,
                              record_address_input=True)

    total_record_bytes = 0
    for f in self._input_files:
      index_data, record_bytes = self._build_index_for_array_record(f)
      total_record_bytes += record_bytes
      append_op.write(index_data)

    patch = append_op.finish()
    if patch is not None:
      patch.storage_statistics_update.record_uncompressed_bytes += total_record_bytes  # pylint: disable=line-too-long

    return patch

  def _build_index_for_array_record(self,
                                    file_path: str) -> Tuple[pa.Table, int]:
    record_field = self._record_field.name
    # TODO: to avoid loading all data into memory at once.
    serialized_records = read_record_file(file_path)

    indxes: List[Dict[str, Any]] = []
    record_uncompressed_bytes = 0
    for sr in serialized_records:
      record_uncompressed_bytes += len(sr)
      record = self._serializer.deserialize({record_field: [sr]})
      indxes.append(self._index_fn(record))

    index_data = pa.Table.from_pylist(indxes, schema=self._physical_schema)
    index_data = index_data.drop(record_field)  # type: ignore[attr-defined]
    index_data = index_data.append_column(
        record_field,
        utils.address_column(file_path, start_row=0, num_rows=len(indxes)))

    return index_data, record_uncompressed_bytes
