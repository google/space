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

from space.core.ops.append import LocalAppendOp
from space.core.ops.read import FileSetReadOp, ReadOptions
from space.core.storage import Storage


class TestFileSetReadOp:

  # TODO: to add tests using Arrow table input.
  def test_read_all_types(self, tmp_path, all_types_schema,
                          all_types_input_data):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=all_types_schema,
                             primary_keys=["int64"],
                             record_fields=[])

    append_op = LocalAppendOp(str(location), storage.metadata)
    # TODO: the test should cover all types supported by column stats.
    input_data = [pa.Table.from_pydict(d) for d in all_types_input_data]
    for batch in input_data:
      append_op.write(batch)

    storage.commit(append_op.finish())

    read_op = FileSetReadOp(str(location), storage.metadata,
                            storage.data_files())
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.concat_tables(input_data)

    # Test FileSetReadOp with filters.
    read_op = FileSetReadOp(
        str(location),
        storage.metadata,
        storage.data_files(),
        # pylint: disable=singleton-comparison
        ReadOptions(filter_=pc.field("bool") == True))
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.Table.from_pydict({
        "int64": [1],
        "float64": [0.1],
        "bool": [True],
        "string": ["a"]
    })

  def test_read_with_record_filters(self, tmp_path, record_fields_schema,
                                    record_fields_input_data):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=record_fields_schema,
                             primary_keys=["int64"],
                             record_fields=["images", "objects"])

    append_op = LocalAppendOp(str(location), storage.metadata)
    input_data = [pa.Table.from_pydict(d) for d in record_fields_input_data]
    for batch in input_data:
      append_op.write(batch)

    storage.commit(append_op.finish())
    data_files = storage.data_files()

    read_op = FileSetReadOp(str(location), storage.metadata, data_files)
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.concat_tables(input_data)
