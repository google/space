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
from space.core.ops.utils import FileOptions
from space.core.ops.delete import FileSetDeleteOp
from space.core.ops.read import FileSetReadOp
from space.core.storage import Storage

_default_file_options = FileOptions()


class TestFileSetDeleteOp:

  # TODO: to add tests using Arrow table input.
  def test_delete_all_types(self, tmp_path, all_types_schema,
                            all_types_input_data):
    location = tmp_path / "dataset"
    storage = Storage.create(location=str(location),
                             schema=all_types_schema,
                             primary_keys=["int64"],
                             record_fields=[])

    append_op = LocalAppendOp(str(location), storage.metadata,
                              _default_file_options)
    # TODO: the test should cover all types supported by column stats.
    input_data = [pa.Table.from_pydict(d) for d in all_types_input_data]
    for batch in input_data:
      append_op.write(batch)

    storage.commit(append_op.finish(), "main")
    old_data_files = storage.data_files()

    delete_op = FileSetDeleteOp(
        str(location),
        storage.metadata,
        storage.data_files(),
        # pylint: disable=singleton-comparison
        pc.field("bool") == False,
        _default_file_options)
    patch = delete_op.delete()
    assert patch is not None
    storage.commit(patch, "main")

    # Verify storage metadata after patch.
    new_data_files = storage.data_files()

    def validate_data_files(data_files, patch_manifests):
      assert len(data_files.index_manifest_files) == 1
      assert len(patch_manifests.index_manifest_files) == 1
      assert data_files.index_manifest_files[
          1] == patch_manifests.index_manifest_files[0]

    validate_data_files(old_data_files, patch.deletion)
    validate_data_files(new_data_files, patch.addition)

    read_op = FileSetReadOp(str(location), storage.metadata,
                            storage.data_files())
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.Table.from_pydict({
        "int64": [1],
        "float64": [0.1],
        "bool": [True],
        "string": ["a"]
    })
