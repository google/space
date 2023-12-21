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

from space.core.manifests import utils


def test_write_parquet_file(tmp_path):
  data_dir = tmp_path / "file.parquet"

  file_path = str(data_dir)
  returned_path = utils.write_parquet_file(
      file_path, pa.schema([("int64", pa.int64())]),
      pa.Table.from_pydict({"int64": [1, 2]}))

  assert data_dir.exists()
  assert returned_path == file_path
