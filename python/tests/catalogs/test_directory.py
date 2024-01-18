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

import os

import pyarrow as pa
import pytest

from space import DatasetInfo, DirCatalog
from space.core.utils import errors


class TestDirectoryCatalog:

  def test_crud(self, tmp_path):
    schema = pa.schema([("f", pa.int64())])
    pks = ["f"]
    records = []

    location = str(tmp_path / "cat")
    cat = DirCatalog(location)

    with pytest.raises(FileNotFoundError):
      cat.datasets()

    os.mkdir(location)
    assert not cat.datasets()

    ds1 = cat.create_dataset("ds1", schema, pks, records)
    ds1_data = {"f": [1, 2, 3]}
    ds1.local().append(ds1_data)

    ds1_loaded = cat.dataset("ds1")
    assert ds1_loaded.local().read_all().to_pydict() == ds1_data

    ds1_info = DatasetInfo("ds1", ds1_loaded.storage.location)
    assert cat.datasets() == [ds1_info]

    ds2 = cat.create_dataset("ds2", schema, pks, records)

    key_fn = lambda ds: ds.location  # pylint: disable=unnecessary-lambda-assignment
    assert sorted(cat.datasets(), key=key_fn) == sorted(
        [ds1_info, DatasetInfo("ds2", ds2.storage.location)], key=key_fn)

    with pytest.raises(errors.StorageExistError) as excinfo:
      cat.create_dataset("ds2", schema, pks, records)

    assert "already exists" in str(excinfo.value)
