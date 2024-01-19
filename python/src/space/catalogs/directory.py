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
"""Directory catalog implementation."""

import os

from typing import List, Union
import pyarrow as pa

from space.catalogs.base import BaseCatalog, DatasetInfo
from space.core.datasets import Dataset
import space.core.proto.metadata_pb2 as meta
from space.core.storage import Storage
from space.core.utils import errors, paths
from space.core.views import MaterializedView, View, load_materialized_view


class DirCatalog(BaseCatalog):
  """A directory catalog consists of datasets with location under the same
  directory.

  TODO: to build file system abstraction instead of directly using `os.path`,
  for extension to more file system types.
  """

  def __init__(self, location):
    self._location = location

  def create_dataset(self, name: str, schema: pa.Schema,
                     primary_keys: List[str],
                     record_fields: List[str]) -> Dataset:
    return Dataset.create(self._dataset_location(name), schema, primary_keys,
                          record_fields)

  def materialize(self, name: str, view: View):
    return view.materialize(self._dataset_location(name))

  def delete_dataset(self, name: str) -> None:
    raise NotImplementedError("delete_dataset has not been implemented")

  def dataset(self, name: str) -> Union[Dataset, MaterializedView]:
    try:
      storage = Storage.load(self._dataset_location(name))
    except FileNotFoundError as e:
      raise errors.StorageNotFoundError(str(e)) from None

    if storage.metadata.type == meta.StorageMetadata.DATASET:
      return Dataset(storage)
    elif storage.metadata.type == meta.StorageMetadata.MATERIALIZED_VIEW:
      return load_materialized_view(storage)

    raise errors.SpaceRuntimeError(
        f"Storage type {storage.metadata.type} is not supported")

  def datasets(self) -> List[DatasetInfo]:
    results = []
    for ds_name in os.listdir(self._location):
      ds_location = self._dataset_location(ds_name)
      if os.path.isdir(ds_location) and os.path.isfile(
          paths.entry_point_path(ds_location)):
        results.append(DatasetInfo(ds_name, ds_location))

    return results

  def _dataset_location(self, name: str) -> str:
    return os.path.join(self._location, name)
