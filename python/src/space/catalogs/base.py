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
"""Catalogs of Space datasets."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import pyarrow as pa

from space.core.datasets import Dataset
from space.core.views import MaterializedView, View


class BaseCatalog(ABC):
  """A catalog is a container of datasets.
  
  Datasets in a catalog scope can be referenced by a dataset name uniquely.
  """

  @abstractmethod
  def create_dataset(self, name: str, schema: pa.Schema,
                     primary_keys: List[str],
                     record_fields: List[str]) -> Dataset:
    """Create a new empty dataset.
    
    Args:
      name: the dataset name.
      schema: the schema of the storage.
      primary_keys: un-enforced primary keys.
      record_fields: fields stored in row format (ArrayRecord).
    """

  def materialize(self, name: str, view: View):
    """Create a new materialized view.
    
    Args:
      name: the materialized view name.
      view: the view to be materialized.
    """

  @abstractmethod
  def delete_dataset(self, name: str) -> None:
    """Delete an existing dataset or materialized view.
    
    Args:
      name: the dataset name.
    """

  @abstractmethod
  def dataset(self, name: str) -> Union[Dataset, MaterializedView]:
    """Get an existing dataset or materialized view.
    
    Args:
      name: the dataset name.
    """

  @abstractmethod
  def datasets(self) -> List[DatasetInfo]:
    """List all datasets and materialized views in the catalog."""


@dataclass
class DatasetInfo:
  """Basic information of a dataset or materialized view."""

  # Dataset name.
  name: str
  # Dataset storage location.
  location: str
  # TODO: to include create time, type; it requires us to store these fields in
  # entry point file to avoid openning metadata file.
