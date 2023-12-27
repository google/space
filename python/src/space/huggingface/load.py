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
"""HuggingFace integration with Space."""

from datasets import dataset_dict, load_dataset  # type: ignore[import-untyped]

import space


def load_space_dataset(location: str) -> dataset_dict.DatasetDict:
  """Load a HuggingFace dataset from a Space dataset.
  
  TODO: to support version (snapshot), column selection and filters.
  """
  space_ds = space.Dataset.load(location)
  return load_dataset("parquet", data_files={"train": space_ds.index_files()})
