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
"""PyTorch dataset integration."""

from typing import List

from torch.utils.data import Dataset

from space import RandomAccessDataSource


class SpaceDataset(Dataset):
  """PyTorch dataset accessing Space storage."""

  def __init__(self,
               location: str,
               feature_fields: List[str],
               deserialize: bool = False):
    self._ds = RandomAccessDataSource(location,
                                      feature_fields=feature_fields,
                                      deserialize=deserialize)

  def __len__(self) -> int:
    return len(self._ds)

  def __getitem__(self, idx):
    return self._ds[idx]
