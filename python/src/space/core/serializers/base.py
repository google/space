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
"""Serializers (and deserializers) for unstructured record fields."""

from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import TypeAlias

# pylint: disable=line-too-long
from tensorflow_datasets.core.dataset_utils import NumpyElem, Tree  # type: ignore[import-untyped]

DeserializedData: TypeAlias = Tree[NumpyElem]


class FieldSerializer(ABC):
  """Abstract serializer of a field.
  
  Used for serializing record fields into bytes to be stored in Space.
  """

  @abstractmethod
  def serialize(self, value: Any) -> bytes:
    """Serialize a value.

    Args:
      value: numpy-like nested dict.
    """

  @abstractmethod
  def deserialize(self, value_bytes: bytes) -> DeserializedData:
    """Deserialize bytes to a value.
    
    Returns:
      Numpy-like nested dict.
    """
