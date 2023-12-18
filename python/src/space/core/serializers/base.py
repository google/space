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
from typing import Any, Dict, List

import pyarrow as pa

DictData = Dict[str, List[Any]]


class TypeSerializer(ABC):
  """Abstract serializer of a type."""

  @abstractmethod
  def serialize(self, value: Any) -> bytes:
    """Serialize a value."""
    return NotImplemented

  @abstractmethod
  def deserialize(self, value_bytes: bytes) -> Any:
    """Deserialize bytes to a value."""
    return NotImplemented


class DictSerializer:
  """A serializer for rows in PyDict format.
  
  PyDict format has the layout {"field": [values...], ...}.
  """

  def __init__(self, schema: pa.Schema):
    self._serializers: Dict[str, TypeSerializer] = {}

    for i in range(len(schema)):
      field = schema.field(i)
      if isinstance(field.type, TypeSerializer):
        self._serializers[field.name] = field.type

  def field_serializer(self, field: str) -> TypeSerializer:
    """Return the serializer for a given field."""
    return self._serializers[field]

  def serialize(self, batch: DictData) -> DictData:
    """Serialize a batch of rows."""
    for name, ser in self._serializers.items():
      if name in batch:
        batch[name] = [ser.serialize(d) for d in batch[name]]

    return batch

  def deserialize(self, batch: DictData) -> DictData:
    """Deserialize a batch of rows."""
    for name, ser in self._serializers.items():
      if name in batch:
        batch[name] = [ser.deserialize(d) for d in batch[name]]

    return batch
