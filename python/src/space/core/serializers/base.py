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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from typing_extensions import TypeAlias

import pyarrow as pa
# pylint: disable=line-too-long
from tensorflow_datasets.core.dataset_utils import NumpyElem, Tree  # type: ignore[import-untyped]

DeserializedData: TypeAlias = Tree[NumpyElem]
DictData: TypeAlias = Dict[str, List[DeserializedData]]


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


class DictSerializer:
  """A serializer (deserializer) of a dict of fields.

  The fields are serialized by FieldSerializer. The dict is usually a Py dict
  converted from an Arrow table, e.g., {"field": [values, ...], ...}
  """

  def __init__(self, serializers: Dict[str, FieldSerializer]):
    self._serializers = serializers

  @classmethod
  def create(cls, logical_schema: pa.Schema) -> DictSerializer:
    """Create a new dictionary serializer."""
    serializers: Dict[str, FieldSerializer] = {}
    for field in logical_schema:
      if isinstance(field.type, FieldSerializer):
        serializers[field.name] = field.type

    return DictSerializer(serializers)

  def field_serializer(self, field: str) -> Optional[FieldSerializer]:
    """Return the FieldSerializer of a given field, or None if not found."""
    if field not in self._serializers:
      return None

    return self._serializers[field]

  def serialize(self, value: DictData) -> DictData:
    """Serialize a value.

    Args:
      value: a dict of numpy-like nested dicts.
    """
    return self._process_dict(value, serialize=True)

  def deserialize(self, value_bytes: DictData) -> DictData:
    """Deserialize a dict of bytes to a dict of values.
    
    Returns:
      A dict of numpy-like nested dicts.
    """
    return self._process_dict(value_bytes, serialize=False)

  def _process_dict(self, value: DictData, serialize: bool) -> DictData:
    result = {}
    for field_name, value_batch in value.items():
      if field_name in self._serializers:
        ser = self._serializers[field_name]
        result[field_name] = [
            ser.serialize(v) if serialize else ser.deserialize(v)
            for v in value_batch
        ]
      else:
        result[field_name] = value_batch

    return result
