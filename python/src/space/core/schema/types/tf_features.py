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
"""Define a custom Arrow type for Tensorflow Dataset Features."""

from __future__ import annotations
from typing import Any, Union
import json
import pyarrow as pa
import tensorflow_datasets as tfds  # type: ignore[import-untyped]
from tensorflow_datasets import features as tf_features

from space.core.serializers import TypeSerializer
from space.core.utils import constants


class TfFeatures(pa.ExtensionType, TypeSerializer):
  """A custom Arrow type for Tensorflow Dataset Features."""

  def __init__(self, features: tf_features.FeaturesDict):
    self._features = features
    self._serialized = json.dumps(features.to_json())
    pa.ExtensionType.__init__(self, pa.binary(), self._serialized)

  def __arrow_ext_serialize__(self) -> bytes:
    return self._serialized.encode(constants.UTF_8)

  @classmethod
  def __arrow_ext_deserialize__(
      cls,
      storage_type: pa.DataType,  # pylint: disable=unused-argument
      serialized: Union[bytes, str]
  ) -> TfFeatures:
    if isinstance(serialized, bytes):
      serialized = serialized.decode(constants.UTF_8)

    features = tf_features.FeaturesDict.from_json(json.loads(serialized))
    return TfFeatures(features)

  def serialize(self, value: Any) -> bytes:
    return self._features.serialize_example(value)

  def deserialize(self, value_bytes: bytes) -> Any:
    return tfds.as_numpy(self._features.deserialize_example(value_bytes))
