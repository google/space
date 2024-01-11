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
from tensorflow_datasets import features as f  # type: ignore[import-untyped]

from space.core.serializers import DeserializedData, FieldSerializer
from space.core.utils.constants import UTF_8


class TfFeatures(pa.ExtensionType, FieldSerializer):
  """A custom Arrow type for Tensorflow Dataset Features."""

  EXTENSION_NAME = "space.tf_features"

  def __init__(self, features_dict: f.FeaturesDict):
    """
    Args:
      features_dict: a Tensorflow Dataset features dict providing serializers
        for a nested dict of Tensors or Numpy arrays, see
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict
    """
    self._features_dict = features_dict
    pa.ExtensionType.__init__(self, pa.binary(), self.EXTENSION_NAME)

  def __arrow_ext_serialize__(self) -> bytes:
    return json.dumps(self._features_dict.to_json()).encode(UTF_8)

  @classmethod
  def __arrow_ext_deserialize__(
      cls,
      storage_type: pa.DataType,  # pylint: disable=unused-argument
      serialized: Union[bytes, str]
  ) -> TfFeatures:
    if isinstance(serialized, bytes):
      serialized = serialized.decode(UTF_8)

    return TfFeatures(f.FeaturesDict.from_json(json.loads(serialized)))

  def serialize(self, value: Any) -> bytes:
    """Serialize value using the provided features_dict."""
    return self._features_dict.serialize_example(value)

  def deserialize(self, value_bytes: bytes) -> DeserializedData:
    """Deserialize value using the provided features_dict."""
    return tfds.as_numpy(self._features_dict.deserialize_example(value_bytes))
