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
"""Utility methods for protos."""

from google.protobuf import message
from google.protobuf import text_format
from google.protobuf.timestamp_pb2 import Timestamp


def proto_to_text(msg: message.Message) -> bytes:
  """Return the text format of a proto."""
  return text_format.MessageToString(msg).encode("utf-8")


def proto_now() -> Timestamp:
  """Return the current time in the proto format."""
  timestamp = Timestamp()
  timestamp.GetCurrentTime()
  return timestamp
