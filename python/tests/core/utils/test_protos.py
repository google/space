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

from space.core.utils import protos
import space.core.proto.metadata_pb2 as meta


def test_proto_to_text():
  text = protos.proto_to_text(meta.StorageMetadata(current_snapshot_id=100))
  assert text.decode("utf-8") == "current_snapshot_id: 100\n"
