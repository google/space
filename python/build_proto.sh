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

#!/bin/bash

set -e

PY_FOLDER=`pwd`
SRC_FOLDER="${PY_FOLDER}/src"

# Build Substrait protos.
cd "${PY_FOLDER}/../substrait/proto"
protoc --python_out="${SRC_FOLDER}" \
  --mypy_out="${SRC_FOLDER}" \
  substrait/*.proto substrait/extensions/*.proto \
  --proto_path=.

# Build Space protos.
cd "${SRC_FOLDER}"
protoc --python_out=. \
  --mypy_out=. \
  space/core/proto/*.proto \
  --proto_path=. \
  --proto_path=../../substrait/proto
