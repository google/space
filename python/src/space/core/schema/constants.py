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
"""Constants for schema."""

FILE_PATH_FIELD = "_FILE"
ROW_ID_FIELD = "_ROW_ID"
FIELD_ID_FIELD = "_FIELD_ID"

NUM_ROWS_FIELD = "_NUM_ROWS"
UNCOMPRESSED_BYTES_FIELD = "_UNCOMPRESSED_BYTES"

# Constants for building column statistics field name.
STATS_FIELD = "_STATS"
MIN_FIELD = "_MIN"
MAX_FIELD = "_MAX"

# Manifest file fields.
INDEX_COMPRESSED_BYTES_FIELD = '_INDEX_COMPRESSED_BYTES'
INDEX_UNCOMPRESSED_BYTES_FIELD = '_INDEX_UNCOMPRESSED_BYTES'
