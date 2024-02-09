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
"""ArrayRecord file utilities."""

from typing import List, Optional

from space.core.utils.lazy_imports_utils import array_record_module as ar


def read_record_file(file_path: str,
                     positions: Optional[List[int]] = None) -> List[bytes]:
  """Read records of an ArrayRecord file.
  
  Args:
   file_path: full file path.
   positions: the position inside the file of the records to read.

  """
  record_reader = ar.ArrayRecordReader(file_path)
  if positions is not None:
    records = record_reader.read(positions)
  else:
    records = record_reader.read_all()

  record_reader.close()
  return records
