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
"""Local runner thats run data and metadata operations locally."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from absl import logging  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.compute as pc

from space.core.ops import FileSetDeleteOp
from space.core.ops import FileSetReadOp
from space.core.ops import LocalAppendOp
from space.core.ops import ReadOptions
from space.core.ops.base import InputData
import space.core.proto.runtime_pb2 as runtime
from space.core.storage import Storage
from space.tf.conversion import LocalConvertTfdsOp, TfdsIndexFn


class BaseRunner(ABC):
  """Abstract base runner class."""

  def __init__(self, storage: Storage):
    self._storage = storage

  @abstractmethod
  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    """Read data from the dataset as an iterator."""

  def read_all(self,
               filter_: Optional[pc.Expression] = None,
               fields: Optional[List[str]] = None,
               snapshot_id: Optional[int] = None,
               reference_read: bool = False) -> pa.Table:
    """Read data from the dataset as an Arrow table."""
    return pa.concat_tables(
        list(self.read(filter_, fields, snapshot_id, reference_read)))

  @abstractmethod
  def append(self, data: InputData) -> runtime.JobResult:
    """Append data into the dataset."""

  @abstractmethod
  def append_tfds(self, tfds_path: str,
                  index_fn: TfdsIndexFn) -> runtime.JobResult:
    """Append data from a Tensorflow Dataset without copying data.
    
    Args:
      tfds_path: the folder of TFDS dataset files, should contain ArrowRecord
        files.
      index_fn: a function that build index fields from each TFDS record.
    """

  @abstractmethod
  def delete(self, filter_: pc.Expression) -> runtime.JobResult:
    """Delete data matching the filter from the dataset."""

  def _try_commit(self, patch: Optional[runtime.Patch]) -> runtime.JobResult:
    if patch is not None:
      self._storage.commit(patch)

    return _job_result(patch)


class LocalRunner(BaseRunner):
  """A runner that runs operations locally."""

  def read(self,
           filter_: Optional[pc.Expression] = None,
           fields: Optional[List[str]] = None,
           snapshot_id: Optional[int] = None,
           reference_read: bool = False) -> Iterator[pa.Table]:
    return iter(
        FileSetReadOp(
            self._storage.location, self._storage.metadata,
            self._storage.data_files(filter_, snapshot_id=snapshot_id),
            ReadOptions(filter_=filter_,
                        fields=fields,
                        reference_read=reference_read)))

  def append(self, data: InputData) -> runtime.JobResult:
    op = LocalAppendOp(self._storage.location, self._storage.metadata)
    op.write(data)
    return self._try_commit(op.finish())

  def append_tfds(self, tfds_path: str,
                  index_fn: TfdsIndexFn) -> runtime.JobResult:
    op = LocalConvertTfdsOp(self._storage.location, self._storage.metadata,
                            tfds_path, index_fn)
    return self._try_commit(op.write())

  def delete(self, filter_: pc.Expression) -> runtime.JobResult:
    ds = self._storage
    op = FileSetDeleteOp(self._storage.location, self._storage.metadata,
                         ds.data_files(filter_), filter_)
    return self._try_commit(op.delete())


def _job_result(patch: Optional[runtime.Patch]) -> runtime.JobResult:
  if patch is None:
    result = runtime.JobResult(state=runtime.JobResult.State.SKIPPED)
  else:
    # TODO: to catch failures and report failed state.
    result = runtime.JobResult(
        state=runtime.JobResult.State.SUCCEEDED,
        storage_statistics_update=patch.storage_statistics_update)

  logging.info(f"Job result:\n{result}")
  return result
