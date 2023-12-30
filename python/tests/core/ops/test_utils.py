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

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from space.core.ops import utils
import space.core.proto.metadata_pb2 as meta
import space.core.proto.runtime_pb2 as rt


def test_update_index_storage_stats_positive():
  base = meta.StorageStatistics(num_rows=100,
                                index_compressed_bytes=200,
                                index_uncompressed_bytes=300)

  utils.update_index_storage_stats(
      base,
      meta.StorageStatistics(num_rows=10,
                             index_compressed_bytes=20,
                             index_uncompressed_bytes=30))
  assert base == meta.StorageStatistics(num_rows=110,
                                        index_compressed_bytes=220,
                                        index_uncompressed_bytes=330)


def test_update_index_storage_stats_negative():
  base = meta.StorageStatistics(num_rows=100,
                                index_compressed_bytes=200,
                                index_uncompressed_bytes=300)

  utils.update_index_storage_stats(
      base,
      meta.StorageStatistics(num_rows=-10,
                             index_compressed_bytes=-20,
                             index_uncompressed_bytes=-30))
  assert base == meta.StorageStatistics(num_rows=90,
                                        index_compressed_bytes=180,
                                        index_uncompressed_bytes=270)


def test_update_record_stats_bytes():
  base = meta.StorageStatistics(num_rows=100,
                                index_compressed_bytes=200,
                                index_uncompressed_bytes=300,
                                record_uncompressed_bytes=1000)

  utils.update_record_stats_bytes(
      base,
      meta.StorageStatistics(num_rows=-10,
                             index_compressed_bytes=-20,
                             record_uncompressed_bytes=-100))
  assert base == meta.StorageStatistics(num_rows=100,
                                        index_compressed_bytes=200,
                                        index_uncompressed_bytes=300,
                                        record_uncompressed_bytes=900)


def test_address_column():
  result = [{
      "_FILE": "data/file.array_record",
      "_ROW_ID": 2
  }, {
      "_FILE": "data/file.array_record",
      "_ROW_ID": 3
  }, {
      "_FILE": "data/file.array_record",
      "_ROW_ID": 4
  }]
  assert utils.address_column("data/file.array_record", 2,
                              3).to_pylist() == result


def test_primary_key_filter(all_types_input_data):
  filter_ = utils.primary_key_filter(["int64", "bool"],
                                     pa.Table.from_pydict(
                                         all_types_input_data[1]))
  # pylint: disable=singleton-comparison
  expected_filter = ((pc.field("int64") == 0) &
                     (pc.field("bool") == False)) | (
                         (pc.field("int64") == 10) &
                         (pc.field("bool") == False))

  assert str(filter_) == str(expected_filter)


def test_primary_key_filter_fail_with_duplicated():
  with pytest.raises(RuntimeError):
    utils.primary_key_filter(["int64", "float64"],
                             pa.Table.from_pydict({
                                 "int64": [1, 2, 1],
                                 "float64": [0.1, 0.2, 0.1],
                                 "bool": [True, False, False],
                                 "string": ["a", "b", "c"]
                             }))


def test_merge_patches():
  append_manifests = meta.ManifestFiles(
      index_manifest_files=["data/index_manifest0"],
      record_manifest_files=["data/record_manifest0"])
  append_patch = rt.Patch(addition=append_manifests,
                          storage_statistics_update=meta.StorageStatistics(
                              num_rows=123,
                              index_compressed_bytes=10,
                              index_uncompressed_bytes=20,
                              record_uncompressed_bytes=30))

  delete_manifests = meta.ManifestFiles(
      index_manifest_files=["data/index_manifest0"])
  delete_patch = rt.Patch(deletion=delete_manifests,
                          storage_statistics_update=meta.StorageStatistics(
                              num_rows=-100,
                              index_compressed_bytes=-1,
                              index_uncompressed_bytes=-2,
                              record_uncompressed_bytes=-3))

  upsert_patch = utils.merge_patches([append_patch, delete_patch])
  assert upsert_patch == rt.Patch(
      addition=append_manifests,
      deletion=delete_manifests,
      storage_statistics_update=meta.StorageStatistics(
          num_rows=23,
          index_compressed_bytes=9,
          index_uncompressed_bytes=18,
          record_uncompressed_bytes=27))
