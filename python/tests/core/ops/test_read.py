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

from typing import List
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tensorflow_datasets import features  # type: ignore[import-untyped]

from space.core.ops import LocalAppendOp
from space.core.ops import FileSetReadOp
import space.core.proto.metadata_pb2 as meta
from space.core.schema.types import TfFeatures
from space.core.storage import Storage


class TestFileSetReadOp:

  # TODO: to add tests using Arrow table input.
  def test_read_all_types(self, tmp_path):
    location = tmp_path / "dataset"
    schema = pa.schema([
        pa.field("int64", pa.int64()),
        pa.field("float64", pa.float64()),
        pa.field("bool", pa.bool_()),
        pa.field("string", pa.string())
    ])
    storage = Storage.create(location=str(location),
                             schema=schema,
                             primary_keys=["int64"],
                             record_fields=[])

    append_op = LocalAppendOp(str(location), storage.metadata)

    # TODO: the test should cover all types supported by column stats.
    input_data = [
        pa.Table.from_pydict({
            "int64": [1, 2, 3],
            "float64": [0.1, 0.2, 0.3],
            "bool": [True, False, False],
            "string": ["a", "b", "c"]
        }),
        pa.Table.from_pydict({
            "int64": [0, 10],
            "float64": [-0.1, 100.0],
            "bool": [False, False],
            "string": ["A", "z"]
        })
    ]

    for batch in input_data:
      append_op.write(batch)

    storage.commit(append_op.finish())
    data_files = storage.data_files()

    read_op = FileSetReadOp(str(location), storage.metadata, data_files)
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.concat_tables(input_data)

    # Test FileSetReadOp with filters.
    read_op = FileSetReadOp(str(location),
                            storage.metadata,
                            data_files,
                            filter_=(pc.field("bool") == True))
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.Table.from_pydict({
        "int64": [1],
        "float64": [0.1],
        "bool": [True],
        "string": ["a"]
    })

  def test_read_with_record_filters(self, tmp_path):
    tf_features_images = features.FeaturesDict(
        {"images": features.Image(shape=(None, None, 3), dtype=np.uint8)})
    tf_features_objects = features.FeaturesDict({
        "objects":
        features.Sequence({
            "bbox": features.BBoxFeature(),
            "id": np.int64
        }),
    })

    location = tmp_path / "dataset"
    schema = pa.schema([
        pa.field("int64", pa.int64()),
        pa.field("string", pa.string()),
        pa.field("images", TfFeatures(tf_features_images)),
        pa.field("objects", TfFeatures(tf_features_objects))
    ])
    storage = Storage.create(location=str(location),
                             schema=schema,
                             primary_keys=["int64"],
                             record_fields=["images", "objects"])

    append_op = LocalAppendOp(str(location), storage.metadata)

    input_data = [
        pa.Table.from_pydict({
            "int64": [1, 2, 3],
            "string": ["a", "b", "c"],
            "images": [b"images0", b"images1", b"images2"],
            "objects": [b"objects0", b"objects1", b"objects2"]
        }),
        pa.Table.from_pydict({
            "int64": [0, 10],
            "string": ["A", "z"],
            "images": [b"images3", b"images4"],
            "objects": [b"objects3", b"objects4"]
        })
    ]

    for batch in input_data:
      append_op.write(batch)

    storage.commit(append_op.finish())
    data_files = storage.data_files()

    read_op = FileSetReadOp(str(location), storage.metadata, data_files)
    results = list(iter(read_op))
    assert len(results) == 1
    assert list(iter(read_op))[0] == pa.concat_tables(input_data)
