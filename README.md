# Space: Storage Framework for Machine Learning Datasets

[![Python CI](https://github.com/google/space/actions/workflows/python-ci.yml/badge.svg?branch=main)](https://github.com/google/space/actions/workflows/python-ci.yml)

<hr/>

Space is a hybrid column/row oriented storage framework for Machine Learning datasets. It brings data warehouse/lake features, e.g., data mutation, version management, OLAP queries, materialized views, to ML datasets, while being able to keep data in original row-oriented files. It is perfect for incrementally building and publishing high quality training datasets.

Space uses [Arrow](https://arrow.apache.org/docs/python/index.html) for its APIs (e.g., schema, filter) and in-memory columnar data processing. It supports the file formats:

- [Parqeut](https://parquet.apache.org/) for storing columnar data.
- [ArrayRecord](https://github.com/google/array_record), a high-performance random access row format for storing bulk ML data. [ArrayRecord](https://www.tensorflow.org/datasets/tfless_tfds) is the successor format in [Tensorflow Datasets](https://www.tensorflow.org/datasets) after [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord).

Data can be moved between Space and other ML datasets (e.g, [TFDS](https://www.tensorflow.org/datasets), [HuggingFace](https://huggingface.co/docs/datasets/index)), with zero or minimized file rewrite, if the file formats and schema are compatible.

Data warehouse/lake features are empowered by a simple [Iceberg](https://iceberg.apache.org/) style, copy-on-write open table format. Data operations can run locally, or distributedly in [Ray](https://docs.ray.io/en/latest/index.html) clusters.

We expect to support more file formats and compute frameworks in future.

## Quick Start

### Install and Setup

Install from code:
```bash
cd python
pip install .[dev]
```

Optionally, setup [GCS FUSE](https://cloud.google.com/storage/docs/gcs-fuse) to use files on Google Cloud Storage (GCS) (or [S3](https://github.com/s3fs-fuse/s3fs-fuse), [Azure](https://github.com/Azure/azure-storage-fuse)):

```bash
gcsfuse <mybucket> "/path/to/<mybucket>"
```

### Read and write data

Create a Space dataset with two index fields (`id`, `image_name`) (store in Parquet) and a record field (`feature`) (store in ArrayRecord). We don't define a concrete type for the record field, so the type is plain `binary`:

```py
import pyarrow as pa
from space import Dataset

schema = pa.schema([
  ("id", pa.int64()),
  ("image_name", pa.string()),
  ("feature", pa.binary())])

ds = Dataset.create(
  "/path/to/<mybucket>/example_ds",
  schema,
  primary_keys=["id"],
  record_fields=["feature"])

# Load the dataset from files later:
# ds = Dataset.load("/path/to/<mybucket>/example_ds")
```

Append, delete some data, then read:
```py
import pyarrow.compute as pc

runner = ds.local()  # or ds.ray()

# Appending data generates a new dataset version `1`.
# Write methods:
# - append(...): no primary key check.
# - insert(...): fail if primary key exists.
# - upsert(...): override if primary key exists.
runner.append({
  "id": [1, 2, 3],
  "image_name": ["1.jpg", "2.jpg", "3.jpg"],
  "binary": [b"somedata1", b"somedata2", b"somedata3"]
})

# Deletion generates a new version `2`.
runner.delete(pc.field("id") == 1)

# Obtain an iterator; read options:
# - filter_: optional, apply a filter (push down to reader).
# - fields: optional, field selection.
# - snapshot_id: optional, time travel back to an old version.
runner.read(
  filter_=pc.field("image_name")=="2.jpg",
  fields=["binary"],
  snapshot_id=1
)

# Read the changes between version 0 and 2.
runner.diff(0, 2)
```

### Transform and materialized views

Space supports transforming a dataset to a view, and materializing the view to files. When the source dataset is updated, refreshing the materialized view incrementally synchronizes changes, which saves processing cost.

```py
# A sample UDF that resizes images.
def resize_image_udf(batch):
  batch["binary"] = resize_image(batch["binary"])
  return batch

# Create a view and materialize it.
view = ds.map_batches(
  fn=resize_image_udf,
  output_schema=ds.schema,
  output_record_fields=["binary"]
)
view_runner = view.ray()
view_runner.read()  # Read dataset files and apply transform on it.

mv = view.materialize("/path/to/<mybucket>/example_mv")

mv_runner = mv.ray()
# Refresh the MV up to version `2`.
mv_runner.refresh(2)
mv_runner.read()  # Directly read from materialized view files.
```

### ML frameworks integration

Space datasets/views support popular ML dataset interfaces to integrate with inference/training frameworks:

```py
from space.tf.data_sources import SpaceDataSource

# To a TFDS random access data source.
tf_datasource = SpaceDataSource(ds, feature_fields=["feature"])

# To a Ray dataset.
ray_dataset = ds.ray_dataset()
```

## More readings

### Examples
- [Load TFDS datasets into Space: COCO as example](notebooks/tfds_coco_tutorial.ipynb)
- [Load custom data and build transform pipeline: Segment Anything as example](notebooks/segment_anything_tutorial.ipynb)

## Staus
Space is a new project under active development.

## Disclaimer
This is not an officially supported Google product.
