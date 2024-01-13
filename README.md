# Space: Unified Storage for Machine Learning

[![Python CI](https://github.com/google/space/actions/workflows/python-ci.yml/badge.svg?branch=main)](https://github.com/google/space/actions/workflows/python-ci.yml)

<hr/>

Unify data in your entire machine learning lifecycle with **Space**, a comprehensive storage solution that seamlessly handles data from ingestion to training.

**Key Features:**
- **Ground Truth Database**
  - Store and manage data locally or in the cloud.
  - Ingest from various sources, including ML datasets, files, and labeling tools.
  - Support data manipulation (append, insert, update, delete) and version control.
- **OLAP Database and Lakehouse**
  - Analyze data distribution using SQL engines like [DuckDB](https://github.com/duckdb/duckdb).
- **Distributed Data Processing Pipelines**
  - Integrate with processing frameworks like [Ray](https://github.com/ray-project/ray) for efficient data transformation.
  - Store processed results as Materialized Views (MVs), and incrementally update MVs when the source is changed.
- **Seamless Training Framework Integration**
  - Access Space datasets and MVs directly via random access interfaces.
  - Convert to popular ML dataset formats (e.g., [TFDS](https://github.com/tensorflow/datasets), [HuggingFace](https://github.com/huggingface/datasets), [Ray](https://github.com/ray-project/ray)).

<img src="docs/pics/overview.png" width="700" />

**Benefits:**
- **Enhanced Efficiency:** Save time and cost by unifying storage and avoiding unnecessary data transfers.
- **Accelerated Insights:** Quickly analyze data with SQL capabilities.
- **Simplified Workflow:** Streamline your entire ML process from data ingestion to training in one graph of transforms and MVs.
- **Ecosystem Integration:** Leverage open source file formats for effortless integration with existing tools.

## Space 101

Space uses [Arrow](https://arrow.apache.org/docs/python/index.html) in the API surface, e.g., schema, filter, data IO. Data operations in Space can run locally or distributedly in [Ray](https://github.com/ray-project/ray) clusters.

Please read [the design](docs/design.md) for more details.

## Onboarding Examples

- [Manage Tensorflow COCO dataset](notebooks/tfds_coco_tutorial.ipynb)
- [Ground truth database of LabelStudio](notebooks/label_studio_tutorial.ipynb)
- [Transforms and materialized views: Segment Anything as example](notebooks/segment_anything_tutorial.ipynb)
- [Incrementally build embedding vector indexes](notebooks/incremental_embedding_index.ipynb)

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

Space has not yet implemented Cloud Storage file systems. FUSE is the current suggested approach.

### Create Empty Datasets

Create a Space dataset with two index fields (`id`, `image_name`) (store in Parquet) and a record field (`feature`) (store in ArrayRecord).

This example uses the plain `binary` type for the record field. Space supports a type `space.TfFeatures` that integrates with the [TFDS feature serializer](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict). See more details in a [TFDS example](/notebooks/tfds_coco_tutorial.ipynb).

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

### Write and Read

Append, delete some data. Each mutation generates a new version of data, represented by an increasing integer ID. We expect to support the [Iceberg](https://iceberg.apache.org/docs/latest/branching/) style tags and branches for better version management.
```py
import pyarrow.compute as pc

# Create a local or Ray runner.
runner = ds.local()  # or ds.ray()

# Appending data generates a new dataset version `snapshot_id=1`.
# Write methods:
# - append(...): no primary key check.
# - insert(...): fail if primary key exists.
# - upsert(...): overwrite if primary key exists.
ids = range(100)
runner.append({
  "id": ids,
  "image_name": [f"{i}.jpg" for i in ids],
  "feature": [f"somedata{i}".encode("utf-8") for i in ids]
})

# Deletion generates a new version `snapshot_id=2`.
runner.delete(pc.field("id") == 1)

# Version management: add tags to snapshots.
ds.add_tag("after_add", 1)
ds.add_tag("after_delete", 2)

# Read options:
# - filter_: optional, apply a filter (push down to reader).
# - fields: optional, field selection.
# - version: optional, snapshot_id or tag, time travel back to an old version.
runner.read_all(
  filter_=pc.field("image_name")=="2.jpg",
  fields=["feature"],
  version="after_add"  # or 1
)

# Read the changes between version 0 and 2.
for change_type, data in runner.diff(0, "after_delete"):
  print(change_type)
  print(data)
  print("===============")
```

### Transform and Materialized Views

Space supports transforming a dataset to a view, and materializing the view to files. The transforms include:

- Mapping batches using a user defined function (UDF).
- Filter using a UDF.
- Joining two views/datasets.

When the source dataset is modified, refreshing the materialized view incrementally synchronizes changes, which saves compute and IO cost. See more details in a [Segment Anything example](/notebooks/segment_anything_tutorial.ipynb). Reading or refreshing views must be the `Ray` runner, because they are implemented based on [Ray transform](https://docs.ray.io/en/latest/data/transforming-data.html).

A materialized view `mv` can be used as a view `mv.view` or a dataset `mv.dataset`. The former always reads data from the source dataset's files and processes all data on-the-fly. The latter directly reads processed data from the MV's files, skips processing data.

#### Example of map_batches

```py
# A sample transform UDF.
# Input is {"field_name": [values, ...], ...}
def modify_feature_udf(batch):
  batch["feature"] = [d + b"123" for d in batch["feature"]]
  return batch

# Create a view and materialize it.
view = ds.map_batches(
  fn=modify_feature_udf,
  output_schema=ds.schema,
  output_record_fields=["feature"]
)

view_runner = view.ray()
# Reading a view will read the source dataset and apply transforms on it.
# It processes all data using `modify_feature_udf` on the fly.
for d in view_runner.read():
  print(d)

mv = view.materialize("/path/to/<mybucket>/example_mv")

mv_runner = mv.ray()
# Refresh the MV up to version `1`.
mv_runner.refresh(1)  # mv_runner.refresh() refresh to the latest version

# Use the MV runner instead of view runner to directly read from materialized
# view files, no data processing any more.
mv_runner.read_all()
```

#### Example of join

See a full example in the [Segment Anything example](/notebooks/segment_anything_tutorial.ipynb). Creating a materialized view of join result is not supported yet.

```py
# If input is a materialized view, using `mv.dataset` instead of `mv.view`
# Only support 1 join key, it must be primary key of both left and right.
joined_view = mv_left.dataset.join(mv_right.dataset, keys=["id"])
```

### ML Frameworks Integration

There are several ways to integrate Space storage with ML frameworks. Space provides a random access data source for reading data in ArrayRecord files:

```py
from space import RandomAccessDataSource

datasource = RandomAccessDataSource(
  # <field-name>: <storage-location>, for reading data from ArrayRecord files.
  {
    "feature": "/path/to/<mybucket>/example_mv",
  },
  # Don't auto deserialize data, because we store them as plain bytes.
  deserialize=False)

len(datasource)
datasource[2]
```

A dataset or view can also be read as a Ray dataset:
```py
ray_ds = ds.ray_dataset()
ray_ds.take(2)
```

Data in Parquet files can be read as a HuggingFace dataset:
```py
from datasets import load_dataset

huggingface_ds = load_dataset("parquet", data_files={"train": ds.index_files()})

```

## Status
Space is a new project under active development.

## Disclaimer
This is not an officially supported Google product.
