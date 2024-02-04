## Cluster Setup and Performance Tuning

Data operations in Space can run distributedly in a Ray cluster. Ray nodes access the same Space dataset files via Cloud Storage or distributed file systems.

## Setup

### Cloud Storage

Setup [GCS FUSE](https://cloud.google.com/storage/docs/gcs-fuse) to use files on Google Cloud Storage (GCS) (or [S3](https://github.com/s3fs-fuse/s3fs-fuse), [Azure](https://github.com/Azure/azure-storage-fuse)):

```bash
gcsfuse <mybucket> "/path/to/<mybucket>"
```

Space has not yet implemented Cloud Storage file systems. FUSE is the current suggested approach.

### Cluster Setup

On the Ray cluster head/worker nodes:
```bash
# Start a Ray head node (IP 123.45.67.89, for example).
# See https://docs.ray.io/en/latest/ray-core/starting-ray.html for details.
ray start --head --port=6379
```

Using [Cloud Storage + FUSE](#cloud-storage) is required in the distributed mode, because the Ray cluster and the client machine should operate on the same directory of files. The mapped local directory paths **must be the same**.

Run the following code on the client machine to connect to the Ray cluster:
```py
import ray

# Connect to the Ray cluster.
ray.init(address="ray://123.45.67.89:10001")
```

## Configure Space Options

Create a Ray runner linking to a Space dataset or view to run operations in the Ray cluster. Use options to tune the performance.

### Data Ingestion

The [WebDataset ingestion example](/notebooks/webdataset_ingestion.ipynb) describes the setup in detail. The options to tune include:

- `max_parallelism`: ingestion workload will run in parallel on Ray nodes, capped by this parallelism

- `array_record_options`: set the [options of ArrayRecord lib](https://github.com/google/array_record/blob/2ac1d904f6be31e5aa2f09549774af65d84bff5a/cpp/array_record_writer.h#L83); Group size is the number of records to serialize together in one chunk. A lower value improves random access latency. However, a larger value is preferred on Cloud Storage, which performs better for batch read. A larger group size reduces the ArrayRecord file size.

```py
# `ds_or_view` is a Space dataset or (materialized) view.
runner = ds_or_view.ray(
  ray_options=RayOptions(max_parallelism=4),
  file_options=FileOptions(
    array_record_options=ArrayRecordOptions(options="group_size:64")
  ))
```

### Data Read

Data read in Ray runner has the following steps:

- Obtain a list of index files to read, based on the filter and version. If a read `batch size` is provided, further split a file into row ranges. Each row range will be a [Ray data block](https://docs.ray.io/en/latest/data/api/doc/ray.data.block.Block.html).

- When reading a block, first read the index file as an Arrow table. If there are record fields, read these fields from ArrayRecord files.

The options to tune include:

- `max_parallelism`: Ray read parallelism, controlls `parallelism` of [Datasource.get_read_tasks](https://docs.ray.io/en/latest/data/api/doc/ray.data.Datasource.get_read_tasks.html#ray.data.Datasource.get_read_tasks)

- `batch_size`: a too small batch size will produce too many Ray blocks and have a negative performance impact. A large batch size will require reading many records from ArrayRecord files for each Ray block, which can be slow.

Examples of setting read batch size in different scenarios:

```py
ray_option = RayOptions(max_parallelism=4)

iterator = ds.ray(ray_option).read(batch_size=64)

mv.ray(ray_option).refresh(batch_size=64)

ray_ds = ds.ray_dataset(ray_option, ReadOptions(batch_size=64))
```

#### Read Data for Training

Users can choose to store data fields in **Parquet** or **ArrayRecord** files (record fields). Space performance is similar to other Parquet based datasets when all fields are in Parquet.

The ArrayRecord reader uses random access read at the granularity of `group size` records. Random access read performs well on local or high performance distributed file systems attached to the reader node (e.g., training VMs). However, the read performance degrades drastically on Cloud Storage, because of too many read RPCs (e.g., per record). The tips for **Cloud Storage** are:

- For quasi sequential read, use a larger `group size` when ingesting data to Space. It can effectively improve read throughput by reducing number of RPC. But it helps only when adjacent records are read together.

- For fully randomized read (the order to read records are shuffled), the Space dataset files should be first cached in a high performance file system.

Random access read training has the benefit of lightweight global shuffling, deterministic training, and checkpointing training state. See the [Grain](https://github.com/google/grain) framework for more details. Integration with Grain is a TODO.
