# Space: Unified Storage for Machine Learning

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
