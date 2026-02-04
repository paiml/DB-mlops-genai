# Week 1: Experiment Tracking with MLflow

**Course 3: MLOps Engineering on Databricks**

## Learning Objectives

1. Understand why experiment tracking matters (reproducibility crisis)
2. Learn the MLflow REST API by building a client from scratch
3. Use MLflow on Databricks for production experiment tracking
4. Compare the trade-offs between SDK convenience and protocol understanding

## Demos

### 1. MLflow Rust Client (`mlflow-client/`)

A type-safe Rust client for the MLflow REST API.

**What it demonstrates:**
- MLflow REST protocol endpoints
- Type-safe experiment/run/metric structures
- Async HTTP with `reqwest`
- JSON serialization with `serde`

**Run locally:**
```bash
# Start MLflow server (in another terminal)
mlflow server --host 0.0.0.0 --port 5000

# Run the demo
cd mlflow-client
cargo run
```

**Key files:**
- `src/main.rs` - Full client implementation with demo

### 2. Databricks Notebook (`databricks/`)

MLflow tracking on Databricks Free Edition.

**What it demonstrates:**
- Native MLflow integration
- Autologging for sklearn
- Experiment organization
- MLflow UI features

**Run on Databricks:**
1. Import `mlflow_tracking.py` into your workspace
2. Attach to a cluster
3. Run all cells

## API Comparison

| Endpoint | Purpose |
|----------|---------|
| `POST /experiments/create` | Create named experiment |
| `POST /runs/create` | Start a new run |
| `POST /runs/log-parameter` | Log hyperparameter |
| `POST /runs/log-metric` | Log metric value |
| `POST /runs/log-batch` | Batch log metrics/params |
| `POST /runs/update` | End run with status |
| `POST /runs/search` | Query runs |

## Key Concepts

### Experiment
A named container for runs. Maps to a project or model type.

### Run
A single execution of ML code. Contains:
- **Parameters**: Hyperparameters (strings)
- **Metrics**: Numeric values over time
- **Artifacts**: Files (models, plots, data)
- **Tags**: Metadata for organization

### Tracking Server
Stores experiments, runs, and artifacts. Can be:
- Local filesystem
- Remote server (MLflow Tracking Server)
- Managed (Databricks)

## Lab Exercises

1. **Lab 1.3**: Create experiments in Databricks
2. **Lab 1.5**: Extend the Rust client with artifact upload
3. **Lab 1.8**: Benchmark Rust client vs Python SDK

## Resources

- [MLflow REST API Reference](https://mlflow.org/docs/latest/rest-api.html)
- [Databricks MLflow Documentation](https://docs.databricks.com/mlflow/index.html)
