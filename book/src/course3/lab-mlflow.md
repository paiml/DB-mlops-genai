# Lab: MLflow Client

Build an MLflow REST client in Rust to understand experiment tracking internals.

## Objectives

- Implement HTTP client for MLflow REST API
- Create experiments and runs
- Log parameters and metrics
- Search and retrieve runs

## Demo Code

See [`demos/course3/week1/mlflow-client/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course3/week1/mlflow-client)

## Lab Exercise

See [`labs/course3/week1/lab_1_5_mlflow_client.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course3/week1)

## Key Implementation

```rust
pub struct MlflowClient {
    base_url: String,
    client: reqwest::Client,
}

impl MlflowClient {
    pub async fn log_metric(
        &self,
        run_id: &str,
        key: &str,
        value: f64,
    ) -> Result<(), MlflowError> {
        let body = json!({
            "run_id": run_id,
            "key": key,
            "value": value,
            "timestamp": Utc::now().timestamp_millis(),
        });
        self.post_void("runs/log-metric", &body).await
    }
}
```

## Validation

Run tests:
```bash
cd demos/course3/week1/mlflow-client
cargo test
```
