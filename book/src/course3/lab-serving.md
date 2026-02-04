# Lab: Inference Server

Build a model serving infrastructure with batching and health checks.

## Objectives

- Implement prediction endpoint
- Add request batching
- Configure health monitoring

## Demo Code

See [`demos/course3/week4/inference-server/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course3/week4/inference-server)

## Lab Exercise

See [`labs/course3/week4/lab_4_5_serving.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course3/week4)

## Key Components

```rust
pub struct InferenceServer {
    model: Box<dyn Model>,
    batcher: RequestBatcher,
    metrics: ServerMetrics,
}

impl InferenceServer {
    pub async fn predict(&self, request: PredictRequest) -> PredictResponse {
        let start = Instant::now();

        let result = self.batcher.add(request).await;

        self.metrics.record_request(start.elapsed());
        result
    }

    pub fn health(&self) -> HealthResponse {
        HealthResponse {
            status: "healthy",
            model_loaded: self.model.is_loaded(),
            requests_processed: self.metrics.total_requests(),
        }
    }
}
```
