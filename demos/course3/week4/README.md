# Week 4: Model Serving and Inference

**Course 3: MLOps Engineering on Databricks**

## Learning Objectives

1. Build an inference server from scratch (realizar concepts)
2. Understand REST API design for model serving
3. Deploy models with Databricks Model Serving
4. Compare latency and scaling characteristics

## Demos

### 1. Inference Server (`inference-server/`)

A Rust HTTP server for model inference using axum.

**What it demonstrates:**
- REST API design (OpenAI-compatible style)
- Request/response handling
- Metrics collection
- Health checks and model info endpoints

**Run locally:**
```bash
cd inference-server
cargo run

# In another terminal:
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[0.1, 0.2, 0.3, 0.4, 0.5]]}'
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | Model information |
| POST | `/v1/predict` | Make predictions |
| GET | `/metrics` | Server metrics |

### 2. Databricks Notebook (`databricks/`)

Databricks Model Serving and batch inference.

**What it demonstrates:**
- Model registration for serving
- Endpoint deployment (pattern)
- Batch inference with Spark UDFs
- A/B testing configuration

**Run on Databricks:**
1. Import `model_serving.py` into your workspace
2. Attach to a cluster
3. Run all cells

## API Design

### Request Format
```json
{
  "inputs": [[0.1, 0.2, 0.3, 0.4, 0.5]],
  "parameters": {
    "return_probabilities": false
  }
}
```

### Response Format
```json
{
  "predictions": [0.85],
  "model": "fraud-detector",
  "version": "1.0.0",
  "latency_ms": 1.23
}
```

## Key Concepts

### Real-time vs Batch Inference

| Aspect | Real-time | Batch |
|--------|-----------|-------|
| Latency | <100ms | Minutes |
| Throughput | 100s/sec | Millions |
| Use case | User-facing | ETL pipelines |
| Technology | HTTP server | Spark jobs |

### Scaling Patterns

1. **Horizontal scaling**: Multiple server instances
2. **Auto-scaling**: Scale based on load
3. **Scale-to-zero**: No cost when idle
4. **Batching**: Group requests for efficiency

### realizar vs Databricks

| Feature | realizar | Databricks |
|---------|---------|------------|
| Latency | 1-10ms | 50-200ms |
| Deployment | Manual | Managed |
| Formats | GGUF, ONNX | MLflow |
| Scaling | Manual | Auto |
| Cost | Infrastructure | Per-request |

## Lab Exercises

1. **Lab 4.3**: Serve a model with realizar
2. **Lab 4.5**: Build batch inference pipeline
3. **Lab 4.8**: Validate with syscall tracing (renacer)

## Performance Benchmarks

Typical latencies for single prediction:

| Platform | Cold Start | Warm |
|----------|-----------|------|
| Local Python | N/A | ~1ms |
| Rust server | N/A | <1ms |
| Databricks Serving | 1-5s | 50-200ms |
| AWS SageMaker | 1-10s | 50-100ms |

**Note:** Cold start applies when scale-to-zero is enabled.

## Resources

- [realizar Documentation](https://docs.rs/realizar)
- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [axum Web Framework](https://docs.rs/axum)
