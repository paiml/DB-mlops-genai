# Course 3: MLOps Engineering Demos

## Week Structure

| Week | Topic | Rust Demo | Databricks Demo |
|------|-------|-----------|-----------------|
| 1 | Tracking & Features | experiment-tracking, feature-pipeline | MLflow, Feature Store |
| 2 | Training & Serving | model-training, inference-server | AutoML, Model Serving |
| 3 | Production & Capstone | quality-gates, capstone | Workflows, Monitoring |

## Running Demos

```bash
# Run specific Rust demo
cd week1/experiment-tracking && cargo run

# Run all Course 3 tests
for dir in week*/*/; do
  if [ -f "$dir/Cargo.toml" ]; then
    (cd "$dir" && cargo test)
  fi
done
```

## Demo Descriptions

### Week 1: Experiment Tracking & Feature Engineering
- **experiment-tracking**: MLflow REST API client for experiment management
- **feature-pipeline**: Feature engineering with SIMD operations
- **databricks-tracking**: MLflow on Databricks
- **databricks-features**: Feature Store integration

### Week 2: Model Training & Serving
- **model-training**: Training pipeline with aprender
- **inference-server**: Model serving with circuit breaker
- **databricks-training**: AutoML and Model Registry
- **databricks-serving**: Databricks Model Serving

### Week 3: Production & Capstone
- **quality-gates**: PMAT quality metrics and thresholds
- **capstone**: End-to-end fraud detection platform
- **databricks-monitoring**: Workflows and monitoring
- **databricks-capstone**: Complete MLOps application
