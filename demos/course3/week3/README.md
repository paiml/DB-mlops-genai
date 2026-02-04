# Week 3: Model Training and Registry

**Course 3: MLOps Engineering on Databricks**

## Learning Objectives

1. Train ML models with aprender (pure Rust ML algorithms)
2. Understand model serialization formats and content addressing
3. Use Databricks AutoML for automated model selection
4. Register and version models with Unity Catalog Model Registry

## Demos

### 1. Model Training (`model-training/`)

Pure Rust ML implementations demonstrating core concepts.

**What it demonstrates:**
- Linear regression with gradient descent
- Decision tree (stump) for classification
- Model cards for metadata
- Serialization for model storage

**Run locally:**
```bash
cd model-training
cargo run
```

**Key files:**
- `src/main.rs` - LinearRegression, DecisionStump, ModelCard

### 2. Databricks Notebook (`databricks/`)

AutoML and Unity Catalog Model Registry.

**What it demonstrates:**
- Manual model training with MLflow tracking
- Databricks AutoML (if available)
- Unity Catalog model registration
- Model versioning and aliases
- Loading models for inference

**Run on Databricks:**
1. Import `automl_registry.py` into your workspace
2. Attach to a ML Runtime cluster
3. Run all cells

## Model Lifecycle

```
Training → Evaluation → Registration → Versioning → Deployment
    │          │            │             │            │
    └──────────┴────────────┴─────────────┴────────────┘
                        MLflow Tracking
```

## Key Concepts

### aprender Algorithms
Pure Rust ML implementations:
- `LinearRegression`: Gradient descent optimization
- `DecisionTree`: Recursive binary splitting
- `RandomForest`: Ensemble of trees
- `KMeans`: Clustering

### pacha Model Registry
Sovereign model management:
- **Content Addressing**: BLAKE3 hash for integrity
- **Signing**: Ed25519 cryptographic signatures
- **Encryption**: ChaCha20-Poly1305 for at-rest protection
- **Lineage**: Full provenance tracking

### Unity Catalog Model Registry
Databricks managed registry:
- **Versioning**: Automatic version tracking
- **Aliases**: Production/staging markers
- **Lineage**: Unity Catalog integration
- **Access Control**: ACL-based permissions

## Comparison

| Aspect | Rust (aprender/pacha) | Databricks |
|--------|----------------------|------------|
| Training | Manual, full control | AutoML, convenience |
| Registry | Cryptographic proof | Platform governance |
| Signing | Ed25519 (explicit) | Implicit (auth) |
| Encryption | ChaCha20-Poly1305 | Platform managed |
| Lineage | Explicit metadata | Unity Catalog |

## Lab Exercises

1. **Lab 3.2**: Train models with aprender
2. **Lab 3.5**: Sign and register models with pacha
3. **Lab 3.8**: Use AutoML and compare results

## Model Card Template

```json
{
  "name": "model-name",
  "version": "1.0.0",
  "description": "What this model does",
  "model_type": "RandomForest",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.93
  },
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "training_data": {
    "n_samples": 10000,
    "n_features": 10
  },
  "author": "team",
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Resources

- [aprender Documentation](https://docs.rs/aprender)
- [pacha Documentation](https://docs.rs/pacha)
- [Databricks Model Registry](https://docs.databricks.com/machine-learning/model-registry/index.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
