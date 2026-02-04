# Week 2: Feature Engineering

**Course 3: MLOps Engineering on Databricks**

## Learning Objectives

1. Understand feature stores and why they matter
2. Build SIMD-accelerated feature computation with trueno
3. Use Databricks Feature Store for managed feature engineering
4. Understand point-in-time correctness and data leakage

## Demos

### 1. SIMD Feature Pipeline (`feature-pipeline/`)

A Rust feature engineering pipeline using trueno for SIMD acceleration.

**What it demonstrates:**
- Z-score normalization with SIMD
- Min-max scaling with SIMD
- Rolling window aggregations
- Log transforms and binning
- SIMD vs scalar performance comparison

**Run locally:**
```bash
cd feature-pipeline
cargo run
```

**Key files:**
- `src/main.rs` - Feature pipeline with transforms and benchmarks

### 2. Databricks Notebook (`databricks/`)

Feature Store on Databricks Free Edition.

**What it demonstrates:**
- Spark SQL feature computation
- Unity Catalog feature tables
- Feature lookup for training
- Point-in-time joins

**Run on Databricks:**
1. Import `feature_store.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Feature Transformations

| Transform | Rust (trueno) | Spark |
|-----------|--------------|-------|
| Z-score | `tensor.sub_scalar(mean).div_scalar(std)` | `(col - mean) / std` |
| Min-max | `tensor.sub_scalar(min).div_scalar(range)` | `(col - min) / (max - min)` |
| Log | `tensor.log()` | `F.log(col)` |
| Rolling | Loop with window slices | `Window.rowsBetween()` |
| Binning | Manual bucket assignment | `F.ntile(n)` |

## Key Concepts

### Feature Store
A centralized repository for features that provides:
- **Discovery**: Find and reuse features
- **Versioning**: Track feature changes
- **Lineage**: Understand feature dependencies
- **Serving**: Online and offline access

### Point-in-Time Correctness
Prevent data leakage by ensuring features only use data available at prediction time:
- Use `timestamp_keys` in feature tables
- Window functions respect temporal ordering
- As-of joins for historical features

### SIMD Acceleration
Single Instruction Multiple Data:
- Process 4-16 floats in one CPU instruction
- 5-15x speedup for vectorizable operations
- trueno abstracts AVX2/AVX-512/NEON

## Lab Exercises

1. **Lab 2.3**: Load and profile data with alimentar
2. **Lab 2.5**: Build feature pipeline with trueno
3. **Lab 2.8**: Create feature store with Delta tables

## Performance Notes

| Data Size | Rust SIMD | Spark | Notes |
|-----------|-----------|-------|-------|
| 100K rows | ~5ms | ~500ms | Spark has startup overhead |
| 1M rows | ~50ms | ~800ms | SIMD scales linearly |
| 10M rows | ~500ms | ~2s | Spark catches up at scale |
| 100M+ rows | Memory-bound | Distributed | Spark wins at scale |

**Rule of thumb:** Use SIMD for low-latency, memory-fit workloads. Use Spark for scale.

## Resources

- [trueno SIMD Documentation](https://docs.rs/trueno)
- [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html)
