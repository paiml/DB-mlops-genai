# Lab: Feature Pipeline

Build a SIMD-accelerated feature computation pipeline.

## Objectives

- Compute feature statistics
- Implement normalization transforms
- Build a composable pipeline

## Demo Code

See [`demos/course3/week2/feature-pipeline/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course3/week2/feature-pipeline)

## Lab Exercise

See [`labs/course3/week2/lab_2_5_feature_pipeline.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course3/week2)

## Key Transforms

```rust
pub fn normalize_zscore(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let stats = compute_statistics(values)?;
    Ok(values.iter()
        .map(|v| (v - stats.mean) / stats.std_dev)
        .collect())
}

pub fn normalize_minmax(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let stats = compute_statistics(values)?;
    let range = stats.max - stats.min;
    Ok(values.iter()
        .map(|v| (v - stats.min) / range)
        .collect())
}
```

## Validation

Run tests:
```bash
cd demos/course3/week2/feature-pipeline
cargo test
```
