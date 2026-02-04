# Week 2: Feature Engineering

## Overview

Build a feature computation engine with SIMD acceleration and zero-copy data loading.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 2.1 | Video | What is a Feature Store? | Concept | 10 min |
| 2.2 | Video | Databricks Feature Store Architecture | Databricks | 10 min |
| 2.3 | Lab | Create Feature Tables in Unity Catalog | Databricks | 35 min |
| 2.4 | Video | SIMD-Accelerated Feature Computation | Sovereign | 10 min |
| 2.5 | Lab | Build Feature Pipeline with trueno | Sovereign | 40 min |
| 2.6 | Video | Point-in-Time Joins and Data Leakage | Concept | 8 min |
| 2.7 | Lab | Feature Lookup and Online Serving | Databricks | 30 min |
| 2.8 | Video | Delta Lake for Feature Versioning | Databricks | 8 min |
| 2.9 | Quiz | Feature Engineering Systems | — | 15 min |

## Sovereign AI Stack Components

- `alimentar` for zero-copy Parquet loading
- `trueno` for SIMD computation
- `delta-rs` for Delta Lake integration

## Key Concepts

### Feature Transformations
- Z-score normalization: `(x - mean) / stddev`
- Min-max scaling: `(x - min) / (max - min)`
- Log transform: `ln(x + 1)`
- Binning: discretize continuous values

### SIMD Acceleration
- Process 8 floats simultaneously with AVX2
- 4-8x speedup over scalar operations
