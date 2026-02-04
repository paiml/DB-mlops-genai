# Week 1: Experiment Tracking with MLflow

## Overview

Understand experiment tracking by implementing an MLflow REST client in Rust.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 1.1 | Video | The Reproducibility Crisis | Concept | 8 min |
| 1.2 | Video | MLflow Architecture: Tracking, Registry, Projects | Databricks | 10 min |
| 1.3 | Lab | Create Experiments in Databricks | Databricks | 30 min |
| 1.4 | Video | MLflow REST Protocol Deep Dive | Concept | 10 min |
| 1.5 | Lab | Build MLflow Client in Rust | Sovereign | 40 min |
| 1.6 | Video | Autologging and Framework Integration | Databricks | 8 min |
| 1.7 | Video | Artifact Storage: DBFS, S3, Unity Catalog | Databricks | 8 min |
| 1.8 | Lab | Compare: Databricks MLflow vs Rust Client | Both | 25 min |
| 1.9 | Quiz | Experiment Tracking Fundamentals | — | 15 min |

## Sovereign AI Stack Components

- `reqwest` for HTTP client
- `serde` for JSON serialization
- `pacha` concepts for artifact storage

## Key Concepts

### MLflow Tracking
- Experiments organize related runs
- Runs contain parameters, metrics, and artifacts
- Metrics can be logged at each training step

### REST API
- `POST /api/2.0/mlflow/experiments/create`
- `POST /api/2.0/mlflow/runs/create`
- `POST /api/2.0/mlflow/runs/log-metric`
- `POST /api/2.0/mlflow/runs/log-batch`
