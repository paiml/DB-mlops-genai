# Week 4: Model Serving and Inference

## Overview

Deploy inference services with realizar and benchmark against Databricks Model Serving.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 4.1 | Video | Inference Patterns: Batch vs Real-time | Concept | 10 min |
| 4.2 | Video | Databricks Model Serving | Databricks | 10 min |
| 4.3 | Lab | Deploy Endpoint in Databricks | Databricks | 35 min |
| 4.4 | Video | Build Inference Server from Scratch | Sovereign | 10 min |
| 4.5 | Lab | Serve Models with realizar | Sovereign | 35 min |
| 4.6 | Video | Batch Inference with Spark | Databricks | 8 min |
| 4.7 | Lab | Batch Scoring Pipeline | Databricks | 30 min |
| 4.8 | Video | Latency Benchmarking | Both | 8 min |
| 4.9 | Quiz | Inference Systems | — | 15 min |

## Sovereign AI Stack Components

- `realizar` for inference server
- `repartir` for distributed batch processing

## Key Concepts

### Inference Patterns
- Real-time: sub-100ms latency requirements
- Batch: throughput-optimized processing
- Streaming: continuous prediction on data flows

### Server Architecture
- Request queuing and batching
- Model loading and caching
- Health checks and metrics
