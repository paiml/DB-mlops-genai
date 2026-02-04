# Week 6: Capstone — Fraud Detection Platform

## Duration
~15 hours

## Overview

Build an end-to-end ML system for fraud detection using both Databricks and the Sovereign AI Stack.

## Architecture

```
alimentar (Parquet) → trueno (SIMD features) → aprender (train)
         ↓                                           ↓
    Delta tables ←── pacha (sign + register) ←── .apr model
         ↓
    realizar (serve) → renacer (validate) → pmat (quality gate)
         ↓
    batuta (orchestrate + privacy tier = Sovereign)
```

## Deliverables

1. **Feature Pipeline**
   - Databricks Feature Store implementation
   - trueno SIMD comparison

2. **Training Pipeline**
   - AutoML experiment
   - aprender model comparison

3. **Model Registry**
   - Unity Catalog registration
   - pacha signing and verification

4. **Inference Endpoint**
   - Model Serving deployment
   - realizar server comparison

5. **Quality Gates**
   - Lakehouse Monitoring setup
   - pmat TDG enforcement

6. **Orchestrated Workflow**
   - Databricks Jobs pipeline
   - batuta orchestration

## Evaluation Criteria

- Functionality: All components working end-to-end
- Performance: Latency and throughput benchmarks
- Quality: pmat score ≥ B
- Documentation: Clear README and API docs
