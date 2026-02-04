# Week 6: Capstone - Sovereign Fraud Detection Platform

**Course 3: MLOps Engineering on Databricks**

## Overview

Build an end-to-end ML system entirely in the Sovereign AI Stack, then compare with the Databricks equivalent.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  alimentar (Parquet) → trueno (SIMD features) → aprender    │
│         ↓                                           ↓       │
│    Delta tables ←── pacha (sign + register) ←── .apr model  │
│         ↓                                                   │
│    realizar (serve) → renacer (validate) → pmat (quality)   │
│         ↓                                                   │
│    batuta (orchestrate + privacy tier = Sovereign)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Capstone Deliverables

| # | Deliverable | Stack Components | Description |
|---|-------------|------------------|-------------|
| 1 | Feature Engine | `alimentar`, `trueno`, `delta-rs` | Load data, compute 6 SIMD features |
| 2 | Training Pipeline | `aprender` | Train logistic regression classifier |
| 3 | Model Registry | `pacha` | Sign model with Ed25519, register with metadata |
| 4 | Inference Server | `realizar` | REST API with <10ms latency target |
| 5 | Quality Gates | `pmat` | TDG score ≥ B, accuracy ≥ 90% |
| 6 | Orchestration | `batuta` | Single command to run full pipeline |
| 7 | Databricks Comparison | Python notebook | Side-by-side with managed equivalent |

## Running the Capstone

### Rust Implementation

```bash
cd capstone
cargo run
```

### Databricks Notebook

Import `databricks/capstone_comparison.py` into your workspace.

## Demo Output

```
╔═══════════════════════════════════════════════════════════════╗
║     Fraud Detection Platform - Course 3 Capstone              ║
╚═══════════════════════════════════════════════════════════════╝

📊 Step 1: Data Generation
   Training samples: 1000
   Test samples: 200

🔧 Step 2: Feature Engineering
   Features computed: 6

🎯 Step 3: Model Training
   Model: fraud-detector v1.0.0

🚀 Step 4: Model Serving
   Avg latency: 0.015ms

📈 Step 5: Quality Evaluation
   Accuracy: 0.85+
   Quality Gate: PASSED ✓
```

## Features Computed

| Feature | Description | Transform |
|---------|-------------|-----------|
| `amount_normalized` | Z-score normalization | `(x - μ) / σ` |
| `amount_log` | Log transform | `ln(x + 1)` |
| `is_high_risk_category` | Binary flag | Categorical lookup |
| `is_night_transaction` | Night hours (0-6, 22-24) | Time bucketing |
| `is_weekend` | Saturday/Sunday | Day check |
| `is_online` | Online vs in-store | Boolean |

## Quality Gates

| Metric | Threshold | Comparator |
|--------|-----------|------------|
| Accuracy | 0.90 | ≥ |
| Latency p99 | 10ms | ≤ |
| Error rate | 0.01 | ≤ |

## Comparison: Sovereign vs Databricks

| Component | Sovereign AI | Databricks |
|-----------|-------------|------------|
| Data loading | alimentar | Spark DataFrame |
| Features | trueno SIMD | Feature Store |
| Training | aprender | AutoML |
| Registry | pacha (signed) | Unity Catalog |
| Serving | realizar | Model Serving |
| Quality | pmat TDG | Custom validation |
| Orchestration | batuta | Workflows |

## Demo Requirements

- Live demo: `batuta` orchestrates full pipeline
- Benchmark: latency comparison (realizar vs Databricks)
- Security audit: Model signatures verified
- Quality report: TDG score ≥ B
