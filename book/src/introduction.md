# MLOps & GenAI Engineering on Databricks

**Courses 3 & 4 of the Databricks Specialization on Coursera**

Platform: Databricks Free Edition | Comparison Layer: Sovereign AI Stack (Rust)

## Design Philosophy

**Dual-layer pedagogy:**

1. **Databricks layer** — Hands-on with MLflow, Feature Store, Model Serving, Vector Search, Foundation Models
2. **Sovereign AI Stack layer** — Build the same concepts from scratch in Rust to understand what platforms abstract

**Why both?**

- Practitioners need to *use* Databricks effectively
- Engineers need to *understand* what's underneath
- "Understand by building" creates deeper retention

## Course Overview

| Course | Title | Duration |
|--------|-------|----------|
| **3** | **MLOps Engineering** | ~30 hours |
| **4** | **GenAI Engineering** | ~34 hours |

## Sovereign AI Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                   batuta (Orchestration)                         │
│              Privacy Tiers · CLI · Stack Coordination            │
├───────────────────┬──────────────────┬───────────────────────────┤
│  realizar         │  entrenar        │      pacha                │
│  (Inference)      │  (Training)      │   (Model Registry)        │
│  GGUF/SafeTensors │  autograd/LoRA   │  Sign/Encrypt/Lineage     │
├───────────────────┴──────────────────┴───────────────────────────┤
│                    aprender                                       │
│         ML Algorithms: regression, trees, clustering              │
├──────────────────────────────────────────────────────────────────┤
│                     trueno                                        │
│         SIMD/GPU Compute (AVX2/AVX-512/NEON, wgpu)               │
├──────────────────────────────────────────────────────────────────┤
│  trueno-rag      │ trueno-db       │ alimentar     │ pmat        │
│  BM25 + Vector   │ GPU Analytics   │ Arrow/Parquet │ Quality     │
└──────────────────┴─────────────────┴───────────────┴─────────────┘
```

## Prerequisites

### Databricks
- Create a free account at [databricks.com](https://www.databricks.com/)
- No paid features required

### Sovereign AI Stack (Rust)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Key crates
cargo install batuta realizar pmat
```

## Getting Started

Begin with [Course 3: MLOps Engineering](./course3/overview.md) or jump directly to [Course 4: GenAI Engineering](./course4/overview.md) if you're already familiar with MLOps concepts.
