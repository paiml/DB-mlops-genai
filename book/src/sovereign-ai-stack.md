# Sovereign AI Stack

The Sovereign AI Stack is a collection of Rust crates for building ML and GenAI systems from first principles.

## Architecture

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

## Component Reference

| Component | Purpose | Course Usage |
|-----------|---------|--------------|
| **trueno** | SIMD tensor operations | Feature computation, embeddings |
| **aprender** | ML algorithms | Model training |
| **realizar** | Inference serving | Model deployment |
| **entrenar** | LoRA/QLoRA training | Fine-tuning |
| **pacha** | Model registry | Signing, encryption |
| **batuta** | Orchestration | Pipeline coordination |
| **trueno-rag** | RAG pipeline | Retrieval + generation |
| **alimentar** | Data loading | Parquet, chunking |
| **pmat** | Quality gates | TDG scoring |

## Installation

```bash
# Install from crates.io
cargo install batuta realizar pmat

# Or add to Cargo.toml
[dependencies]
trueno = "0.11"
aprender = "0.24"
realizar = "0.5"
pacha = "0.2"
batuta = "0.4"
alimentar = "0.2"
pmat = "2.213"
```

## Privacy Tiers

The Sovereign AI Stack supports three privacy tiers:

| Tier | Description | Data Location |
|------|-------------|---------------|
| **Sovereign** | Air-gapped, on-premises | Never leaves local infrastructure |
| **Private** | Cloud but encrypted | Your cloud account, E2E encrypted |
| **Standard** | Managed services | Third-party APIs allowed |

Configure in `batuta.toml`:

```toml
[privacy]
tier = "sovereign"  # or "private", "standard"
allowed_endpoints = ["localhost", "*.internal.corp"]
```
