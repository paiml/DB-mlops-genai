# Sovereign AI Stack Examples

Rust examples demonstrating the Sovereign AI Stack components.

## Structure

```
sovereign/
  trueno/           # SIMD tensor operations
  aprender/         # ML algorithms
  realizar/         # Inference serving
  entrenar/         # LoRA/QLoRA training
  pacha/            # Model registry
  batuta/           # Orchestration
  trueno-rag/       # RAG pipelines
```

## Running Examples

```bash
cd trueno && cargo run --example simd_similarity
cd aprender && cargo run --example random_forest
cd realizar && cargo run --example serve_model
```
