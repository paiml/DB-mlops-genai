# Week 5: Fine-Tuning and Model Security

## Overview

Fine-tune models with LoRA/QLoRA and implement secure model distribution.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 5.1 | Video | When to Fine-Tune vs RAG | Concept | 10 min |
| 5.2 | Video | Databricks Fine-Tuning | Databricks | 10 min |
| 5.3 | Lab | Fine-Tune in Databricks | Databricks | 40 min |
| 5.4 | Video | LoRA/QLoRA from Scratch | Sovereign | 10 min |
| 5.5 | Lab | Fine-Tune with entrenar | Sovereign | 45 min |
| 5.6 | Video | Model Encryption and Signing | Sovereign | 10 min |
| 5.7 | Lab | Secure Model Pipeline with pacha | Sovereign | 35 min |
| 5.8 | Video | EU AI Act and Governance | Concept | 8 min |
| 5.9 | Quiz | Fine-Tuning and Security | — | 15 min |

## Sovereign AI Stack Components

- `entrenar` for LoRA/QLoRA training
- `pacha` for ChaCha20-Poly1305 encryption

## Key Concepts

### LoRA (Low-Rank Adaptation)
- Freeze base model weights
- Add trainable low-rank matrices
- Scaling factor: `alpha / r`
- Target modules: q_proj, v_proj, k_proj

### QLoRA
- Quantized base model (4-bit)
- Double quantization for memory efficiency
- Paged optimizers for large batches

### Fine-Tuning vs RAG
| Aspect | Fine-Tuning | RAG |
|--------|-------------|-----|
| Knowledge | Baked into weights | Retrieved at runtime |
| Updates | Requires retraining | Update index only |
| Cost | Higher compute | Lower compute |
| Use case | Style/behavior change | Knowledge access |
