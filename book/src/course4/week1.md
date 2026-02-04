# Week 1: Foundation Models and LLM Serving

## Overview

Understand LLM serving by building a tokenizer and inference server in Rust.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 1.1 | Video | The GenAI Landscape | Concept | 10 min |
| 1.2 | Video | Databricks Foundation Model APIs | Databricks | 10 min |
| 1.3 | Lab | Query Models in Playground | Databricks | 25 min |
| 1.4 | Video | GGUF Format and Quantization | Sovereign | 10 min |
| 1.5 | Lab | Serve Local Model with realizar | Sovereign | 35 min |
| 1.6 | Video | Tokenization Deep Dive | Concept | 10 min |
| 1.7 | Lab | Build BPE Tokenizer | Sovereign | 30 min |
| 1.8 | Video | External Models and AI Gateway | Databricks | 8 min |
| 1.9 | Quiz | LLM Serving Fundamentals | — | 15 min |

## Sovereign AI Stack Components

- `realizar` for GGUF inference
- `tokenizers` crate for BPE

## Key Concepts

### Tokenization
- BPE (Byte-Pair Encoding) algorithm
- Vocabulary and merge rules
- Special tokens: `<|endoftext|>`, `<|pad|>`

### Model Quantization
- FP16, INT8, INT4 representations
- GGUF format: Q4_K_M, Q5_K_M, Q8_0
- Memory vs accuracy trade-offs
