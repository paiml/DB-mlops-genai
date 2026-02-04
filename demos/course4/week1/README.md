# Week 1: LLM Serving and Tokenization

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Understand LLM architecture and serving patterns
2. Implement tokenization for text processing
3. Use Databricks Foundation Model APIs
4. Compare managed vs self-hosted LLM serving

## Demos

### 1. LLM Serving (`llm-serving/`)

Rust implementation demonstrating LLM serving concepts.

**What it demonstrates:**
- BPE-style tokenization with special tokens
- Completion API (text generation)
- Chat API (OpenAI-compatible)
- GGUF quantization concepts

**Run locally:**
```bash
cd llm-serving
cargo run
```

### 2. Databricks Notebook (`databricks/`)

Foundation Model APIs on Databricks.

**What it demonstrates:**
- Foundation Model API usage
- Chat and completion endpoints
- Token estimation and cost calculation
- Model serving endpoint configuration

**Run on Databricks:**
1. Import `foundation_models.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### Tokenization

Converting text to numerical tokens for model input:

| Token Type | Example | Purpose |
|------------|---------|---------|
| BOS | `<s>` | Start of sequence |
| EOS | `</s>` | End of sequence |
| PAD | `<pad>` | Padding for batching |
| UNK | `<unk>` | Unknown tokens |

### Quantization

Reducing model precision for efficient inference:

| Format | Bits | Size Reduction | Quality Impact |
|--------|------|----------------|----------------|
| FP16 | 16 | Baseline | None |
| Q8_0 | 8 | 50% | Minimal |
| Q4_K_M | 4 | 25% | Small |

### API Endpoints

OpenAI-compatible endpoints:

```
POST /v1/completions    # Text completion
POST /v1/chat/completions  # Chat interface
GET  /v1/models         # List models
```

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Model Access | Foundation APIs | Self-hosted GGUF |
| Scaling | Auto-scaling | Manual |
| Latency | ~100-500ms | <50ms local |
| Cost | Per-token | Infrastructure |
| Privacy | Managed | Sovereign |

## Lab Exercises

1. **Lab 1.1**: Implement custom tokenizer
2. **Lab 1.2**: Call Foundation Model APIs
3. **Lab 1.3**: Compare token costs across models

## Resources

- [Databricks Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)
- [realizar Documentation](https://docs.rs/realizar)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
