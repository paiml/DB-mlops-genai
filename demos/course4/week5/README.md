# Week 5: LLM Fine-Tuning

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Understand LoRA and QLoRA fine-tuning techniques
2. Prepare training data in standard formats
3. Configure training hyperparameters
4. Use Databricks model training APIs

## Demos

### 1. Fine-Tuning (`fine-tuning/`)

Rust implementation demonstrating fine-tuning concepts.

**What it demonstrates:**
- Training data formatting (Alpaca, ChatML)
- LoRA configuration (rank, alpha, targets)
- QLoRA memory optimization
- Training simulation with metrics

**Run locally:**
```bash
cd fine-tuning
cargo run
```

### 2. Databricks Notebook (`databricks/`)

LLM fine-tuning with Databricks APIs.

**What it demonstrates:**
- Training data preparation
- LoRA/QLoRA configuration
- Training arguments
- Databricks training API

**Run on Databricks:**
1. Import `model_training.py` into your workspace
2. Attach to a GPU cluster
3. Run all cells

## Key Concepts

### LoRA (Low-Rank Adaptation)

Instead of training all parameters, LoRA adds small trainable matrices:

```
W' = W + BA
```

Where:
- W: Original frozen weights
- B: Low-rank matrix (d × r)
- A: Low-rank matrix (r × d)
- r: Rank (typically 8-64)

### QLoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| bits | 4 | Quantization bits |
| quant_type | nf4 | Normal Float 4-bit |
| double_quant | true | Nested quantization |

### Training Data Formats

**Alpaca:**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**ChatML:**
```
<|user|>
{instruction}
{input}
<|assistant|>
{output}
```

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Framework | Managed | Custom |
| Models | Foundation Models | Any GGUF |
| LoRA | Supported | Full control |
| Quantization | Built-in | Manual |

## Lab Exercises

1. **Lab 5.1**: Prepare instruction-tuning dataset
2. **Lab 5.2**: Configure LoRA parameters
3. **Lab 5.3**: Fine-tune with Databricks API

## Resources

- [Databricks Model Training](https://docs.databricks.com/en/machine-learning/train-model/index.html)
- [entrenar Documentation](https://docs.rs/entrenar)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
