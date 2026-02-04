# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Fine-Tuning on Databricks
# MAGIC
# MAGIC **Course 4, Week 5: Fine-Tuning with LoRA/QLoRA**
# MAGIC
# MAGIC This notebook demonstrates LLM fine-tuning patterns.
# MAGIC Compare with entrenar for self-hosted fine-tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import List, Dict
from dataclasses import dataclass

print("Fine-Tuning Demo - Course 4 Week 5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Training Data Preparation

# COMMAND ----------

@dataclass
class TrainingSample:
    instruction: str
    input: str
    output: str

    def format_alpaca(self) -> str:
        if self.input:
            return f"""### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
{self.output}"""
        else:
            return f"""### Instruction:
{self.instruction}

### Response:
{self.output}"""

    def format_chatml(self) -> str:
        content = self.instruction
        if self.input:
            content += f"\n{self.input}"
        return f"""<|user|>
{content}
<|assistant|>
{self.output}"""


# Sample training data
samples = [
    TrainingSample(
        "Summarize the following text.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "ML automates model building from data."
    ),
    TrainingSample(
        "Translate to Spanish.",
        "Hello, how are you?",
        "Hola, ¿cómo estás?"
    ),
    TrainingSample(
        "Classify the sentiment as positive or negative.",
        "This product exceeded all my expectations!",
        "Positive"
    ),
    TrainingSample(
        "Answer the question.",
        "What is the capital of Japan?",
        "The capital of Japan is Tokyo."
    ),
]

print("Training Data Formats:\n")
print("ALPACA FORMAT:")
print("-" * 40)
print(samples[0].format_alpaca())
print()
print("CHATML FORMAT:")
print("-" * 40)
print(samples[0].format_chatml())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. LoRA Configuration

# COMMAND ----------

lora_config = {
    "r": 8,                           # Rank of adaptation matrices
    "lora_alpha": 16,                 # Scaling factor (alpha/r)
    "lora_dropout": 0.05,             # Dropout probability
    "target_modules": [               # Which layers to adapt
        "q_proj",                     # Query projection
        "v_proj",                     # Value projection
        "k_proj",                     # Key projection
        "o_proj",                     # Output projection
    ],
    "bias": "none",                   # Don't train bias terms
    "task_type": "CAUSAL_LM"          # Causal language modeling
}

# Calculate trainable parameters
model_params = 7_000_000_000  # 7B model
hidden_dim = 4096
num_layers = 32
lora_params_per_layer = lora_config["r"] * hidden_dim * 2 * len(lora_config["target_modules"])
total_lora_params = lora_params_per_layer * num_layers

print("LoRA Configuration:")
print(json.dumps(lora_config, indent=2))
print()
print(f"Model parameters: {model_params:,}")
print(f"LoRA parameters: {total_lora_params:,}")
print(f"Trainable: {total_lora_params / model_params * 100:.4f}%")
print(f"Scaling factor (alpha/r): {lora_config['lora_alpha'] / lora_config['r']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. QLoRA Configuration

# COMMAND ----------

qlora_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",      # Normal Float 4-bit
    "bnb_4bit_use_double_quant": True,  # Nested quantization
}

# Memory calculation
fp16_memory_gb = model_params * 2 / (1024 ** 3)  # 2 bytes per param
q4_memory_gb = model_params * 0.5 / (1024 ** 3)  # ~0.5 bytes per param with double quant
memory_savings = (1 - q4_memory_gb / fp16_memory_gb) * 100

print("QLoRA Configuration:")
print(json.dumps(qlora_config, indent=2))
print()
print(f"FP16 memory: {fp16_memory_gb:.1f} GB")
print(f"4-bit memory: {q4_memory_gb:.1f} GB")
print(f"Memory savings: {memory_savings:.0f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training Arguments

# COMMAND ----------

training_args = {
    "output_dir": "/tmp/llm-finetuned",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "max_seq_length": 512,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
}

effective_batch = training_args["per_device_train_batch_size"] * training_args["gradient_accumulation_steps"]

print("Training Arguments:")
print(json.dumps(training_args, indent=2))
print()
print(f"Effective batch size: {effective_batch}")
print(f"Total epochs: {training_args['num_train_epochs']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Training Loop Simulation

# COMMAND ----------

def simulate_training(num_samples: int, training_args: dict) -> List[Dict]:
    """Simulate training loop and return metrics."""
    import math

    effective_batch = (
        training_args["per_device_train_batch_size"] *
        training_args["gradient_accumulation_steps"]
    )
    steps_per_epoch = num_samples // effective_batch
    total_steps = steps_per_epoch * training_args["num_train_epochs"]

    metrics = []
    loss = 2.5

    for step in range(total_steps):
        # Simulate loss decrease with noise
        loss = loss * 0.995 + 0.05 * math.sin(step * 0.1)
        loss = max(0.1, loss)

        # Learning rate with warmup and cosine decay
        if step < training_args["warmup_steps"]:
            lr = training_args["learning_rate"] * (step / training_args["warmup_steps"])
        else:
            progress = (step - training_args["warmup_steps"]) / (total_steps - training_args["warmup_steps"])
            lr = training_args["learning_rate"] * 0.5 * (1 + math.cos(math.pi * progress))

        if step % 10 == 0:
            metrics.append({
                "step": step,
                "epoch": step / steps_per_epoch,
                "loss": loss,
                "learning_rate": lr
            })

    return metrics


# Simulate training
num_samples = 1000
metrics = simulate_training(num_samples, training_args)

print(f"Training simulation with {num_samples} samples:")
print(f"Total steps: {len(metrics) * 10}")
print()
print("Step  | Epoch | Loss   | LR")
print("-" * 40)
for m in metrics[:5]:
    print(f"{m['step']:5} | {m['epoch']:.2f}  | {m['loss']:.4f} | {m['learning_rate']:.2e}")
print("...")
for m in metrics[-3:]:
    print(f"{m['step']:5} | {m['epoch']:.2f}  | {m['loss']:.4f} | {m['learning_rate']:.2e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Databricks Model Training API
# MAGIC
# MAGIC Reference configuration for Databricks fine-tuning.

# COMMAND ----------

databricks_training_config = {
    "model": "databricks/dolly-v2-7b",
    "train_data_path": "dbfs:/datasets/training/train.jsonl",
    "eval_data_path": "dbfs:/datasets/training/eval.jsonl",
    "register_to": "main.models.my_finetuned_model",
    "training_duration": "3ep",  # 3 epochs
    "learning_rate": "2e-4",
    "context_length": 512,
    "task_type": "INSTRUCTION_FINETUNE"
}

print("Databricks Training API Configuration:")
print(json.dumps(databricks_training_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks | Sovereign AI (entrenar) |
# MAGIC |---------|------------|-------------------------|
# MAGIC | **Framework** | Managed | Custom |
# MAGIC | **Models** | Foundation Models | Any GGUF/HF |
# MAGIC | **LoRA** | Supported | Full control |
# MAGIC | **Quantization** | Built-in | Manual |
# MAGIC | **Distributed** | Auto | Manual |
# MAGIC | **Monitoring** | MLflow | Custom |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices

# COMMAND ----------

best_practices = """
Fine-Tuning Best Practices:

1. **Data Quality**
   - Clean, diverse training examples
   - Consistent formatting
   - Balance across categories

2. **LoRA Configuration**
   - Start with r=8, alpha=16
   - Include q/v/k projections
   - Use low dropout (0.05)

3. **Training**
   - Use gradient accumulation for larger batch
   - Cosine LR schedule with warmup
   - Monitor loss curve for overfitting

4. **QLoRA for Memory**
   - 4-bit quantization reduces memory 75%
   - Use double quantization
   - bfloat16 compute dtype

5. **Evaluation**
   - Hold out validation set
   - Track perplexity and task metrics
   - Compare with base model
"""

print(best_practices)

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. LoRA reduces trainable parameters to <1%")
print("2. QLoRA enables 7B model on consumer GPU")
print("3. Data formatting crucial for quality")
print("4. Warmup + cosine schedule is standard")
print("5. Databricks provides managed fine-tuning")
