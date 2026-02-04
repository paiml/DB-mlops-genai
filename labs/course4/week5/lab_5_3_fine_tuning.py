# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5.3: Fine-Tuning Configuration
# MAGIC
# MAGIC **Course 4, Week 5: Fine-Tuning**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Configure LoRA parameters
# MAGIC - Prepare training data
# MAGIC - Understand training hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import json
from typing import List, Dict
from dataclasses import dataclass, asdict

print("Fine-Tuning Lab - Course 4 Week 5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Training Data Format
# MAGIC
# MAGIC Implement data formatting for instruction tuning.

# COMMAND ----------

@dataclass
class TrainingSample:
    instruction: str
    input: str
    output: str

    def format_alpaca(self) -> str:
        """Format as Alpaca-style prompt."""
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


# Test formatting
sample = TrainingSample(
    instruction="Summarize the following text.",
    input="Machine learning is a method of data analysis.",
    output="ML automates data analysis."
)
print(sample.format_alpaca())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: LoRA Configuration
# MAGIC
# MAGIC TODO: Implement LoRA configuration class.

# COMMAND ----------

@dataclass
class LoRAConfig:
    r: int = 8              # Rank
    alpha: int = 16         # Scaling factor
    dropout: float = 0.05   # Dropout rate
    target_modules: List[str] = None  # Modules to adapt

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

    def scaling_factor(self) -> float:
        """Calculate the LoRA scaling factor (alpha/r)."""
        # TODO: Implement
        # YOUR CODE HERE
        pass

    def estimated_params(self, hidden_dim: int, num_layers: int) -> int:
        """
        Estimate trainable parameters.

        TODO: Calculate: r * hidden_dim * 2 * num_target_modules * num_layers
        """
        # YOUR CODE HERE
        pass

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return asdict(self)


# Test LoRA config
# config = LoRAConfig(r=8, alpha=16)
# print(f"Scaling factor: {config.scaling_factor()}")
# print(f"Estimated params (7B model): {config.estimated_params(4096, 32):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Training Arguments
# MAGIC
# MAGIC TODO: Implement training configuration.

# COMMAND ----------

@dataclass
class TrainingArgs:
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512
    weight_decay: float = 0.01
    bf16: bool = True

    def effective_batch_size(self) -> int:
        """Calculate effective batch size with accumulation."""
        # TODO: Implement
        # YOUR CODE HERE
        pass

    def steps_per_epoch(self, dataset_size: int) -> int:
        """Calculate training steps per epoch."""
        # TODO: Implement
        # YOUR CODE HERE
        pass

    def total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        # TODO: Implement
        # YOUR CODE HERE
        pass

    def to_dict(self) -> Dict:
        return asdict(self)


# Test training args
# args = TrainingArgs()
# print(f"Effective batch size: {args.effective_batch_size()}")
# print(f"Total steps (1000 samples): {args.total_steps(1000)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Dataset Preparation
# MAGIC
# MAGIC TODO: Create a dataset preparation pipeline.

# COMMAND ----------

class DatasetPreparer:
    """Prepare dataset for fine-tuning."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.samples: List[TrainingSample] = []

    def add_sample(self, instruction: str, input: str, output: str) -> None:
        """Add a training sample."""
        # TODO: Create and store TrainingSample
        # YOUR CODE HERE
        pass

    def add_from_dict(self, data: Dict) -> None:
        """Add sample from dictionary."""
        # TODO: Extract fields and call add_sample
        # YOUR CODE HERE
        pass

    def format_all(self, format_type: str = "alpaca") -> List[str]:
        """Format all samples."""
        # TODO: Return list of formatted prompts
        # YOUR CODE HERE
        pass

    def split(self, train_ratio: float = 0.9) -> tuple:
        """Split into train/eval sets."""
        # TODO: Return (train_samples, eval_samples)
        # YOUR CODE HERE
        pass

    def save_jsonl(self, path: str) -> None:
        """Save samples as JSONL."""
        # TODO: Write samples to file
        # YOUR CODE HERE
        pass


# Test dataset preparation
# preparer = DatasetPreparer()
# preparer.add_sample("Translate to French", "Hello", "Bonjour")
# preparer.add_sample("Summarize", "Long text here...", "Short summary")
# formatted = preparer.format_all()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Full Configuration
# MAGIC
# MAGIC TODO: Create a complete fine-tuning configuration.

# COMMAND ----------

def create_finetune_config(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    lora_config: LoRAConfig = None,
    training_args: TrainingArgs = None
) -> Dict:
    """
    Create complete fine-tuning configuration.

    TODO: Return dict with:
    - model: model configuration
    - lora: LoRA configuration
    - training: training arguments
    - data: dataset configuration
    """
    # YOUR CODE HERE
    pass


# Test configuration
# config = create_finetune_config(
#     model_name="meta-llama/Llama-2-7b",
#     dataset_path="/data/train.jsonl",
#     output_dir="/models/finetuned",
#     lora_config=LoRAConfig(r=16, alpha=32),
#     training_args=TrainingArgs(num_epochs=5)
# )
# print(json.dumps(config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check LoRA config
        lora = LoRAConfig(r=8, alpha=16)
        scaling = lora.scaling_factor()
        checks.append(("LoRA scaling", scaling is not None and abs(scaling - 2.0) < 0.01))

        params = lora.estimated_params(4096, 32)
        checks.append(("LoRA params estimate", params is not None and params > 0))

        # Check training args
        args = TrainingArgs(batch_size=4, gradient_accumulation_steps=4)
        effective = args.effective_batch_size()
        checks.append(("Effective batch size", effective == 16))

        # Check dataset preparer
        preparer = DatasetPreparer()
        preparer.add_sample("Test", "", "Output")
        checks.append(("Dataset preparer", len(preparer.samples) == 1))

    except Exception as e:
        checks.append(("Implementation complete", False))
        print(f"Error: {e}")

    # Display results
    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Lab complete.")
    else:
        print("\n⚠️ Some checks failed. Review your code.")

validate_lab()
