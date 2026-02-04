# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning on Databricks
# MAGIC
# MAGIC **Course 4, Week 3: Fine-Tuning**
# MAGIC
# MAGIC This notebook demonstrates fine-tuning foundation models on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prepare Training Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create training data table
# MAGIC CREATE TABLE IF NOT EXISTS main.genai_course.training_data (
# MAGIC   instruction STRING,
# MAGIC   input STRING,
# MAGIC   output STRING
# MAGIC );
# MAGIC
# MAGIC -- Insert sample training examples
# MAGIC INSERT INTO main.genai_course.training_data VALUES
# MAGIC   ('Summarize the text', 'MLflow is an open-source platform for ML lifecycle.', 'MLflow manages the ML lifecycle.'),
# MAGIC   ('Explain the concept', 'What is gradient descent?', 'Gradient descent optimizes model parameters by following the gradient.'),
# MAGIC   ('Answer the question', 'What is RAG?', 'RAG combines retrieval with generation for grounded responses.');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Format Data for Fine-Tuning

# COMMAND ----------

from datasets import Dataset
import pandas as pd

# Load training data
df = spark.table("main.genai_course.training_data").toPandas()

# Format as instruction-following dataset
def format_sample(row):
    if row['input']:
        return f"### Instruction:\n{row['instruction']}\n\n### Input:\n{row['input']}\n\n### Response:\n{row['output']}"
    return f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}"

df['text'] = df.apply(format_sample, axis=1)
dataset = Dataset.from_pandas(df[['text']])

print(f"Training samples: {len(dataset)}")
print("\nSample formatted text:")
print(dataset[0]['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Fine-Tuning Run

# COMMAND ----------

from databricks.model_training import foundation_model as fm

# Configure training run
run = fm.create(
    model="meta-llama/Llama-2-7b-hf",
    train_data_path="main.genai_course.training_data",
    task_type="INSTRUCTION_FINETUNE",
    register_to="main.genai_course",
    training_duration="1ep",  # 1 epoch
    learning_rate="1e-4",
    context_length=512
)

print(f"Run name: {run.name}")
print(f"Run status: {run.get_events()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. LoRA Configuration

# COMMAND ----------

# LoRA-specific configuration
lora_config = {
    "lora_r": 8,              # Rank
    "lora_alpha": 16,         # Scaling factor
    "lora_dropout": 0.05,     # Dropout
    "target_modules": ["q_proj", "v_proj"]  # Target layers
}

# QLoRA for memory efficiency
qlora_config = {
    **lora_config,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16"
}

print("LoRA Config:", lora_config)
print("QLoRA Config:", qlora_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Monitor Training

# COMMAND ----------

import time

# Monitor training progress
while run.get_status() == "RUNNING":
    events = run.get_events()
    if events:
        latest = events[-1]
        print(f"Step: {latest.get('step', 'N/A')}, Loss: {latest.get('loss', 'N/A'):.4f}")
    time.sleep(30)

print(f"\nFinal status: {run.get_status()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deploy Fine-Tuned Model

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

w = WorkspaceClient()

# Create serving endpoint for fine-tuned model
w.serving_endpoints.create(
    name="course4-finetuned-model",
    config=EndpointCoreConfigInput(
        served_models=[{
            "model_name": "main.genai_course.my_finetuned_model",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    )
)

print("Endpoint created: course4-finetuned-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Fine-Tuned Model

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint="course4-finetuned-model",
    inputs={
        "messages": [
            {"role": "user", "content": "Explain what RAG is."}
        ],
        "max_tokens": 100
    }
)

print("Fine-tuned model response:")
print(response["choices"][0]["message"]["content"])
