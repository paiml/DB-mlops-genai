# Databricks notebook source
# MAGIC %md
# MAGIC # Foundation Models on Databricks
# MAGIC
# MAGIC **Course 4, Week 1: LLM Serving**
# MAGIC
# MAGIC This notebook demonstrates Databricks Foundation Model APIs.
# MAGIC Compare this with the Rust implementation to understand the underlying concepts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import mlflow.deployments

# Get the deployment client
client = mlflow.deployments.get_deploy_client("databricks")

# List available endpoints
endpoints = client.list_endpoints()
print(f"Available endpoints: {len(endpoints)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Query Foundation Model

# COMMAND ----------

# Query a foundation model (using DBRX or Llama)
response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print("Response:", response["choices"][0]["message"]["content"])
print(f"Tokens: {response['usage']['total_tokens']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Completion API

# COMMAND ----------

# Text completion (non-chat)
response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.1
    }
)

print("Completion:", response["choices"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embeddings

# COMMAND ----------

# Generate embeddings
response = client.predict(
    endpoint="databricks-bge-large-en",
    inputs={
        "input": ["Machine learning is fascinating", "Deep learning uses neural networks"]
    }
)

embeddings = response["data"]
print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0]['embedding'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare: Databricks vs Rust
# MAGIC
# MAGIC | Feature | Databricks | Rust Demo |
# MAGIC |---------|-----------|-----------|
# MAGIC | Models | Pre-trained, hosted | Simulated/local |
# MAGIC | Scaling | Automatic | Manual |
# MAGIC | API | REST/Python SDK | Custom |
# MAGIC | Cost | Pay-per-token | Compute cost |
# MAGIC
# MAGIC
# MAGIC The Python SDK provides convenience; the Rust client provides understanding of the internals.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Token Counting

# COMMAND ----------

from transformers import AutoTokenizer

# Load tokenizer for comparison
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "Hello, how are you doing today?"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
