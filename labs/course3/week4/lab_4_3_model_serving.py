# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.3: Model Serving
# MAGIC
# MAGIC **Course 3, Week 4: Model Serving**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Deploy a model to a serving endpoint
# MAGIC - Make predictions via REST API
# MAGIC - Monitor inference latency

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import json
import time
from typing import List, Dict

print("Model Serving Lab - Course 3 Week 4")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Prepare Model for Serving
# MAGIC
# MAGIC First, ensure you have a registered model from Lab 3.2.

# COMMAND ----------

import mlflow

# TODO: Get your registered model
model_name = "YOUR_MODEL_NAME"  # Update with your model name from Lab 3.2
model_version = 1

# Load the model
try:
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model: {model_name} version {model_version}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you completed Lab 3.2 and registered a model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Create Serving Endpoint (Simulation)
# MAGIC
# MAGIC We'll simulate a serving endpoint since creating real endpoints requires permissions.

# COMMAND ----------

class ModelEndpoint:
    """Simulated model serving endpoint."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.request_count = 0
        self.total_latency = 0

    def predict(self, instances: List[List[float]]) -> Dict:
        """Make predictions on input instances."""
        start = time.time()

        predictions = self.model.predict(instances).tolist()

        latency_ms = (time.time() - start) * 1000
        self.request_count += 1
        self.total_latency += latency_ms

        return {
            "predictions": predictions,
            "latency_ms": latency_ms
        }

    def metrics(self) -> Dict:
        """Get endpoint metrics."""
        return {
            "request_count": self.request_count,
            "avg_latency_ms": self.total_latency / max(1, self.request_count)
        }


# TODO: Create endpoint with your model
# endpoint = ModelEndpoint(model, "my-endpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Make Predictions
# MAGIC
# MAGIC TODO: Send prediction requests to the endpoint.

# COMMAND ----------

# Sample input data
import numpy as np

test_instances = np.random.randn(10, 10).tolist()  # 10 samples, 10 features

# TODO: Make predictions
# Hint: Use endpoint.predict()

# YOUR CODE HERE
# response = endpoint.predict(test_instances)
# print(f"Predictions: {response['predictions']}")
# print(f"Latency: {response['latency_ms']:.2f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Batch Predictions
# MAGIC
# MAGIC TODO: Process multiple batches and track metrics.

# COMMAND ----------

# TODO: Send 100 batch requests and track latency
# Hint: Use a loop and endpoint.metrics()

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Latency Analysis
# MAGIC
# MAGIC TODO: Analyze prediction latencies.

# COMMAND ----------

# TODO: Calculate latency statistics
# - Average latency
# - p50 latency
# - p99 latency

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check endpoint exists
        checks.append(("Endpoint created", 'endpoint' in dir()))

        # Check predictions made
        if 'endpoint' in dir():
            metrics = endpoint.metrics()
            checks.append(("Predictions made", metrics["request_count"] > 0))
            checks.append(("Multiple batches", metrics["request_count"] >= 10))
    except Exception as e:
        checks.append(("Setup complete", False))

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
