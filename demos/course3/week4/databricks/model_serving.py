# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving on Databricks
# MAGIC
# MAGIC **Course 3, Week 4: Model Serving and Inference**
# MAGIC
# MAGIC This notebook demonstrates Databricks Model Serving.
# MAGIC Compare with the Rust inference server to understand what the platform automates.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Prerequisites

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import requests
import json
import time
import numpy as np

client = MlflowClient()
print("MLflow client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train and Register a Model
# MAGIC
# MAGIC First, let's create a model to serve.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# COMMAND ----------

# Register model
model_name = "serving-demo-model"

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name=model_name,
    )

    run_id = run.info.run_id
    print(f"Registered model: {model_name}")
    print(f"Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Model Serving Endpoint
# MAGIC
# MAGIC Deploy the model to a serverless endpoint.

# COMMAND ----------

# Note: Model Serving requires Databricks workspace with serving enabled
# This code shows the pattern - actual deployment requires workspace setup

endpoint_name = "serving-demo-endpoint"

# Endpoint configuration
endpoint_config = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": f"{model_name}-1",
                    "traffic_percentage": 100,
                }
            ]
        },
    },
}

print("Endpoint configuration:")
print(json.dumps(endpoint_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deploy Endpoint (if workspace supports it)

# COMMAND ----------

# Uncomment to deploy (requires Model Serving enabled in workspace)
#
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
#
# w = WorkspaceClient()
#
# # Create endpoint
# try:
#     endpoint = w.serving_endpoints.create_and_wait(
#         name=endpoint_name,
#         config=EndpointCoreConfigInput(
#             served_models=[
#                 ServedModelInput(
#                     model_name=model_name,
#                     model_version="1",
#                     workload_size="Small",
#                     scale_to_zero_enabled=True,
#                 )
#             ]
#         ),
#     )
#     print(f"Endpoint created: {endpoint.name}")
#     print(f"State: {endpoint.state}")
# except Exception as e:
#     print(f"Endpoint creation: {e}")

print("Note: Endpoint deployment requires Model Serving enabled in workspace")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Query Endpoint (Pattern)
# MAGIC
# MAGIC How to query a deployed endpoint.

# COMMAND ----------

def query_endpoint(endpoint_url, token, data):
    """Query a Databricks Model Serving endpoint."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        endpoint_url,
        headers=headers,
        json=data,
    )

    return response.json()


# Example query format
sample_request = {
    "inputs": X_test[:3].tolist()
}

print("Sample request format:")
print(json.dumps(sample_request, indent=2))

# Expected response format
sample_response = {
    "predictions": [0, 1, 0],
    "model_name": model_name,
    "model_version": "1",
}

print("\nExpected response format:")
print(json.dumps(sample_response, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch Inference with Spark
# MAGIC
# MAGIC For large-scale inference, use Spark UDFs.

# COMMAND ----------

from pyspark.sql.functions import struct, col
from pyspark.sql.types import DoubleType

# Create test DataFrame
feature_cols = [f"feature_{i}" for i in range(5)]
test_data = [(float(y_test[i]), *X_test[i].tolist()) for i in range(len(y_test))]
schema = "label double, " + ", ".join([f"{c} double" for c in feature_cols])

test_df = spark.createDataFrame(test_data, schema)
display(test_df.limit(5))

# COMMAND ----------

# Load model as Spark UDF
model_uri = f"models:/{model_name}/1"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type=DoubleType())

# Apply predictions
predictions_df = test_df.withColumn(
    "prediction",
    predict_udf(struct(*[col(c) for c in feature_cols]))
)

display(predictions_df.limit(10))

# COMMAND ----------

# Calculate accuracy
from pyspark.sql.functions import when, sum as spark_sum

accuracy_df = predictions_df.withColumn(
    "correct",
    when(col("label") == col("prediction"), 1).otherwise(0)
)

total = accuracy_df.count()
correct = accuracy_df.agg(spark_sum("correct")).collect()[0][0]
batch_accuracy = correct / total

print(f"Batch inference accuracy: {batch_accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. A/B Testing Pattern
# MAGIC
# MAGIC Route traffic between model versions.

# COMMAND ----------

ab_test_config = {
    "name": "ab-test-endpoint",
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": "1",
                "workload_size": "Small",
                "name": "model-v1",
            },
            {
                "model_name": model_name,
                "model_version": "2",  # Assuming v2 exists
                "workload_size": "Small",
                "name": "model-v2",
            },
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": "model-v1", "traffic_percentage": 90},
                {"served_model_name": "model-v2", "traffic_percentage": 10},
            ]
        },
    },
}

print("A/B Test Configuration:")
print(json.dumps(ab_test_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: Databricks vs Rust Server
# MAGIC
# MAGIC | Feature | Databricks Model Serving | Rust (realizar) |
# MAGIC |---------|-------------------------|-----------------|
# MAGIC | **Deployment** | Managed, click/API | Manual, container |
# MAGIC | **Scaling** | Auto-scale, scale-to-zero | Manual scaling |
# MAGIC | **Latency** | ~50-200ms (cold start) | ~1-10ms (warm) |
# MAGIC | **Cost** | Pay per request | Infrastructure cost |
# MAGIC | **A/B Testing** | Built-in | Manual routing |
# MAGIC | **Monitoring** | Built-in | Custom metrics |
# MAGIC | **Model Formats** | MLflow models | GGUF, ONNX, custom |
# MAGIC | **Sovereignty** | Cloud-dependent | Full control |
# MAGIC
# MAGIC **Key Insight:** Databricks provides convenience; Rust provides control and performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Latency Benchmarking
# MAGIC
# MAGIC Compare local inference vs serving endpoint.

# COMMAND ----------

# Local inference benchmark
n_iterations = 100
sample_input = X_test[:1]

start = time.time()
for _ in range(n_iterations):
    _ = model.predict(sample_input)
local_time = (time.time() - start) / n_iterations * 1000

print(f"Local inference: {local_time:.2f} ms per prediction")

# Note: Endpoint latency would be measured with actual endpoint
# Typical: 50-200ms including network overhead
print("Endpoint inference: ~50-200ms (includes network, cold start)")
print(f"\nSpeedup potential with local Rust: {local_time/1:.2f}x to {200/local_time:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Clean Up

# COMMAND ----------

# Uncomment to delete endpoint
# w.serving_endpoints.delete(endpoint_name)

print("Demo complete!")
print("\nKey takeaways:")
print("1. Model Serving provides managed deployment")
print("2. Spark UDFs enable batch inference at scale")
print("3. A/B testing enables safe rollouts")
print("4. Rust servers offer lower latency for edge cases")
