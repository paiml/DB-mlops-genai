# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training and Registry on Databricks
# MAGIC
# MAGIC **Course 3, Week 3: Model Training and Registry**
# MAGIC
# MAGIC This notebook demonstrates AutoML and Unity Catalog Model Registry.
# MAGIC Compare with the Rust implementation to understand what the platform automates.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Initialize clients
client = MlflowClient()
print("MLflow client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Training Data
# MAGIC
# MAGIC Creating a synthetic fraud detection dataset.

# COMMAND ----------

# Generate synthetic data
np.random.seed(42)
n_samples = 10000
n_features = 10

# Features
X = np.random.randn(n_samples, n_features)

# Target (binary classification with some pattern)
weights = np.array([0.5, -0.3, 0.8, -0.2, 0.4, -0.5, 0.3, -0.1, 0.6, -0.4])
y = (X @ weights + np.random.randn(n_samples) * 0.5 > 0).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Manual Model Training with MLflow Tracking
# MAGIC
# MAGIC First, let's train models manually with explicit tracking.

# COMMAND ----------

# Set experiment
experiment_name = "/Users/demo/mlops-week3-models"
mlflow.set_experiment(experiment_name)

# Train Logistic Regression
with mlflow.start_run(run_name="logistic-regression") as run:
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 100)

    # Train
    lr_model = LogisticRegression(C=1.0, max_iter=100, random_state=42)
    lr_model.fit(X_train, y_train)

    # Evaluate
    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)

    # Log model
    mlflow.sklearn.log_model(lr_model, "model")

    print(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    lr_run_id = run.info.run_id

# COMMAND ----------

# Train Random Forest
with mlflow.start_run(run_name="random-forest") as run:
    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)

    # Log model
    mlflow.sklearn.log_model(rf_model, "model")

    # Log feature importances
    importance_dict = {f"feature_{i}": imp for i, imp in enumerate(rf_model.feature_importances_)}
    mlflow.log_dict(importance_dict, "feature_importances.json")

    print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    rf_run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Databricks AutoML (if available)
# MAGIC
# MAGIC AutoML automatically tries multiple algorithms and hyperparameters.

# COMMAND ----------

# Convert to Spark DataFrame for AutoML
feature_cols = [f"feature_{i}" for i in range(n_features)]
data = [(int(y[i]), *X[i].tolist()) for i in range(len(y))]
schema = StructType([StructField("label", IntegerType())] +
                    [StructField(f"feature_{i}", FloatType()) for i in range(n_features)])

df = spark.createDataFrame(data, schema)
display(df.limit(5))

# COMMAND ----------

# AutoML (uncomment if AutoML is available in your workspace)
# from databricks import automl
#
# summary = automl.classify(
#     dataset=df,
#     target_col="label",
#     primary_metric="f1",
#     timeout_minutes=10,
# )
#
# print(f"Best model: {summary.best_trial.model_path}")
# print(f"Best F1: {summary.best_trial.metrics['f1']}")

print("Note: AutoML requires Databricks ML Runtime. Showing manual training instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Model in Unity Catalog
# MAGIC
# MAGIC Unity Catalog provides centralized model governance.

# COMMAND ----------

# Define model name in Unity Catalog
catalog = "main"
schema_name = "default"
model_name = "fraud_detector"
full_model_name = f"{catalog}.{schema_name}.{model_name}"

# Register the best model (Random Forest in this case)
model_uri = f"runs:/{rf_run_id}/model"

try:
    # Register model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=full_model_name,
    )
    print(f"Registered model: {full_model_name}")
    print(f"Version: {registered_model.version}")
except Exception as e:
    print(f"Note: Unity Catalog registration requires proper setup: {e}")
    print("Using MLflow Model Registry instead...")

    # Fallback to MLflow registry
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    print(f"Registered in MLflow: {model_name} v{registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Versioning and Aliases
# MAGIC
# MAGIC Manage model lifecycle with versions and aliases.

# COMMAND ----------

# Set alias for production
try:
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=registered_model.version,
    )
    print(f"Set 'production' alias to version {registered_model.version}")
except Exception as e:
    print(f"Alias setting: {e}")

# List model versions
try:
    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"\nModel versions for {model_name}:")
    for v in versions:
        print(f"  v{v.version}: {v.status} (run_id: {v.run_id[:8]}...)")
except Exception as e:
    print(f"Version listing: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load Model for Inference
# MAGIC
# MAGIC Load registered model for predictions.

# COMMAND ----------

# Load model by alias
try:
    model_uri = f"models:/{model_name}@production"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from: {model_uri}")

    # Make predictions
    sample_predictions = loaded_model.predict(X_test[:5])
    print(f"Sample predictions: {sample_predictions}")
except Exception as e:
    print(f"Loading by alias: {e}")
    # Fallback to version
    model_uri = f"models:/{model_name}/{registered_model.version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks | Sovereign AI (pacha) |
# MAGIC |---------|------------|---------------------|
# MAGIC | **Model Storage** | Unity Catalog / MLflow | Content-addressed (BLAKE3) |
# MAGIC | **Versioning** | Automatic | Semantic versioning |
# MAGIC | **Signing** | Implicit (workspace auth) | Ed25519 signatures |
# MAGIC | **Encryption** | At-rest (platform) | ChaCha20-Poly1305 (explicit) |
# MAGIC | **Lineage** | Unity Catalog | Explicit metadata |
# MAGIC | **Access Control** | ACLs | Cryptographic (keys) |
# MAGIC | **AutoML** | Built-in | Manual (aprender) |
# MAGIC
# MAGIC **Key Insight:** Databricks provides convenience and governance; pacha provides cryptographic proof and sovereignty.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Card (Best Practice)
# MAGIC
# MAGIC Document model metadata for governance.

# COMMAND ----------

model_card = {
    "name": model_name,
    "version": registered_model.version,
    "description": "Binary classifier for fraud detection",
    "model_type": "RandomForestClassifier",
    "metrics": {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
    },
    "parameters": {
        "n_estimators": 100,
        "max_depth": 10,
    },
    "training_data": {
        "n_samples": len(X_train),
        "n_features": n_features,
        "class_balance": dict(zip(*np.unique(y_train, return_counts=True))),
    },
    "author": "mlops-course",
    "created_at": str(datetime.now()),
}

import json
from datetime import datetime

print("Model Card:")
print(json.dumps(model_card, indent=2, default=str))

# Log model card as artifact
with mlflow.start_run(run_id=rf_run_id):
    mlflow.log_dict(model_card, "model_card.json")
