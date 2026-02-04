# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Tracking on Databricks
# MAGIC
# MAGIC **Course 3, Week 1: Experiment Tracking with MLflow**
# MAGIC
# MAGIC This notebook demonstrates MLflow tracking on Databricks Free Edition.
# MAGIC Compare this with the Rust implementation to understand what the platform abstracts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from datetime import datetime

# Databricks automatically configures MLflow tracking URI
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Version: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create an Experiment
# MAGIC
# MAGIC On Databricks, experiments are stored in the workspace and integrated with Unity Catalog.

# COMMAND ----------

# Create a unique experiment name
experiment_name = f"/Users/{spark.conf.get('spark.databricks.notebook.path').split('/')[2]}/mlops-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Set or create the experiment
mlflow.set_experiment(experiment_name)
print(f"Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log a Training Run
# MAGIC
# MAGIC This demonstrates the same workflow as our Rust client, but using the Python SDK.

# COMMAND ----------

# Simulate training data
np.random.seed(42)

with mlflow.start_run(run_name="sklearn-demo") as run:
    # Log parameters (hyperparameters)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "adam")

    # Simulate training loop with metrics
    for epoch in range(10):
        # Simulate decreasing loss and increasing accuracy
        loss = 1.0 / (epoch + 1) + 0.1 + np.random.normal(0, 0.02)
        accuracy = min(1.0 - loss + 0.05 + np.random.normal(0, 0.01), 0.99)

        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # Log final metrics
    mlflow.log_metric("final_loss", loss)
    mlflow.log_metric("final_accuracy", accuracy)

    # Log tags for organization
    mlflow.set_tag("framework", "demo")
    mlflow.set_tag("course", "mlops-week1")

    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query Runs via the Client API
# MAGIC
# MAGIC The MlflowClient provides programmatic access - same as our Rust REST client.

# COMMAND ----------

client = MlflowClient()

# Get experiment info
experiment = client.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Lifecycle Stage: {experiment.lifecycle_stage}")

# COMMAND ----------

# Search runs in the experiment
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.final_accuracy DESC"],
)

print(f"\nFound {len(runs)} run(s):")
for run in runs:
    print(f"  - {run.info.run_id[:8]}... | accuracy: {run.data.metrics.get('final_accuracy', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Autologging (Databricks Feature)
# MAGIC
# MAGIC Databricks supports automatic logging for popular ML frameworks.
# MAGIC This is a platform convenience that our Rust client doesn't have.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model - MLflow automatically logs everything!
with mlflow.start_run(run_name="autolog-demo"):
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Autologging captures:
    # - All hyperparameters
    # - Training metrics
    # - Model artifact
    # - Feature importances

    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare: SDK vs REST API
# MAGIC
# MAGIC | Feature | Python SDK | Rust REST Client |
# MAGIC |---------|-----------|------------------|
# MAGIC | Autologging | ✅ Built-in | ❌ Manual |
# MAGIC | Type Safety | ❌ Runtime | ✅ Compile-time |
# MAGIC | Framework Integration | ✅ Deep | ⚠️ Manual |
# MAGIC | Performance | Good | Better |
# MAGIC | Databricks Integration | ✅ Native | ⚠️ Manual auth |
# MAGIC
# MAGIC **Key Insight:** The Python SDK provides convenience; the Rust client provides understanding.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. View in MLflow UI
# MAGIC
# MAGIC Click **Experiments** in the left sidebar to see:
# MAGIC - All logged runs
# MAGIC - Parameter comparisons
# MAGIC - Metric visualizations
# MAGIC - Artifact browser

# COMMAND ----------

# Print direct link to the experiment
print(f"\n🔗 View experiment in MLflow UI:")
print(f"   Experiment: {experiment_name}")
