# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1.4: Build MLflow Rust Client (Part 1)
# MAGIC
# MAGIC **Course 3, Week 1: Experiment Tracking**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Understand MLflow REST API structure
# MAGIC - Implement experiment creation and retrieval
# MAGIC - Log parameters and metrics programmatically

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Explore MLflow REST API
# MAGIC
# MAGIC First, let's understand the MLflow REST API endpoints.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List existing experiments
experiments = client.search_experiments()
print(f"Found {len(experiments)} experiments")

for exp in experiments[:3]:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Create an Experiment
# MAGIC
# MAGIC TODO: Create a new experiment using the MLflow client.

# COMMAND ----------

# TODO: Create a new experiment
# Hint: Use client.create_experiment() or mlflow.set_experiment()

experiment_name = "/Users/YOUR_NAME/lab-1-4-experiment"  # TODO: Update with your name

# YOUR CODE HERE
# experiment_id = ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Log Parameters and Metrics
# MAGIC
# MAGIC TODO: Start a run and log parameters and metrics.

# COMMAND ----------

# TODO: Start a run and log data
# Hint: Use mlflow.start_run() context manager

# YOUR CODE HERE
# with mlflow.start_run():
#     mlflow.log_param("learning_rate", ...)
#     mlflow.log_metric("accuracy", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Query Runs
# MAGIC
# MAGIC TODO: Search for runs and display their metrics.

# COMMAND ----------

# TODO: Search for runs in your experiment
# Hint: Use client.search_runs()

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation
# MAGIC
# MAGIC Run this cell to validate your work.

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Experiment exists
    try:
        exp = client.get_experiment_by_name(experiment_name)
        checks.append(("Experiment created", exp is not None))
    except:
        checks.append(("Experiment created", False))

    # Check 2: Runs exist
    try:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        checks.append(("Runs logged", len(runs) > 0))
    except:
        checks.append(("Runs logged", False))

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
