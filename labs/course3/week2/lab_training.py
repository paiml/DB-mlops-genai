# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3.2: Train Models with MLflow
# MAGIC
# MAGIC **Course 3, Week 3: Model Training**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Train a classification model
# MAGIC - Log model artifacts with MLflow
# MAGIC - Compare model versions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

print("Model Training Lab - Course 3 Week 3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Create Dataset
# MAGIC
# MAGIC Generate a synthetic classification dataset.

# COMMAND ----------

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Train Logistic Regression
# MAGIC
# MAGIC EXERCISE: Train a logistic regression model and log with MLflow.

# COMMAND ----------

# EXERCISE: Set experiment
experiment_name = "/Users/YOUR_NAME/lab-3-2-training"  # Update with your name
mlflow.set_experiment(experiment_name)

# EXERCISE: Train and log logistic regression
# Hint: Use mlflow.start_run() and mlflow.sklearn.log_model()

# YOUR CODE HERE
# with mlflow.start_run(run_name="logistic_regression"):
#     model = LogisticRegression(...)
#     model.fit(X_train, y_train)
#
#     # Log parameters
#     mlflow.log_param("model_type", "logistic_regression")
#     mlflow.log_param("C", ...)
#
#     # Evaluate
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # Log metrics
#     mlflow.log_metric("accuracy", accuracy)
#
#     # Log model
#     mlflow.sklearn.log_model(model, "model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Train Random Forest
# MAGIC
# MAGIC EXERCISE: Train a random forest model with different hyperparameters.

# COMMAND ----------

# EXERCISE: Train random forest with MLflow logging
# Try different values for n_estimators and max_depth

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Compare Models
# MAGIC
# MAGIC EXERCISE: Search for runs and compare metrics.

# COMMAND ----------

# EXERCISE: Search runs and compare
# Hint: Use mlflow.search_runs()

# YOUR CODE HERE
# runs_df = mlflow.search_runs(...)
# display(runs_df[["run_id", "params.model_type", "metrics.accuracy"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Register Best Model
# MAGIC
# MAGIC EXERCISE: Register the best model to the MLflow Model Registry.

# COMMAND ----------

# EXERCISE: Find best run and register model
# Hint: Sort by accuracy and use mlflow.register_model()

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check runs exist
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        checks.append(("Runs logged", len(runs) >= 2))

        # Check metrics logged
        has_accuracy = "metrics.accuracy" in runs.columns
        checks.append(("Accuracy logged", has_accuracy))

        # Check model types
        if "params.model_type" in runs.columns:
            model_types = runs["params.model_type"].unique()
            checks.append(("Multiple models trained", len(model_types) >= 2))
    except Exception as e:
        checks.append(("Experiment setup", False))

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
