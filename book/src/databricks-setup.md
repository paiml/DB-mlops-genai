# Databricks Setup

This guide covers setting up Databricks Free Edition for the courses.

## Create Account

1. Go to [databricks.com](https://www.databricks.com/)
2. Click "Try Databricks Free"
3. Sign up with your email
4. Verify your account

## Workspace Setup

### Create Cluster

1. Navigate to **Compute** in the sidebar
2. Click **Create Cluster**
3. Select the smallest instance type
4. Enable auto-termination (15 minutes)

### Install Libraries

For Python notebooks:
```python
%pip install mlflow databricks-feature-store
```

## Features Used

### Course 3: MLOps

| Feature | Purpose |
|---------|---------|
| Experiments | MLflow tracking |
| Catalog | Model registry |
| Jobs | Pipeline orchestration |
| SQL Warehouses | Feature computation |
| Playground | Model testing |

### Course 4: GenAI

| Feature | Purpose |
|---------|---------|
| Playground | Foundation Models |
| Vector Search | Semantic retrieval |
| Genie | AI/BI demo |
| Experiments | Evaluation tracking |
| Jobs | RAG orchestration |

## Notebook Conventions

All Databricks notebooks in this repository use:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Title

# COMMAND ----------
# Code cell
```

## Running Labs

1. Import notebook into Databricks workspace
2. Attach to running cluster
3. Run cells sequentially
4. Complete TODO sections

## Troubleshooting

### Cluster won't start
- Check your free tier limits
- Ensure auto-termination is enabled
- Try a smaller instance type

### MLflow not found
```python
%pip install mlflow --quiet
dbutils.library.restartPython()
```

### Feature Store issues
```python
%pip install databricks-feature-store --quiet
dbutils.library.restartPython()
```
