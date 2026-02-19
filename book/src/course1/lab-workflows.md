# Lab: Jobs & Workflows

Build a parameterized ETL pipeline, create dashboard-ready queries, and understand Databricks job scheduling.

## Objectives

- Create parameterized notebooks with widgets
- Build an Extract-Transform-Load pipeline
- Write dashboard-ready SQL queries
- Understand job scheduling and workflow orchestration

## Lab Exercise

See [`labs/course1/week3/lab_workflows.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week3)

## Key Tasks

1. **Widgets** — Create text and dropdown widgets for runtime parameters
2. **ETL pipeline** — Extract raw orders, transform (filter + enrich), load to Delta
3. **Dashboard queries** — Revenue by category, daily trends, top products
4. **Job concepts** — Answer questions about cluster types, retries, and parameter passing

## Key Concepts

- **Widgets:** `dbutils.widgets.text()`, `dbutils.widgets.dropdown()`
- **Job clusters:** Auto-created and terminated — best for scheduled workloads
- **Workflows:** Multi-task DAG with dependency ordering
- **Dashboards:** SQL queries connected to SQL Warehouses for visualization

## Validation

The lab includes a `validate_lab()` function that checks:
- Parameters are configured
- Gold Delta table was created with data
- Revenue column exists in output
- Only completed orders were loaded
