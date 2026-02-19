# Lab: Lakehouse Concepts

Explore the data lakehouse architecture hands-on: compare architectures, inspect platform components, and create your first Delta table.

## Objectives

- Identify key properties of a data lakehouse
- Compare lakehouse vs data warehouse vs data lake
- Create a Delta table and inspect the transaction log
- Verify the Databricks environment

## Lab Exercise

See [`labs/course1/week1/lab_lakehouse.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week1)

## Key Tasks

1. **Verify environment** — Print Spark version and runtime info
2. **Architecture comparison** — Build a DataFrame comparing warehouse/lake/lakehouse features
3. **Create Delta table** — Write sample data as a Delta table
4. **Inspect history** — Use `DESCRIBE HISTORY` to view the transaction log

## Validation

The lab includes a `validate_lab()` function that checks:
- Spark environment is running
- Delta table was created with at least 5 rows
- Architecture comparison DataFrame has all 3 architectures
