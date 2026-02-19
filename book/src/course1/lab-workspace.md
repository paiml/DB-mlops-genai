# Lab: Workspace & Catalog

Navigate the Databricks workspace, explore the Unity Catalog hierarchy, browse DBFS, and inspect compute resources.

## Objectives

- Navigate the Databricks Workspace UI
- Explore Unity Catalog (Metastore > Catalog > Schema > Table)
- Use DBFS to browse files and sample datasets
- Inspect cluster configuration

## Lab Exercise

See [`labs/course1/week1/lab_workspace.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week1)

## Key Tasks

1. **Catalog exploration** — List catalogs and schemas using SQL
2. **Create schema and table** — Build a `lab_workspace.cities` table with data
3. **File system exploration** — Browse `/databricks-datasets/` with dbutils
4. **Compute inspection** — Print cluster and runtime configuration

## Validation

The lab includes a `validate_lab()` function that checks:
- Schema `lab_workspace` was created
- Cities table exists with at least 3 rows
