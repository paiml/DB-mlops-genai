# Lab: Using Notebooks

Practice using Databricks notebooks: magic commands for multi-language cells, dbutils for file operations, loading data, and visualizations.

## Objectives

- Use magic commands to switch between Python, SQL, and Markdown
- Work with dbutils for file system operations
- Load data from CSV files
- Use display() for rich visualizations

## Lab Exercise

See [`labs/course1/week2/lab_notebooks.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week2)

## Key Tasks

1. **Magic commands** — Write Python and SQL cells in the same notebook
2. **dbutils exploration** — List sample datasets, preview file contents
3. **Load data** — Read a CSV file with schema inference
4. **Visualization** — Use display() to create charts from aggregated data

## Validation

The lab includes a `validate_lab()` function that checks:
- Python magic command executed correctly
- DataFrame was loaded with data
