# Week 2: Spark Fundamentals

## Overview

Master Apache Spark on Databricks: use notebooks with magic commands and utilities, load and preview data, then apply core DataFrame operations — select, filter, groupBy, aggregations, and joins.

## Topics

| # | Type | Title | Duration |
|---|------|-------|----------|
| 2.1.1 | Video | Using Notebooks | 10 min |
| 2.1.2 | Video | Magic Commands & Utilities | 8 min |
| 2.1.3 | Video | Loading & Previewing Data | 10 min |
| 2.2.1 | Video | Spark Core Concepts | 12 min |
| 2.2.2 | Video | Select & Filter Operations | 10 min |
| 2.2.3 | Video | GroupBy, Aggregations & Joins | 12 min |
| — | Lab | Using Notebooks | 30 min |
| — | Lab | Spark Operations | 45 min |
| — | Quiz | Spark Fundamentals | 15 min |

## Key Concepts

### Databricks Notebooks

- Support **Python**, **SQL**, **Scala**, **R** in the same notebook
- **Magic commands:** `%python`, `%sql`, `%scala`, `%r`, `%md`, `%sh`, `%fs`, `%run`
- **dbutils:** File system ops (`fs`), notebook chaining (`notebook`), widgets, secrets
- **display():** Rich visualizations built into Databricks

### Spark Core Architecture

- **SparkSession:** Entry point (`spark` variable, auto-created on Databricks)
- **DataFrame:** Distributed collection of rows with named columns
- **Lazy evaluation:** Transformations build a plan; actions trigger execution
- **Catalyst Optimizer:** Optimizes the query plan regardless of API used

### Transformations vs Actions

| Transformations (Lazy) | Actions (Eager) |
|-----------------------|-----------------|
| `select()` | `show()` |
| `filter()` / `where()` | `count()` |
| `groupBy()` | `collect()` |
| `join()` | `first()` |
| `orderBy()` | `take(n)` |
| `withColumn()` | `write.*` |

### Core Operations

- **select()** — Choose and transform columns
- **filter() / where()** — Select rows by condition
- **groupBy().agg()** — Group rows and compute aggregates (sum, avg, count, max, min)
- **join()** — Combine DataFrames (inner, left, right, full)
- **orderBy()** — Sort results

### Data Formats

| Format | Command | Use Case |
|--------|---------|----------|
| CSV | `spark.read.csv()` | Simple tabular data |
| JSON | `spark.read.json()` | Semi-structured data |
| Parquet | `spark.read.parquet()` | Columnar analytics |
| Delta | `spark.read.format("delta")` | Lakehouse tables |

## Demo Code

- [`demos/course1/week2/databricks-notebooks/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week2/databricks-notebooks) — Notebooks, magic commands, data loading
- [`demos/course1/week2/databricks-spark/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week2/databricks-spark) — Spark operations (select, filter, groupBy, join)
