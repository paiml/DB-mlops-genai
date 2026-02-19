# Week 3: Delta Lake & Workflows

## Overview

Build reliable data pipelines with Delta Lake — ACID transactions, schema enforcement, DML operations (INSERT, UPDATE, MERGE), and time travel. Then orchestrate pipelines with Databricks Jobs, Dashboards, and Workflows.

## Topics

| # | Type | Title | Duration |
|---|------|-------|----------|
| 3.1.1 | Video | What Is Delta Lake | 10 min |
| 3.1.2 | Video | Delta Lake Concepts | 12 min |
| 3.1.3 | Video | Creating Delta Tables | 10 min |
| 3.2.1 | Video | Insert, Update & Merge | 12 min |
| 3.2.2 | Video | Time Travel | 8 min |
| 3.3.1 | Video | Jobs, Dashboards & Workflows | 12 min |
| — | Lab | Delta Tables | 45 min |
| — | Lab | Jobs & Workflows | 30 min |
| — | Quiz | Delta Lake & Workflows | 15 min |

## Key Concepts

### Delta Lake Architecture

```
Delta Table
├── _delta_log/              # Transaction log (JSON + Parquet)
│   ├── 00000000000000.json  # Version 0
│   ├── 00000000000001.json  # Version 1
│   └── 00000000000010.checkpoint.parquet
└── part-00000-*.parquet     # Data files (standard Parquet)
```

The **transaction log** records every change, enabling ACID guarantees.

### Delta Lake Features

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| ACID Transactions | Atomic, consistent writes | No corrupt/partial data |
| Schema Enforcement | Validates data on write | Data quality |
| Schema Evolution | Add columns safely | Agile development |
| Time Travel | Query historical versions | Auditing, rollback |
| MERGE (Upsert) | INSERT + UPDATE + DELETE | Efficient CDC |
| Auto-Optimize | Compacts small files | Query performance |

### DML Operations

- **INSERT:** `df.write.format("delta").mode("append")`
- **UPDATE:** `UPDATE table SET col = val WHERE condition`
- **MERGE:** Match on key — update if exists, insert if not
- **Time Travel:** `SELECT * FROM table VERSION AS OF n`

### Databricks Workflows

- **Job:** Scheduled execution of a notebook or script
- **Task:** Single unit of work within a workflow
- **Workflow:** Multi-task DAG with dependencies
- **Dashboard:** SQL-powered visualizations connected to SQL Warehouses
- **Widgets:** Parameterize notebooks for reusable pipelines

### Certification Topics

Key accreditation concepts from this week:
1. Delta Lake provides ACID transactions via the transaction log
2. MERGE combines INSERT, UPDATE, and DELETE in one atomic operation
3. Time travel enables querying any previous version of the data
4. Schema enforcement prevents bad data; schema evolution adds columns safely
5. Jobs use job clusters (auto-created, auto-terminated) for scheduled workloads
6. Workflows orchestrate multi-step pipelines with DAG dependencies

## Demo Code

- [`demos/course1/week3/databricks-delta/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week3/databricks-delta) — Delta tables, DML, time travel
- [`demos/course1/week3/databricks-workflows/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week3/databricks-workflows) — Jobs, dashboards, workflows
