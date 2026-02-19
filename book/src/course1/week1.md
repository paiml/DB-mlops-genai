# Week 1: Lakehouse Architecture & Platform

## Overview

Understand the evolution of data architectures from data warehouses to data lakes to the data lakehouse. Explore the Databricks platform: workspace navigation, Unity Catalog hierarchy, and compute resources.

## Topics

| # | Type | Title | Duration |
|---|------|-------|----------|
| 1.1.1 | Video | Data Architecture Evolution | 8 min |
| 1.1.2 | Video | Lakehouse Architecture | 10 min |
| 1.1.3 | Video | Databricks and the Lakehouse | 8 min |
| 1.2.1 | Video | Databricks Overview | 10 min |
| 1.2.2 | Video | Workspace, Catalog & Data | 12 min |
| 1.3.1 | Video | Compute Resources | 8 min |
| — | Lab | Lakehouse Concepts | 30 min |
| — | Lab | Workspace & Catalog | 30 min |
| — | Quiz | Lakehouse Architecture | 15 min |

## Key Concepts

### Data Architecture Evolution

| Era | Architecture | Strengths | Weaknesses |
|-----|-------------|-----------|------------|
| 1980s–2000s | Data Warehouse | ACID, schema, BI | Expensive, rigid, no unstructured |
| 2010s | Data Lake | Cheap, flexible, any format | No ACID, quality issues, "data swamp" |
| 2020s+ | Data Lakehouse | Best of both | Requires modern platform |

### Lakehouse Properties

A data lakehouse provides:
- **ACID transactions** on data lake storage (via Delta Lake)
- **Schema enforcement and evolution** for data quality
- **Direct BI access** to source data (no ETL to warehouse)
- **Unified batch and streaming** in one architecture
- **Open formats** (Parquet + Delta) — no vendor lock-in
- **Governance** via Unity Catalog

### Databricks Platform Architecture

- **Control Plane:** Managed by Databricks — workspace UI, job scheduling, notebooks
- **Data Plane:** Runs in your cloud account — compute clusters, data storage, processing
- **Unity Catalog:** Three-level namespace (Metastore > Catalog > Schema > Table)
- **Compute Options:** All-purpose clusters, job clusters, SQL warehouses, serverless

### Certification Topics

Key accreditation concepts from this week:
1. A data lakehouse combines warehouse reliability with lake flexibility
2. Delta Lake provides ACID transactions on data lake storage
3. Unity Catalog provides unified governance across all data assets
4. The control plane is managed by Databricks; the data plane runs in your cloud
5. Photon accelerates SQL queries without requiring code changes
6. Open formats prevent vendor lock-in

## Demo Code

- [`demos/course1/week1/databricks-lakehouse/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week1/databricks-lakehouse) — Lakehouse architecture exploration
- [`demos/course1/week1/databricks-workspace/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course1/week1/databricks-workspace) — Workspace, Catalog & Compute
