# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse Architecture on Databricks
# MAGIC
# MAGIC **Course 1, Week 1: Lakehouse Architecture & Platform**
# MAGIC
# MAGIC This notebook explores the data lakehouse architecture — how it combines
# MAGIC the best of data warehouses and data lakes into a single platform.
# MAGIC
# MAGIC **Certification Alignment:** Databricks Accredited Lakehouse Platform Fundamentals

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The Data Architecture Evolution
# MAGIC
# MAGIC | Era | Architecture | Strengths | Weaknesses |
# MAGIC |-----|-------------|-----------|------------|
# MAGIC | 1980s-2000s | **Data Warehouse** | ACID, schema, BI | Expensive, rigid, no unstructured |
# MAGIC | 2010s | **Data Lake** | Cheap, flexible, any format | No ACID, quality issues, slow BI |
# MAGIC | 2020s+ | **Data Lakehouse** | Best of both | Requires modern platform |
# MAGIC
# MAGIC The lakehouse combines warehouse reliability with lake flexibility.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lakehouse Key Properties
# MAGIC
# MAGIC A data lakehouse provides:
# MAGIC - **ACID transactions** on data lake storage (via Delta Lake)
# MAGIC - **Schema enforcement and evolution** for data quality
# MAGIC - **Direct BI access** to source data (no ETL to warehouse)
# MAGIC - **Unified batch and streaming** in one architecture
# MAGIC - **Open formats** (Parquet + Delta) — no vendor lock-in
# MAGIC - **Governance** via Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Databricks Lakehouse Platform Components
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────┐
# MAGIC │              Unity Catalog (Governance)              │
# MAGIC ├──────────────┬──────────────┬────────────────────────┤
# MAGIC │  Databricks  │  Databricks  │   Databricks           │
# MAGIC │  SQL         │  ML/DS       │   Data Engineering     │
# MAGIC ├──────────────┴──────────────┴────────────────────────┤
# MAGIC │              Delta Lake (Storage Layer)              │
# MAGIC ├──────────────────────────────────────────────────────┤
# MAGIC │         Apache Spark (Compute Engine)                │
# MAGIC ├──────────────────────────────────────────────────────┤
# MAGIC │         Photon (Accelerated Query Engine)            │
# MAGIC └──────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Control Plane vs Data Plane
# MAGIC
# MAGIC Databricks separates the **control plane** (managed by Databricks) from the
# MAGIC **data plane** (runs in your cloud account):
# MAGIC
# MAGIC - **Control Plane:** Workspace UI, job scheduling, notebooks, cluster management
# MAGIC - **Data Plane:** Compute clusters, data storage, actual processing
# MAGIC
# MAGIC This separation ensures your data never leaves your cloud account.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Explore Your Lakehouse Environment

# COMMAND ----------

# Verify we're running on Databricks
try:
    print(f"Spark Version: {spark.version}")
    print(f"Databricks Runtime: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion', 'N/A')}")
except NameError:
    print("Not running on Databricks — use Databricks Free Edition to run this notebook")
    print("Sign up at: https://www.databricks.com/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Delta Lake — The Storage Foundation
# MAGIC
# MAGIC Delta Lake provides the reliability layer that makes the lakehouse possible:
# MAGIC
# MAGIC | Feature | Data Lake (Parquet) | Delta Lake |
# MAGIC |---------|-------------------|------------|
# MAGIC | ACID Transactions | No | Yes |
# MAGIC | Schema Enforcement | No | Yes |
# MAGIC | Time Travel | No | Yes |
# MAGIC | Streaming + Batch | Separate | Unified |
# MAGIC | Small File Compaction | Manual | Auto-optimize |
# MAGIC
# MAGIC We'll dive deep into Delta Lake in Week 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Photon — Accelerated Queries
# MAGIC
# MAGIC Photon is Databricks' native vectorized query engine written in C++:
# MAGIC - Up to **7x faster** than standard Spark SQL
# MAGIC - Compatible with Spark APIs — no code changes needed
# MAGIC - Automatically used in SQL Warehouses and Photon-enabled clusters

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Quiz Preparation
# MAGIC
# MAGIC **Key concepts for the Lakehouse Fundamentals Accreditation:**
# MAGIC
# MAGIC 1. A data lakehouse combines the reliability of data warehouses with the flexibility of data lakes
# MAGIC 2. Delta Lake provides ACID transactions on top of data lake storage
# MAGIC 3. Unity Catalog provides unified governance across all data assets
# MAGIC 4. The control plane is managed by Databricks; the data plane runs in your cloud
# MAGIC 5. Photon accelerates SQL queries without requiring code changes
# MAGIC 6. Open formats (Delta/Parquet) prevent vendor lock-in

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC - **Demo:** Workspace, Catalog & Data navigation
# MAGIC - **Lab:** Explore Lakehouse concepts hands-on
# MAGIC - **Video:** Databricks Overview and Compute
