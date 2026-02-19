# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Workspace, Catalog & Compute
# MAGIC
# MAGIC **Course 1, Week 1: Lakehouse Architecture & Platform**
# MAGIC
# MAGIC This notebook demonstrates the core Databricks workspace components:
# MAGIC Workspace navigation, Unity Catalog hierarchy, and compute resources.
# MAGIC
# MAGIC **Certification Alignment:** Databricks Accredited Lakehouse Platform Fundamentals

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Workspace Overview
# MAGIC
# MAGIC The Databricks Workspace is your central hub:
# MAGIC - **Home:** Personal notebooks and files
# MAGIC - **Workspace:** Shared notebooks, libraries, repos
# MAGIC - **Catalog:** Browse data assets (Unity Catalog)
# MAGIC - **Workflows:** Scheduled jobs and pipelines
# MAGIC - **Compute:** Clusters and SQL Warehouses
# MAGIC - **Experiments:** MLflow tracking (Course 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog Hierarchy
# MAGIC
# MAGIC Unity Catalog organizes data in a three-level namespace:
# MAGIC
# MAGIC ```
# MAGIC Metastore (top-level container)
# MAGIC └── Catalog (logical grouping — e.g., "production", "dev")
# MAGIC     └── Schema (database — e.g., "sales", "analytics")
# MAGIC         ├── Table (managed or external)
# MAGIC         ├── View (virtual table from query)
# MAGIC         └── Function (UDFs)
# MAGIC ```
# MAGIC
# MAGIC **Key terms for certification:**
# MAGIC - **Metastore:** Top-level container for all data assets
# MAGIC - **Catalog:** Logical grouping (like a database server)
# MAGIC - **Schema:** Collection of tables/views (like a database)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Explore Available Catalogs

# COMMAND ----------

# List catalogs (Unity Catalog)
try:
    catalogs = spark.sql("SHOW CATALOGS")
    catalogs.show()
except Exception as e:
    print(f"Unity Catalog may not be enabled: {e}")
    print("On Databricks Free Edition, use the default catalog: 'hive_metastore'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Browse Schemas and Tables

# COMMAND ----------

# List schemas in the default catalog
try:
    schemas = spark.sql("SHOW SCHEMAS")
    schemas.show(truncate=False)
except Exception as e:
    print(f"Error listing schemas: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compute Resources
# MAGIC
# MAGIC Databricks offers several compute options:
# MAGIC
# MAGIC | Compute Type | Best For | Key Feature |
# MAGIC |-------------|----------|-------------|
# MAGIC | **All-Purpose Cluster** | Interactive development | Shared, configurable |
# MAGIC | **Job Cluster** | Scheduled workloads | Auto-created, auto-terminated |
# MAGIC | **SQL Warehouse** | BI and SQL analytics | Optimized for SQL, Photon |
# MAGIC | **Serverless** | On-demand compute | No cluster management |
# MAGIC
# MAGIC On **Databricks Free Edition**, you get a single-node cluster for development.

# COMMAND ----------

# Check current cluster configuration
try:
    print(f"Spark Version: {spark.version}")
    print(f"Cluster ID: {spark.conf.get('spark.databricks.clusterUsageTags.clusterId', 'N/A')}")
    print(f"Driver Memory: {spark.conf.get('spark.driver.memory', 'N/A')}")
    print(f"Executor Cores: {spark.conf.get('spark.executor.cores', 'N/A')}")
except Exception as e:
    print(f"Cluster info unavailable: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Governance with Unity Catalog
# MAGIC
# MAGIC Unity Catalog provides a **single governance solution** across all workloads:
# MAGIC
# MAGIC - **Access Control:** Fine-grained permissions on catalogs, schemas, tables
# MAGIC - **Data Lineage:** Track data flow from source to consumption
# MAGIC - **Auditing:** Full audit trail of who accessed what data
# MAGIC - **Data Sharing:** Delta Sharing for cross-organization sharing
# MAGIC
# MAGIC **Certification note:** Unity Catalog is the recommended governance layer
# MAGIC for all Databricks deployments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. DBFS (Databricks File System)
# MAGIC
# MAGIC DBFS provides a distributed file system mounted to the workspace:

# COMMAND ----------

# Explore DBFS
try:
    files = dbutils.fs.ls("/")
    for f in files[:10]:
        print(f"  {f.name:30s} {'DIR' if f.isDir() else f'{f.size:>10,} bytes'}")
except NameError:
    print("dbutils not available — run this notebook on Databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Component | Purpose | Certification Focus |
# MAGIC |-----------|---------|-------------------|
# MAGIC | Workspace | Central development hub | Navigation, collaboration |
# MAGIC | Unity Catalog | Data governance | Metastore/Catalog/Schema hierarchy |
# MAGIC | Compute | Processing resources | Cluster types, SQL Warehouses |
# MAGIC | DBFS | File storage | Data access patterns |
# MAGIC
# MAGIC **Next:** Week 2 — Using Notebooks and Spark
