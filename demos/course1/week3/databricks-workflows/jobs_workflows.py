# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Jobs, Dashboards & Workflows
# MAGIC
# MAGIC **Course 1, Week 3: Delta Lake & Workflows**
# MAGIC
# MAGIC This notebook demonstrates production workflow patterns on Databricks:
# MAGIC Jobs for scheduling, Dashboards for visualization, and Workflows for
# MAGIC multi-step pipelines.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Databricks Jobs
# MAGIC
# MAGIC A **Job** is a scheduled or triggered execution of a notebook, script, or JAR:
# MAGIC
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | **Schedule** | Cron-based or event-triggered |
# MAGIC | **Cluster** | Uses job clusters (auto-created, auto-terminated) |
# MAGIC | **Retries** | Configurable retry on failure |
# MAGIC | **Alerts** | Email/webhook notifications |
# MAGIC | **Parameters** | Pass runtime parameters to notebooks |
# MAGIC
# MAGIC Jobs are created via the **Workflows** UI or the **Jobs API**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simulated ETL Pipeline
# MAGIC
# MAGIC Let's build a simple ETL pipeline that could be scheduled as a Job:

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

# EXTRACT: Generate raw transaction data
raw_transactions = spark.createDataFrame(
    [
        ("2024-01-15 10:30:00", "T001", "Electronics", 299.99, "completed"),
        ("2024-01-15 11:00:00", "T002", "Books", 24.99, "completed"),
        ("2024-01-15 11:30:00", "T003", "Electronics", 149.99, "pending"),
        ("2024-01-15 12:00:00", "T004", "Clothing", 79.99, "completed"),
        ("2024-01-15 12:30:00", "T005", "Books", 34.99, "failed"),
        ("2024-01-15 13:00:00", "T006", "Electronics", 899.99, "completed"),
        ("2024-01-15 13:30:00", "T007", "Clothing", 129.99, "completed"),
        ("2024-01-15 14:00:00", "T008", "Books", 19.99, "completed"),
    ],
    ["timestamp", "transaction_id", "category", "amount", "status"],
)

print(f"Extracted {raw_transactions.count()} raw transactions")
raw_transactions.show()

# COMMAND ----------

# TRANSFORM: Clean and enrich data
transformed = (
    raw_transactions.withColumn("timestamp", F.to_timestamp("timestamp"))
    .withColumn("date", F.to_date("timestamp"))
    .withColumn("hour", F.hour("timestamp"))
    .filter(F.col("status") == "completed")  # Only completed transactions
    .withColumn("processed_at", F.current_timestamp())
)

print(f"Transformed: {transformed.count()} valid transactions (filtered from {raw_transactions.count()})")
transformed.show()

# COMMAND ----------

# LOAD: Write to Delta table
try:
    transformed.write.format("delta").mode("overwrite").saveAsTable("course1_daily_transactions")
    print("Loaded to Delta table: course1_daily_transactions")
except Exception as e:
    print(f"Note: {e}")
    transformed.write.format("delta").mode("overwrite").save("/tmp/course1_daily_transactions")
    print("Loaded to /tmp/course1_daily_transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dashboard Queries
# MAGIC
# MAGIC On Databricks, you can create **SQL Dashboards** from queries.
# MAGIC Here are example queries for a sales dashboard:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Revenue by category (Dashboard chart: bar chart)
# MAGIC SELECT
# MAGIC     category,
# MAGIC     COUNT(*) AS num_transactions,
# MAGIC     ROUND(SUM(amount), 2) AS total_revenue,
# MAGIC     ROUND(AVG(amount), 2) AS avg_transaction
# MAGIC FROM course1_daily_transactions
# MAGIC GROUP BY category
# MAGIC ORDER BY total_revenue DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Hourly transaction volume (Dashboard chart: line chart)
# MAGIC SELECT
# MAGIC     hour,
# MAGIC     COUNT(*) AS transaction_count,
# MAGIC     ROUND(SUM(amount), 2) AS hourly_revenue
# MAGIC FROM course1_daily_transactions
# MAGIC GROUP BY hour
# MAGIC ORDER BY hour

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Step Workflows
# MAGIC
# MAGIC Databricks Workflows orchestrate multiple tasks:
# MAGIC
# MAGIC ```
# MAGIC Workflow: Daily Sales Pipeline
# MAGIC ├── Task 1: Extract (notebook: extract_raw_data)
# MAGIC ├── Task 2: Transform (notebook: clean_and_enrich)
# MAGIC │   └── depends_on: Task 1
# MAGIC ├── Task 3: Load (notebook: write_to_delta)
# MAGIC │   └── depends_on: Task 2
# MAGIC └── Task 4: Quality Check (notebook: validate_output)
# MAGIC     └── depends_on: Task 3
# MAGIC ```
# MAGIC
# MAGIC **Workflow features:**
# MAGIC - **DAG execution:** Tasks run in dependency order
# MAGIC - **Conditional logic:** Run tasks based on previous results
# MAGIC - **Error handling:** Retry policies per task
# MAGIC - **Cluster reuse:** Share clusters across tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Job Parameters and Widgets
# MAGIC
# MAGIC Parameterize notebooks for reusable pipelines:

# COMMAND ----------

# Create widgets for parameterization
try:
    dbutils.widgets.text("date", "2024-01-15", "Processing Date")
    dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "append"], "Write Mode")

    processing_date = dbutils.widgets.get("date")
    write_mode = dbutils.widgets.get("mode")
    print(f"Processing date: {processing_date}, Mode: {write_mode}")
except NameError:
    processing_date = "2024-01-15"
    write_mode = "overwrite"
    print(f"Default params — date: {processing_date}, mode: {write_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monitoring and Alerts
# MAGIC
# MAGIC | Feature | Purpose | Configuration |
# MAGIC |---------|---------|---------------|
# MAGIC | **Run History** | View past job executions | Workflows UI |
# MAGIC | **Email Alerts** | Notify on success/failure | Job settings |
# MAGIC | **Webhook** | Trigger external systems | Job settings |
# MAGIC | **Ganglia Metrics** | Cluster performance | Compute UI |
# MAGIC | **Query History** | SQL execution tracking | SQL Warehouse UI |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Creating a Job (UI Steps)
# MAGIC
# MAGIC To create a job in Databricks:
# MAGIC
# MAGIC 1. Navigate to **Workflows** in the left sidebar
# MAGIC 2. Click **Create Job**
# MAGIC 3. Name your job (e.g., "Daily Sales ETL")
# MAGIC 4. Add a task:
# MAGIC    - Select task type: **Notebook**
# MAGIC    - Choose this notebook
# MAGIC    - Select cluster: **Job Cluster** (recommended) or existing cluster
# MAGIC 5. Set schedule: e.g., daily at 6:00 AM UTC
# MAGIC 6. Configure alerts: email on failure
# MAGIC 7. Click **Create**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Component | Purpose | Key Feature |
# MAGIC |-----------|---------|-------------|
# MAGIC | **Jobs** | Scheduled notebook execution | Cron schedules, retries |
# MAGIC | **Dashboards** | SQL-based visualizations | Connected to SQL Warehouse |
# MAGIC | **Workflows** | Multi-task orchestration | DAG dependencies |
# MAGIC | **Widgets** | Parameterized notebooks | Reusable pipelines |
# MAGIC
# MAGIC **Certification note:** Understand how Jobs, Workflows, and Dashboards
# MAGIC fit into the Databricks platform for production workloads.

# COMMAND ----------

# Clean up
try:
    spark.sql("DROP TABLE IF EXISTS course1_daily_transactions")
    dbutils.widgets.removeAll()
except Exception:
    pass
