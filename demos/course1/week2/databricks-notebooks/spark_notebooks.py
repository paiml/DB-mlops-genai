# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Notebooks & Spark Basics
# MAGIC
# MAGIC **Course 1, Week 2: Spark Fundamentals**
# MAGIC
# MAGIC This notebook covers using Databricks notebooks effectively:
# MAGIC magic commands, utilities, loading data, and previewing results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Notebook Basics
# MAGIC
# MAGIC Databricks notebooks support:
# MAGIC - **Python** (default), **SQL**, **Scala**, **R** in the same notebook
# MAGIC - **Markdown** cells for documentation
# MAGIC - **Magic commands** to switch languages per cell
# MAGIC - **Widgets** for parameterized notebooks
# MAGIC - **Visualizations** built into display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Magic Commands
# MAGIC
# MAGIC Switch languages within a notebook using magic commands:
# MAGIC - `%python` — Python (default)
# MAGIC - `%sql` — SQL
# MAGIC - `%scala` — Scala
# MAGIC - `%r` — R
# MAGIC - `%md` — Markdown
# MAGIC - `%sh` — Shell commands
# MAGIC - `%fs` — DBFS filesystem commands
# MAGIC - `%run` — Run another notebook

# COMMAND ----------

# Python cell (default)
message = "Hello from Python!"
print(message)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SQL cell using magic command
# MAGIC SELECT 'Hello from SQL!' AS greeting

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Databricks Utilities (dbutils)
# MAGIC
# MAGIC `dbutils` provides helper functions for common tasks:

# COMMAND ----------

# File system utilities
try:
    # List files
    dbutils.fs.ls("/databricks-datasets/")[:5]
except NameError:
    print("dbutils not available — run on Databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common dbutils Commands
# MAGIC
# MAGIC | Command | Purpose |
# MAGIC |---------|---------|
# MAGIC | `dbutils.fs.ls(path)` | List files |
# MAGIC | `dbutils.fs.head(path)` | Preview file content |
# MAGIC | `dbutils.fs.cp(src, dst)` | Copy files |
# MAGIC | `dbutils.fs.rm(path)` | Remove files |
# MAGIC | `dbutils.notebook.run(path)` | Run another notebook |
# MAGIC | `dbutils.widgets.text(name, default)` | Create input widget |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Loading Data

# COMMAND ----------

# Load a CSV from Databricks sample datasets
try:
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/databricks-datasets/samples/population-vs-price/data_geo.csv")
    )
    print(f"Loaded {df.count()} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns}")
except Exception as e:
    print(f"Dataset not available: {e}")
    print("Creating sample data instead...")

    data = [
        ("San Francisco", "CA", 864816, 1200000),
        ("New York", "NY", 8336817, 680000),
        ("Austin", "TX", 978908, 550000),
        ("Seattle", "WA", 737015, 820000),
        ("Denver", "CO", 715522, 600000),
    ]
    df = spark.createDataFrame(data, ["city", "state", "population", "median_home_price"])
    print(f"Created sample data: {df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Previewing Data

# COMMAND ----------

# show() — text output
df.show(5)

# COMMAND ----------

# display() — Databricks rich visualization
try:
    display(df)
except NameError:
    print("display() requires Databricks runtime")
    df.show()

# COMMAND ----------

# Schema inspection
df.printSchema()

# COMMAND ----------

# Summary statistics
df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Common Data Formats
# MAGIC
# MAGIC | Format | Read Command | Use Case |
# MAGIC |--------|-------------|----------|
# MAGIC | CSV | `spark.read.csv(path)` | Simple tabular data |
# MAGIC | JSON | `spark.read.json(path)` | Semi-structured data |
# MAGIC | Parquet | `spark.read.parquet(path)` | Columnar analytics |
# MAGIC | Delta | `spark.read.format("delta")` | Lakehouse tables |
# MAGIC | ORC | `spark.read.orc(path)` | Hive-compatible |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary
# MAGIC
# MAGIC - Databricks notebooks support multiple languages via magic commands
# MAGIC - `dbutils` provides filesystem, notebook, and widget utilities
# MAGIC - Data can be loaded from DBFS, cloud storage, or sample datasets
# MAGIC - `display()` provides rich visualizations; `show()` gives text output
# MAGIC
# MAGIC **Next:** Spark Core Concepts — DataFrames, transformations, and actions
