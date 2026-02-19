# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Workspace & Catalog
# MAGIC
# MAGIC **Course 1, Week 1: Lakehouse Architecture & Platform**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Navigate the Databricks Workspace
# MAGIC - Explore the Unity Catalog hierarchy (Metastore > Catalog > Schema > Table)
# MAGIC - Use DBFS to browse files
# MAGIC - Inspect compute resources

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Catalog Exploration
# MAGIC
# MAGIC EXERCISE: List all available catalogs and schemas.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: List all catalogs
# MAGIC -- YOUR CODE HERE (Hint: SHOW CATALOGS)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: List schemas in the default catalog
# MAGIC -- YOUR CODE HERE (Hint: SHOW SCHEMAS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Create Your Own Schema
# MAGIC
# MAGIC EXERCISE: Create a schema and table in the catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Create a schema for this lab
# MAGIC -- YOUR CODE HERE (Hint: CREATE SCHEMA IF NOT EXISTS lab_workspace)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Create a table in your schema
# MAGIC -- Create a table called lab_workspace.cities with columns: name, state, population
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Insert at least 3 cities
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Query your table
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: File System Exploration
# MAGIC
# MAGIC EXERCISE: Explore DBFS and Databricks sample datasets.

# COMMAND ----------

# EXERCISE: List the contents of /databricks-datasets/
# Hint: Use dbutils.fs.ls() or %fs magic command
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE: Find and preview one sample dataset
# Hint: Use dbutils.fs.head() to preview the first few bytes
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Compute Information
# MAGIC
# MAGIC EXERCISE: Inspect your current cluster.

# COMMAND ----------

# EXERCISE: Print cluster and runtime information
# Hint: Use spark.conf.get() for configuration values
# Get: spark version, driver memory, number of executor cores
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Schema exists
    try:
        spark.sql("SHOW TABLES IN lab_workspace")
        checks.append(("Schema created", True))
    except Exception:
        checks.append(("Schema created", False))

    # Check 2: Cities table exists with data
    try:
        df = spark.sql("SELECT * FROM lab_workspace.cities")
        checks.append(("Cities table with data", df.count() >= 3))
    except Exception:
        checks.append(("Cities table with data", False))

    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! Lab complete.")
    else:
        print("\nSome checks failed. Review your code above.")

validate_lab()

# COMMAND ----------

# Clean up
try:
    spark.sql("DROP SCHEMA IF EXISTS lab_workspace CASCADE")
except Exception:
    pass
