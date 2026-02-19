# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Lakehouse Concepts
# MAGIC
# MAGIC **Course 1, Week 1: Lakehouse Architecture & Platform**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Identify the key properties of a data lakehouse
# MAGIC - Compare lakehouse vs data warehouse vs data lake
# MAGIC - Explore Delta Lake as the lakehouse storage layer
# MAGIC - Understand the Databricks platform architecture

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Platform Exploration
# MAGIC
# MAGIC EXERCISE: Verify your Databricks environment.

# COMMAND ----------

# EXERCISE: Print the Spark version and verify the environment
# YOUR CODE HERE
# Hint: Use spark.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Architecture Comparison
# MAGIC
# MAGIC EXERCISE: Complete the comparison table by filling in the missing values.

# COMMAND ----------

# EXERCISE: Create a DataFrame comparing architectures
# Fill in the missing values (True/False) based on what you learned
architectures = spark.createDataFrame(
    [
        ("Data Warehouse", True, True, False, False, True),
        ("Data Lake", False, False, True, True, False),
        # EXERCISE: Add the Data Lakehouse row with correct values
        # ("Data Lakehouse", ?, ?, ?, ?, ?),
    ],
    ["architecture", "acid_transactions", "schema_enforcement", "unstructured_data", "low_cost_storage", "bi_support"],
)

architectures.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Delta Lake Basics
# MAGIC
# MAGIC EXERCISE: Create a simple Delta table and inspect its properties.

# COMMAND ----------

# EXERCISE: Create a Delta table with sample data
# Hint: Use spark.createDataFrame() and .write.format("delta")

# Step 1: Create sample data (at least 5 rows with columns: id, name, value)
# YOUR CODE HERE

# Step 2: Write as Delta table named "lab1_lakehouse"
# YOUR CODE HERE

# Step 3: Query the table to verify
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Explore the Transaction Log
# MAGIC
# MAGIC EXERCISE: Look at the Delta transaction log to understand how ACID works.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: View the history of your table
# MAGIC -- Hint: DESCRIBE HISTORY <table_name>
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Spark is running
    try:
        version = spark.version
        checks.append(("Spark environment", True))
    except Exception:
        checks.append(("Spark environment", False))

    # Check 2: Delta table exists
    try:
        df = spark.sql("SELECT * FROM lab1_lakehouse")
        checks.append(("Delta table created", df.count() >= 5))
    except Exception:
        checks.append(("Delta table created", False))

    # Check 3: Architecture comparison has 3 rows
    try:
        checks.append(("Architecture comparison", architectures.count() == 3))
    except Exception:
        checks.append(("Architecture comparison", False))

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
    spark.sql("DROP TABLE IF EXISTS lab1_lakehouse")
except Exception:
    pass
