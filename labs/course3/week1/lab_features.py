# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2.5: Build a Feature Pipeline
# MAGIC
# MAGIC **Course 3, Week 2: Feature Engineering**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Implement feature transformations
# MAGIC - Build a reusable feature pipeline
# MAGIC - Compute statistical features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType
import math

print("Feature Pipeline Lab - Course 3 Week 2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Create Sample Data
# MAGIC
# MAGIC We'll work with transaction data.

# COMMAND ----------

# Sample transaction data
data = [
    (1, 100.50, "grocery", 10),
    (2, 250.00, "electronics", 14),
    (3, 15.99, "grocery", 8),
    (4, 899.99, "electronics", 20),
    (5, 45.00, "clothing", 12),
    (6, 1200.00, "electronics", 22),
    (7, 35.50, "grocery", 9),
    (8, 75.00, "clothing", 15),
]

schema = StructType([
    StructField("transaction_id", IntegerType(), False),
    StructField("amount", FloatType(), False),
    StructField("category", StringType(), False),
    StructField("hour", IntegerType(), False),
])

df = spark.createDataFrame(data, schema)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Implement Z-Score Normalization
# MAGIC
# MAGIC EXERCISE: Compute z-score normalized amount.
# MAGIC
# MAGIC Formula: `z = (x - mean) / stddev`

# COMMAND ----------

# EXERCISE: Compute mean and standard deviation of amount
# Hint: Use F.mean() and F.stddev()

# YOUR CODE HERE
# stats = df.agg(...)
# mean_amount = ...
# std_amount = ...

# EXERCISE: Add z-score column
# df_with_zscore = df.withColumn("amount_zscore", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Implement Log Transform
# MAGIC
# MAGIC EXERCISE: Apply log transformation to amount.
# MAGIC
# MAGIC Formula: `log_amount = ln(amount + 1)`

# COMMAND ----------

# EXERCISE: Add log-transformed amount column
# Hint: Use F.log()

# YOUR CODE HERE
# df_with_log = df_with_zscore.withColumn("amount_log", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Time-Based Features
# MAGIC
# MAGIC EXERCISE: Create time bucket features.

# COMMAND ----------

# EXERCISE: Create time bucket feature
# - 0-6: night
# - 6-12: morning
# - 12-18: afternoon
# - 18-24: evening

# Hint: Use F.when() and F.col()

# YOUR CODE HERE
# df_with_time = df_with_log.withColumn("time_bucket", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Category Encoding
# MAGIC
# MAGIC EXERCISE: One-hot encode the category column.

# COMMAND ----------

# EXERCISE: Create binary columns for each category
# Hint: Use F.when() for each category

# YOUR CODE HERE
# df_encoded = df_with_time.withColumn("is_grocery", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = _run_validation_checks()
    _display_results(checks)


def _run_validation_checks():
    """Run all validation checks."""
    checks = []
    required_cols = ["amount_zscore", "amount_log", "time_bucket"]

    try:
        final_df = df_encoded if 'df_encoded' in dir() else df
        cols = final_df.columns
        checks.extend([(f"Column '{col}' exists", col in cols) for col in required_cols])
        checks.extend(_check_zscore_normalization(final_df, cols))
    except Exception:
        checks.append(("Pipeline complete", False))

    return checks


def _check_zscore_normalization(df, cols):
    """Check if z-score is properly normalized."""
    if "amount_zscore" not in cols:
        return []
    stats = df.agg(F.mean("amount_zscore").alias("mean")).collect()[0]
    return [("Z-score mean ≈ 0", abs(stats["mean"]) < 0.1)]


def _display_results(checks):
    """Display validation results."""
    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = all(passed for _, passed in checks)
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
    msg = "🎉 All checks passed!" if all_passed else "⚠️ Some checks failed."
    print(f"\n{msg}")

validate_lab()
