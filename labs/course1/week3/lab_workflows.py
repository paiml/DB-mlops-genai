# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Jobs & Workflows
# MAGIC
# MAGIC **Course 1, Week 3: Delta Lake & Workflows**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Build a parameterized ETL notebook
# MAGIC - Use widgets for runtime parameters
# MAGIC - Create dashboard-ready queries
# MAGIC - Understand job scheduling concepts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Parameterized Notebook
# MAGIC
# MAGIC EXERCISE: Create widgets and use them in your pipeline.

# COMMAND ----------

# EXERCISE: Create two widgets:
# 1. "start_date" (text widget, default "2024-01-01")
# 2. "category_filter" (dropdown widget, options: "All", "Electronics", "Books", "Clothing")
# YOUR CODE HERE

# Hint:
# dbutils.widgets.text("start_date", "2024-01-01", "Start Date")
# dbutils.widgets.dropdown("category_filter", "All", ["All", "Electronics", "Books", "Clothing"], "Category")

# COMMAND ----------

# EXERCISE: Read the widget values into variables
# YOUR CODE HERE
# start_date = dbutils.widgets.get("start_date")
# category_filter = dbutils.widgets.get("category_filter")

# Fallback for non-Databricks environments
try:
    start_date = dbutils.widgets.get("start_date")
    category_filter = dbutils.widgets.get("category_filter")
except NameError:
    start_date = "2024-01-01"
    category_filter = "All"

print(f"Parameters: start_date={start_date}, category_filter={category_filter}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Build an ETL Pipeline
# MAGIC
# MAGIC EXERCISE: Implement Extract, Transform, Load steps.

# COMMAND ----------

from pyspark.sql import functions as F

# EXTRACT: Raw order data
raw_orders = spark.createDataFrame(
    [
        ("2024-01-01", "O001", "Electronics", "Laptop", 999.99, 1, "completed"),
        ("2024-01-01", "O002", "Books", "Python Guide", 49.99, 2, "completed"),
        ("2024-01-02", "O003", "Electronics", "Phone", 699.99, 1, "cancelled"),
        ("2024-01-02", "O004", "Clothing", "Jacket", 129.99, 3, "completed"),
        ("2024-01-03", "O005", "Books", "ML Handbook", 69.99, 1, "completed"),
        ("2024-01-03", "O006", "Electronics", "Tablet", 449.99, 2, "completed"),
        ("2024-01-04", "O007", "Clothing", "Shoes", 89.99, 2, "completed"),
        ("2024-01-04", "O008", "Electronics", "Earbuds", 79.99, 5, "completed"),
        ("2024-01-05", "O009", "Books", "Data Science", 59.99, 3, "pending"),
        ("2024-01-05", "O010", "Clothing", "Hat", 29.99, 4, "completed"),
    ],
    ["order_date", "order_id", "category", "product", "price", "quantity", "status"],
)

print(f"Extracted {raw_orders.count()} raw orders")

# COMMAND ----------

# EXERCISE: TRANSFORM the data
# 1. Filter to only "completed" orders
# 2. Filter by start_date (order_date >= start_date)
# 3. Filter by category_filter (if not "All")
# 4. Add a "revenue" column (price * quantity)
# 5. Add a "processed_at" timestamp column
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE: LOAD the transformed data into a Delta table "lab_orders_gold"
# Use overwrite mode
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Dashboard Queries
# MAGIC
# MAGIC EXERCISE: Write queries that could power a dashboard.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Revenue by category (for a bar chart)
# MAGIC -- Columns: category, total_revenue, order_count
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Daily revenue trend (for a line chart)
# MAGIC -- Columns: order_date, daily_revenue, cumulative_revenue
# MAGIC -- YOUR CODE HERE (Hint: Use SUM() OVER(ORDER BY order_date) for cumulative)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Top products by revenue (for a table/leaderboard)
# MAGIC -- Columns: product, category, total_revenue, total_quantity
# MAGIC -- Order by total_revenue DESC, limit 5
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Job Configuration (Conceptual)
# MAGIC
# MAGIC EXERCISE: Answer these questions about scheduling this notebook as a job.

# COMMAND ----------

# MAGIC %md
# MAGIC **Q1:** What cluster type should you use for a scheduled daily job?
# MAGIC
# MAGIC **Your answer:** <!-- Replace: Job Cluster / All-Purpose / SQL Warehouse -->
# MAGIC
# MAGIC **Q2:** If this job fails, what retry configuration would you set?
# MAGIC
# MAGIC **Your answer:** <!-- Replace: describe retry strategy -->
# MAGIC
# MAGIC **Q3:** How would you pass the start_date parameter when running as a job?
# MAGIC
# MAGIC **Your answer:** <!-- Replace: describe parameter passing -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Parameters set
    checks.append(("Parameters configured", start_date is not None and category_filter is not None))

    # Check 2: Gold table exists
    try:
        df = spark.sql("SELECT * FROM lab_orders_gold")
        checks.append(("Gold table created", df.count() > 0))
    except Exception:
        checks.append(("Gold table created", False))
        df = None

    # Check 3: Revenue column exists
    if df:
        checks.append(("Revenue column", "revenue" in df.columns))

    # Check 4: Only completed orders
    if df:
        statuses = [row.status for row in df.select("status").distinct().collect()]
        checks.append(("Filtered to completed", statuses == ["completed"]))

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
    spark.sql("DROP TABLE IF EXISTS lab_orders_gold")
    dbutils.widgets.removeAll()
except Exception:
    pass
