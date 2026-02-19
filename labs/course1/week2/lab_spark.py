# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Spark Operations
# MAGIC
# MAGIC **Course 1, Week 2: Spark Fundamentals**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Use select() to choose and transform columns
# MAGIC - Use filter() to select rows by condition
# MAGIC - Use groupBy() with aggregation functions
# MAGIC - Perform joins between DataFrames
# MAGIC - Understand lazy evaluation vs actions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Lab Data

# COMMAND ----------

from pyspark.sql import functions as F

# Sales data
sales = spark.createDataFrame(
    [
        ("2024-01-01", "S001", "Electronics", "Laptop", 999.99, 2, "West"),
        ("2024-01-01", "S002", "Books", "Python Guide", 49.99, 5, "East"),
        ("2024-01-02", "S003", "Electronics", "Phone", 699.99, 3, "West"),
        ("2024-01-02", "S004", "Clothing", "Jacket", 129.99, 4, "North"),
        ("2024-01-03", "S005", "Books", "Data Science", 59.99, 8, "East"),
        ("2024-01-03", "S006", "Electronics", "Tablet", 449.99, 2, "South"),
        ("2024-01-04", "S007", "Clothing", "Shoes", 89.99, 6, "West"),
        ("2024-01-04", "S008", "Electronics", "Earbuds", 79.99, 10, "North"),
        ("2024-01-05", "S009", "Books", "ML Handbook", 69.99, 3, "South"),
        ("2024-01-05", "S010", "Clothing", "Hat", 29.99, 15, "East"),
    ],
    ["date", "sale_id", "category", "product", "price", "quantity", "region"],
)

# Region lookup
regions = spark.createDataFrame(
    [
        ("West", "Pacific", "Sarah"),
        ("East", "Atlantic", "Mike"),
        ("North", "Central", "Lisa"),
        ("South", "Gulf", "Tom"),
    ],
    ["region", "territory", "manager"],
)

print(f"Sales: {sales.count()} rows")
print(f"Regions: {regions.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Select Operations
# MAGIC
# MAGIC EXERCISE: Use select() to create derived columns.

# COMMAND ----------

# EXERCISE 1a: Select sale_id, product, price, quantity
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 1b: Create a total_revenue column (price * quantity)
# and a discounted_price column (price * 0.9)
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Filter Operations
# MAGIC
# MAGIC EXERCISE: Use filter() to find specific rows.

# COMMAND ----------

# EXERCISE 2a: Filter sales where price > 100
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 2b: Filter Electronics sales in the West region
# Hint: Use & for AND conditions
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 2c: Filter sales between Jan 2 and Jan 4 (inclusive)
# Hint: Use .where() with BETWEEN
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: GroupBy & Aggregations
# MAGIC
# MAGIC EXERCISE: Compute summary statistics.

# COMMAND ----------

# EXERCISE 3a: Total revenue by category
# Columns: category, total_revenue (sum of price*quantity), num_sales (count)
# Order by total_revenue descending
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 3b: Average price and total quantity by region
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 3c: Find the most expensive product in each category
# Hint: groupBy("category").agg(F.max("price"))
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Joins
# MAGIC
# MAGIC EXERCISE: Combine sales with region information.

# COMMAND ----------

# EXERCISE 4a: Inner join sales with regions on the region column
# Show: sale_id, product, price, region, territory, manager
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE 4b: Find total revenue by territory and manager
# (Join first, then groupBy territory and manager)
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: SQL Equivalent
# MAGIC
# MAGIC EXERCISE: Register as views and write SQL queries.

# COMMAND ----------

sales.createOrReplaceTempView("sales")
regions.createOrReplaceTempView("regions")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Write a SQL query that finds the top 3 products by total revenue
# MAGIC -- (revenue = price * quantity)
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Sales data loaded
    checks.append(("Sales data loaded", sales.count() == 10))

    # Check 2: Can perform select
    try:
        result = sales.select("sale_id", "product", "price", "quantity")
        checks.append(("Select operation", len(result.columns) == 4))
    except Exception:
        checks.append(("Select operation", False))

    # Check 3: Can perform filter
    try:
        result = sales.filter(F.col("price") > 100)
        checks.append(("Filter operation", result.count() > 0))
    except Exception:
        checks.append(("Filter operation", False))

    # Check 4: Can perform groupBy
    try:
        result = sales.groupBy("category").agg(F.count("*").alias("n"))
        checks.append(("GroupBy operation", result.count() == 3))
    except Exception:
        checks.append(("GroupBy operation", False))

    # Check 5: Can perform join
    try:
        result = sales.join(regions, "region", "inner")
        checks.append(("Join operation", result.count() == 10))
    except Exception:
        checks.append(("Join operation", False))

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
