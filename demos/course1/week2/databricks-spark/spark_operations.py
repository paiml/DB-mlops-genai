# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Core Concepts & Operations
# MAGIC
# MAGIC **Course 1, Week 2: Spark Fundamentals**
# MAGIC
# MAGIC This notebook covers Apache Spark fundamentals on Databricks:
# MAGIC DataFrames, transformations, actions, and common operations
# MAGIC (select, filter, groupBy, aggregations, joins).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Spark Core Concepts
# MAGIC
# MAGIC **Key terminology:**
# MAGIC - **SparkSession:** Entry point to Spark functionality (`spark`)
# MAGIC - **DataFrame:** Distributed collection of rows with named columns
# MAGIC - **Transformation:** Lazy operation that defines a computation (select, filter, groupBy)
# MAGIC - **Action:** Triggers computation and returns results (show, count, collect)
# MAGIC - **Catalyst Optimizer:** Spark's query optimizer that plans execution
# MAGIC
# MAGIC **Lazy evaluation:** Transformations are not executed until an action is called.
# MAGIC This allows Spark to optimize the entire query plan.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Sample DataFrames

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Employees DataFrame
employees = spark.createDataFrame(
    [
        (1, "Alice", "Engineering", 120000, "San Francisco"),
        (2, "Bob", "Engineering", 115000, "Seattle"),
        (3, "Carol", "Marketing", 95000, "New York"),
        (4, "Dave", "Marketing", 90000, "New York"),
        (5, "Eve", "Sales", 105000, "Austin"),
        (6, "Frank", "Sales", 98000, "Denver"),
        (7, "Grace", "Engineering", 130000, "San Francisco"),
        (8, "Hank", "Data Science", 125000, "Seattle"),
        (9, "Iris", "Data Science", 118000, "Austin"),
        (10, "Jack", "Engineering", 110000, "Denver"),
    ],
    ["id", "name", "department", "salary", "city"],
)

# Departments DataFrame (for joins)
departments = spark.createDataFrame(
    [
        ("Engineering", "VP Engineering", 50),
        ("Marketing", "VP Marketing", 20),
        ("Sales", "VP Sales", 30),
        ("Data Science", "Chief Data Officer", 15),
    ],
    ["dept_name", "head", "headcount"],
)

print(f"Employees: {employees.count()} rows")
print(f"Departments: {departments.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Select Operations
# MAGIC
# MAGIC `select()` chooses which columns to include in the result.

# COMMAND ----------

# Select specific columns
employees.select("name", "department", "salary").show()

# COMMAND ----------

# Select with expressions
employees.select(
    "name",
    "salary",
    (F.col("salary") * 1.1).alias("salary_with_raise"),
    F.upper(F.col("department")).alias("dept_upper"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Filter Operations
# MAGIC
# MAGIC `filter()` (or `where()`) selects rows that match a condition.

# COMMAND ----------

# Filter by condition
employees.filter(F.col("salary") > 110000).show()

# COMMAND ----------

# Multiple conditions
employees.filter(
    (F.col("department") == "Engineering") & (F.col("city") == "San Francisco")
).show()

# COMMAND ----------

# Using SQL-style where
employees.where("salary BETWEEN 100000 AND 120000").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. GroupBy & Aggregations
# MAGIC
# MAGIC `groupBy()` groups rows by column values, then apply aggregation functions.

# COMMAND ----------

# Average salary by department
employees.groupBy("department").agg(
    F.avg("salary").alias("avg_salary"),
    F.count("*").alias("employee_count"),
    F.max("salary").alias("max_salary"),
    F.min("salary").alias("min_salary"),
).orderBy("avg_salary", ascending=False).show()

# COMMAND ----------

# Group by multiple columns
employees.groupBy("department", "city").agg(
    F.count("*").alias("count"),
    F.sum("salary").alias("total_salary"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Joins
# MAGIC
# MAGIC Combine DataFrames using `join()`. Common join types:
# MAGIC - `inner` (default): Only matching rows from both sides
# MAGIC - `left`: All rows from left + matching from right
# MAGIC - `right`: All rows from right + matching from left
# MAGIC - `full`: All rows from both sides

# COMMAND ----------

# Inner join — employees with department info
joined = employees.join(
    departments,
    employees.department == departments.dept_name,
    "inner",
)
joined.select("name", "department", "salary", "head", "headcount").show()

# COMMAND ----------

# Left join with additional department not in employees
extra_dept = spark.createDataFrame(
    [("Research", "VP Research", 5)],
    ["dept_name", "head", "headcount"],
)
all_depts = departments.union(extra_dept)

left_joined = all_depts.join(
    employees,
    all_depts.dept_name == employees.department,
    "left",
)
left_joined.select("dept_name", "head", "name", "salary").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. SQL Interface
# MAGIC
# MAGIC Register a DataFrame as a temp view to query with SQL:

# COMMAND ----------

employees.createOrReplaceTempView("employees")
departments.createOrReplaceTempView("departments")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SQL query on registered views
# MAGIC SELECT
# MAGIC     e.department,
# MAGIC     d.head,
# MAGIC     COUNT(*) AS num_employees,
# MAGIC     ROUND(AVG(e.salary), 0) AS avg_salary
# MAGIC FROM employees e
# MAGIC JOIN departments d ON e.department = d.dept_name
# MAGIC GROUP BY e.department, d.head
# MAGIC ORDER BY avg_salary DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Transformations vs Actions
# MAGIC
# MAGIC | Transformations (Lazy) | Actions (Eager) |
# MAGIC |-----------------------|-----------------|
# MAGIC | `select()` | `show()` |
# MAGIC | `filter()` / `where()` | `count()` |
# MAGIC | `groupBy()` | `collect()` |
# MAGIC | `join()` | `first()` |
# MAGIC | `orderBy()` | `take(n)` |
# MAGIC | `withColumn()` | `write.*` |
# MAGIC
# MAGIC **Key insight:** Chain transformations freely — Spark optimizes the plan.
# MAGIC Only actions trigger computation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Explain — View the Query Plan

# COMMAND ----------

# Show how Spark will execute a query
(
    employees.filter(F.col("salary") > 100000)
    .groupBy("department")
    .agg(F.avg("salary").alias("avg_salary"))
    .orderBy("avg_salary", ascending=False)
    .explain(True)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Spark uses lazy evaluation — transformations build a plan, actions execute it
# MAGIC - `select()`, `filter()`, `groupBy()`, `join()` are core transformations
# MAGIC - `show()`, `count()`, `collect()` are common actions
# MAGIC - SQL and DataFrame APIs are interchangeable
# MAGIC - Catalyst optimizer ensures efficient execution regardless of API used
# MAGIC
# MAGIC **Next:** Week 3 — Delta Lake and Workflows
