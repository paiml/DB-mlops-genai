# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Using Notebooks
# MAGIC
# MAGIC **Course 1, Week 2: Spark Fundamentals**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Use magic commands to switch languages
# MAGIC - Work with dbutils for file operations
# MAGIC - Load data from multiple formats
# MAGIC - Use display() for visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Magic Commands
# MAGIC
# MAGIC EXERCISE: Use different language cells in one notebook.

# COMMAND ----------

# Python cell
python_result = 42 * 2
print(f"Python says: {python_result}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Write a SQL query that returns your name and today's date
# MAGIC -- YOUR CODE HERE (Hint: SELECT 'Your Name' AS name, current_date() AS today)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Working with dbutils
# MAGIC
# MAGIC EXERCISE: Explore the Databricks file system.

# COMMAND ----------

# EXERCISE: List available Databricks sample datasets
# Use dbutils.fs.ls("/databricks-datasets/") and print the first 10 entries
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE: Use dbutils.fs.head() to preview a file
# Hint: Try "/databricks-datasets/README.md"
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Loading Data
# MAGIC
# MAGIC EXERCISE: Load data from a CSV file.

# COMMAND ----------

# EXERCISE: Load the airlines dataset (CSV)
# Path: /databricks-datasets/airlines/
# Options: header=true, inferSchema=true
# YOUR CODE HERE

# Hint:
# df = (spark.read
#     .format("csv")
#     .option("header", "true")
#     .option("inferSchema", "true")
#     .load("/databricks-datasets/airlines/part-00000"))

# COMMAND ----------

# EXERCISE: Show the schema of the loaded DataFrame
# YOUR CODE HERE (Hint: df.printSchema())

# COMMAND ----------

# EXERCISE: Show the first 5 rows
# YOUR CODE HERE (Hint: df.show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Visualization
# MAGIC
# MAGIC EXERCISE: Use display() to create a chart.

# COMMAND ----------

# EXERCISE: Create a summary and visualize it
# Step 1: Group by a column and count
# Step 2: Use display() to show the result
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Python variable exists
    try:
        checks.append(("Python magic command", python_result == 84))
    except NameError:
        checks.append(("Python magic command", False))

    # Check 2: DataFrame loaded
    try:
        checks.append(("Data loaded", df.count() > 0))
    except NameError:
        checks.append(("Data loaded", False))

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
