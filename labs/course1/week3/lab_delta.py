# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Delta Tables
# MAGIC
# MAGIC **Course 1, Week 3: Delta Lake & Workflows**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Create and manage Delta tables
# MAGIC - Perform INSERT, UPDATE, and MERGE operations
# MAGIC - Use time travel to query historical data
# MAGIC - Understand schema enforcement and evolution

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Create a Delta Table
# MAGIC
# MAGIC EXERCISE: Create a Delta table for an inventory system.

# COMMAND ----------

# EXERCISE: Create an inventory DataFrame with columns:
# sku (string), product_name (string), category (string), quantity (int), unit_price (double)
# Include at least 6 products across 2+ categories
# YOUR CODE HERE

# COMMAND ----------

# EXERCISE: Write the DataFrame as a Delta table named "lab_inventory"
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Verify the table was created
# MAGIC -- YOUR CODE HERE (Hint: SELECT * FROM lab_inventory)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: INSERT New Records
# MAGIC
# MAGIC EXERCISE: Add new products to the inventory.

# COMMAND ----------

# EXERCISE: Create a DataFrame with 2 new products and INSERT (append) them
# YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify the insert
# MAGIC SELECT COUNT(*) AS total_products FROM lab_inventory

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: UPDATE Records
# MAGIC
# MAGIC EXERCISE: Update prices for a category.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Increase all prices by 5% for one category of your choice
# MAGIC -- YOUR CODE HERE (Hint: UPDATE lab_inventory SET ... WHERE ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: MERGE (Upsert)
# MAGIC
# MAGIC EXERCISE: Perform a MERGE operation to update existing and insert new products.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Create a source view with updates and new records
# MAGIC -- Include at least 1 existing SKU (update) and 1 new SKU (insert)
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Write a MERGE statement
# MAGIC -- Match on SKU
# MAGIC -- WHEN MATCHED: update quantity and unit_price
# MAGIC -- WHEN NOT MATCHED: insert all columns
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Time Travel
# MAGIC
# MAGIC EXERCISE: Query previous versions of the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: View the full history of the table
# MAGIC -- YOUR CODE HERE (Hint: DESCRIBE HISTORY lab_inventory)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Query the original version (version 0) and compare with current
# MAGIC -- YOUR CODE HERE (Hint: SELECT * FROM lab_inventory VERSION AS OF 0)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- EXERCISE: Write a query that shows price changes between version 0 and current
# MAGIC -- Join current with VERSION AS OF 0 on SKU
# MAGIC -- Show: sku, product_name, original_price, current_price, price_change
# MAGIC -- YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Schema Enforcement
# MAGIC
# MAGIC EXERCISE: Test that Delta Lake enforces the schema.

# COMMAND ----------

# EXERCISE: Try to insert a DataFrame with wrong schema (missing columns)
# This should FAIL - verify that Delta Lake catches the error
# YOUR CODE HERE
# Hint: Try creating a DataFrame with only 2-3 columns and appending to lab_inventory

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    # Check 1: Table exists
    try:
        df = spark.sql("SELECT * FROM lab_inventory")
        checks.append(("Delta table exists", True))
    except Exception:
        checks.append(("Delta table exists", False))
        df = None

    # Check 2: Table has data
    if df:
        checks.append(("Table has data", df.count() >= 6))

    # Check 3: Multiple versions exist (operations were performed)
    try:
        history = spark.sql("DESCRIBE HISTORY lab_inventory")
        checks.append(("Multiple versions (DML performed)", history.count() >= 3))
    except Exception:
        checks.append(("Multiple versions (DML performed)", False))

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
    spark.sql("DROP TABLE IF EXISTS lab_inventory")
except Exception:
    pass
