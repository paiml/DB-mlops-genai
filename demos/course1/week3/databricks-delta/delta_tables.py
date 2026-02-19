# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Lake on Databricks
# MAGIC
# MAGIC **Course 1, Week 3: Delta Lake & Workflows**
# MAGIC
# MAGIC This notebook demonstrates Delta Lake — the storage layer that makes
# MAGIC the lakehouse possible. Covers creating Delta tables, DML operations
# MAGIC (INSERT, UPDATE, MERGE), and time travel.
# MAGIC
# MAGIC **Certification Alignment:** Delta Lake is a core exam topic for the
# MAGIC Databricks Accredited Lakehouse Platform Fundamentals.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What Is Delta Lake?
# MAGIC
# MAGIC Delta Lake is an open-source storage framework that brings reliability to data lakes:
# MAGIC
# MAGIC - **ACID Transactions:** Every write is atomic — no partial/corrupt data
# MAGIC - **Schema Enforcement:** Prevents bad data from entering tables
# MAGIC - **Schema Evolution:** Safely add new columns over time
# MAGIC - **Time Travel:** Query previous versions of your data
# MAGIC - **Unified Batch + Streaming:** Same table for both workloads
# MAGIC - **Audit History:** Full log of every operation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Delta Lake Architecture
# MAGIC
# MAGIC ```
# MAGIC Delta Table
# MAGIC ├── _delta_log/              # Transaction log (JSON + Parquet)
# MAGIC │   ├── 00000000000000.json  # Version 0
# MAGIC │   ├── 00000000000001.json  # Version 1
# MAGIC │   └── 00000000000010.checkpoint.parquet  # Checkpoint
# MAGIC └── part-00000-*.parquet     # Data files (standard Parquet)
# MAGIC ```
# MAGIC
# MAGIC The **transaction log** (`_delta_log/`) is what gives Delta Lake its superpowers.
# MAGIC Every change is recorded as a JSON commit file.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating Delta Tables

# COMMAND ----------

# Create a Delta table from a DataFrame
from pyspark.sql import functions as F

data = [
    (1, "Laptop", "Electronics", 999.99, 50),
    (2, "Headphones", "Electronics", 79.99, 200),
    (3, "Desk Chair", "Furniture", 349.99, 30),
    (4, "Monitor", "Electronics", 449.99, 75),
    (5, "Standing Desk", "Furniture", 599.99, 20),
]
products = spark.createDataFrame(data, ["id", "name", "category", "price", "stock"])

# Write as Delta table
try:
    products.write.format("delta").mode("overwrite").saveAsTable("course1_products")
    print("Delta table 'course1_products' created successfully")
except Exception as e:
    print(f"Table creation note: {e}")
    # Fallback: write to temp path
    products.write.format("delta").mode("overwrite").save("/tmp/course1_products")
    print("Delta table written to /tmp/course1_products")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View table properties
# MAGIC DESCRIBE EXTENDED course1_products

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. INSERT Operations

# COMMAND ----------

# Insert new rows
new_products = spark.createDataFrame(
    [
        (6, "Keyboard", "Electronics", 129.99, 150),
        (7, "Bookshelf", "Furniture", 199.99, 40),
    ],
    ["id", "name", "category", "price", "stock"],
)

try:
    new_products.write.format("delta").mode("append").saveAsTable("course1_products")
    print("Inserted 2 new products")
    spark.sql("SELECT * FROM course1_products ORDER BY id").show()
except Exception as e:
    print(f"Insert note: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. UPDATE Operations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 10% price increase for Electronics
# MAGIC UPDATE course1_products
# MAGIC SET price = price * 1.10
# MAGIC WHERE category = 'Electronics'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify the update
# MAGIC SELECT * FROM course1_products WHERE category = 'Electronics' ORDER BY id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MERGE (Upsert) Operations
# MAGIC
# MAGIC MERGE is the most powerful DML in Delta Lake — it combines INSERT, UPDATE,
# MAGIC and DELETE in a single atomic operation.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create source data for merge (some existing, some new)
# MAGIC CREATE OR REPLACE TEMP VIEW product_updates AS
# MAGIC SELECT * FROM VALUES
# MAGIC     (1, 'Laptop Pro', 'Electronics', 1099.99, 45),   -- UPDATE existing
# MAGIC     (8, 'Webcam', 'Electronics', 69.99, 100),         -- INSERT new
# MAGIC     (9, 'Filing Cabinet', 'Furniture', 159.99, 25)    -- INSERT new
# MAGIC AS updates(id, name, category, price, stock)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- MERGE: update if exists, insert if not
# MAGIC MERGE INTO course1_products AS target
# MAGIC USING product_updates AS source
# MAGIC ON target.id = source.id
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         target.name = source.name,
# MAGIC         target.price = source.price,
# MAGIC         target.stock = source.stock
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id, name, category, price, stock)
# MAGIC     VALUES (source.id, source.name, source.category, source.price, source.stock)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify merge results
# MAGIC SELECT * FROM course1_products ORDER BY id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Time Travel
# MAGIC
# MAGIC Delta Lake keeps a history of every change. You can query any previous version.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View table history
# MAGIC DESCRIBE HISTORY course1_products

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query a previous version (version 0 = original data)
# MAGIC SELECT * FROM course1_products VERSION AS OF 0 ORDER BY id

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare current vs previous version
# MAGIC SELECT
# MAGIC     curr.id,
# MAGIC     curr.name AS current_name,
# MAGIC     curr.price AS current_price,
# MAGIC     prev.price AS original_price,
# MAGIC     ROUND(curr.price - prev.price, 2) AS price_change
# MAGIC FROM course1_products curr
# MAGIC JOIN course1_products VERSION AS OF 0 prev ON curr.id = prev.id
# MAGIC WHERE curr.price != prev.price
# MAGIC ORDER BY curr.id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Schema Enforcement & Evolution

# COMMAND ----------

# Schema enforcement — this will FAIL (wrong schema)
try:
    bad_data = spark.createDataFrame(
        [(100, "Widget", 9.99)],
        ["id", "name", "price"],  # Missing category, stock columns
    )
    bad_data.write.format("delta").mode("append").saveAsTable("course1_products")
except Exception as e:
    print(f"Schema enforcement caught: {type(e).__name__}")
    print("Delta Lake prevents writing data with mismatched schema!")

# COMMAND ----------

# Schema evolution — add a new column with mergeSchema
try:
    extra = spark.createDataFrame(
        [(10, "Lamp", "Furniture", 49.99, 60, "2024-01-15")],
        ["id", "name", "category", "price", "stock", "added_date"],
    )
    extra.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
        "course1_products"
    )
    print("Schema evolved — added 'added_date' column")
except Exception as e:
    print(f"Schema evolution note: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Key Concepts for Certification
# MAGIC
# MAGIC | Concept | What It Does | Why It Matters |
# MAGIC |---------|-------------|----------------|
# MAGIC | ACID Transactions | Atomic, consistent writes | No corrupt data |
# MAGIC | Schema Enforcement | Validates data on write | Data quality |
# MAGIC | Schema Evolution | Add columns safely | Agile development |
# MAGIC | Time Travel | Query historical versions | Auditing, rollback |
# MAGIC | MERGE | Upsert in one operation | Efficient CDC |
# MAGIC | Transaction Log | Records every change | Audit trail |
# MAGIC
# MAGIC **Next:** Jobs, Dashboards & Workflows

# COMMAND ----------

# Clean up
try:
    spark.sql("DROP TABLE IF EXISTS course1_products")
    print("Cleanup complete")
except Exception:
    pass
