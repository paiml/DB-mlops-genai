# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering with Databricks Feature Store
# MAGIC
# MAGIC **Course 3, Week 2: Feature Engineering**
# MAGIC
# MAGIC This notebook demonstrates Feature Store on Databricks Free Edition.
# MAGIC Compare with the Rust SIMD implementation to understand the trade-offs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, TimestampType
from databricks.feature_engineering import FeatureEngineeringClient
from datetime import datetime, timedelta
import numpy as np

# Initialize Feature Engineering client
fe = FeatureEngineeringClient()

print("Feature Engineering client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Sample Transaction Data
# MAGIC
# MAGIC Simulating transaction data for feature engineering.

# COMMAND ----------

# Generate sample data
np.random.seed(42)
n_samples = 100000

# Create transaction data
data = [
    (
        i,  # transaction_id
        i % 1000,  # customer_id
        float(np.sin(i * 0.01) * 100 + 500 + (i % 7) * 10),  # amount
        datetime.now() - timedelta(days=np.random.randint(0, 365)),  # timestamp
    )
    for i in range(n_samples)
]

schema = StructType([
    StructField("transaction_id", IntegerType(), False),
    StructField("customer_id", IntegerType(), False),
    StructField("amount", FloatType(), False),
    StructField("timestamp", TimestampType(), False),
])

transactions_df = spark.createDataFrame(data, schema)
transactions_df.cache()

print(f"Created {transactions_df.count()} transactions")
display(transactions_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compute Features with Spark
# MAGIC
# MAGIC Feature computation using Spark SQL - compare with trueno SIMD.

# COMMAND ----------

# Compute statistics for normalization
stats = transactions_df.agg(
    F.mean("amount").alias("mean_amount"),
    F.stddev("amount").alias("std_amount"),
    F.min("amount").alias("min_amount"),
    F.max("amount").alias("max_amount"),
).collect()[0]

mean_amount = stats["mean_amount"]
std_amount = stats["std_amount"]
min_amount = stats["min_amount"]
max_amount = stats["max_amount"]

print(f"Amount statistics:")
print(f"  Mean: {mean_amount:.2f}")
print(f"  Std:  {std_amount:.2f}")
print(f"  Min:  {min_amount:.2f}")
print(f"  Max:  {max_amount:.2f}")

# COMMAND ----------

# Compute features
features_df = transactions_df.select(
    "transaction_id",
    "customer_id",
    "amount",
    "timestamp",
    # Z-score normalization
    ((F.col("amount") - mean_amount) / std_amount).alias("amount_normalized"),
    # Min-max scaling
    ((F.col("amount") - min_amount) / (max_amount - min_amount)).alias("amount_scaled"),
    # Log transform
    F.log(F.col("amount") + 1e-8).alias("amount_log"),
    # Binning (using ntile)
    F.ntile(10).over(Window.orderBy("amount")).alias("amount_bin"),
)

from pyspark.sql.window import Window

# Recompute with window
window_spec = Window.partitionBy("customer_id").orderBy("timestamp").rowsBetween(-6, 0)

features_df = transactions_df.select(
    "transaction_id",
    "customer_id",
    "amount",
    "timestamp",
    ((F.col("amount") - mean_amount) / std_amount).alias("amount_normalized"),
    ((F.col("amount") - min_amount) / (max_amount - min_amount)).alias("amount_scaled"),
    F.log(F.col("amount") + 1e-8).alias("amount_log"),
    F.avg("amount").over(window_spec).alias("amount_rolling_7"),
    F.ntile(10).over(Window.orderBy("amount")).alias("amount_bin"),
)

display(features_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Feature Table in Unity Catalog
# MAGIC
# MAGIC Register the features for reuse across models.

# COMMAND ----------

# Define the feature table name
catalog = "main"  # Or your catalog name
schema_name = "default"
table_name = "transaction_features"
full_table_name = f"{catalog}.{schema_name}.{table_name}"

# Create feature table (if it doesn't exist)
try:
    fe.create_table(
        name=full_table_name,
        primary_keys=["transaction_id"],
        timestamp_keys=["timestamp"],
        df=features_df,
        description="Transaction features for fraud detection",
    )
    print(f"Created feature table: {full_table_name}")
except Exception as e:
    if "already exists" in str(e):
        print(f"Feature table {full_table_name} already exists, updating...")
        fe.write_table(
            name=full_table_name,
            df=features_df,
            mode="merge",
        )
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Read Features Back
# MAGIC
# MAGIC Demonstrate feature lookup for model training.

# COMMAND ----------

# Read the feature table
feature_table = fe.read_table(name=full_table_name)
display(feature_table.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Lookup for Training
# MAGIC
# MAGIC Use FeatureLookup to join features with training labels.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

# Simulated training labels
training_labels = spark.createDataFrame([
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 0),
    (4, 1),
], ["transaction_id", "is_fraud"])

# Create feature lookups
feature_lookups = [
    FeatureLookup(
        table_name=full_table_name,
        feature_names=["amount_normalized", "amount_scaled", "amount_log", "amount_rolling_7"],
        lookup_key=["transaction_id"],
    )
]

# Create training set
training_set = fe.create_training_set(
    df=training_labels,
    feature_lookups=feature_lookups,
    label="is_fraud",
)

training_df = training_set.load_df()
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Performance Comparison
# MAGIC
# MAGIC | Aspect | Spark/Feature Store | Rust/trueno SIMD |
# MAGIC |--------|---------------------|------------------|
# MAGIC | **Scalability** | Distributed, handles TB+ | Single-node, memory-bound |
# MAGIC | **Latency** | Higher (cluster overhead) | Lower (direct CPU) |
# MAGIC | **Feature Registry** | Built-in | Manual (pacha) |
# MAGIC | **Point-in-time** | Built-in | Manual implementation |
# MAGIC | **Online Serving** | Managed | Custom (realizar) |
# MAGIC | **Understanding** | Black box | Full control |
# MAGIC
# MAGIC **Key Insight:** Feature Store handles scale and governance; SIMD shows what happens at the CPU level.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. View in Unity Catalog
# MAGIC
# MAGIC Navigate to **Catalog > main > default > transaction_features** to see:
# MAGIC - Feature lineage
# MAGIC - Usage statistics
# MAGIC - Data preview
# MAGIC - Permissions
