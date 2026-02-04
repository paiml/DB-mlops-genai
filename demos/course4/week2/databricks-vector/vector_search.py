# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search on Databricks
# MAGIC
# MAGIC **Course 4, Week 2: Vector Search**
# MAGIC
# MAGIC This notebook demonstrates Databricks Vector Search for semantic similarity.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Vector Search

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Initialize client
vsc = VectorSearchClient()

# List existing endpoints
endpoints = vsc.list_endpoints()
print(f"Existing endpoints: {endpoints}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Vector Search Endpoint

# COMMAND ----------

# Create endpoint (if not exists)
endpoint_name = "course4-demo-endpoint"

try:
    vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    print(f"Created endpoint: {endpoint_name}")
except Exception as e:
    print(f"Endpoint may already exist: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Delta Table with Embeddings

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create source table
# MAGIC CREATE TABLE IF NOT EXISTS main.genai_course.documents (
# MAGIC   id STRING,
# MAGIC   content STRING,
# MAGIC   category STRING
# MAGIC );
# MAGIC
# MAGIC -- Insert sample data
# MAGIC INSERT INTO main.genai_course.documents VALUES
# MAGIC   ('doc1', 'Machine learning is a subset of artificial intelligence', 'ml'),
# MAGIC   ('doc2', 'Deep learning uses neural networks with many layers', 'ml'),
# MAGIC   ('doc3', 'RAG combines retrieval with generation', 'genai'),
# MAGIC   ('doc4', 'Vector databases enable similarity search', 'genai');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Index

# COMMAND ----------

# Create index with auto-sync from Delta table
index_name = "main.genai_course.doc_index"

index = vsc.create_delta_sync_index(
    endpoint_name=endpoint_name,
    index_name=index_name,
    source_table_name="main.genai_course.documents",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
)

print(f"Created index: {index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Similarity Search

# COMMAND ----------

# Wait for index to be ready
index = vsc.get_index(endpoint_name, index_name)
index.wait_until_ready()

# Perform similarity search
results = index.similarity_search(
    query_text="How does deep learning work?",
    columns=["id", "content", "category"],
    num_results=3
)

print("Search Results:")
for row in results.get("result", {}).get("data_array", []):
    print(f"  ID: {row[0]}, Score: {row[-1]:.4f}")
    print(f"  Content: {row[1][:50]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Filtered Search

# COMMAND ----------

# Search with metadata filter
results = index.similarity_search(
    query_text="neural networks",
    columns=["id", "content", "category"],
    filters={"category": "ml"},
    num_results=2
)

print("Filtered Results (category=ml):")
for row in results.get("result", {}).get("data_array", []):
    print(f"  {row[0]}: {row[1][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compare: Databricks vs Rust
# MAGIC
# MAGIC | Feature | Databricks | Rust Demo |
# MAGIC |---------|-----------|-----------|
# MAGIC | Embedding | Managed endpoints | Hash-based |
# MAGIC | Index | Auto-sync from Delta | In-memory |
# MAGIC | Scaling | Automatic | Manual |
# MAGIC | Filtering | SQL-like | Programmatic |
