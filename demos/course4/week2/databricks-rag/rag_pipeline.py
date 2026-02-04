# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Pipeline on Databricks
# MAGIC
# MAGIC **Course 4, Week 2: RAG Pipelines**
# MAGIC
# MAGIC This notebook demonstrates building RAG applications with Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import mlflow.deployments

vsc = VectorSearchClient()
llm_client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Document Chunking

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample document
document = """
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.
It tackles four primary functions: tracking experiments, packaging code into reproducible runs,
sharing and deploying models, and providing a central model registry.

The MLflow Tracking component provides an API and UI for logging parameters, metrics,
code versions, and artifacts during ML experiments. It supports Python, R, Java, and REST APIs.

MLflow Models is a standard format for packaging machine learning models that can be used
in a variety of downstream tools—for example, real-time serving through a REST API or
batch inference on Apache Spark.
"""

# Chunk the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_text(document)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Index Chunks

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create chunks table
# MAGIC CREATE TABLE IF NOT EXISTS main.genai_course.rag_chunks (
# MAGIC   chunk_id STRING,
# MAGIC   content STRING,
# MAGIC   source_doc STRING
# MAGIC );

# COMMAND ----------

# Insert chunks into Delta table
from pyspark.sql import Row

chunk_rows = [
    Row(chunk_id=f"chunk_{i}", content=chunk, source_doc="mlflow_docs")
    for i, chunk in enumerate(chunks)
]

df = spark.createDataFrame(chunk_rows)
df.write.mode("overwrite").saveAsTable("main.genai_course.rag_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Query Function

# COMMAND ----------

def rag_query(question: str, top_k: int = 3) -> str:
    """Execute RAG query with retrieval and generation."""

    # Step 1: Retrieve relevant chunks
    index = vsc.get_index("course4-demo-endpoint", "main.genai_course.chunks_index")

    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "content"],
        num_results=top_k
    )

    # Step 2: Build context
    context_chunks = []
    for row in results.get("result", {}).get("data_array", []):
        context_chunks.append(row[1])

    context = "\n\n".join(context_chunks)

    # Step 3: Generate response
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    response = llm_client.predict(
        endpoint="databricks-dbrx-instruct",
        inputs={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }
    )

    return {
        "answer": response["choices"][0]["message"]["content"],
        "sources": [r[0] for r in results.get("result", {}).get("data_array", [])],
        "context_length": len(context)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test RAG Pipeline

# COMMAND ----------

# Test the RAG pipeline
result = rag_query("What is MLflow used for?")

print("Question: What is MLflow used for?")
print(f"\nAnswer: {result['answer']}")
print(f"\nSources: {result['sources']}")
print(f"Context length: {result['context_length']} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluation

# COMMAND ----------

# Simple relevance check
def evaluate_response(question, answer, expected_keywords):
    """Evaluate RAG response quality."""
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(found) / len(expected_keywords)

test_cases = [
    {
        "question": "What does MLflow Tracking do?",
        "keywords": ["logging", "parameters", "metrics", "experiments"]
    },
    {
        "question": "What is MLflow Models?",
        "keywords": ["packaging", "format", "serving", "inference"]
    }
]

for test in test_cases:
    result = rag_query(test["question"])
    score = evaluate_response(test["question"], result["answer"], test["keywords"])
    print(f"Q: {test['question']}")
    print(f"Score: {score:.2f}")
    print()
