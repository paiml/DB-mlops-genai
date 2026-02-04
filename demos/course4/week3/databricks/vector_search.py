# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search on Databricks
# MAGIC
# MAGIC **Course 4, Week 3: Vector Search and Embeddings**
# MAGIC
# MAGIC This notebook demonstrates Databricks Vector Search capabilities.
# MAGIC Compare with trueno-rag for self-hosted vector search.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import List, Dict, Tuple
import math

print("Vector Search Demo - Course 4 Week 3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vector Embeddings

# COMMAND ----------

class SimpleEmbedding:
    """Demonstration embedding (use real models in production)."""

    def __init__(self, values: List[float]):
        self.values = values
        self.dimension = len(values)

    def normalize(self) -> 'SimpleEmbedding':
        norm = math.sqrt(sum(v * v for v in self.values))
        if norm < 1e-10:
            return self
        return SimpleEmbedding([v / norm for v in self.values])

    def dot(self, other: 'SimpleEmbedding') -> float:
        return sum(a * b for a, b in zip(self.values, other.values))

    def cosine_similarity(self, other: 'SimpleEmbedding') -> float:
        dot = self.dot(other)
        norm_self = math.sqrt(sum(v * v for v in self.values))
        norm_other = math.sqrt(sum(v * v for v in other.values))
        if norm_self < 1e-10 or norm_other < 1e-10:
            return 0.0
        return dot / (norm_self * norm_other)


class HashEmbedder:
    """Simple hash-based embedder for demonstration."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, text: str) -> SimpleEmbedding:
        values = [0.0] * self.dimension
        for i, word in enumerate(text.lower().split()):
            for j, char in enumerate(word):
                idx = (ord(char) * (i + 1) + j) % self.dimension
                values[idx] += 0.1
        return SimpleEmbedding(values).normalize()


# Demo embeddings
embedder = HashEmbedder(128)
texts = ["machine learning", "deep learning", "cooking recipes"]
embeddings = [embedder.embed(t) for t in texts]

print("Embedding examples:")
for text, emb in zip(texts, embeddings):
    print(f"  '{text}': dim={emb.dimension}, first 5 values: {emb.values[:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Similarity Metrics

# COMMAND ----------

print("Similarity between embeddings:\n")

for i, (t1, e1) in enumerate(zip(texts, embeddings)):
    for t2, e2 in zip(texts[i+1:], embeddings[i+1:]):
        sim = e1.cosine_similarity(e2)
        print(f"  '{t1}' vs '{t2}': {sim:.4f}")

# ML and DL should be more similar than ML and cooking
print("\nExpected: ML/DL similarity > ML/cooking similarity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Databricks Vector Search Index
# MAGIC
# MAGIC Vector Search provides managed vector indices with Delta table sync.

# COMMAND ----------

# Index configuration (for reference - actual creation via Databricks UI/API)
index_config = {
    "name": "product_search_index",
    "primary_key": "product_id",
    "index_type": "DELTA_SYNC",
    "delta_sync_index_spec": {
        "source_table": "catalog.schema.products",
        "embedding_source_columns": [
            {
                "name": "description_embedding",
                "embedding_model_endpoint_name": "databricks-bge-large-en"
            }
        ],
        "pipeline_type": "TRIGGERED"
    }
}

print("Vector Search Index Configuration:")
print(json.dumps(index_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Embedding Models on Databricks

# COMMAND ----------

# Available embedding models
embedding_models = {
    "databricks-bge-large-en": {
        "provider": "BAAI",
        "dimension": 1024,
        "max_tokens": 512,
        "use_case": "General text embedding"
    },
    "databricks-gte-large-en": {
        "provider": "Alibaba",
        "dimension": 1024,
        "max_tokens": 512,
        "use_case": "General text embedding"
    },
    "text-embedding-3-small": {
        "provider": "OpenAI (via Databricks)",
        "dimension": 1536,
        "max_tokens": 8191,
        "use_case": "High quality embeddings"
    }
}

print("Available Embedding Models:\n")
for name, info in embedding_models.items():
    print(f"  {name}:")
    print(f"    Provider: {info['provider']}")
    print(f"    Dimension: {info['dimension']}")
    print(f"    Max tokens: {info['max_tokens']}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Vector Search Query

# COMMAND ----------

def simulate_vector_search(query: str, documents: List[Dict], embedder: HashEmbedder, top_k: int = 3) -> List[Dict]:
    """Simulate vector search (actual implementation uses Databricks API)."""
    query_emb = embedder.embed(query)

    # Score all documents
    scored = []
    for doc in documents:
        doc_emb = embedder.embed(doc["content"])
        score = query_emb.cosine_similarity(doc_emb)
        scored.append({
            "id": doc["id"],
            "content": doc["content"],
            "score": score
        })

    # Sort by score and return top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# Sample documents
documents = [
    {"id": "1", "content": "Machine learning algorithms learn patterns from data"},
    {"id": "2", "content": "Deep neural networks have multiple hidden layers"},
    {"id": "3", "content": "Natural language processing handles text analysis"},
    {"id": "4", "content": "Computer vision recognizes objects in images"},
    {"id": "5", "content": "Reinforcement learning uses rewards for training"},
]

# Search
queries = ["neural network training", "text understanding", "image recognition"]

print("Vector Search Results:\n")
for query in queries:
    print(f"Query: '{query}'")
    results = simulate_vector_search(query, documents, embedder, top_k=2)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:50]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Text Chunking

# COMMAND ----------

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


# Demo chunking
long_text = """
Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. Deep learning is a
specialized form of machine learning that uses neural networks with many layers.
These systems can recognize patterns in data, make predictions, and even generate
new content. Applications include natural language processing, computer vision,
speech recognition, and autonomous vehicles.
"""

chunks = chunk_text(long_text.strip(), chunk_size=20, overlap=5)
print(f"Original text: {len(long_text.split())} words")
print(f"Number of chunks: {len(chunks)}")
print()

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk.split())} words):")
    print(f"  {chunk[:60]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hybrid Search

# COMMAND ----------

def hybrid_search(
    query: str,
    documents: List[Dict],
    embedder: HashEmbedder,
    alpha: float = 0.5,
    top_k: int = 3
) -> List[Dict]:
    """Combine vector and keyword search."""
    query_emb = embedder.embed(query)
    query_words = set(query.lower().split())

    scored = []
    for doc in documents:
        # Vector score
        doc_emb = embedder.embed(doc["content"])
        vector_score = query_emb.cosine_similarity(doc_emb)

        # Keyword score (simple word overlap)
        doc_words = set(doc["content"].lower().split())
        keyword_score = len(query_words & doc_words) / max(len(query_words), 1)

        # Combined score
        combined = alpha * vector_score + (1 - alpha) * keyword_score

        scored.append({
            "id": doc["id"],
            "content": doc["content"],
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "combined_score": combined
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored[:top_k]


# Demo hybrid search
query = "neural network learning"
print(f"Hybrid Search: '{query}'\n")
results = hybrid_search(query, documents, embedder, alpha=0.7)

for r in results:
    print(f"  [{r['combined_score']:.4f}] = {r['vector_score']:.4f} (vector) + {r['keyword_score']:.4f} (keyword)")
    print(f"    {r['content'][:50]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks Vector Search | Sovereign AI (trueno-rag) |
# MAGIC |---------|-------------------------|---------------------------|
# MAGIC | **Index Type** | Delta Sync, Direct | Custom (HNSW, Flat) |
# MAGIC | **Embedding** | Managed models | Self-hosted |
# MAGIC | **Scaling** | Auto-scaling | Manual |
# MAGIC | **Integration** | Unity Catalog | File-based |
# MAGIC | **Updates** | Real-time sync | Manual refresh |
# MAGIC
# MAGIC **Key Insight:** Databricks offers managed simplicity; trueno-rag offers full control.

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. Vector embeddings capture semantic meaning")
print("2. Cosine similarity measures embedding closeness")
print("3. Chunking preserves context for long documents")
print("4. Hybrid search combines vector and keyword matching")
print("5. Databricks Vector Search provides managed indices")
