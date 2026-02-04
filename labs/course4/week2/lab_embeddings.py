# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3.3: Generate and Store Embeddings
# MAGIC
# MAGIC **Course 4, Week 3: Vector Search**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Generate text embeddings
# MAGIC - Compute similarity scores
# MAGIC - Build a simple vector store

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

print("Embeddings Lab - Course 4 Week 3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Simple Embedding Function
# MAGIC
# MAGIC We'll create a hash-based embedding for demonstration.

# COMMAND ----------

def simple_embed(text: str, dim: int = 64) -> List[float]:
    """
    Generate a simple hash-based embedding.
    (In production, use transformer models like BGE or E5)
    """
    values = [0.0] * dim
    for i, word in enumerate(text.lower().split()):
        for j, char in enumerate(word):
            idx = (ord(char) * (i + 1) + j) % dim
            values[idx] += 0.1

    # Normalize to unit length
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0:
        values = [v / norm for v in values]

    return values


# Test embedding
text = "machine learning is powerful"
embedding = simple_embed(text)
print(f"Text: {text}")
print(f"Embedding dim: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"Norm: {math.sqrt(sum(v*v for v in embedding)):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Similarity Functions
# MAGIC
# MAGIC EXERCISE: Implement cosine and euclidean similarity.

# COMMAND ----------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    TODO: Implement formula: cos(θ) = (a·b) / (|a| × |b|)
    """
    # YOUR CODE HERE
    pass


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Compute euclidean distance between two vectors.

    TODO: Implement formula: d = √(Σ(a_i - b_i)²)
    """
    # YOUR CODE HERE
    pass


# Test your implementations
# emb1 = simple_embed("machine learning")
# emb2 = simple_embed("deep learning")
# emb3 = simple_embed("cooking recipes")

# print(f"ML vs DL (cosine): {cosine_similarity(emb1, emb2):.4f}")
# print(f"ML vs Cooking (cosine): {cosine_similarity(emb1, emb3):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Vector Store
# MAGIC
# MAGIC EXERCISE: Implement a simple in-memory vector store.

# COMMAND ----------

@dataclass
class Document:
    id: str
    content: str
    embedding: List[float] = None


class VectorStore:
    """Simple in-memory vector store."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.documents: List[Document] = []

    def add(self, doc_id: str, content: str) -> None:
        """
        Add document to store with auto-generated embedding.

        TODO:
        1. Generate embedding for content
        2. Create Document with id, content, embedding
        3. Add to documents list
        """
        # YOUR CODE HERE
        pass

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        TODO:
        1. Generate embedding for query
        2. Compute similarity with all documents
        3. Return top_k results sorted by similarity
        """
        # YOUR CODE HERE
        pass

    def __len__(self) -> int:
        return len(self.documents)


# Test your vector store
# store = VectorStore(dim=64)
# store.add("doc1", "Machine learning algorithms learn from data")
# store.add("doc2", "Deep learning uses neural networks")
# store.add("doc3", "Cooking requires fresh ingredients")

# results = store.search("neural network training", top_k=2)
# for doc, score in results:
#     print(f"[{score:.4f}] {doc.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Batch Operations
# MAGIC
# MAGIC EXERCISE: Add batch insert and batch search.

# COMMAND ----------

def batch_add(store: VectorStore, documents: List[Dict]) -> int:
    """
    Add multiple documents at once.

    TODO:
    - documents is list of {"id": str, "content": str}
    - Return count of added documents
    """
    # YOUR CODE HERE
    pass


def batch_search(store: VectorStore, queries: List[str], top_k: int = 3) -> List[List[Tuple]]:
    """
    Search multiple queries at once.

    TODO: Return list of results for each query
    """
    # YOUR CODE HERE
    pass


# Test batch operations
# docs = [
#     {"id": "1", "content": "Python programming"},
#     {"id": "2", "content": "Java development"},
#     {"id": "3", "content": "Data science with Python"},
# ]
# batch_add(store, docs)
# results = batch_search(store, ["Python code", "machine learning"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Persistence (Simulation)
# MAGIC
# MAGIC EXERCISE: Implement save/load for the vector store.

# COMMAND ----------

import json

def save_store(store: VectorStore, path: str) -> None:
    """
    Save vector store to JSON file.

    TODO: Serialize documents with embeddings
    """
    # YOUR CODE HERE
    pass


def load_store(path: str) -> VectorStore:
    """
    Load vector store from JSON file.

    TODO: Deserialize and recreate VectorStore
    """
    # YOUR CODE HERE
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check similarity functions
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        c = [0.0, 1.0]

        sim_same = cosine_similarity(a, b)
        sim_orth = cosine_similarity(a, c)
        checks.append(("Cosine similarity works", abs(sim_same - 1.0) < 0.01 and abs(sim_orth) < 0.01))

        dist = euclidean_distance(a, c)
        checks.append(("Euclidean distance works", abs(dist - math.sqrt(2)) < 0.01))

        # Check vector store
        store = VectorStore(dim=64)
        store.add("test", "test content")
        checks.append(("VectorStore add works", len(store) == 1))

        results = store.search("test query", top_k=1)
        checks.append(("VectorStore search works", len(results) == 1))

    except Exception as e:
        checks.append(("Implementation complete", False))
        print(f"Error: {e}")

    # Display results
    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Lab complete.")
    else:
        print("\n⚠️ Some checks failed. Review your code.")

validate_lab()
