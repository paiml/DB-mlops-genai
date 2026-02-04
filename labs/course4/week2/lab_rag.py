# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.5: End-to-End RAG Pipeline
# MAGIC
# MAGIC **Course 4, Week 4: RAG Pipelines**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Build complete RAG pipeline
# MAGIC - Implement chunking strategy
# MAGIC - Evaluate retrieval quality

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import math
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

print("RAG Pipeline Lab - Course 4 Week 4")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Document Chunking
# MAGIC
# MAGIC Implement chunking with overlap.

# COMMAND ----------

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[Dict]:
    """
    Split text into overlapping chunks.

    Returns list of {"text": str, "start": int, "end": int}
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "start": start,
            "end": end
        })

        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


# Test chunking
sample_text = """
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience. Deep learning is a specialized form of
machine learning using neural networks. Natural language processing combines
linguistics with machine learning to process human language.
"""

chunks = chunk_text(sample_text.strip(), chunk_size=20, overlap=5)
print(f"Created {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {len(chunk['text'].split())} words")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: RAG Pipeline Class
# MAGIC
# MAGIC EXERCISE: Implement a complete RAG pipeline.

# COMMAND ----------

@dataclass
class Document:
    id: str
    title: str
    content: str


class RAGPipeline:
    """Complete RAG pipeline."""

    def __init__(self, chunk_size: int = 50, chunk_overlap: int = 10, embed_dim: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_dim = embed_dim
        self.chunks: List[Dict] = []
        self.documents: Dict[str, Document] = {}

    def _embed(self, text: str) -> List[float]:
        """Simple embedding function."""
        values = [0.0] * self.embed_dim
        for i, word in enumerate(text.lower().split()):
            for j, c in enumerate(word):
                idx = (ord(c) * (i + 1) + j) % self.embed_dim
                values[idx] += 0.1
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm > 0 else values

    def ingest(self, document: Document) -> int:
        """
        Ingest document into the pipeline.

        TODO:
        1. Store document
        2. Chunk the content
        3. Embed each chunk
        4. Store chunks with embeddings
        5. Return number of chunks created
        """
        # YOUR CODE HERE
        pass

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks for query.

        TODO:
        1. Embed the query
        2. Compute similarity with all chunks
        3. Return top_k chunks with scores
        """
        # YOUR CODE HERE
        pass

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer from context (simulated).

        TODO: Return a simple response based on context
        """
        # YOUR CODE HERE (simple pattern matching for demo)
        pass

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Complete RAG query: retrieve + generate.

        TODO:
        1. Retrieve relevant chunks
        2. Build context from chunks
        3. Generate answer
        4. Return dict with answer, sources, and context
        """
        # YOUR CODE HERE
        pass


# Test your RAG pipeline
# rag = RAGPipeline()
# rag.ingest(Document("1", "ML Guide", "Machine learning enables automation..."))
# result = rag.query("What is machine learning?")
# print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Evaluation Metrics
# MAGIC
# MAGIC EXERCISE: Implement RAG evaluation metrics.

# COMMAND ----------

def evaluate_retrieval(retrieved_ids: List[str], relevant_ids: List[str]) -> Dict:
    """
    Evaluate retrieval quality.

    TODO: Calculate:
    - precision: relevant retrieved / total retrieved
    - recall: relevant retrieved / total relevant
    - f1: harmonic mean of precision and recall
    """
    # YOUR CODE HERE
    pass


def evaluate_answer(answer: str, ground_truth: str) -> Dict:
    """
    Evaluate answer quality.

    TODO: Calculate:
    - word_overlap: proportion of ground truth words in answer
    - answer_length: number of words
    """
    # YOUR CODE HERE
    pass


# Test evaluation
# retrieved = ["doc1", "doc2", "doc3"]
# relevant = ["doc1", "doc3", "doc5"]
# metrics = evaluate_retrieval(retrieved, relevant)
# print(f"Retrieval metrics: {metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Multi-Document RAG
# MAGIC
# MAGIC EXERCISE: Test RAG with multiple documents.

# COMMAND ----------

# Create a knowledge base
documents = [
    Document("ml", "Machine Learning Basics",
             "Machine learning is a method of data analysis that automates analytical model building. "
             "It uses algorithms that iteratively learn from data, allowing computers to find hidden insights."),
    Document("dl", "Deep Learning Introduction",
             "Deep learning is a subset of machine learning using neural networks with multiple layers. "
             "These networks can learn representations of data with multiple levels of abstraction."),
    Document("nlp", "Natural Language Processing",
             "NLP is a field focused on the interaction between computers and humans through language. "
             "It involves programming computers to process and analyze large amounts of natural language data."),
]

# EXERCISE: Build RAG pipeline and test queries
# rag = RAGPipeline()
# for doc in documents:
#     rag.ingest(doc)

# Test queries
# queries = [
#     "What is machine learning?",
#     "How does deep learning work?",
#     "What is NLP used for?",
# ]

# for q in queries:
#     result = rag.query(q)
#     print(f"Q: {q}")
#     print(f"A: {result['answer']}")
#     print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check RAG pipeline
        rag = RAGPipeline()
        chunks = rag.ingest(Document("test", "Test", "This is test content for the RAG pipeline"))
        checks.append(("Ingest works", chunks > 0))

        retrieved = rag.retrieve("test content", top_k=1)
        checks.append(("Retrieve works", len(retrieved) > 0))

        result = rag.query("What is this about?")
        checks.append(("Query works", "answer" in result))

        # Check evaluation
        if 'evaluate_retrieval' in dir():
            metrics = evaluate_retrieval(["a", "b"], ["a", "c"])
            checks.append(("Evaluation works", "precision" in metrics))

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
