# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Applications on Databricks
# MAGIC
# MAGIC **Course 4, Week 4: RAG Pipelines**
# MAGIC
# MAGIC This notebook demonstrates building RAG applications with Databricks.
# MAGIC Compare with trueno-rag for self-hosted RAG.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import List, Dict, Optional
import math

print("RAG Pipeline Demo - Course 4 Week 4")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Document Processing

# COMMAND ----------

class Document:
    """Document with chunking support."""

    def __init__(self, doc_id: str, content: str, metadata: Dict = None):
        self.id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.chunks = []

    def chunk(self, chunk_size: int = 100, overlap: int = 20) -> List[Dict]:
        """Split document into overlapping chunks."""
        words = self.content.split()
        self.chunks = []

        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            self.chunks.append({
                "chunk_id": f"{self.id}-chunk-{chunk_idx}",
                "text": chunk_text,
                "doc_id": self.id,
                "metadata": self.metadata
            })

            if end >= len(words):
                break

            start += chunk_size - overlap
            chunk_idx += 1

        return self.chunks


# Create sample documents
documents = [
    Document(
        "ml-guide",
        """Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        It focuses on developing algorithms that can access data and use it to learn
        for themselves. The process begins with observations or data, such as examples,
        direct experience, or instruction, to look for patterns in data and make
        better decisions in the future.""",
        {"source": "ml-textbook", "topic": "fundamentals"}
    ),
    Document(
        "dl-guide",
        """Deep learning is a machine learning technique that uses neural networks
        with many layers. These deep neural networks attempt to simulate the behavior
        of the human brain in processing data and creating patterns for decision making.
        Deep learning drives many artificial intelligence applications and services
        that improve automation, performing analytical and physical tasks without
        human intervention.""",
        {"source": "dl-textbook", "topic": "deep-learning"}
    ),
    Document(
        "nlp-guide",
        """Natural language processing combines computational linguistics with machine
        learning and deep learning models to process human language. It enables computers
        to understand text and spoken words in much the same way human beings can.
        NLP combines computational linguistics—rule-based modeling of human language—with
        statistical, machine learning, and deep learning models.""",
        {"source": "nlp-textbook", "topic": "nlp"}
    ),
]

# Process documents
all_chunks = []
for doc in documents:
    chunks = doc.chunk(chunk_size=30, overlap=10)
    all_chunks.extend(chunks)
    print(f"Document '{doc.id}': {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simple Vector Store

# COMMAND ----------

class SimpleEmbedding:
    """Hash-based embedding for demonstration."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        values = [0.0] * self.dim
        for i, word in enumerate(text.lower().split()):
            for j, c in enumerate(word):
                idx = (ord(c) * (i + 1) + j) % self.dim
                values[idx] += 0.1

        # Normalize
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 1e-10:
            values = [v / norm for v in values]
        return values

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)


class VectorStore:
    """Simple in-memory vector store."""

    def __init__(self, dim: int = 128):
        self.embedder = SimpleEmbedding(dim)
        self.chunks = []
        self.embeddings = []

    def add(self, chunks: List[Dict]):
        for chunk in chunks:
            self.chunks.append(chunk)
            self.embeddings.append(self.embedder.embed(chunk["text"]))

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_emb = self.embedder.embed(query)

        scored = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            score = self.embedder.cosine_similarity(query_emb, emb)
            scored.append({"chunk": chunk, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Build vector store
vector_store = VectorStore(dim=128)
vector_store.add(all_chunks)

print(f"Vector store contains {len(vector_store.chunks)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Pipeline

# COMMAND ----------

class RAGPipeline:
    """Complete RAG pipeline."""

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k
        self.prompt_template = """Answer the question based on the context below.
If the context doesn't contain relevant information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks."""
        return self.vector_store.search(query, self.top_k)

    def generate(self, query: str, context: str) -> str:
        """Generate answer (simulated - use Foundation Models in production)."""
        q_lower = query.lower()

        if "machine learning" in q_lower:
            return "Machine learning is a subset of AI that enables systems to learn from data and improve over time without explicit programming."
        elif "deep learning" in q_lower:
            return "Deep learning uses neural networks with multiple layers to process data and learn complex patterns, inspired by the human brain."
        elif "nlp" in q_lower or "natural language" in q_lower:
            return "NLP combines linguistics with machine learning to enable computers to understand and process human language."
        else:
            return f"Based on the context, {context[:100]}..."

    def query(self, question: str) -> Dict:
        """Execute full RAG pipeline."""
        # Retrieve
        retrieved = self.retrieve(question)

        # Build context
        context = "\n\n".join([r["chunk"]["text"] for r in retrieved])

        # Generate
        answer = self.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": r["chunk"]["chunk_id"],
                    "score": r["score"],
                    "preview": r["chunk"]["text"][:50] + "..."
                }
                for r in retrieved
            ],
            "context": context
        }


# Create RAG pipeline
rag = RAGPipeline(vector_store, top_k=2)

# Test queries
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "What is natural language processing?"
]

print("RAG Pipeline Results:\n")
for query in queries:
    result = rag.query(query)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['chunk_id']} (score: {src['score']:.3f})")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. RAG with Databricks Foundation Models
# MAGIC
# MAGIC In production, replace the simulated generation with Foundation Model APIs.

# COMMAND ----------

# Production RAG configuration (reference)
rag_config = {
    "vector_search": {
        "endpoint_name": "my-vector-search-endpoint",
        "index_name": "catalog.schema.documents_index",
        "columns": ["chunk_id", "text", "metadata"]
    },
    "embedding_model": {
        "endpoint_name": "databricks-bge-large-en",
        "dimension": 1024
    },
    "generation_model": {
        "endpoint_name": "databricks-dbrx-instruct",
        "max_tokens": 500,
        "temperature": 0.1
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.7
    }
}

print("Production RAG Configuration:")
print(json.dumps(rag_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG Evaluation

# COMMAND ----------

def evaluate_rag(response: Dict, ground_truth: str = None) -> Dict:
    """Evaluate RAG response quality."""

    # Retrieval metrics
    avg_score = sum(s["score"] for s in response["sources"]) / len(response["sources"]) if response["sources"] else 0

    # Context relevance (simple word overlap)
    query_words = set(response["question"].lower().split())
    context_words = set(response["context"].lower().split())
    context_relevance = len(query_words & context_words) / len(query_words) if query_words else 0

    # Answer quality (if ground truth available)
    if ground_truth:
        answer_words = set(response["answer"].lower().split())
        truth_words = set(ground_truth.lower().split())
        faithfulness = len(answer_words & truth_words) / len(truth_words) if truth_words else 0
    else:
        faithfulness = None

    return {
        "retrieval_score": avg_score,
        "context_relevance": context_relevance,
        "answer_faithfulness": faithfulness,
        "num_sources": len(response["sources"])
    }


# Evaluate our RAG responses
result = rag.query("What is machine learning?")
ground_truth = "Machine learning is AI that learns from data"
metrics = evaluate_rag(result, ground_truth)

print("RAG Evaluation Metrics:")
for metric, value in metrics.items():
    if value is not None:
        print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks RAG | Sovereign AI (trueno-rag) |
# MAGIC |---------|---------------|---------------------------|
# MAGIC | **Vector Store** | Vector Search | Custom index |
# MAGIC | **Embedding** | Managed models | Self-hosted |
# MAGIC | **Generation** | Foundation Models | realizar |
# MAGIC | **Scaling** | Auto-scaling | Manual |
# MAGIC | **Integration** | Unity Catalog | File-based |
# MAGIC
# MAGIC **Key Insight:** Databricks provides end-to-end managed RAG; trueno-rag offers full control.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices

# COMMAND ----------

best_practices = """
RAG Best Practices:

1. **Chunking Strategy**
   - Optimal chunk size: 100-500 tokens
   - Use overlap (10-20%) for context
   - Preserve semantic boundaries

2. **Retrieval Quality**
   - Use high-quality embeddings
   - Tune top_k for precision/recall
   - Consider hybrid search (vector + keyword)

3. **Context Management**
   - Don't exceed model context limits
   - Order chunks by relevance
   - Include source attribution

4. **Generation**
   - Use clear system prompts
   - Set appropriate temperature
   - Instruct to cite sources

5. **Evaluation**
   - Track retrieval precision
   - Measure answer faithfulness
   - Monitor for hallucinations
"""

print(best_practices)

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. RAG combines retrieval with generation")
print("2. Chunk overlap preserves context")
print("3. Vector search enables semantic retrieval")
print("4. Evaluation tracks quality metrics")
print("5. Production RAG uses managed services")
