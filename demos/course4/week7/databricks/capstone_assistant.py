# Databricks notebook source
# MAGIC %md
# MAGIC # Enterprise Knowledge Assistant Capstone
# MAGIC
# MAGIC **Course 4, Week 7: End-to-End GenAI System**
# MAGIC
# MAGIC This notebook demonstrates a complete enterprise knowledge assistant.
# MAGIC Combines all concepts from Course 4: RAG, Production, Fine-tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                 Enterprise Knowledge Assistant              │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │                                                             │
# MAGIC │  Documents → Chunk → Embed → Vector Index                   │
# MAGIC │       ↓                            ↓                        │
# MAGIC │    Metadata ──────────────── Hybrid Search                  │
# MAGIC │                                    ↓                        │
# MAGIC │  Query → Guardrails → Retrieve → Augment → Generate         │
# MAGIC │                                             ↓               │
# MAGIC │                                       Response → Log        │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import math

print("Enterprise Knowledge Assistant - Course 4 Capstone")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Document Processing

# COMMAND ----------

@dataclass
class Document:
    id: str
    title: str
    content: str
    category: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    embedding: List[float] = field(default_factory=list)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 100, overlap: int = 20, dim: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.dim = dim

    def process(self, doc: Document) -> List[Chunk]:
        words = doc.content.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            text = " ".join(words[start:end])
            embedding = self._embed(text)

            chunks.append(Chunk(
                id=f"{doc.id}-chunk-{len(chunks)}",
                doc_id=doc.id,
                text=text,
                embedding=embedding
            ))

            if end >= len(words):
                break
            start += self.chunk_size - self.overlap

        return chunks

    def _embed(self, text: str) -> List[float]:
        values = [0.0] * self.dim
        for i, word in enumerate(text.lower().split()):
            for j, c in enumerate(word):
                idx = (ord(c) * (i + 1) + j) % self.dim
                values[idx] += 0.1
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm > 1e-10 else values


# Create documents
documents = [
    Document("hr-001", "Employee Handbook",
             "Company policies require all employees to follow the code of conduct. "
             "Vacation time accrues monthly. Remote work needs manager approval. "
             "Benefits enrollment opens annually in November.",
             "HR"),
    Document("eng-001", "ML Platform Guide",
             "Machine learning models are trained using distributed compute. "
             "The feature store holds preprocessed features. Model registry tracks versions. "
             "Deployments use containerized serving endpoints.",
             "Engineering"),
    Document("data-001", "Data Architecture",
             "We use the medallion architecture. Bronze holds raw data. "
             "Silver contains cleaned and validated data. Gold has business aggregates. "
             "Data quality checks run at each layer.",
             "Data"),
]

processor = DocumentProcessor(chunk_size=30, overlap=5, dim=128)

all_chunks = []
for doc in documents:
    chunks = processor.process(doc)
    all_chunks.extend(chunks)
    print(f"Processed '{doc.title}': {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Store

# COMMAND ----------

class VectorStore:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.chunks = []

    def add(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[tuple]:
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scored = [(c, cosine_sim(query_embedding, c.embedding)) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


vector_store = VectorStore(dim=128)
vector_store.add(all_chunks)

print(f"Vector store contains {len(vector_store.chunks)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guardrails

# COMMAND ----------

class Guardrails:
    def __init__(self):
        self.blocked = ["password", "secret", "confidential", "ssn"]
        self.max_length = 4096

    def check(self, text: str) -> tuple:
        text_lower = text.lower()

        if len(text) > self.max_length:
            return False, f"Input too long: {len(text)}"

        for pattern in self.blocked:
            if pattern in text_lower:
                return False, f"Blocked pattern: {pattern}"

        return True, "OK"


guardrails = Guardrails()

# Test guardrails
test_inputs = ["What is ML?", "What is the admin password?", "Tell me company secrets"]
print("Guardrail Tests:\n")
for text in test_inputs:
    passed, msg = guardrails.check(text)
    status = "✓" if passed else "✗"
    print(f"  {status} \"{text}\" -> {msg}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Knowledge Assistant

# COMMAND ----------

class KnowledgeAssistant:
    def __init__(self, processor: DocumentProcessor, store: VectorStore, docs: Dict[str, Document]):
        self.processor = processor
        self.store = store
        self.docs = {d.id: d for d in docs}
        self.guardrails = Guardrails()

    def query(self, question: str, top_k: int = 3) -> Dict:
        # 1. Guardrails
        passed, msg = self.guardrails.check(question)
        if not passed:
            return {"error": msg, "blocked": True}

        # 2. Embed query
        query_emb = self.processor._embed(question)

        # 3. Retrieve
        results = self.store.search(query_emb, top_k)

        # 4. Build context
        context = "\n\n".join([r[0].text for r in results])

        # 5. Generate (simulated)
        answer = self._generate(question, context)

        # 6. Build response
        sources = []
        for chunk, score in results:
            doc = self.docs.get(chunk.doc_id)
            sources.append({
                "doc_id": chunk.doc_id,
                "title": doc.title if doc else "",
                "preview": chunk.text[:100] + "...",
                "score": score
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "blocked": False
        }

    def _generate(self, question: str, context: str) -> str:
        q_lower = question.lower()
        c_lower = context.lower()

        if "machine learning" in q_lower or "ml" in q_lower:
            return "Based on the ML Platform Guide, machine learning models are trained using distributed compute and deployed via containerized serving endpoints."
        elif "vacation" in q_lower or "time off" in q_lower:
            return "According to the Employee Handbook, vacation time accrues monthly. Please contact HR for your specific balance."
        elif "medallion" in q_lower or "data architecture" in q_lower:
            return "The company uses medallion architecture with Bronze (raw), Silver (cleaned), and Gold (aggregated) layers."
        elif context:
            return f"Based on the knowledge base: {context[:200]}..."
        else:
            return "I couldn't find relevant information in the knowledge base."


# Create assistant
doc_map = {d.id: d for d in documents}
assistant = KnowledgeAssistant(processor, vector_store, documents)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Query the Assistant

# COMMAND ----------

queries = [
    "How does machine learning work on the platform?",
    "What is the vacation policy?",
    "Explain the medallion architecture",
    "What is the admin password?",  # Should be blocked
]

print("Knowledge Assistant Queries:\n")
for q in queries:
    result = assistant.query(q, top_k=2)

    if result.get("blocked"):
        print(f"Q: {q}")
        print(f"🛡️ BLOCKED: {result['error']}\n")
    else:
        print(f"Q: {q}")
        print(f"A: {result['answer'][:100]}...")
        print("Sources:")
        for src in result["sources"]:
            print(f"  - {src['title']} [{src['score']:.3f}]")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Production Configuration

# COMMAND ----------

production_config = {
    "endpoint": {
        "name": "knowledge-assistant-prod",
        "served_entities": [
            {
                "name": "rag-pipeline",
                "workload_size": "Medium",
                "scale_to_zero_enabled": False
            }
        ]
    },
    "vector_search": {
        "index_name": "knowledge_base_index",
        "embedding_model": "databricks-bge-large-en"
    },
    "foundation_model": {
        "endpoint": "databricks-dbrx-instruct",
        "max_tokens": 500,
        "temperature": 0.1
    },
    "guardrails": {
        "input_filters": ["pii", "prompt_injection"],
        "output_filters": ["toxicity", "hallucination"]
    },
    "monitoring": {
        "inference_table": "main.logs.assistant_requests",
        "alert_on_error_rate": 0.01,
        "alert_on_latency_p99": 1000
    }
}

print("Production Configuration:")
print(json.dumps(production_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Capstone Deliverables
# MAGIC
# MAGIC | # | Deliverable | Stack Components | Status |
# MAGIC |---|-------------|------------------|--------|
# MAGIC | 1 | Document Ingestion | alimentar, trueno | ✓ |
# MAGIC | 2 | Vector Index | trueno-rag, Vector Search | ✓ |
# MAGIC | 3 | RAG Pipeline | Foundation Models | ✓ |
# MAGIC | 4 | Guardrails | Custom filters | ✓ |
# MAGIC | 5 | Production Serving | realizar, Model Serving | ✓ |
# MAGIC | 6 | Monitoring | pmat, Inference Tables | ✓ |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comparison: Sovereign vs Databricks
# MAGIC
# MAGIC | Component | Sovereign AI | Databricks |
# MAGIC |-----------|-------------|------------|
# MAGIC | Embedding | trueno | BGE-Large |
# MAGIC | Vector Store | trueno-rag | Vector Search |
# MAGIC | Generation | realizar | Foundation Models |
# MAGIC | Serving | realizar | Model Serving |
# MAGIC | Monitoring | pmat | Inference Tables |
# MAGIC | Orchestration | batuta | Workflows |

# COMMAND ----------

print("Capstone Complete!")
print("\nKey achievements:")
print("1. Built document ingestion pipeline")
print("2. Implemented vector-based retrieval")
print("3. Created RAG question-answering system")
print("4. Added safety guardrails")
print("5. Configured production deployment")
print("\nThis assistant can be deployed on Databricks Model Serving")
print("with Vector Search and Foundation Model APIs.")
