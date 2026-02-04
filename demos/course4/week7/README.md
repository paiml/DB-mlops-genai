# Week 7: Capstone - Enterprise Knowledge Assistant

**Course 4: GenAI Engineering on Databricks**

## Overview

Build an end-to-end enterprise knowledge assistant that combines all concepts from Course 4.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Enterprise Knowledge Assistant              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents → Chunk → Embed → Vector Index                   │
│       ↓                            ↓                        │
│    Metadata ──────────────── Hybrid Search                  │
│                                    ↓                        │
│  Query → Guardrails → Retrieve → Augment → Generate         │
│                                             ↓               │
│                                       Response → Log        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Capstone Deliverables

| # | Deliverable | Stack Components | Description |
|---|-------------|------------------|-------------|
| 1 | Document Pipeline | alimentar, trueno | Ingest and chunk documents |
| 2 | Vector Index | trueno-rag | Build searchable embeddings |
| 3 | RAG Pipeline | realizar | Retrieve and generate answers |
| 4 | Safety Layer | Custom | Input/output guardrails |
| 5 | Production API | realizar | REST endpoint with metrics |
| 6 | Monitoring | pmat | Track quality and latency |

## Running the Capstone

### Rust Implementation

```bash
cd capstone
cargo run
```

### Databricks Notebook

Import `databricks/capstone_assistant.py` into your workspace.

## Demo Output

```
╔═══════════════════════════════════════════════════════════════╗
║     Enterprise Knowledge Assistant - Course 4 Capstone        ║
╚═══════════════════════════════════════════════════════════════╝

📚 Document Ingestion
   Documents: 3
   Chunks: 15

❓ Knowledge Queries
   Q: How does machine learning work?
   A: Based on the ML Platform Guide...

📊 Performance Metrics
   Success rate: 95%
   Avg latency: 50ms
```

## Key Features

- **Document Processing**: Chunking with overlap for context preservation
- **Vector Search**: Semantic similarity for relevant retrieval
- **RAG Pipeline**: Context-augmented generation
- **Guardrails**: PII detection, blocked patterns, length limits
- **Metrics**: Request tracking, latency, success rate

## Comparison: Sovereign vs Databricks

| Component | Sovereign AI | Databricks |
|-----------|-------------|------------|
| Embedding | trueno | BGE-Large |
| Vector Store | trueno-rag | Vector Search |
| Generation | realizar | Foundation Models |
| Serving | realizar | Model Serving |
| Monitoring | pmat | Inference Tables |

## Resources

- [Databricks RAG Tutorial](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/index.html)
- [Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
