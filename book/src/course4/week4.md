# Week 4: RAG Pipelines

## Overview

Build production RAG systems with document chunking, retrieval, and generation.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 4.1 | Video | RAG Architecture | Concept | 10 min |
| 4.2 | Video | Databricks RAG Components | Databricks | 10 min |
| 4.3 | Lab | Build RAG Pipeline in Databricks | Databricks | 45 min |
| 4.4 | Video | Document Chunking Strategies | Concept | 10 min |
| 4.5 | Lab | Chunking Pipeline with alimentar | Sovereign | 35 min |
| 4.6 | Video | Context Window Management | Sovereign | 8 min |
| 4.7 | Lab | End-to-End RAG with trueno-rag | Sovereign | 45 min |
| 4.8 | Video | RAG Evaluation Metrics | Concept | 10 min |
| 4.9 | Lab | Evaluate RAG Quality | Both | 35 min |
| 4.10 | Quiz | RAG Systems | — | 15 min |

## Sovereign AI Stack Components

- `trueno-rag` for full RAG pipeline
- `alimentar` for document chunking
- `pmat` for evaluation metrics

## Key Concepts

### RAG Pipeline
1. **Chunk**: Split documents into retrievable segments
2. **Embed**: Convert chunks to vector representations
3. **Index**: Store in vector database
4. **Retrieve**: Find relevant chunks for query
5. **Generate**: Synthesize answer with LLM

### Chunking Strategies
- Fixed size with overlap
- Sentence-based splitting
- Semantic chunking
