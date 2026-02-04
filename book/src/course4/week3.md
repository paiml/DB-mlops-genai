# Week 3: Embeddings and Vector Search

## Overview

Build SIMD-accelerated vector search with trueno and implement HNSW indexing.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 3.1 | Video | What Are Embeddings? | Concept | 10 min |
| 3.2 | Video | Databricks Vector Search | Databricks | 10 min |
| 3.3 | Lab | Create Vector Search Index | Databricks | 35 min |
| 3.4 | Video | SIMD Similarity: Cosine, Dot Product | Sovereign | 10 min |
| 3.5 | Lab | Build SIMD Vector Search with trueno | Sovereign | 35 min |
| 3.6 | Video | HNSW: Approximate Nearest Neighbors | Concept | 10 min |
| 3.7 | Lab | Implement HNSW Index | Sovereign | 40 min |
| 3.8 | Video | Hybrid Search: BM25 + Vector | Sovereign | 8 min |
| 3.9 | Lab | Hybrid Retrieval with trueno-rag | Sovereign | 35 min |
| 3.10 | Quiz | Vector Search | — | 15 min |

## Sovereign AI Stack Components

- `trueno` for SIMD computation
- `trueno-rag` for BM25 + HNSW
- `trueno-db` for GPU analytics

## Key Concepts

### Similarity Metrics
- Cosine similarity: `dot(a, b) / (||a|| * ||b||)`
- Euclidean distance: `sqrt(sum((a - b)^2))`
- Dot product: `sum(a * b)`

### HNSW Algorithm
- Hierarchical navigable small world graphs
- O(log n) search complexity
- Configurable M and ef parameters
