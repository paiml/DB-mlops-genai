# Week 3: Vector Search and Embeddings

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Understand vector embeddings and similarity metrics
2. Build and query vector indices
3. Implement text chunking strategies
4. Use Databricks Vector Search

## Demos

### 1. Vector Search (`vector-search/`)

Rust implementation demonstrating vector search concepts.

**What it demonstrates:**
- Vector embeddings (normalization, dimension)
- Similarity metrics (cosine, euclidean)
- Vector index operations (add, search)
- Text chunking with overlap

**Run locally:**
```bash
cd vector-search
cargo run
```

### 2. Databricks Notebook (`databricks/`)

Databricks Vector Search integration.

**What it demonstrates:**
- Embedding model selection
- Index configuration
- Semantic search queries
- Hybrid search (vector + keyword)

**Run on Databricks:**
1. Import `vector_search.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### Similarity Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| Cosine | `a·b / (|a||b|)` | [-1, 1] | Text similarity |
| Euclidean | `√Σ(a-b)²` | [0, ∞) | Spatial distance |
| Dot Product | `Σ(a*b)` | (-∞, ∞) | Normalized vectors |

### Index Types

| Type | Complexity | Memory | Use Case |
|------|------------|--------|----------|
| Flat | O(n) | Low | Small datasets |
| HNSW | O(log n) | High | Production |
| IVF | O(√n) | Medium | Large datasets |

### Chunking Strategy

```
Document: [---------------long text---------------]
Chunks:   [----1----][----2----][----3----]
                  overlap    overlap
```

- Chunk size: 100-500 tokens typical
- Overlap: 10-20% of chunk size
- Preserves context across boundaries

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Index Type | Delta Sync | HNSW/Flat |
| Embedding | Managed | Self-hosted |
| Scaling | Auto | Manual |
| Integration | Unity Catalog | File-based |

## Lab Exercises

1. **Lab 3.1**: Build vector index from documents
2. **Lab 3.2**: Compare embedding models
3. **Lab 3.3**: Implement hybrid search

## Resources

- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [trueno-rag Documentation](https://docs.rs/trueno-rag)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
