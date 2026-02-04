# Week 4: RAG Pipelines

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Understand RAG architecture and components
2. Build document ingestion and chunking pipelines
3. Implement retrieval-augmented generation
4. Evaluate RAG system quality

## Demos

### 1. RAG Pipeline (`rag-pipeline/`)

Rust implementation demonstrating RAG concepts.

**What it demonstrates:**
- Document chunking with overlap
- Vector-based retrieval
- Context-aware generation
- RAG evaluation metrics

**Run locally:**
```bash
cd rag-pipeline
cargo run
```

### 2. Databricks Notebook (`databricks/`)

RAG applications with Databricks Vector Search and Foundation Models.

**What it demonstrates:**
- Document processing pipeline
- Vector store operations
- RAG query execution
- Quality evaluation

**Run on Databricks:**
1. Import `rag_application.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### RAG Architecture

```
Query → Embed → Retrieve → Augment → Generate → Answer
              ↓
         Vector Store
              ↑
    Documents → Chunk → Embed → Index
```

### Chunking Strategy

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| Chunk size | 100-500 tokens | Balance context vs specificity |
| Overlap | 10-20% | Preserve boundary context |
| Separator | Sentence/Paragraph | Semantic boundaries |

### RAG Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Retrieval Precision | Relevance of retrieved chunks | > 0.7 |
| Context Relevance | Query-context alignment | > 0.6 |
| Answer Faithfulness | Grounded in context | > 0.8 |

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Vector Store | Vector Search | trueno-rag |
| Embedding | Managed models | Self-hosted |
| Generation | Foundation Models | realizar |
| Scaling | Auto | Manual |

## Lab Exercises

1. **Lab 4.1**: Build document ingestion pipeline
2. **Lab 4.2**: Implement RAG with evaluation
3. **Lab 4.3**: Compare chunking strategies

## Resources

- [Databricks RAG Documentation](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html)
- [trueno-rag Documentation](https://docs.rs/trueno-rag)
