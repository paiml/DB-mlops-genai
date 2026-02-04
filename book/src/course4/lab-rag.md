# Lab: RAG Pipeline

Build an end-to-end RAG system with chunking, retrieval, and generation.

## Objectives

- Implement document chunking
- Build retrieval pipeline
- Generate contextual answers

## Demo Code

See [`demos/course4/week4/rag-pipeline/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course4/week4/rag-pipeline)

## Lab Exercise

See [`labs/course4/week4/lab_4_7_rag.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course4/week4)

## Key Implementation

```rust
pub struct RagPipeline {
    chunker: TextChunker,
    vector_store: VectorStore,
    generator: Generator,
}

impl RagPipeline {
    pub fn query(&self, question: &str) -> RagResponse {
        // 1. Embed query
        let query_embedding = self.embed(question);

        // 2. Retrieve relevant chunks
        let results = self.vector_store.search(&query_embedding, 3);

        // 3. Build context
        let context = results.iter()
            .map(|r| r.chunk.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        // 4. Generate answer
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, question
        );
        let answer = self.generator.generate(&prompt);

        RagResponse { answer, sources: results }
    }
}
```
