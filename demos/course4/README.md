# Course 4: GenAI Engineering Demos

## Week Structure

| Week | Topic | Rust Demo | Databricks Demo |
|------|-------|-----------|-----------------|
| 1 | LLM & Prompts | llm-serving, prompt-engineering | Foundation Models, AI Functions |
| 2 | Vectors & RAG | vector-search, rag-pipeline | Vector Search, RAG |
| 3 | Fine-tune & Production | fine-tuning, production, capstone | Fine-tuning, AI Gateway |

## Running Demos

```bash
# Run specific Rust demo
cd week1/llm-serving && cargo run

# Run all Course 4 tests
for dir in week*/*/; do
  if [ -f "$dir/Cargo.toml" ]; then
    (cd "$dir" && cargo test)
  fi
done
```

## Demo Descriptions

### Week 1: Foundation Models & Prompt Engineering
- **llm-serving**: Tokenization, completion API, chat API, quantization
- **prompt-engineering**: Templates, few-shot, chain-of-thought, system prompts
- **databricks-llm**: Foundation Model APIs
- **databricks-prompts**: AI Functions and prompt patterns

### Week 2: Vector Search & RAG
- **vector-search**: Embeddings, similarity metrics, indexing, filtered search
- **rag-pipeline**: Chunking, retrieval, context injection, reranking
- **databricks-vector**: Databricks Vector Search
- **databricks-rag**: End-to-end RAG with Vector Search

### Week 3: Fine-tuning & Production
- **fine-tuning**: LoRA, QLoRA, training configuration, adapter management
- **production**: Rate limiting, logging, quality gates, A/B testing
- **capstone**: Complete GenAI application integrating all concepts
- **databricks-finetuning**: Foundation Model fine-tuning
- **databricks-production**: AI Gateway, inference tables, guardrails
