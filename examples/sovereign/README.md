# Sovereign AI Stack Examples

Pure Rust examples demonstrating the Sovereign AI Stack components. These examples complement the main course demos by providing standalone implementations of key concepts.

## Structure

```
sovereign/
├── course3/                         # MLOps Engineering
│   ├── week1-aprender-features/     # Feature engineering
│   ├── week2-realizar-serving/      # Model serving
│   └── week3-pmat-quality/          # Quality gates
└── course4/                         # GenAI Engineering
    ├── week1-realizar-llm/          # LLM inference
    ├── week2-trueno-rag/            # RAG pipelines
    └── week3-entrenar-lora/         # Fine-tuning
```

## Course 3: MLOps Engineering

### Week 1: Feature Engineering with aprender
- StandardScaler (Z-score normalization)
- MinMaxScaler (scale to [0, 1])
- LabelEncoder (categorical encoding)
- train_test_split

```bash
cd course3/week1-aprender-features && cargo run
```

### Week 2: Model Serving with realizar
- Model metadata and schema
- Inference request/response
- Circuit breaker pattern
- Health checks

```bash
cd course3/week2-realizar-serving && cargo run
```

### Week 3: Quality Gates with pmat
- Code metrics collection
- TDG (Technical Debt Grade) scoring
- Quality gate evaluation
- Metric trend analysis

```bash
cd course3/week3-pmat-quality && cargo run
```

## Course 4: GenAI Engineering

### Week 1: LLM Inference with realizar
- GGUF model metadata
- Tokenization (BPE-style)
- Text generation
- KV cache management
- Prompt templates

```bash
cd course4/week1-realizar-llm && cargo run
```

### Week 2: RAG Pipeline with trueno-rag
- Document chunking with overlap
- Dense embeddings
- Vector similarity search
- Cross-encoder reranking
- Hybrid search (vector + keyword)

```bash
cd course4/week2-trueno-rag && cargo run
```

### Week 3: Fine-Tuning with entrenar
- LoRA configuration (rank, alpha, targets)
- QLoRA (4-bit quantization)
- Training data formatting
- Learning rate scheduling
- Adapter merging

```bash
cd course4/week3-entrenar-lora && cargo run
```

## Running All Examples

```bash
# Course 3
for dir in course3/week*/; do
  echo "=== Running $dir ==="
  (cd "$dir" && cargo run)
done

# Course 4
for dir in course4/week*/; do
  echo "=== Running $dir ==="
  (cd "$dir" && cargo run)
done
```

## Running Tests

```bash
# All examples
for dir in course*/week*/; do
  echo "=== Testing $dir ==="
  (cd "$dir" && cargo test)
done
```

## Mapping to Databricks

| Sovereign Stack | Databricks Equivalent |
|-----------------|----------------------|
| aprender        | Feature Store, MLflow |
| realizar        | Model Serving |
| pmat            | MLflow, Workflows |
| trueno-rag      | Vector Search, RAG |
| entrenar        | Fine-tuning APIs |
