# Falsification Checklist

This document contains testable claims about the MLOps & GenAI Engineering courses.
Each claim can be falsified through the tests and demonstrations in this repository.

## Course 3: MLOps Engineering

### Week 1: Experiment Tracking & Feature Engineering

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C3-1.1 | MLflow REST API can track experiments via HTTP | `demos/course3/week1/experiment-tracking` | ✅ Verified |
| C3-1.2 | Feature pipelines can transform raw data into model inputs | `demos/course3/week1/feature-pipeline` | ✅ Verified |
| C3-1.3 | Rust can implement type-safe MLflow client | `cargo test` in experiment-tracking | ✅ 28 tests pass |

### Week 2: Model Training & Serving

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C3-2.1 | Models can be trained with configurable hyperparameters | `demos/course3/week2/model-training` | ✅ Verified |
| C3-2.2 | Inference servers can implement circuit breaker pattern | `demos/course3/week2/inference-server` | ✅ Verified |
| C3-2.3 | Health checks can detect service degradation | `cargo test` in inference-server | ✅ 27 tests pass |

### Week 3: Production & Capstone

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C3-3.1 | Quality gates can enforce TDG scoring thresholds | `demos/course3/week3/quality-gates` | ✅ Verified |
| C3-3.2 | Capstone integrates all MLOps components | `demos/course3/week3/capstone` | ✅ Verified |
| C3-3.3 | Quality metrics can track trends over time | `cargo test` in quality-gates | ✅ 32 tests pass |

## Course 4: GenAI Engineering

### Week 1: LLM & Prompt Engineering

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C4-1.1 | LLM tokenization handles special tokens correctly | `demos/course4/week1/llm-serving` | ✅ Verified |
| C4-1.2 | Prompt templates can substitute variables safely | `demos/course4/week1/prompt-engineering` | ✅ Verified |
| C4-1.3 | Chain-of-thought prompting produces reasoning steps | `cargo test` in prompt-engineering | ✅ 12 tests pass |
| C4-1.4 | Few-shot examples improve classification | `cargo test` in prompt-engineering | ✅ Verified |

### Week 2: Vector Search & RAG

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C4-2.1 | Embeddings can be normalized for cosine similarity | `demos/course4/week2/vector-search` | ✅ Verified |
| C4-2.2 | Vector indices support filtered search | `demos/course4/week2/vector-search` | ✅ Verified |
| C4-2.3 | RAG pipelines chunk documents with overlap | `demos/course4/week2/rag-pipeline` | ✅ Verified |
| C4-2.4 | Reranking improves retrieval relevance | `cargo test` in rag-pipeline | ✅ 13 tests pass |
| C4-2.5 | Hybrid search combines keyword and semantic | `cargo test` in rag-pipeline | ✅ Verified |

### Week 3: Fine-Tuning & Production

| ID | Claim | Test | Status |
|----|-------|------|--------|
| C4-3.1 | LoRA reduces trainable parameters significantly | `demos/course4/week3/fine-tuning` | ✅ Verified |
| C4-3.2 | QLoRA provides additional memory savings | `demos/course4/week3/fine-tuning` | ✅ Verified |
| C4-3.3 | Rate limiting prevents request overload | `demos/course4/week3/production` | ✅ Verified |
| C4-3.4 | Quality gates can reject low-quality responses | `demos/course4/week3/production` | ✅ Verified |
| C4-3.5 | A/B testing splits traffic correctly | `cargo test` in production | ✅ 11 tests pass |
| C4-3.6 | Capstone integrates all GenAI components | `demos/course4/week3/capstone` | ✅ 13 tests pass |

## Sovereign AI Stack Examples

### Course 3 Examples

| ID | Claim | Test | Status |
|----|-------|------|--------|
| S3-1.1 | StandardScaler normalizes features to zero mean | `examples/sovereign/course3/week1-aprender-features` | ✅ 28 tests pass |
| S3-2.1 | Circuit breaker opens after failure threshold | `examples/sovereign/course3/week2-realizar-serving` | ✅ 27 tests pass |
| S3-3.1 | PMAT scoring enforces quality thresholds | `examples/sovereign/course3/week3-pmat-quality` | ✅ 32 tests pass |

### Course 4 Examples

| ID | Claim | Test | Status |
|----|-------|------|--------|
| S4-1.1 | GGUF metadata extraction works correctly | `examples/sovereign/course4/week1-realizar-llm` | ✅ 31 tests pass |
| S4-2.1 | RAG chunker handles overlap correctly | `examples/sovereign/course4/week2-trueno-rag` | ✅ 30 tests pass |
| S4-3.1 | LoRA config calculates scaling correctly | `examples/sovereign/course4/week3-entrenar-lora` | ✅ 39 tests pass |

## Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| Course 3 Demos | 87 | ✅ All Pass |
| Course 4 Demos | 86 | ✅ All Pass |
| Sovereign Examples Course 3 | 87 | ✅ All Pass |
| Sovereign Examples Course 4 | 100 | ✅ All Pass |
| **Total** | **360** | ✅ All Pass |

## How to Verify

```bash
# Run all tests
make check

# Run specific course tests
make test-course3
make test-course4

# Run Rust tests for demos
for dir in demos/course*/week*/*/; do
  if [ -f "$dir/Cargo.toml" ]; then
    (cd "$dir" && cargo test)
  fi
done

# Run Sovereign examples tests
for dir in examples/sovereign/course*/*/; do
  (cd "$dir" && cargo test)
done
```

## Falsification Criteria

A claim is considered **falsified** if:
1. The associated test fails
2. The demo produces incorrect output
3. The claimed behavior is not observable

A claim is considered **verified** if:
1. All associated tests pass
2. The demo produces expected output
3. The claimed behavior is observable and correct
