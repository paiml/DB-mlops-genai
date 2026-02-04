# Falsification Specification: MLOps & GenAI Courses

This specification defines testable claims for Popperian falsification analysis.
Each claim can be falsified through code analysis and test execution.

## Course 3: MLOps Engineering

### Section: Experiment Tracking

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C3-1.1 | MLflow client must validate experiment IDs | `experiment_id.*[0-9]+` | high |
| C3-1.2 | Run tracking must include timestamps | `started_at\|timestamp\|created` | medium |
| C3-1.3 | Metrics must be numeric values | `log_metric.*f64\|f32\|float` | high |
| C3-1.4 | Parameters must be string key-value pairs | `log_param.*String\|str` | medium |

### Section: Feature Engineering

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C3-2.1 | StandardScaler must compute mean and std | `mean\|std\|standard_dev` | high |
| C3-2.2 | MinMaxScaler must track min and max values | `min_val\|max_val\|min\|max` | high |
| C3-2.3 | Train/test split must preserve ratios | `test_ratio\|train_size\|split` | medium |
| C3-2.4 | Feature pipelines must handle missing values | `missing\|null\|None\|NaN` | medium |

### Section: Model Training

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C3-3.1 | Training must track loss over epochs | `loss\|epoch\|iteration` | high |
| C3-3.2 | Hyperparameters must be configurable | `learning_rate\|lr\|batch_size` | medium |
| C3-3.3 | Model checkpoints must be serializable | `save\|load\|serialize\|checkpoint` | high |
| C3-3.4 | Validation must prevent overfitting | `validation\|val_loss\|early_stop` | medium |

### Section: Inference Server

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C3-4.1 | Circuit breaker must track failure counts | `failure_count\|failures\|threshold` | critical |
| C3-4.2 | Health checks must return status codes | `health\|status\|ready\|live` | high |
| C3-4.3 | Request routing must handle timeouts | `timeout\|duration\|elapsed` | high |
| C3-4.4 | Batch inference must aggregate results | `batch\|bulk\|aggregate` | medium |

### Section: Quality Gates

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C3-5.1 | Quality scores must be bounded 0-100 | `score.*0.*100\|percentage\|percent` | high |
| C3-5.2 | Thresholds must be configurable | `threshold\|min_score\|gate` | medium |
| C3-5.3 | Trends must track historical data | `trend\|history\|previous` | medium |
| C3-5.4 | Reports must include pass/fail status | `pass\|fail\|status\|result` | high |

## Course 4: GenAI Engineering

### Section: LLM Serving

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-1.1 | Tokenizer must handle special tokens | `special_token\|bos\|eos\|pad` | critical |
| C4-1.2 | Token IDs must be non-negative integers | `token_id.*u32\|usize\|index` | high |
| C4-1.3 | Vocabulary must map tokens to IDs | `vocab\|token_to_id\|encode` | high |
| C4-1.4 | Decoding must reverse encoding | `decode\|id_to_token\|detokenize` | high |

### Section: Prompt Engineering

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-2.1 | Templates must validate variable names | `variable\|placeholder\|\{\{.*\}\}` | high |
| C4-2.2 | System prompts must be distinct from user | `system\|user\|assistant\|role` | medium |
| C4-2.3 | Few-shot examples must be formatted | `example\|few_shot\|shot` | medium |
| C4-2.4 | Chain-of-thought must include reasoning | `reasoning\|step\|think\|chain` | medium |

### Section: Vector Search

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-3.1 | Embeddings must be normalized for cosine | `normalize\|unit_vector\|norm` | critical |
| C4-3.2 | Similarity scores must be bounded | `similarity.*-?1.*1\|cosine\|dot` | high |
| C4-3.3 | HNSW must maintain graph connectivity | `neighbor\|edge\|layer\|graph` | high |
| C4-3.4 | Metadata filtering must support predicates | `filter\|predicate\|where\|metadata` | medium |

### Section: RAG Pipeline

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-4.1 | Chunking must respect overlap settings | `overlap\|stride\|chunk_size` | high |
| C4-4.2 | Retrieval must return ranked results | `rank\|score\|top_k\|retrieve` | high |
| C4-4.3 | Context injection must preserve order | `context\|inject\|augment` | medium |
| C4-4.4 | Reranking must improve relevance | `rerank\|cross_encoder\|relevance` | medium |
| C4-4.5 | Hybrid search must combine BM25 and dense | `hybrid\|bm25\|sparse\|dense` | medium |

### Section: Fine-Tuning

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-5.1 | LoRA rank must be positive integer | `rank.*[1-9]\|lora_r\|low_rank` | critical |
| C4-5.2 | Alpha scaling must be configurable | `alpha\|lora_alpha\|scaling` | high |
| C4-5.3 | Target modules must be specified | `target_module\|q_proj\|v_proj` | high |
| C4-5.4 | QLoRA must reduce memory usage | `quantiz\|4bit\|8bit\|bnb` | high |
| C4-5.5 | Adapters must be mergeable | `merge\|adapter\|combine` | medium |

### Section: Production

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| C4-6.1 | Rate limiting must track request counts | `rate_limit\|requests_per\|quota` | critical |
| C4-6.2 | Quality gates must reject low scores | `quality_gate\|min_quality\|reject` | high |
| C4-6.3 | A/B testing must split traffic correctly | `a_b_test\|variant\|split\|traffic` | high |
| C4-6.4 | Cost tracking must log token usage | `cost\|token_count\|usage\|billing` | medium |
| C4-6.5 | Logging must include request IDs | `request_id\|trace_id\|correlation` | medium |

## Test Coverage Requirements

### Section: Minimum Coverage

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| TC-1 | All demos must have tests | `#\[test\]\|#\[cfg\(test\)\]` | critical |
| TC-2 | Test functions must have assertions | `assert\|expect\|should` | high |
| TC-3 | Edge cases must be tested | `edge\|boundary\|empty\|zero` | medium |
| TC-4 | Error paths must be tested | `error\|err\|fail\|invalid` | high |

## Security Requirements

### Section: Input Validation

| ID | Invariant | Pattern | Severity |
|----|-----------|---------|----------|
| SEC-1 | User input must be sanitized | `sanitize\|escape\|validate` | critical |
| SEC-2 | File paths must be validated | `path.*valid\|canonicalize` | high |
| SEC-3 | Numeric inputs must be bounded | `clamp\|min\|max\|bound` | medium |
