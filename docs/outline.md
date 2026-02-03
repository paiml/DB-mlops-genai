# MLOps & GenAI Engineering with Databricks

**Courses 3 & 4 of the Databricks Specialization on Coursera**

Platform: Databricks Free Edition | Comparison Layer: Sovereign AI Stack (Rust)

---

## Specialization Context

| Course | Title | Status |
|--------|-------|--------|
| 1 | Lakehouse Fundamentals | Prerequisite |
| 2 | Data Engineering on Databricks | Prerequisite |
| **3** | **MLOps Engineering** | This course |
| **4** | **GenAI Engineering** | This course |
| 5 | Production MLOps & Governance | Follow-on |

---

## Design Philosophy

**Dual-layer pedagogy:**
1. **Databricks layer** — Hands-on with MLflow, Feature Store, Model Serving, Vector Search, Foundation Models
2. **Sovereign AI Stack layer** — Build the same concepts from scratch in Rust to understand what platforms abstract

**Why both?**
- Practitioners need to *use* Databricks effectively
- Engineers need to *understand* what's underneath
- "Understand by building" creates deeper retention

---

## Databricks Free Edition Features Used

### Course 3: MLOps
- Experiments (MLflow Tracking)
- Catalog (Unity Catalog for model registry)
- Jobs & Pipelines (orchestration)
- SQL Warehouses (feature computation)
- Playground (model testing)

### Course 4: GenAI
- Playground (Foundation Models)
- Vector Search (via Catalog)
- Genie (AI/BI demo)
- Experiments (evaluation tracking)
- Jobs & Pipelines (RAG orchestration)

---

## Sovereign AI Stack Components

```
┌──────────────────────────────────────────────────────────────────┐
│                   batuta (Orchestration)                         │
│              Privacy Tiers · CLI · Stack Coordination            │
├───────────────────┬──────────────────┬───────────────────────────┤
│  realizar         │  entrenar        │      pacha                │
│  (Inference)      │  (Training)      │   (Model Registry)        │
│  GGUF/SafeTensors │  autograd/LoRA   │  Sign/Encrypt/Lineage     │
├───────────────────┴──────────────────┴───────────────────────────┤
│                    aprender                                       │
│         ML Algorithms: regression, trees, clustering              │
├──────────────────────────────────────────────────────────────────┤
│                     trueno                                        │
│         SIMD/GPU Compute (AVX2/AVX-512/NEON, wgpu)               │
├──────────────────────────────────────────────────────────────────┤
│  trueno-rag      │ trueno-db       │ alimentar     │ pmat        │
│  BM25 + Vector   │ GPU Analytics   │ Arrow/Parquet │ Quality     │
└──────────────────┴─────────────────┴───────────────┴─────────────┘
```

---

# Course 3: MLOps Engineering on Databricks

## Subtitle
Build and Deploy ML Systems with MLflow, Feature Store, and Model Serving

## Description
Master the complete MLOps lifecycle on Databricks: experiment tracking with MLflow, feature engineering with Feature Store, model management with Unity Catalog, and deployment with Model Serving. Understand each component deeply by building equivalent systems from scratch with the Sovereign AI Stack.

## Learning Outcomes
1. Track experiments and manage model lifecycle with MLflow on Databricks
2. Build and serve features using Databricks Feature Store and SQL Warehouses
3. Register, version, and govern models with Unity Catalog
4. Deploy models for batch and real-time inference
5. Implement quality gates and monitoring for production ML

## Duration
~30 hours | 38 videos | 12 labs | 5 quizzes | 1 capstone

---

### Week 1: Experiment Tracking with MLflow

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 1.1 | Video | The Reproducibility Crisis | Concept | 8 min |
| 1.2 | Video | MLflow Architecture: Tracking, Registry, Projects | Databricks | 10 min |
| 1.3 | Lab | Create Experiments in Databricks | Databricks | 30 min |
| 1.4 | Video | MLflow REST Protocol Deep Dive | Concept | 10 min |
| 1.5 | Lab | Build MLflow Client in Rust | Sovereign | 40 min |
| 1.6 | Video | Autologging and Framework Integration | Databricks | 8 min |
| 1.7 | Video | Artifact Storage: DBFS, S3, Unity Catalog | Databricks | 8 min |
| 1.8 | Lab | Compare: Databricks MLflow vs Rust Client | Both | 25 min |
| 1.9 | Quiz | Experiment Tracking Fundamentals | — | 15 min |

**Sovereign AI Stack:** `reqwest`, `serde` for HTTP client; `pacha` concepts for artifact storage

---

### Week 2: Feature Engineering

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 2.1 | Video | What is a Feature Store? | Concept | 10 min |
| 2.2 | Video | Databricks Feature Store Architecture | Databricks | 10 min |
| 2.3 | Lab | Create Feature Tables in Unity Catalog | Databricks | 35 min |
| 2.4 | Video | SIMD-Accelerated Feature Computation | Sovereign | 10 min |
| 2.5 | Lab | Build Feature Pipeline with trueno | Sovereign | 40 min |
| 2.6 | Video | Point-in-Time Joins and Data Leakage | Concept | 8 min |
| 2.7 | Lab | Feature Lookup and Online Serving | Databricks | 30 min |
| 2.8 | Video | Delta Lake for Feature Versioning | Databricks | 8 min |
| 2.9 | Quiz | Feature Engineering Systems | — | 15 min |

**Sovereign AI Stack:** `alimentar` (zero-copy Parquet), `trueno` (SIMD), `delta-rs`

---

### Week 3: Model Training and Registry

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 3.1 | Video | ML Algorithms: From Scratch to AutoML | Concept | 10 min |
| 3.2 | Lab | Train Models with aprender | Sovereign | 40 min |
| 3.3 | Video | Databricks AutoML | Databricks | 10 min |
| 3.4 | Lab | AutoML Experiment in Databricks | Databricks | 30 min |
| 3.5 | Video | Model Registry with Unity Catalog | Databricks | 10 min |
| 3.6 | Video | Model Signing and Security | Sovereign | 8 min |
| 3.7 | Lab | Register and Sign Models with pacha | Sovereign | 35 min |
| 3.8 | Video | Model Lineage and Governance | Databricks | 8 min |
| 3.9 | Quiz | Training and Registry | — | 15 min |

**Sovereign AI Stack:** `aprender` (ML algorithms), `pacha` (Ed25519 signing, BLAKE3 hashing)

---

### Week 4: Model Serving and Inference

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 4.1 | Video | Inference Patterns: Batch vs Real-time | Concept | 10 min |
| 4.2 | Video | Databricks Model Serving | Databricks | 10 min |
| 4.3 | Lab | Deploy Endpoint in Databricks | Databricks | 35 min |
| 4.4 | Video | Build Inference Server from Scratch | Sovereign | 10 min |
| 4.5 | Lab | Serve Models with realizar | Sovereign | 35 min |
| 4.6 | Video | Batch Inference with Spark | Databricks | 8 min |
| 4.7 | Lab | Batch Scoring Pipeline | Databricks | 30 min |
| 4.8 | Video | Latency Benchmarking | Both | 8 min |
| 4.9 | Quiz | Inference Systems | — | 15 min |

**Sovereign AI Stack:** `realizar` (inference server), `repartir` (distributed batch)

---

### Week 5: Production Quality and Orchestration

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 5.1 | Video | MLOps Maturity Model | Concept | 8 min |
| 5.2 | Video | Databricks Workflows for ML | Databricks | 10 min |
| 5.3 | Lab | Build ML Pipeline with Jobs | Databricks | 35 min |
| 5.4 | Video | Quality Gates with pmat | Sovereign | 8 min |
| 5.5 | Lab | Enforce TDG Quality Score | Sovereign | 25 min |
| 5.6 | Video | Monitoring and Drift Detection | Databricks | 10 min |
| 5.7 | Video | batuta Orchestration | Sovereign | 8 min |
| 5.8 | Quiz | Production MLOps | — | 15 min |

**Sovereign AI Stack:** `batuta` (orchestration), `pmat` (quality), `renacer` (syscall tracing)

---

### Week 6: Capstone — Fraud Detection Platform

**Duration:** ~15 hours

**Deliverables:**
1. Feature pipeline (Databricks Feature Store + trueno comparison)
2. Training pipeline (AutoML + aprender comparison)
3. Model registry (Unity Catalog + pacha comparison)
4. Inference endpoint (Model Serving + realizar comparison)
5. Quality gates (Lakehouse Monitoring + pmat)
6. Orchestrated workflow (Databricks Jobs + batuta)

---

# Course 4: GenAI Engineering on Databricks

## Subtitle
Build LLM Applications with Foundation Models, Vector Search, and RAG

## Description
Construct production GenAI systems on Databricks: serve foundation models, implement vector search for semantic retrieval, build RAG pipelines, and fine-tune models for domain adaptation. Understand the internals by building equivalent systems with the Sovereign AI Stack.

## Learning Outcomes
1. Serve and query foundation models on Databricks
2. Generate embeddings and build vector search indexes
3. Implement production RAG pipelines with hybrid retrieval
4. Fine-tune models with LoRA/QLoRA for domain adaptation
5. Deploy privacy-aware GenAI systems with proper governance

## Duration
~34 hours | 40 videos | 12 labs | 5 quizzes | 1 capstone

---

### Week 1: Foundation Models and LLM Serving

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 1.1 | Video | The GenAI Landscape | Concept | 10 min |
| 1.2 | Video | Databricks Foundation Model APIs | Databricks | 10 min |
| 1.3 | Lab | Query Models in Playground | Databricks | 25 min |
| 1.4 | Video | GGUF Format and Quantization | Sovereign | 10 min |
| 1.5 | Lab | Serve Local Model with realizar | Sovereign | 35 min |
| 1.6 | Video | Tokenization Deep Dive | Concept | 10 min |
| 1.7 | Lab | Build BPE Tokenizer | Sovereign | 30 min |
| 1.8 | Video | External Models and AI Gateway | Databricks | 8 min |
| 1.9 | Quiz | LLM Serving Fundamentals | — | 15 min |

**Sovereign AI Stack:** `realizar` (GGUF inference), `tokenizers` crate

---

### Week 2: Prompt Engineering and Structured Output

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 2.1 | Video | Prompt Engineering Patterns | Concept | 10 min |
| 2.2 | Lab | Effective Prompts in Databricks | Databricks | 30 min |
| 2.3 | Video | Structured Output Extraction | Concept | 8 min |
| 2.4 | Lab | JSON Mode and Schema Enforcement | Databricks | 30 min |
| 2.5 | Video | Build Type-Safe Prompt Templates | Sovereign | 10 min |
| 2.6 | Lab | Rust Prompt Template Engine | Sovereign | 35 min |
| 2.7 | Video | Privacy Tiers: Sovereign/Private/Standard | Sovereign | 10 min |
| 2.8 | Lab | Multi-Backend Gateway with batuta | Sovereign | 35 min |
| 2.9 | Quiz | Prompt Engineering | — | 15 min |

**Sovereign AI Stack:** `batuta` (privacy tiers, backend routing), `serde`

---

### Week 3: Embeddings and Vector Search

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

**Sovereign AI Stack:** `trueno` (SIMD), `trueno-rag` (BM25 + HNSW), `trueno-db`

---

### Week 4: RAG Pipelines

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

**Sovereign AI Stack:** `trueno-rag` (full pipeline), `alimentar` (chunking), `pmat` (evaluation)

---

### Week 5: Fine-Tuning and Model Security

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 5.1 | Video | When to Fine-Tune vs RAG | Concept | 10 min |
| 5.2 | Video | Databricks Fine-Tuning | Databricks | 10 min |
| 5.3 | Lab | Fine-Tune in Databricks | Databricks | 40 min |
| 5.4 | Video | LoRA/QLoRA from Scratch | Sovereign | 10 min |
| 5.5 | Lab | Fine-Tune with entrenar | Sovereign | 45 min |
| 5.6 | Video | Model Encryption and Signing | Sovereign | 10 min |
| 5.7 | Lab | Secure Model Pipeline with pacha | Sovereign | 35 min |
| 5.8 | Video | EU AI Act and Governance | Concept | 8 min |
| 5.9 | Quiz | Fine-Tuning and Security | — | 15 min |

**Sovereign AI Stack:** `entrenar` (LoRA/QLoRA), `pacha` (ChaCha20-Poly1305 encryption)

---

### Week 6: Production Deployment

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 6.1 | Video | GenAI Production Patterns | Concept | 10 min |
| 6.2 | Video | Databricks Model Serving for LLMs | Databricks | 10 min |
| 6.3 | Lab | Deploy GenAI Endpoint | Databricks | 35 min |
| 6.4 | Video | Cost Control and Circuit Breakers | Sovereign | 8 min |
| 6.5 | Lab | Production Config with batuta | Sovereign | 30 min |
| 6.6 | Quiz | Production Deployment | — | 15 min |

**Sovereign AI Stack:** `batuta` (cost limits, observability), `renacer` (audit trails)

---

### Week 7: Capstone — Enterprise Knowledge Assistant

**Duration:** ~18 hours

**Deliverables:**
1. Document ingestion (Databricks + alimentar)
2. Embedding pipeline (Vector Search + trueno)
3. Hybrid retrieval (Databricks RAG + trueno-rag)
4. Generation layer (Foundation Models + realizar)
5. Fine-tuned model (Databricks + entrenar)
6. Security package (Unity Catalog + pacha)
7. Production deployment (Model Serving + batuta)
8. Evaluation suite (pmat quality gates)

---

## Certification Alignment

| Skill | Databricks Cert | Course |
|-------|-----------------|--------|
| MLflow Tracking & Registry | ML Associate | 3 |
| Feature Engineering | ML Associate | 3 |
| Model Serving | ML Associate | 3 |
| Foundation Model APIs | GenAI Engineer | 4 |
| Vector Search & RAG | GenAI Engineer | 4 |
| Fine-Tuning | GenAI Engineer | 4 |
| Workflows & Governance | ML Professional | 3, 4 |

---

## batuta Cookbook Recipes Used

| Recipe | Course | Lab |
|--------|--------|-----|
| `ml-serving` | 3 | Week 4: Serve with realizar |
| `registry-pacha` | 3 | Week 3: Sign models |
| `data-alimentar` | 3 | Week 2: Feature pipeline |
| `rag-pipeline` | 4 | Week 4: End-to-end RAG |
| `rag-semantic-search` | 4 | Week 3: HNSW index |
| `training-lora` | 4 | Week 5: Fine-tune |

---

## Language Distribution

| Language | Course 3 | Course 4 |
|----------|----------|----------|
| Python/SQL (Databricks) | ~55% | ~50% |
| Rust (Sovereign) | ~35% | ~40% |
| Conceptual | ~10% | ~10% |
