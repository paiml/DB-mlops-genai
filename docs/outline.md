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

# Course 3: MLOps Engineering on Databricks (3 Weeks)

## Subtitle
Build and Deploy ML Systems with MLflow, Feature Store, and Model Serving

## Learning Outcomes
1. Track experiments and manage model lifecycle with MLflow on Databricks
2. Build and serve features using Databricks Feature Store and SQL Warehouses
3. Register, version, and govern models with Unity Catalog
4. Deploy models for batch and real-time inference
5. Implement quality gates and monitoring for production ML

## Duration
~20 hours | 3 weeks | 6 labs | 3 quizzes | 1 capstone

---

### Week 1: Experiment Tracking & Feature Engineering

**Topics:**
- MLflow Architecture: Tracking, Registry, Projects
- Feature Store concepts and implementation
- Delta Lake for versioning

| Type | Title | Platform |
|------|-------|----------|
| Video | The Reproducibility Crisis | Concept |
| Video | MLflow Architecture | Databricks |
| Lab | Create Experiments in Databricks | Databricks |
| Lab | Build MLflow Client in Rust | Sovereign |
| Video | Feature Store Architecture | Databricks |
| Lab | Build Feature Pipeline | Both |
| Quiz | Tracking & Features | — |

**Sovereign AI Stack:** `reqwest`, `serde`, `alimentar`, `trueno`

---

### Week 2: Model Training & Serving

**Topics:**
- ML algorithms and AutoML
- Model registry with Unity Catalog
- Inference patterns (batch vs real-time)

| Type | Title | Platform |
|------|-------|----------|
| Video | ML Algorithms: Scratch to AutoML | Concept |
| Lab | Train Models with aprender | Sovereign |
| Lab | AutoML in Databricks | Databricks |
| Video | Model Registry & Signing | Databricks |
| Video | Inference Server Architecture | Sovereign |
| Lab | Serve Models with realizar | Sovereign |
| Lab | Deploy Endpoint in Databricks | Databricks |
| Quiz | Training & Serving | — |

**Sovereign AI Stack:** `aprender`, `pacha`, `realizar`

---

### Week 3: Production & Capstone

**Topics:**
- Quality gates and monitoring
- Orchestration with Databricks Workflows
- End-to-end ML pipeline

| Type | Title | Platform |
|------|-------|----------|
| Video | MLOps Maturity Model | Concept |
| Video | Databricks Workflows | Databricks |
| Lab | Build ML Pipeline with Jobs | Databricks |
| Video | Quality Gates with pmat | Sovereign |
| Lab | Enforce TDG Quality Score | Sovereign |
| Capstone | Fraud Detection Platform | Both |
| Quiz | Production MLOps | — |

**Sovereign AI Stack:** `batuta`, `pmat`, `renacer`

---

# Course 4: GenAI Engineering on Databricks (3 Weeks)

## Subtitle
Build LLM Applications with Foundation Models, Vector Search, and RAG

## Learning Outcomes
1. Serve and query foundation models on Databricks
2. Generate embeddings and build vector search indexes
3. Implement production RAG pipelines with hybrid retrieval
4. Fine-tune models with LoRA/QLoRA for domain adaptation
5. Deploy privacy-aware GenAI systems with proper governance

## Duration
~22 hours | 3 weeks | 6 labs | 3 quizzes | 1 capstone

---

### Week 1: Foundation Models & Prompt Engineering

**Topics:**
- LLM architectures and tokenization
- GGUF format and quantization
- Prompt engineering patterns

| Type | Title | Platform |
|------|-------|----------|
| Video | The GenAI Landscape | Concept |
| Video | Databricks Foundation Model APIs | Databricks |
| Lab | Query Models in Playground | Databricks |
| Lab | Serve Local Model with realizar | Sovereign |
| Video | Tokenization Deep Dive | Concept |
| Lab | Build BPE Tokenizer | Sovereign |
| Video | Prompt Engineering Patterns | Concept |
| Lab | Prompt Template Engine | Sovereign |
| Quiz | LLM & Prompts | — |

**Sovereign AI Stack:** `realizar`, `tokenizers`

---

### Week 2: Vector Search & RAG Pipelines

**Topics:**
- Embeddings and similarity search
- HNSW approximate nearest neighbors
- RAG architecture and chunking

| Type | Title | Platform |
|------|-------|----------|
| Video | What Are Embeddings? | Concept |
| Video | Databricks Vector Search | Databricks |
| Lab | Create Vector Search Index | Databricks |
| Lab | SIMD Vector Search with trueno | Sovereign |
| Video | RAG Architecture | Concept |
| Lab | Build RAG Pipeline | Databricks |
| Lab | End-to-End RAG with trueno-rag | Sovereign |
| Video | RAG Evaluation Metrics | Concept |
| Quiz | Vectors & RAG | — |

**Sovereign AI Stack:** `trueno`, `trueno-rag`, `alimentar`

---

### Week 3: Fine-Tuning & Production

**Topics:**
- LoRA/QLoRA fine-tuning
- Model security and governance
- Production deployment patterns

| Type | Title | Platform |
|------|-------|----------|
| Video | When to Fine-Tune vs RAG | Concept |
| Lab | Fine-Tune in Databricks | Databricks |
| Lab | Fine-Tune with entrenar (LoRA) | Sovereign |
| Video | Model Security & Signing | Sovereign |
| Video | Production Patterns | Concept |
| Lab | Production Deployment | Databricks |
| Lab | Configure with batuta | Sovereign |
| Capstone | Enterprise Knowledge Assistant | Both |
| Quiz | Fine-Tuning & Production | — |

**Sovereign AI Stack:** `entrenar`, `pacha`, `batuta`, `pmat`

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

---

## Language Distribution

| Language | Course 3 | Course 4 |
|----------|----------|----------|
| Python/SQL (Databricks) | ~80% | ~80% |
| Rust (Sovereign) | ~20% | ~20% |
