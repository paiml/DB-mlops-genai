# Systems ML & GenAI Engineering with the Sovereign AI Stack

**Courses 3 & 4 — Redesigned for Approach 2: "Understand by Building"**

Rust-first • Sovereign AI Stack • Minimal Python • Databricks as comparison layer

---

## Design Philosophy

**Core principle:** Every managed platform feature maps to something you can build, benchmark, and understand in Rust. Learners don't just call `mlflow.log_metric()` — they understand the HTTP protocol underneath by implementing a client in Rust. They don't just click "Create Vector Search Index" — they implement HNSW from SIMD primitives in `trueno`.

**The Sovereign AI Stack** replaces the typical Python ML ecosystem:

```
┌──────────────────────────────────────────────────────────────────┐
│                   batuta v0.4 (Orchestration)                    │
│              Privacy Tiers · CLI · Stack Coordination            │
├───────────────────┬──────────────────┬───────────────────────────┤
│  realizar v0.5    │  entrenar v0.5   │      pacha v0.2           │
│  (Inference)      │  (Training)      │   (Model Registry)        │
│  GGUF/SafeTensors │  autograd/LoRA   │  Sign/Encrypt/Lineage     │
├───────────────────┴──────────────────┴───────────────────────────┤
│                    aprender v0.24                                 │
│         ML Algorithms: regression, trees, clustering, NAS        │
├──────────────────────────────────────────────────────────────────┤
│                     trueno v0.11                                 │
│         SIMD/GPU Compute (AVX2/AVX-512/NEON, wgpu)              │
├──────────────────────────────────────────────────────────────────┤
│  trueno-rag 0.1  │ trueno-db 0.3  │ alimentar 0.2 │ pmat 2.213 │
│  BM25 + Vector   │ GPU Analytics  │ Arrow/Parquet │ Quality    │
└──────────────────┴────────────────┴───────────────┴─────────────┘
```

**Python budget per course:** ≤5 videos (≤15% of content). Used only for:
- Databricks SDK/notebook demos that require it
- Brief "here's the managed equivalent" comparison moments
- MLflow autologging (it's Python-only)

**Databricks role:** Comparison layer, not primary platform. Every module ends with a "Managed Platform Parallel" segment showing the Databricks equivalent of what was just built.

---

## Sovereign AI Stack Component Map

| Stack Component | Course 3 Role | Course 4 Role |
|---|---|---|
| **trueno** | SIMD feature computation, benchmarking | Embedding math, similarity search, SIMD dot products |
| **aprender** | ML algorithms (regression, trees, clustering) | — |
| **realizar** | ONNX/GGUF inference serving | Foundation model inference, LLM serving |
| **entrenar** | — | LoRA/QLoRA fine-tuning |
| **pacha** | Model signing, versioning, registry | Model encryption, sovereign distribution |
| **batuta** | Orchestration, privacy tiers | Privacy tiers, sovereign deployment |
| **alimentar** | Zero-copy Parquet/Arrow loading | Document loading for RAG |
| **trueno-db** | — | GPU-accelerated retrieval |
| **trueno-rag** | — | BM25+vector RAG pipeline |
| **pmat** | Quality gates, TDG scoring | Quality gates for RAG evaluation |
| **renacer** | Syscall tracing for validation | Inference audit trails |
| **depyler** | — | Transpile LangChain pattern to Rust (bonus) |

---

# Course 3: Systems ML Engineering

## Subtitle
Build ML Infrastructure from First Principles with the Sovereign AI Stack

## Course Description
Understand ML systems by building them. Construct an experiment tracking client, feature computation engine, model registry, and inference server entirely in Rust using the Sovereign AI Stack. Then compare each component against its managed Databricks equivalent (MLflow, Feature Store, AutoML, Model Serving) to understand what platforms abstract away — and what they can't.

## Prerequisites
- Rust basics (ownership, borrowing, traits, cargo)
- Courses 1–2 of the specialization (Lakehouse + Data Engineering)
- Conceptual familiarity with ML (what a model is, training vs inference)

## Learning Outcomes
1. Implement an experiment tracking client by building against the MLflow REST protocol in Rust
2. Build SIMD-accelerated feature computation pipelines using trueno and alimentar
3. Train models with aprender and manage them with pacha's signed registry
4. Deploy inference services with realizar and benchmark against Databricks Model Serving
5. Enforce production quality with pmat and renacer validation

## Total: ~38 videos + 12 labs + 5 readings + 5 quizzes + 1 capstone

---

### Week 1: The ML Tracking Protocol (8 videos)

**Theme:** Understand experiment tracking by implementing an MLflow REST client in Rust.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 1.1 | Video | Why Track Experiments? | — | The reproducibility crisis: parameters, metrics, artifacts, lineage | 10 min |
| 1.2 | Video | MLflow REST Protocol Deep Dive | — | Endpoints, data model (experiments → runs → params/metrics/artifacts), JSON schemas | 10 min |
| 1.3 | Video | Rust HTTP Client Foundations | `reqwest`, `serde` | Designing a type-safe MLflow client: structs for Run, Experiment, Metric | 10 min |
| 1.4 | Lab | Build MLflow Rust Client (Part 1) | `reqwest`, `serde` | Implement create_experiment, create_run, log_param, log_metric against local MLflow server | 40 min |
| 1.5 | Video | Artifact Storage Internals | `pacha` concepts | How MLflow stores artifacts (filesystem, S3, DBFS) — relate to pacha's content-addressed storage | 8 min |
| 1.6 | Lab | Build MLflow Rust Client (Part 2) | `reqwest`, `tokio` | Add log_artifact (upload files), search_runs (query API), batch logging with async | 35 min |
| 1.7 | Video | Autologging — The Python Exception | Python (brief) | 5-min demo: `mlflow.autolog()` in a Databricks notebook, how it hooks framework callbacks. Why this is Python-only and what Rust alternatives look like | 8 min |
| 1.8 | Video | Managed Platform Parallel: MLflow on Databricks | — | Managed MLflow vs self-hosted, Unity Catalog integration, Databricks-specific autologging | 8 min |
| 1.9 | Reading | MLflow Tracking Server Architecture | — | Deep dive: backend store, artifact store, tracking URI resolution | 12 min |
| 1.10 | Quiz | Experiment Tracking Internals | — | 10 questions on REST protocol, run lifecycle, artifact storage | 15 min |

---

### Week 2: Feature Engineering as Systems Design (8 videos)

**Theme:** Build a feature computation engine with SIMD acceleration and zero-copy data loading.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 2.1 | Video | What is a Feature Store? | — | Features vs raw data, online vs offline, point-in-time correctness, why it's a systems problem | 10 min |
| 2.2 | Video | Zero-Copy Data Loading | `alimentar` | Parquet/Arrow loading with alimentar: memory mapping, columnar access, encryption-at-rest | 10 min |
| 2.3 | Lab | Load and Profile Data with alimentar | `alimentar`, `trueno` | Load Parquet dataset, compute statistics (mean, stddev, null rate) using trueno SIMD | 35 min |
| 2.4 | Video | SIMD Feature Computation | `trueno` | AVX2/AVX-512 for feature engineering: vector normalization, binning, one-hot encoding at hardware speed | 10 min |
| 2.5 | Lab | Build a Feature Pipeline | `trueno`, `alimentar` | End-to-end: load raw Parquet → compute 20 features with trueno SIMD → write to Delta (via delta-rs) | 40 min |
| 2.6 | Video | Point-in-Time Joins | `alimentar` | Temporal correctness: why naive joins cause data leakage, implementing as-of joins in Rust | 8 min |
| 2.7 | Video | Feature Tables as Delta | `delta-rs` | Store feature tables in Delta format: versioning, time travel for feature auditing | 8 min |
| 2.8 | Lab | Build a Feature Store | `alimentar`, `delta-rs`, `trueno` | Create a mini feature store: register features, compute on demand, serve from Delta tables with time-travel | 35 min |
| 2.9 | Video | Managed Platform Parallel: Databricks Feature Store | Python (brief) | 5-min demo: FeatureLookup, online store sync, Unity Catalog integration — mapping to what we just built | 8 min |
| 2.10 | Reading | Feature Store Architecture Patterns | — | Netflix, Uber, Spotify approaches. How alimentar + trueno maps to these patterns | 12 min |
| 2.11 | Quiz | Feature Engineering Systems | — | 10 questions on SIMD computation, point-in-time, Delta-backed features | 15 min |

---

### Week 3: ML Algorithms and Model Formats (7 videos)

**Theme:** Train models with aprender, understand model serialization, and implement quality gates.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 3.1 | Video | ML from Scratch in Rust | `aprender` | Regression, decision trees, random forests, clustering — all in pure Rust with aprender | 10 min |
| 3.2 | Lab | Train Models with aprender | `aprender`, `trueno` | Train regression + random forest on tabular dataset, benchmark training speed vs scikit-learn | 40 min |
| 3.3 | Video | Model Serialization Formats | `aprender`, `realizar` | .apr format, ONNX, GGUF, SafeTensors — what's inside each, why format matters for deployment | 10 min |
| 3.4 | Video | The Model Registry Problem | `pacha` | Versioning, lineage, access control — why a registry is essential and what it must do | 8 min |
| 3.5 | Lab | Sign and Register Models with pacha | `pacha` | Generate Ed25519 keys, sign model artifacts, register in pacha, verify before loading | 35 min |
| 3.6 | Video | BLAKE3 Content Addressing | `pacha` | How pacha uses BLAKE3 hashes for model integrity — compare to MLflow artifact hashing | 8 min |
| 3.7 | Video | Model Encryption at Rest | `pacha` | ChaCha20-Poly1305 encryption for model distribution — when and why you encrypt models | 8 min |
| 3.8 | Video | Managed Platform Parallel: MLflow Model Registry + AutoML | Python (brief) | Demo: register model in UC, transition stages, run AutoML — what the managed version automates | 8 min |
| 3.9 | Reading | Model Governance: Signatures, Encryption, Lineage | `pacha` | Deep dive: pacha's security model vs MLflow Model Registry vs Databricks Unity Catalog | 12 min |
| 3.10 | Quiz | Models and Registry | — | 10 questions on aprender algorithms, serialization, pacha signing/encryption | 15 min |

---

### Week 4: Inference Systems (8 videos)

**Theme:** Deploy models for batch and real-time inference with realizar, then benchmark against managed serving.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 4.1 | Video | Inference Server Architecture | `realizar` | Request routing, batching, model loading, OpenAI-compatible API design | 10 min |
| 4.2 | Video | realizar: GGUF/SafeTensors Inference | `realizar` | How realizar loads and runs models: quantization (Q4_K, Q8_0), SIMD via trueno | 10 min |
| 4.3 | Lab | Serve a Model with realizar | `realizar` | `realizar serve --demo --port 8080`, send requests, inspect throughput/latency | 30 min |
| 4.4 | Video | Batch Inference at Scale | `realizar`, `repartir` | Distributed batch scoring: work-stealing with repartir, zero-copy Arrow output | 8 min |
| 4.5 | Lab | Build Batch Inference Pipeline | `realizar`, `alimentar`, `repartir` | Load dataset with alimentar → batch score with realizar → write results to Delta | 35 min |
| 4.6 | Video | A/B Testing Infrastructure | `batuta` | Request routing for model comparison, metric collection, statistical significance | 8 min |
| 4.7 | Video | Syscall Tracing for Inference Validation | `renacer` | Trace what your inference server actually does at the OS level — no hidden network calls | 8 min |
| 4.8 | Lab | Validate with renacer | `renacer` | Run renacer trace on realizar serving, verify no unexpected syscalls, confirm sovereignty | 25 min |
| 4.9 | Video | Managed Platform Parallel: Databricks Model Serving | Python (brief) | Demo: create serverless endpoint, test predictions, compare latency against our Rust server | 8 min |
| 4.10 | Reading | Inference Optimization: Quantization, Batching, Caching | `realizar`, `trueno` | Deep dive into what makes realizar competitive with llama.cpp | 12 min |
| 4.11 | Quiz | Inference Systems | — | 10 questions on serving architecture, batch inference, A/B testing, validation | 15 min |

---

### Week 5: Quality, Monitoring, and Orchestration (7 videos)

**Theme:** Production quality gates with pmat, monitoring, and orchestration with batuta.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 5.1 | Video | Production Quality with pmat | `pmat` | TDG scoring, mutation testing, coverage enforcement — Toyota Way applied to ML code | 10 min |
| 5.2 | Lab | Enforce Quality Gates | `pmat` | Run `pmat rust-project-score` on our ML code, interpret TDG grade, fix to reach A- | 30 min |
| 5.3 | Video | Privacy Tiers: Sovereign, Private, Standard | `batuta` | When and why data can't leave your infrastructure — regulatory mapping | 8 min |
| 5.4 | Video | Orchestrating the ML Pipeline | `batuta` | batuta as conductor: coordinate alimentar → aprender → pacha → realizar | 8 min |
| 5.5 | Lab | End-to-End Orchestrated Pipeline | `batuta` | Use `batuta oracle` to recommend components, wire together a training → registry → serving pipeline | 35 min |
| 5.6 | Video | Monitoring and Drift Detection | `pmat`, `renacer` | Statistical drift detection in Rust, inference logging, anomaly detection | 8 min |
| 5.7 | Video | Managed Platform Parallel: Databricks Workflows + Lakehouse Monitoring | — | How Databricks Jobs, monitoring, and alerts map to batuta + pmat | 8 min |
| 5.8 | Quiz | Production Quality | — | 10 questions on pmat, privacy tiers, orchestration, monitoring | 15 min |

---

### Week 6: Capstone — Sovereign Fraud Detection Platform

**Duration:** ~15 hours across 2 weeks

**The build:** An end-to-end ML system entirely in the Sovereign AI Stack.

#### Architecture
```
alimentar (Parquet) → trueno (SIMD features) → aprender (train)
         ↓                                           ↓
    Delta tables ←── pacha (sign + register) ←── .apr model
         ↓
    realizar (serve) → renacer (validate) → pmat (quality gate)
         ↓
    batuta (orchestrate + privacy tier = Sovereign)
```

#### Capstone Deliverables

| Deliverable | Stack Components | Description |
|---|---|---|
| 1. Feature Engine | `alimentar`, `trueno`, `delta-rs` | Load transaction data, compute 30+ SIMD-accelerated features, store as Delta feature tables |
| 2. Training Pipeline | `aprender` | Train random forest + gradient boosted trees, benchmark against each other |
| 3. Model Registry | `pacha` | Sign models with Ed25519, encrypt with ChaCha20-Poly1305, register with lineage metadata |
| 4. Inference Server | `realizar` | OpenAI-compatible REST API serving the fraud model, <10ms p99 latency target |
| 5. Validation Suite | `renacer`, `pmat` | Syscall trace proving no external calls (Sovereign tier), TDG score ≥ A- |
| 6. Orchestration | `batuta` | Single `batuta` command to run full pipeline: train → sign → deploy → validate |
| 7. Databricks Comparison | Python notebook | Brief notebook showing the Databricks-managed equivalent (MLflow + Feature Store + Model Serving) side-by-side |

#### Demo Requirements
- Live demo: `batuta` orchestrates full pipeline from raw data to served model
- Benchmark: latency comparison (realizar vs Databricks Model Serving)
- Security audit: `renacer` trace showing zero external network calls
- Quality report: `pmat rust-project-score` showing A- or higher

---

# Course 4: GenAI Systems Engineering

## Subtitle
Build LLM Infrastructure from the Ground Up with the Sovereign AI Stack

## Course Description
Construct every layer of a GenAI system in Rust: from tokenization and embedding math to vector search indexes, RAG pipelines, and sovereign LLM serving. Use realizar for inference, trueno for SIMD similarity search, trueno-rag for retrieval, entrenar for fine-tuning, and pacha for model security. Then compare each component against Databricks Foundation Models, Vector Search, and RAG tooling to understand the full stack.

## Prerequisites
- Course 3 of this specialization (Systems ML Engineering)
- Familiarity with language models (what tokens are, what a transformer does conceptually)
- Comfort with Rust (from Course 3 you'll have significant Rust experience)

## Learning Outcomes
1. Serve LLMs locally with realizar and understand GGUF quantization internals
2. Implement embedding generation and vector similarity search using trueno SIMD primitives
3. Build production RAG pipelines with trueno-rag combining BM25 and vector retrieval
4. Fine-tune models with entrenar's LoRA/QLoRA and manage sovereign model distribution with pacha
5. Deploy privacy-preserving GenAI systems using batuta's Sovereign privacy tier

## Total: ~40 videos + 12 labs + 5 readings + 5 quizzes + 1 capstone

---

### Week 1: LLM Serving and Tokenization (8 videos)

**Theme:** Understand LLM inference by serving models locally with realizar and building tokenization from scratch.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 1.1 | Video | The Sovereign AI Thesis for GenAI | `batuta` | Why organizations need local LLM inference: data residency, cost control, latency, EU AI Act | 10 min |
| 1.2 | Video | GGUF Format Deep Dive | `realizar` | Inside a GGUF file: header, tensors, metadata, quantization types (Q4_K_M, Q8_0) | 10 min |
| 1.3 | Lab | Inspect and Serve a GGUF Model | `realizar` | Download a small model, inspect with realizar, serve with `realizar serve`, query the OpenAI-compatible API | 35 min |
| 1.4 | Video | Tokenization from Scratch | HF `tokenizers` (Rust-native) | BPE algorithm step-by-step: vocabulary, merges, encoding, decoding — implemented in Rust | 10 min |
| 1.5 | Lab | Build a BPE Tokenizer | `tokenizers` crate | Implement BPE training on a small corpus, compare output against HuggingFace tokenizer | 30 min |
| 1.6 | Video | Quantization: Why and How | `realizar`, `trueno` | Weight quantization math: FP16 → Q8_0 → Q4_K, SIMD dequantization with trueno | 10 min |
| 1.7 | Video | realizar vs llama.cpp Performance | `realizar` | Head-to-head benchmark: tokens/sec, memory usage, cold start — brutally honest results | 8 min |
| 1.8 | Video | Managed Platform Parallel: Databricks Foundation Models | Python (brief) | Demo: `ai_query()`, pay-per-token, provisioned throughput — what the managed layer abstracts | 8 min |
| 1.9 | Reading | LLM Inference Optimization Landscape | `realizar`, `trueno` | KV cache, speculative decoding, continuous batching — where the field is heading | 12 min |
| 1.10 | Quiz | LLM Serving Internals | — | 10 questions on GGUF, tokenization, quantization, serving architecture | 15 min |

---

### Week 2: Prompt Engineering and External Models (6 videos)

**Theme:** Master prompt engineering patterns from Rust, connect to external providers when Sovereign tier allows.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 2.1 | Video | Prompt Engineering as Systems Design | `batuta` | System prompts, chat templates, few-shot patterns — building a type-safe prompt builder in Rust | 10 min |
| 2.2 | Lab | Build a Rust Prompt Template Engine | `serde`, `tera` | Create a template engine: variables, conditionals, few-shot injection, token budget management | 35 min |
| 2.3 | Video | Structured Output Extraction | `realizar`, `serde` | JSON mode, schema enforcement, constrained generation — parsing LLM output into Rust structs | 8 min |
| 2.4 | Lab | Structured Extraction Pipeline | `realizar`, `serde` | Prompt a local model → extract structured data → validate against Rust types → handle failures | 30 min |
| 2.5 | Video | External Model Gateway | `batuta` | BackendSelector: routing between local (realizar), VPC, and cloud (OpenAI/Anthropic) based on privacy tier | 10 min |
| 2.6 | Lab | Build a Multi-Backend Gateway | `batuta`, `reqwest` | Configure batuta with Sovereign/Private/Standard tiers, route requests to realizar (local) or external API, implement failover | 35 min |
| 2.7 | Video | Managed Platform Parallel: Databricks External Models | Python (brief) | Demo: external endpoints for OpenAI/Anthropic/Cohere on Databricks — mapping to our gateway | 6 min |
| 2.8 | Reading | Privacy Tiers in Practice | `batuta` | Healthcare (Sovereign), financial services (Private), general (Standard) — real deployment patterns | 10 min |
| 2.9 | Quiz | Prompt Engineering and Routing | — | 10 questions on prompt patterns, structured output, privacy tiers, backend selection | 15 min |

---

### Week 3: Embeddings and Vector Search from First Principles (9 videos)

**Theme:** Implement vector similarity search by hand with trueno SIMD, then see what managed services automate.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 3.1 | Video | What Are Embeddings? | `trueno` | Vectors as meaning: the geometry of semantic space, dimensionality, distance metrics | 10 min |
| 3.2 | Video | Generating Embeddings in Rust | `realizar` | Use realizar to run an embedding model (BGE/GTE), batched encoding, output as Arrow arrays | 8 min |
| 3.3 | Lab | Generate and Store Embeddings | `realizar`, `alimentar`, `delta-rs` | Embed a document corpus → store vectors + metadata in Delta tables via delta-rs | 35 min |
| 3.4 | Video | Similarity Search: Brute Force with SIMD | `trueno` | Cosine similarity with AVX-512: why SIMD matters, benchmark scalar vs SIMD (12x+ speedup) | 10 min |
| 3.5 | Lab | SIMD Similarity Search | `trueno` | Implement brute-force k-NN with trueno SIMD dot product, benchmark on 1M vectors | 30 min |
| 3.6 | Video | HNSW: Approximate Nearest Neighbors | `trueno` | Hierarchical Navigable Small World graphs: construction, search, recall vs speed tradeoff | 10 min |
| 3.7 | Lab | Build an HNSW Index | `trueno` | Implement HNSW index construction and search, tune ef_construction and M parameters, measure recall@10 | 40 min |
| 3.8 | Video | Hybrid Search: BM25 + Vector | `trueno-rag` | Why keyword search still matters, BM25 scoring, reciprocal rank fusion with vector results | 8 min |
| 3.9 | Lab | Hybrid Search with trueno-rag | `trueno-rag` | Build BM25 index + vector index, implement fusion, compare against vector-only and BM25-only baselines | 35 min |
| 3.10 | Video | Managed Platform Parallel: Databricks Vector Search | Python (brief) | Demo: Delta Sync index, managed embeddings, similarity queries — what trueno-rag builds from scratch | 8 min |
| 3.11 | Reading | Vector Search Architecture: Tradeoffs at Scale | `trueno`, `trueno-db` | FAISS vs Pinecone vs Milvus vs trueno: honest comparison of approaches | 12 min |
| 3.12 | Quiz | Embeddings and Vector Search | — | 10 questions on embedding models, SIMD similarity, HNSW, hybrid search | 15 min |

---

### Week 4: RAG Pipelines (8 videos)

**Theme:** Build a complete RAG system with trueno-rag, then evaluate it rigorously.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 4.1 | Video | RAG Architecture from First Principles | `trueno-rag` | Retrieve → Augment → Generate: data flow, failure modes, quality bottlenecks | 10 min |
| 4.2 | Video | Document Chunking Strategies | `alimentar` | Fixed-size, sentence, semantic, recursive — implement each in Rust, measure retrieval impact | 10 min |
| 4.3 | Lab | Build Chunking Pipeline | `alimentar`, `trueno-rag` | Implement 4 chunking strategies, embed each, measure retrieval precision/recall for same queries | 40 min |
| 4.4 | Video | Context Window Management | `realizar` | Token budgeting: retrieved chunks + system prompt + user query must fit. Dynamic selection in Rust | 8 min |
| 4.5 | Lab | End-to-End RAG Pipeline | `trueno-rag`, `realizar` | Full pipeline: chunk documents → embed → index → retrieve → augment prompt → generate with realizar | 45 min |
| 4.6 | Video | Reranking and Filtering | `trueno-rag` | Cross-encoder reranking, metadata filtering, diversity-aware selection | 8 min |
| 4.7 | Video | RAG Evaluation Framework | `pmat` | Retrieval precision, answer faithfulness, hallucination detection — building a Rust evaluation harness | 10 min |
| 4.8 | Lab | Evaluate RAG Quality | `pmat`, `trueno-rag` | Build evaluation dataset, measure retrieval@k, answer quality scores, identify failure cases | 35 min |
| 4.9 | Video | Managed Platform Parallel: Databricks RAG + LangChain | Python (brief) | Demo: Databricks RAG with Vector Search + DBRX + LangChain — compare against our Rust pipeline | 8 min |
| 4.10 | Reading | RAG Failure Modes and Mitigations | `trueno-rag` | When RAG goes wrong: retrieval failures, context poisoning, hallucination amplification | 12 min |
| 4.11 | Quiz | RAG Systems | — | 10 questions on chunking, retrieval, augmentation, evaluation | 15 min |

---

### Week 5: Fine-Tuning and Model Security (8 videos)

**Theme:** Fine-tune LLMs with entrenar and secure the entire model lifecycle with pacha.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 5.1 | Video | Why Fine-Tune? RAG vs Fine-Tuning Decision | — | When retrieval isn't enough: style adaptation, domain knowledge, format control | 10 min |
| 5.2 | Video | LoRA/QLoRA in Rust | `entrenar` | Low-rank adaptation: the math, why it works, entrenar's autograd implementation | 10 min |
| 5.3 | Lab | Fine-Tune with entrenar | `entrenar` | QLoRA fine-tune a small model on domain-specific data, measure before/after on eval set | 45 min |
| 5.4 | Video | Sovereign Model Distribution | `pacha` | The full security lifecycle: train → sign → encrypt → distribute → verify → decrypt → serve | 10 min |
| 5.5 | Lab | Secure Model Pipeline | `pacha` | `pacha keygen` → `pacha sign` → `pacha encrypt` → distribute → `pacha verify` → `pacha decrypt` → `realizar serve` | 35 min |
| 5.6 | Video | Model Lineage and Audit Trails | `pacha`, `renacer` | Track every operation: who trained, what data, which version, syscall audit with renacer | 8 min |
| 5.7 | Video | EU AI Act Compliance | `batuta`, `pacha` | Articles 10, 13, 15: how the Sovereign AI Stack addresses data governance, transparency, accuracy | 8 min |
| 5.8 | Video | Managed Platform Parallel: Databricks Mosaic + Unity Catalog | Python (brief) | How Databricks handles fine-tuning + model governance — mapping to entrenar + pacha | 8 min |
| 5.9 | Reading | Model Security Threat Models | `pacha` | Supply-chain attacks on models, adversarial weights, why signing and encryption matter | 10 min |
| 5.10 | Quiz | Fine-Tuning and Security | — | 10 questions on LoRA, model signing, encryption, compliance | 15 min |

---

### Week 6: Production Deployment and Orchestration (5 videos)

**Theme:** Wire everything together with batuta for production deployment.

| # | Type | Title | Sovereign AI Stack | Description | Duration |
|---|------|-------|-------------------|-------------|----------|
| 6.1 | Video | Production Architecture Patterns | `batuta` | Sovereign deployment: air-gapped, VPC, hybrid — choosing the right tier | 10 min |
| 6.2 | Video | Cost Control with Circuit Breakers | `batuta` | Daily budget enforcement, spillover routing, Muda (waste elimination) | 8 min |
| 6.3 | Lab | Configure Production Deployment | `batuta` | Set up batuta with privacy tier, budget limits, fallback routing, Prometheus metrics | 35 min |
| 6.4 | Video | Observability: Metrics and Tracing | `batuta`, `pmat` | Prometheus integration, distributed tracing, quality dashboards | 8 min |
| 6.5 | Video | Continuous Improvement (Kaizen) | `pmat` | TDG trending, regression detection, automated quality enforcement in CI/CD | 8 min |
| 6.6 | Quiz | Production Deployment | — | 10 questions on privacy tiers, cost control, observability | 15 min |

---

### Week 7: Capstone — Sovereign Enterprise Knowledge Assistant

**Duration:** ~18 hours across 2 weeks

**The build:** A complete GenAI system — from document ingestion to served answers — entirely in the Sovereign AI Stack, deployable in an air-gapped environment.

#### Architecture
```
alimentar (docs) → chunking → realizar (embed) → trueno-rag (index)
                                                        │
User query → trueno-rag (retrieve: BM25 + vector + rerank)
                  │
                  ↓
         realizar (generate with augmented prompt)
                  │
         pacha (signed model, encrypted at rest)
                  │
         batuta (orchestrate, Sovereign tier, budget limits)
                  │
         pmat (quality gate: eval scores, TDG ≥ A-)
                  │
         renacer (syscall audit: zero external calls)
```

#### Capstone Deliverables

| Deliverable | Stack Components | Description |
|---|---|---|
| 1. Document Pipeline | `alimentar`, `trueno-rag` | Ingest markdown/PDF corpus, chunk with multiple strategies, benchmark each |
| 2. Embedding Engine | `realizar`, `trueno` | Generate embeddings with local model, SIMD-accelerated storage and indexing |
| 3. Hybrid Search | `trueno-rag` | BM25 + HNSW vector search with reciprocal rank fusion, metadata filtering |
| 4. RAG Pipeline | `trueno-rag`, `realizar` | Full retrieve → augment → generate pipeline with context window management |
| 5. Fine-Tuned Model | `entrenar`, `pacha` | QLoRA-adapted model for domain, signed and encrypted with pacha |
| 6. Serving Layer | `realizar`, `batuta` | OpenAI-compatible API, Sovereign privacy tier, cost circuit breakers |
| 7. Evaluation Suite | `pmat` | Retrieval precision, answer quality, hallucination rate, regression tests |
| 8. Audit Package | `renacer`, `pacha` | Syscall trace proving sovereignty, model lineage, compliance documentation |
| 9. Databricks Comparison | Python notebook | Brief notebook showing equivalent with Databricks FM + Vector Search + LangChain |

#### Demo Requirements
- Live demo: Ask questions of the knowledge assistant, show retrieval + generation in real-time
- Sovereignty proof: `renacer` trace confirming zero external network calls during inference
- Quality dashboard: Evaluation metrics, TDG score, retrieval precision
- Security audit: Model signatures verified, encryption at rest confirmed
- Benchmark: Latency and quality comparison against Databricks-managed equivalent

---

## Combined Asset Counts

### Course 3: Systems ML Engineering

| Asset Type | Count | Total Duration |
|---|---|---|
| Videos | 38 | ~5.5 hrs |
| Labs | 12 | ~7 hrs |
| Readings | 5 | ~1 hr |
| Quizzes | 5 | ~1.25 hrs |
| Capstone | 1 | ~15 hrs |
| **Total** | **61 assets** | **~30 hrs** |

### Course 4: GenAI Systems Engineering

| Asset Type | Count | Total Duration |
|---|---|---|
| Videos | 40 | ~5.7 hrs |
| Labs | 12 | ~8 hrs |
| Readings | 5 | ~1 hr |
| Quizzes | 5 | ~1.25 hrs |
| Capstone | 1 | ~18 hrs |
| **Total** | **63 assets** | **~34 hrs** |

### Language Distribution

| Language | Course 3 | Course 4 |
|---|---|---|
| **Rust** | ~82% (32/38 videos) | ~85% (34/40 videos) |
| **Python** | ~10% (4/38 videos) | ~10% (4/40 videos) |
| **Conceptual** | ~8% (2/38 videos) | ~5% (2/40 videos) |

---

## Certification Alignment

| Sovereign AI Stack Skill | Databricks Cert Mapping |
|---|---|
| MLflow REST protocol + pacha registry | ML Associate: Model management |
| alimentar + trueno feature engineering | ML Associate: Feature engineering |
| aprender training + realizar serving | ML Associate: Model training & deployment |
| realizar LLM serving | GenAI Engineer: Foundation model serving |
| trueno-rag vector search + RAG | GenAI Engineer: Vector search & RAG |
| entrenar fine-tuning | GenAI Engineer: Model customization |
| batuta orchestration + pmat quality | ML Professional: Production MLOps |
| pacha security + renacer audit | ML Professional: Governance & security |

---

## Toyota Production System Principles Throughout

| Principle | Course 3 Application | Course 4 Application |
|---|---|---|
| **Jidoka** (Stop on defect) | Rust type system prevents invalid model states; pmat stops on quality regression | pacha refuses to serve unsigned models; trueno-rag returns confidence scores |
| **Poka-Yoke** (Error-proof) | Privacy tiers prevent accidental data leakage | Sovereign tier blocks all external API calls at the network level |
| **Genchi Genbutsu** (Go see) | Benchmark every claim: aprender vs sklearn, realizar vs Databricks | SIMD vs scalar benchmarks, RAG recall measurements, fine-tune before/after |
| **Muda** (Eliminate waste) | SIMD computation eliminates CPU waste in features | Cost circuit breakers prevent token waste; quantization reduces memory waste |
| **Kaizen** (Continuous improvement) | pmat TDG trending over project lifetime | Evaluation suite catches quality regressions between model versions |
