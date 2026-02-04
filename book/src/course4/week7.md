# Week 7: Capstone — Enterprise Knowledge Assistant

## Duration
~18 hours

## Overview

Build a complete enterprise knowledge assistant with RAG, fine-tuning, and production deployment.

## Architecture

```
Documents → alimentar (chunk) → trueno (embed) → trueno-rag (index)
                                                        ↓
User Query → trueno-rag (retrieve) → realizar (generate) → Response
                     ↑                      ↑
              Hybrid BM25+Vector    Fine-tuned Model
                                          ↓
                                    pacha (signed)
                                          ↓
                                    batuta (deploy)
```

## Deliverables

1. **Document Ingestion**
   - Databricks pipeline
   - alimentar chunking comparison

2. **Embedding Pipeline**
   - Vector Search index
   - trueno SIMD embeddings

3. **Hybrid Retrieval**
   - Databricks RAG
   - trueno-rag BM25 + vector

4. **Generation Layer**
   - Foundation Models API
   - realizar inference

5. **Fine-Tuned Model**
   - Databricks fine-tuning
   - entrenar LoRA

6. **Security Package**
   - Unity Catalog governance
   - pacha encryption and signing

7. **Production Deployment**
   - Model Serving endpoint
   - batuta orchestration

8. **Evaluation Suite**
   - pmat quality gates
   - RAG evaluation metrics

## Evaluation Criteria

- Retrieval quality: MRR@10 ≥ 0.7
- Answer quality: Human evaluation ≥ 4/5
- Latency: p95 < 2 seconds
- Security: All models signed, PII filtered
- Quality: pmat score ≥ B
