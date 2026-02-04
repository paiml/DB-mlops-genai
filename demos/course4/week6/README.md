# Week 6: GenAI Production Deployment

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Implement safety guardrails for GenAI systems
2. Configure rate limiting and quotas
3. Set up A/B testing for model comparison
4. Monitor production GenAI systems

## Demos

### 1. Production Server (`production/`)

Rust implementation demonstrating production patterns.

**What it demonstrates:**
- Input/output guardrails (PII, blocked patterns)
- Rate limiting (requests, tokens)
- A/B testing with deterministic routing
- Production metrics tracking

**Run locally:**
```bash
cd production
cargo run
```

### 2. Databricks Notebook (`databricks/`)

GenAI production on Databricks.

**What it demonstrates:**
- Guardrail implementation
- Inference table configuration
- Endpoint configuration
- Monitoring setup

**Run on Databricks:**
1. Import `genai_production.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### Guardrails

| Type | Purpose | Implementation |
|------|---------|----------------|
| Input filters | Block harmful prompts | Pattern matching, PII detection |
| Output filters | Prevent harmful responses | Content classification |
| Length limits | Control costs | Token/char limits |

### Rate Limiting

```
Requests per minute: 60
Tokens per minute: 100,000
Per-user limits: 10 req/min
```

### A/B Testing

```python
# Deterministic routing
user_hash = hash(user_id) % 100
if user_hash < 50:
    model = "control"
else:
    model = "treatment"
```

### Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Latency p99 | Response time | > 500ms |
| Error rate | Failed requests | > 1% |
| Token usage | Cost tracking | Budget-based |
| Guardrail hits | Safety violations | Any |

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Guardrails | Custom | Custom |
| Rate Limiting | Built-in | Manual |
| A/B Testing | MLflow | Manual |
| Monitoring | Inference Tables | Custom |

## Lab Exercises

1. **Lab 6.1**: Implement custom guardrails
2. **Lab 6.2**: Configure A/B test experiment
3. **Lab 6.3**: Set up production monitoring

## Resources

- [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [Inference Tables](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html)
