# Week 6: Production Deployment

## Overview

Deploy GenAI systems with guardrails, rate limiting, and monitoring.

## Topics

| # | Type | Title | Platform | Duration |
|---|------|-------|----------|----------|
| 6.1 | Video | GenAI Production Patterns | Concept | 10 min |
| 6.2 | Video | Databricks Model Serving for LLMs | Databricks | 10 min |
| 6.3 | Lab | Deploy GenAI Endpoint | Databricks | 35 min |
| 6.4 | Video | Cost Control and Circuit Breakers | Sovereign | 8 min |
| 6.5 | Lab | Production Config with batuta | Sovereign | 30 min |
| 6.6 | Quiz | Production Deployment | — | 15 min |

## Sovereign AI Stack Components

- `batuta` for cost limits and observability
- `renacer` for audit trails

## Key Concepts

### Guardrails
- Input validation (length, blocked patterns)
- Output filtering (PII, harmful content)
- Fail-safe defaults

### Rate Limiting
- Requests per minute
- Tokens per minute
- Per-user quotas

### Monitoring
- Latency percentiles (p50, p95, p99)
- Error rates and types
- Token usage and costs
- Guardrail trigger rates

### A/B Testing
- Traffic splitting
- Metric collection
- Statistical significance
