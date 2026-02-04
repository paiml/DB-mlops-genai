# Week 5: Production Quality and Orchestration

**Course 3: MLOps Engineering on Databricks**

## Learning Objectives

1. Implement quality gates with pmat TDG scoring
2. Build ML pipelines with dependency orchestration
3. Monitor for data and model drift
4. Use Databricks Workflows for production pipelines

## Demos

### 1. Quality Gates (`quality-gates/`)

Rust implementation of quality gates and orchestration.

**What it demonstrates:**
- TDG (Toyota Development Grade) scoring
- Quality gate evaluation with thresholds
- Pipeline orchestration with dependencies
- Drift detection with PSI

**Run locally:**
```bash
cd quality-gates
cargo run
```

### 2. Databricks Notebook (`databricks/`)

Databricks Workflows and Lakehouse Monitoring.

**What it demonstrates:**
- Workflow definition with task dependencies
- Quality gate validation tasks
- Monitoring statistics
- Drift detection and alerting

**Run on Databricks:**
1. Import `workflows_monitoring.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### TDG (Toyota Development Grade)

Quality scoring inspired by Toyota Production System:

| Component | Weight | Metrics |
|-----------|--------|---------|
| Complexity | 25% | Cyclomatic, cognitive |
| Coverage | 35% | Line, branch, mutation |
| Documentation | 15% | Doc coverage, README |
| Security | 25% | Vulnerabilities, advisories |

### Quality Gates

Automated checks before deployment:

```rust
QualityGate::new("production")
    .add_check("accuracy", 0.95, GreaterOrEqual)
    .add_check("latency_p99", 100.0, LessOrEqual)
    .add_check("error_rate", 0.01, LessOrEqual)
```

### Population Stability Index (PSI)

Drift detection metric:

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No drift |
| 0.1 - 0.25 | Moderate drift |
| > 0.25 | Significant drift |

### Pipeline Orchestration

Dependency-aware execution:

```
ingest → features → train → validate → deploy → monitor
```

## Lab Exercises

1. **Lab 5.2**: Enforce TDG quality gates
2. **Lab 5.5**: Build end-to-end orchestrated pipeline
3. **Lab 5.6**: Configure drift detection and alerting

## Comparison

| Feature | pmat/batuta | Databricks |
|---------|-------------|------------|
| TDG Scoring | Native | Custom |
| Quality Gates | Built-in | Task-based |
| Orchestration | CLI | Workflows GUI |
| Drift Detection | Manual | Built-in |
| Alerting | Webhook | SQL Alerts |

## Resources

- [pmat Documentation](https://docs.rs/pmat)
- [batuta Documentation](https://docs.rs/batuta)
- [Databricks Workflows](https://docs.databricks.com/workflows/index.html)
- [Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/index.html)
