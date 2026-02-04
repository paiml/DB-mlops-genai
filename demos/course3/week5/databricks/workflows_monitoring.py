# Databricks notebook source
# MAGIC %md
# MAGIC # Production Quality with Databricks Workflows and Monitoring
# MAGIC
# MAGIC **Course 3, Week 5: Production Quality and Orchestration**
# MAGIC
# MAGIC This notebook demonstrates Databricks Workflows and Lakehouse Monitoring.
# MAGIC Compare with pmat quality gates and batuta orchestration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta
import json

print("Databricks Workflows and Monitoring Demo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Databricks Workflows
# MAGIC
# MAGIC Workflows orchestrate multi-task jobs with dependencies.

# COMMAND ----------

# Workflow definition (JSON format used by Databricks Jobs API)
workflow_definition = {
    "name": "fraud-detection-pipeline",
    "tasks": [
        {
            "task_key": "ingest_data",
            "description": "Load raw transaction data",
            "notebook_task": {
                "notebook_path": "/Repos/ml-pipeline/ingest"
            },
            "cluster_spec": {"existing_cluster_id": "cluster-123"}
        },
        {
            "task_key": "feature_engineering",
            "description": "Compute features",
            "depends_on": [{"task_key": "ingest_data"}],
            "notebook_task": {
                "notebook_path": "/Repos/ml-pipeline/features"
            }
        },
        {
            "task_key": "train_model",
            "description": "Train fraud detection model",
            "depends_on": [{"task_key": "feature_engineering"}],
            "notebook_task": {
                "notebook_path": "/Repos/ml-pipeline/train"
            }
        },
        {
            "task_key": "validate_model",
            "description": "Validate model quality",
            "depends_on": [{"task_key": "train_model"}],
            "notebook_task": {
                "notebook_path": "/Repos/ml-pipeline/validate"
            }
        },
        {
            "task_key": "deploy_model",
            "description": "Deploy to serving endpoint",
            "depends_on": [{"task_key": "validate_model"}],
            "notebook_task": {
                "notebook_path": "/Repos/ml-pipeline/deploy"
            }
        }
    ],
    "schedule": {
        "quartz_cron_expression": "0 0 6 * * ?",  # Daily at 6 AM
        "timezone_id": "UTC"
    },
    "email_notifications": {
        "on_failure": ["team@example.com"]
    }
}

print("Workflow Definition:")
print(json.dumps(workflow_definition, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Quality Gates in Workflows
# MAGIC
# MAGIC Implement quality gates as validation tasks.

# COMMAND ----------

def quality_gate_task(metrics: dict, thresholds: dict) -> bool:
    """
    Quality gate validation task.
    Returns True if all checks pass, False otherwise.
    """
    results = []

    for metric, threshold in thresholds.items():
        value = metrics.get(metric, 0)
        comparator = threshold.get("comparator", ">=")
        threshold_value = threshold.get("value", 0)

        if comparator == ">=":
            passed = value >= threshold_value
        elif comparator == "<=":
            passed = value <= threshold_value
        else:
            passed = value == threshold_value

        results.append({
            "metric": metric,
            "value": value,
            "threshold": threshold_value,
            "passed": passed
        })

        print(f"{'✓' if passed else '✗'} {metric}: {value:.4f} (threshold: {threshold_value})")

    all_passed = all(r["passed"] for r in results)
    print(f"\nQuality Gate: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


# Example usage
model_metrics = {
    "accuracy": 0.96,
    "f1_score": 0.94,
    "latency_p99_ms": 85.0,
    "error_rate": 0.005
}

thresholds = {
    "accuracy": {"comparator": ">=", "value": 0.95},
    "f1_score": {"comparator": ">=", "value": 0.90},
    "latency_p99_ms": {"comparator": "<=", "value": 100.0},
    "error_rate": {"comparator": "<=", "value": 0.01}
}

gate_passed = quality_gate_task(model_metrics, thresholds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Lakehouse Monitoring
# MAGIC
# MAGIC Monitor data quality and model performance.

# COMMAND ----------

# Create sample monitoring data
monitoring_data = [
    (datetime.now() - timedelta(days=i), 0.95 + (i % 5) * 0.005, 80 + i % 20, 0.005 + (i % 3) * 0.001)
    for i in range(30)
]

monitoring_df = spark.createDataFrame(
    monitoring_data,
    ["timestamp", "accuracy", "latency_ms", "error_rate"]
)

display(monitoring_df)

# COMMAND ----------

# Calculate monitoring statistics
stats = monitoring_df.agg(
    F.mean("accuracy").alias("avg_accuracy"),
    F.stddev("accuracy").alias("std_accuracy"),
    F.mean("latency_ms").alias("avg_latency"),
    F.percentile_approx("latency_ms", 0.99).alias("p99_latency"),
    F.mean("error_rate").alias("avg_error_rate")
).collect()[0]

print("Monitoring Statistics (Last 30 days):")
print(f"  Accuracy: {stats['avg_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
print(f"  Latency: {stats['avg_latency']:.1f}ms (p99: {stats['p99_latency']:.1f}ms)")
print(f"  Error Rate: {stats['avg_error_rate']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Drift Detection
# MAGIC
# MAGIC Monitor for data and concept drift.

# COMMAND ----------

def calculate_psi(baseline: list, current: list, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    PSI < 0.1: No drift
    0.1 <= PSI < 0.25: Moderate drift
    PSI >= 0.25: Significant drift
    """
    import numpy as np

    # Create bins based on baseline
    min_val = min(min(baseline), min(current))
    max_val = max(max(baseline), max(current))
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate proportions
    baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
    current_hist, _ = np.histogram(current, bins=bin_edges)

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    baseline_prop = (baseline_hist + eps) / (len(baseline) + eps * bins)
    current_prop = (current_hist + eps) / (len(current) + eps * bins)

    # Calculate PSI
    psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

    return psi


# Simulate baseline and current distributions
import random
random.seed(42)

baseline_amounts = [random.gauss(100, 20) for _ in range(1000)]
current_amounts_ok = [random.gauss(102, 21) for _ in range(1000)]  # Slight shift
current_amounts_drift = [random.gauss(150, 30) for _ in range(1000)]  # Significant drift

psi_ok = calculate_psi(baseline_amounts, current_amounts_ok)
psi_drift = calculate_psi(baseline_amounts, current_amounts_drift)

print("Drift Detection Results:")
print(f"  No drift scenario: PSI = {psi_ok:.4f} ({'OK' if psi_ok < 0.1 else 'DRIFT'})")
print(f"  Drift scenario: PSI = {psi_drift:.4f} ({'OK' if psi_drift < 0.1 else 'DRIFT'})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alerting Configuration
# MAGIC
# MAGIC Set up alerts for quality degradation.

# COMMAND ----------

alert_config = {
    "name": "model-quality-alert",
    "condition": {
        "op": "OR",
        "rules": [
            {"metric": "accuracy", "op": "<", "value": 0.90},
            {"metric": "error_rate", "op": ">", "value": 0.05},
            {"metric": "psi", "op": ">", "value": 0.25}
        ]
    },
    "actions": [
        {"type": "email", "recipients": ["mlops-team@example.com"]},
        {"type": "slack", "channel": "#ml-alerts"},
        {"type": "pagerduty", "service": "ml-oncall"}
    ],
    "evaluation_interval": "5m"
}

print("Alert Configuration:")
print(json.dumps(alert_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks | Sovereign AI (pmat/batuta) |
# MAGIC |---------|------------|---------------------------|
# MAGIC | **Orchestration** | Workflows (GUI/API) | batuta CLI |
# MAGIC | **Scheduling** | Cron, event-based | External (cron) |
# MAGIC | **Quality Gates** | Custom tasks | pmat built-in |
# MAGIC | **Monitoring** | Lakehouse Monitoring | Custom metrics |
# MAGIC | **Drift Detection** | Built-in | Manual (trueno) |
# MAGIC | **Alerting** | SQL Alerts | External (webhook) |
# MAGIC | **TDG Scoring** | N/A | pmat native |
# MAGIC
# MAGIC **Key Insight:** Databricks provides integrated monitoring; pmat provides deeper code quality analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices
# MAGIC
# MAGIC 1. **Quality Gates**: Block deployments that don't meet thresholds
# MAGIC 2. **Monitoring**: Track key metrics over time
# MAGIC 3. **Drift Detection**: Monitor feature distributions
# MAGIC 4. **Alerting**: Notify on quality degradation
# MAGIC 5. **Automation**: Schedule regular retraining

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. Workflows orchestrate multi-stage ML pipelines")
print("2. Quality gates enforce deployment standards")
print("3. Lakehouse Monitoring tracks performance over time")
print("4. PSI detects distribution drift")
print("5. Alerts enable proactive issue response")
