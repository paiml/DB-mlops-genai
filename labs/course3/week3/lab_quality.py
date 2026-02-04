# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5.2: Enforce Quality Gates
# MAGIC
# MAGIC **Course 3, Week 5: Production Quality**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Implement quality gate checks
# MAGIC - Define thresholds for metrics
# MAGIC - Block deployments that don't meet standards

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

print("Quality Gates Lab - Course 3 Week 5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Define Quality Gate Structure

# COMMAND ----------

class Comparator(Enum):
    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    EQUAL = "=="


@dataclass
class QualityCheck:
    metric: str
    threshold: float
    comparator: Comparator

    def evaluate(self, value: float) -> bool:
        if self.comparator == Comparator.GREATER_OR_EQUAL:
            return value >= self.threshold
        elif self.comparator == Comparator.LESS_OR_EQUAL:
            return value <= self.threshold
        else:
            return value == self.threshold


# Example check
accuracy_check = QualityCheck("accuracy", 0.90, Comparator.GREATER_OR_EQUAL)
print(f"Check: {accuracy_check.metric} {accuracy_check.comparator.value} {accuracy_check.threshold}")
print(f"Evaluate 0.95: {accuracy_check.evaluate(0.95)}")
print(f"Evaluate 0.85: {accuracy_check.evaluate(0.85)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Implement Quality Gate
# MAGIC
# MAGIC EXERCISE: Create a QualityGate class that holds multiple checks.

# COMMAND ----------

# EXERCISE: Implement QualityGate class

class QualityGate:
    def __init__(self, name: str):
        self.name = name
        self.checks: List[QualityCheck] = []

    def add_check(self, metric: str, threshold: float, comparator: Comparator) -> 'QualityGate':
        # EXERCISE: Add a check to the list
        # YOUR CODE HERE
        pass
        return self

    def evaluate(self, metrics: Dict[str, float]) -> Dict:
        # EXERCISE: Evaluate all checks against provided metrics
        # Return dict with: passed (bool), results (list of check results)
        # YOUR CODE HERE
        pass


# Test your implementation
# gate = QualityGate("production")
# gate.add_check("accuracy", 0.90, Comparator.GREATER_OR_EQUAL)
# gate.add_check("latency_p99_ms", 100, Comparator.LESS_OR_EQUAL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Define Production Gate
# MAGIC
# MAGIC EXERCISE: Create a quality gate with production-ready thresholds.

# COMMAND ----------

# EXERCISE: Define a production quality gate with these checks:
# - accuracy >= 0.90
# - f1_score >= 0.85
# - latency_p99_ms <= 100
# - error_rate <= 0.01

# YOUR CODE HERE
# production_gate = QualityGate("production")
# production_gate.add_check(...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Evaluate Model Metrics
# MAGIC
# MAGIC EXERCISE: Test your gate against sample model metrics.

# COMMAND ----------

# Sample model metrics
model_metrics_good = {
    "accuracy": 0.95,
    "f1_score": 0.92,
    "latency_p99_ms": 85,
    "error_rate": 0.005
}

model_metrics_bad = {
    "accuracy": 0.88,
    "f1_score": 0.80,
    "latency_p99_ms": 150,
    "error_rate": 0.02
}

# EXERCISE: Evaluate both models
# result_good = production_gate.evaluate(model_metrics_good)
# result_bad = production_gate.evaluate(model_metrics_bad)

# YOUR CODE HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Quality Report
# MAGIC
# MAGIC EXERCISE: Generate a quality report from evaluation results.

# COMMAND ----------

def generate_report(gate_name: str, result: Dict) -> str:
    """Generate a formatted quality report."""
    # EXERCISE: Implement report generation
    # Include:
    # - Gate name
    # - Overall pass/fail status
    # - Individual check results
    # YOUR CODE HERE
    pass


# Test
# report = generate_report("production", result_good)
# print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check QualityGate class
        gate = QualityGate("test")
        gate.add_check("test_metric", 0.5, Comparator.GREATER_OR_EQUAL)
        checks.append(("QualityGate implemented", len(gate.checks) == 1))

        # Check evaluate method
        result = gate.evaluate({"test_metric": 0.6})
        checks.append(("Evaluate works", result is not None and result.get("passed", False)))

        # Check production gate
        if 'production_gate' in dir():
            checks.append(("Production gate defined", len(production_gate.checks) >= 3))
    except Exception as e:
        checks.append(("Implementation complete", False))

    # Display results
    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Lab complete.")
    else:
        print("\n⚠️ Some checks failed. Review your code.")

validate_lab()
