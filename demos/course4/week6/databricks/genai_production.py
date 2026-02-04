# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI Production on Databricks
# MAGIC
# MAGIC **Course 4, Week 6: Production Deployment**
# MAGIC
# MAGIC This notebook demonstrates production patterns for GenAI systems.
# MAGIC Compare with realizar for self-hosted production serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field

print("GenAI Production Demo - Course 4 Week 6")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Guardrails

# COMMAND ----------

@dataclass
class GuardrailResult:
    passed: bool
    violations: List[str] = field(default_factory=list)


class Guardrails:
    """Input/output safety checks."""

    def __init__(self):
        self.blocked_patterns = ["password", "credit card", "ssn", "api key"]
        self.max_prompt_length = 4096
        self.max_output_length = 2048

    def check_input(self, text: str) -> GuardrailResult:
        violations = []
        text_lower = text.lower()

        # Length check
        if len(text) > self.max_prompt_length:
            violations.append(f"Prompt too long: {len(text)} > {self.max_prompt_length}")

        # Blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in text_lower:
                violations.append(f"Blocked pattern: {pattern}")

        # PII detection
        if "@" in text and "." in text:
            violations.append("PII detected: email address")

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    def check_output(self, text: str) -> GuardrailResult:
        violations = []

        if len(text) > self.max_output_length:
            violations.append(f"Output too long: {len(text)} > {self.max_output_length}")

        return GuardrailResult(passed=len(violations) == 0, violations=violations)


# Test guardrails
guardrails = Guardrails()

test_inputs = [
    "What is machine learning?",
    "My password is secret123",
    "Contact user@email.com",
]

print("Guardrail Tests:\n")
for text in test_inputs:
    result = guardrails.check_input(text)
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"{status}: \"{text[:40]}...\"")
    for v in result.violations:
        print(f"  - {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rate Limiting

# COMMAND ----------

class RateLimiter:
    """Token and request rate limiting."""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.current_requests = 0
        self.current_tokens = 0

    def check(self) -> bool:
        return (self.current_requests < self.requests_per_minute and
                self.current_tokens < self.tokens_per_minute)

    def record(self, tokens: int):
        self.current_requests += 1
        self.current_tokens += tokens

    def reset(self):
        self.current_requests = 0
        self.current_tokens = 0

    def usage(self) -> Dict[str, float]:
        return {
            "requests": self.current_requests / self.requests_per_minute,
            "tokens": self.current_tokens / self.tokens_per_minute
        }


# Test rate limiter
limiter = RateLimiter(requests_per_minute=10, tokens_per_minute=10000)

print("Rate Limiter Test:\n")
for i in range(5):
    if limiter.check():
        limiter.record(500)
        print(f"Request {i+1}: allowed")
    else:
        print(f"Request {i+1}: rate limited")

usage = limiter.usage()
print(f"\nUsage: {usage['requests']:.0%} requests, {usage['tokens']:.0%} tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. A/B Testing

# COMMAND ----------

@dataclass
class Variant:
    name: str
    model: str
    weight: float


class ABRouter:
    """Deterministic A/B test routing."""

    def __init__(self):
        self.experiments = {}

    def add_experiment(self, name: str, variants: List[Variant]):
        self.experiments[name] = variants

    def route(self, experiment: str, user_id: str) -> Optional[Variant]:
        if experiment not in self.experiments:
            return None

        variants = self.experiments[experiment]
        user_hash = hash(user_id) % 100 / 100

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if user_hash < cumulative:
                return variant

        return variants[-1]


# Test A/B routing
router = ABRouter()
router.add_experiment("model-test", [
    Variant("control", "llama-7b", 0.5),
    Variant("treatment", "llama-13b", 0.5),
])

print("A/B Test Routing:\n")
for user_id in ["user-1", "user-2", "user-3", "user-4", "user-5"]:
    variant = router.route("model-test", user_id)
    print(f"  {user_id} -> {variant.name} ({variant.model})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Production Metrics

# COMMAND ----------

@dataclass
class Metrics:
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    guardrail_violations: int = 0

    def record(self, latency_ms: float, tokens: int, success: bool):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        if success:
            self.successful_requests += 1

    def summary(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_requests),
            "total_tokens": self.total_tokens,
            "guardrail_violations": self.guardrail_violations
        }


# Simulate metrics collection
metrics = Metrics()
import random
random.seed(42)

for _ in range(100):
    latency = random.uniform(50, 200)
    tokens = random.randint(50, 500)
    success = random.random() > 0.05
    metrics.record(latency, tokens, success)

print("Production Metrics:\n")
summary = metrics.summary()
print(f"  Total requests: {summary['total_requests']}")
print(f"  Success rate: {summary['success_rate']:.1%}")
print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
print(f"  Total tokens: {summary['total_tokens']:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inference Tables
# MAGIC
# MAGIC Databricks inference tables for logging and monitoring.

# COMMAND ----------

inference_table_schema = {
    "table_name": "catalog.schema.model_inference_logs",
    "columns": [
        {"name": "request_id", "type": "STRING"},
        {"name": "timestamp", "type": "TIMESTAMP"},
        {"name": "model_name", "type": "STRING"},
        {"name": "model_version", "type": "STRING"},
        {"name": "prompt", "type": "STRING"},
        {"name": "completion", "type": "STRING"},
        {"name": "prompt_tokens", "type": "INT"},
        {"name": "completion_tokens", "type": "INT"},
        {"name": "latency_ms", "type": "DOUBLE"},
        {"name": "user_id", "type": "STRING"},
        {"name": "experiment_variant", "type": "STRING"},
    ],
    "partition_columns": ["timestamp"],
    "auto_capture": True
}

print("Inference Table Schema:")
print(json.dumps(inference_table_schema, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Serving Endpoint Configuration

# COMMAND ----------

endpoint_config = {
    "name": "genai-production-endpoint",
    "config": {
        "served_entities": [
            {
                "name": "llama-7b",
                "external_model": {
                    "name": "databricks-llama-2-70b-chat",
                    "provider": "databricks-model-serving",
                    "task": "llm/v1/chat"
                },
                "workload_size": "Medium",
                "scale_to_zero_enabled": False
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": "llama-7b", "traffic_percentage": 100}
            ]
        },
        "auto_capture_config": {
            "catalog_name": "main",
            "schema_name": "inference",
            "table_name_prefix": "genai_logs"
        }
    },
    "tags": [
        {"key": "team", "value": "ml-platform"},
        {"key": "environment", "value": "production"}
    ]
}

print("Model Serving Endpoint Configuration:")
print(json.dumps(endpoint_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks | Sovereign AI |
# MAGIC |---------|------------|--------------|
# MAGIC | **Guardrails** | Custom code | Custom code |
# MAGIC | **Rate Limiting** | Built-in | Manual |
# MAGIC | **A/B Testing** | Custom + MLflow | Manual |
# MAGIC | **Metrics** | Inference tables | Custom |
# MAGIC | **Monitoring** | Lakehouse Monitor | Custom |
# MAGIC | **Scaling** | Auto-scaling | Manual |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices

# COMMAND ----------

best_practices = """
GenAI Production Best Practices:

1. **Safety First**
   - Implement input/output guardrails
   - Detect and redact PII
   - Block known harmful patterns

2. **Rate Limiting**
   - Limit by requests AND tokens
   - Per-user and global limits
   - Graceful degradation

3. **A/B Testing**
   - Deterministic routing by user
   - Track metrics per variant
   - Statistical significance before decisions

4. **Monitoring**
   - Log all requests/responses
   - Track latency, tokens, errors
   - Alert on anomalies

5. **Scaling**
   - Use auto-scaling endpoints
   - Consider scale-to-zero for cost
   - Monitor queue depth
"""

print(best_practices)

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. Guardrails protect against harmful inputs/outputs")
print("2. Rate limiting prevents abuse and controls costs")
print("3. A/B testing enables model comparison")
print("4. Inference tables provide audit trail")
print("5. Databricks provides managed production infrastructure")
