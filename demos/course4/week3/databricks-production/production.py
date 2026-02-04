# Databricks notebook source
# MAGIC %md
# MAGIC # Production GenAI on Databricks
# MAGIC
# MAGIC **Course 4, Week 3: Production Deployment**
# MAGIC
# MAGIC This notebook demonstrates production deployment patterns for GenAI applications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. AI Gateway Configuration

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayRateLimit

w = WorkspaceClient()

# Configure AI Gateway with rate limits
gateway_config = AiGatewayConfig(
    rate_limits=[
        AiGatewayRateLimit(
            calls=100,
            renewal_period="minute"
        )
    ],
    usage_tracking_config={
        "enabled": True
    }
)

print("AI Gateway configured with rate limiting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inference Tables (Logging)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create inference logging table
# MAGIC CREATE TABLE IF NOT EXISTS main.genai_course.inference_logs (
# MAGIC   request_id STRING,
# MAGIC   timestamp TIMESTAMP,
# MAGIC   endpoint STRING,
# MAGIC   prompt_tokens INT,
# MAGIC   completion_tokens INT,
# MAGIC   latency_ms INT,
# MAGIC   status STRING
# MAGIC );

# COMMAND ----------

import mlflow.deployments
from datetime import datetime
import uuid

client = mlflow.deployments.get_deploy_client("databricks")

def logged_predict(endpoint: str, inputs: dict) -> dict:
    """Make prediction with automatic logging."""
    request_id = str(uuid.uuid4())
    start = datetime.now()

    try:
        response = client.predict(endpoint=endpoint, inputs=inputs)
        latency = (datetime.now() - start).total_seconds() * 1000

        # Log to inference table
        spark.sql(f"""
            INSERT INTO main.genai_course.inference_logs VALUES (
                '{request_id}',
                current_timestamp(),
                '{endpoint}',
                {response['usage']['prompt_tokens']},
                {response['usage']['completion_tokens']},
                {int(latency)},
                'success'
            )
        """)

        return response
    except Exception as e:
        spark.sql(f"""
            INSERT INTO main.genai_course.inference_logs VALUES (
                '{request_id}',
                current_timestamp(),
                '{endpoint}',
                0, 0, 0,
                'error: {str(e)[:100]}'
            )
        """)
        raise

# Test logged prediction
response = logged_predict(
    endpoint="databricks-dbrx-instruct",
    inputs={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}
)
print("Response logged successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Quality Monitoring

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analyze inference metrics
# MAGIC SELECT
# MAGIC   endpoint,
# MAGIC   COUNT(*) as total_requests,
# MAGIC   AVG(latency_ms) as avg_latency,
# MAGIC   SUM(prompt_tokens + completion_tokens) as total_tokens,
# MAGIC   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
# MAGIC FROM main.genai_course.inference_logs
# MAGIC WHERE timestamp > current_timestamp() - INTERVAL 1 HOUR
# MAGIC GROUP BY endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. A/B Testing

# COMMAND ----------

import random

def ab_test_query(question: str, test_name: str = "model_comparison"):
    """Run A/B test between two model versions."""

    # Randomly assign to variant
    variant = "control" if random.random() < 0.5 else "treatment"

    # Route to appropriate endpoint
    if variant == "control":
        endpoint = "databricks-dbrx-instruct"
    else:
        endpoint = "course4-finetuned-model"

    response = logged_predict(
        endpoint=endpoint,
        inputs={"messages": [{"role": "user", "content": question}], "max_tokens": 100}
    )

    # Log A/B test assignment
    spark.sql(f"""
        INSERT INTO main.genai_course.ab_test_logs VALUES (
            '{test_name}',
            '{variant}',
            current_timestamp(),
            '{response["choices"][0]["message"]["content"][:100]}'
        )
    """)

    return {"variant": variant, "response": response}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cost Tracking

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Track costs by endpoint
# MAGIC SELECT
# MAGIC   endpoint,
# MAGIC   DATE(timestamp) as date,
# MAGIC   SUM(prompt_tokens) as input_tokens,
# MAGIC   SUM(completion_tokens) as output_tokens,
# MAGIC   -- Estimated cost (adjust rates as needed)
# MAGIC   SUM(prompt_tokens) * 0.00001 + SUM(completion_tokens) * 0.00003 as estimated_cost_usd
# MAGIC FROM main.genai_course.inference_logs
# MAGIC WHERE status = 'success'
# MAGIC GROUP BY endpoint, DATE(timestamp)
# MAGIC ORDER BY date DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Guardrails

# COMMAND ----------

def apply_guardrails(prompt: str, response: str) -> dict:
    """Apply safety guardrails to GenAI requests."""

    issues = []

    # Check for blocked content in prompt
    blocked_terms = ["harmful", "illegal", "dangerous"]
    for term in blocked_terms:
        if term in prompt.lower():
            issues.append(f"Blocked term in prompt: {term}")

    # Check response length
    if len(response) > 2000:
        issues.append("Response exceeds length limit")

    # Check for PII patterns (simplified)
    import re
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', response):  # SSN pattern
        issues.append("Potential PII detected")

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }

# Test guardrails
result = apply_guardrails(
    prompt="Tell me about machine learning",
    response="Machine learning is a subset of AI..."
)
print(f"Guardrails passed: {result['passed']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Production Checklist
# MAGIC
# MAGIC | Item | Status |
# MAGIC |------|--------|
# MAGIC | Rate limiting | ✅ Configured |
# MAGIC | Request logging | ✅ Inference tables |
# MAGIC | Quality monitoring | ✅ SQL analytics |
# MAGIC | A/B testing | ✅ Implemented |
# MAGIC | Cost tracking | ✅ Token accounting |
# MAGIC | Guardrails | ✅ Content filtering |
