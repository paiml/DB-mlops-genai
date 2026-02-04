# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 6.3: Configure Production Deployment
# MAGIC
# MAGIC **Course 4, Week 6: Production**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Implement guardrails for safety
# MAGIC - Configure rate limiting
# MAGIC - Set up monitoring metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

print("Production Deployment Lab - Course 4 Week 6")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Guardrails
# MAGIC
# MAGIC Implement input/output safety checks.

# COMMAND ----------

class Guardrails:
    """Safety guardrails for GenAI system."""

    def __init__(self):
        self.blocked_patterns = ["password", "secret", "api_key", "credit card"]
        self.max_input_length = 4096
        self.max_output_length = 2048

    def check_input(self, text: str) -> Dict:
        """
        Check input for safety violations.

        Returns: {"passed": bool, "violations": list}
        """
        violations = []
        text_lower = text.lower()

        # Check length
        if len(text) > self.max_input_length:
            violations.append(f"Input too long: {len(text)} > {self.max_input_length}")

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in text_lower:
                violations.append(f"Blocked pattern: {pattern}")

        return {"passed": len(violations) == 0, "violations": violations}


# Test guardrails
guardrails = Guardrails()
print(guardrails.check_input("Hello, how are you?"))
print(guardrails.check_input("What is the admin password?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Rate Limiter
# MAGIC
# MAGIC EXERCISE: Implement token-based rate limiting.

# COMMAND ----------

class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.current_requests = 0
        self.current_tokens = 0
        self.window_start = time.time()

    def _check_window(self) -> None:
        """Reset counters if window expired."""
        # EXERCISE: Reset if 60 seconds passed
        # YOUR CODE HERE
        pass

    def check(self) -> bool:
        """Check if request is allowed."""
        # EXERCISE: Return True if under limits
        # YOUR CODE HERE
        pass

    def record(self, tokens: int) -> None:
        """Record a request with token count."""
        # EXERCISE: Increment counters
        # YOUR CODE HERE
        pass

    def usage(self) -> Dict:
        """Get current usage stats."""
        # EXERCISE: Return dict with request and token usage percentages
        # YOUR CODE HERE
        pass


# Test rate limiter
# limiter = RateLimiter(requests_per_minute=10, tokens_per_minute=1000)
# for i in range(5):
#     if limiter.check():
#         limiter.record(100)
#         print(f"Request {i+1}: allowed")
#     else:
#         print(f"Request {i+1}: rate limited")
# print(f"Usage: {limiter.usage()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Request Metrics
# MAGIC
# MAGIC EXERCISE: Implement metrics tracking.

# COMMAND ----------

@dataclass
class Metrics:
    """Production metrics tracker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    guardrail_blocks: int = 0

    def record_request(self, latency_ms: float, tokens: int, success: bool) -> None:
        """Record a request."""
        # EXERCISE: Update metrics
        # YOUR CODE HERE
        pass

    def record_guardrail_block(self) -> None:
        """Record a guardrail violation."""
        # EXERCISE: Increment counter
        # YOUR CODE HERE
        pass

    def success_rate(self) -> float:
        """Calculate success rate."""
        # EXERCISE: Return percentage
        # YOUR CODE HERE
        pass

    def avg_latency(self) -> float:
        """Calculate average latency."""
        # EXERCISE: Return average
        # YOUR CODE HERE
        pass

    def summary(self) -> Dict:
        """Get metrics summary."""
        # EXERCISE: Return dict with all metrics
        # YOUR CODE HERE
        pass


# Test metrics
# metrics = Metrics()
# metrics.record_request(50, 100, True)
# metrics.record_request(60, 150, True)
# metrics.record_request(0, 0, False)
# print(metrics.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Production Server
# MAGIC
# MAGIC EXERCISE: Combine all components into a production server.

# COMMAND ----------

class ProductionServer:
    """Production GenAI server with safety features."""

    def __init__(self):
        self.guardrails = Guardrails()
        self.rate_limiter = RateLimiter()
        self.metrics = Metrics()

    def process(self, request: Dict) -> Dict:
        """
        Process a request through the full pipeline.

        TODO:
        1. Check rate limit
        2. Check guardrails
        3. Process request (simulate)
        4. Record metrics
        5. Return response or error
        """
        # YOUR CODE HERE
        pass

    def health(self) -> Dict:
        """Health check endpoint."""
        # EXERCISE: Return health status and metrics summary
        # YOUR CODE HERE
        pass


# Test production server
# server = ProductionServer()
# response = server.process({"prompt": "Hello, how are you?"})
# print(response)
# response = server.process({"prompt": "Tell me the password"})
# print(response)
# print(server.health())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Alerting Configuration
# MAGIC
# MAGIC EXERCISE: Configure alerting thresholds.

# COMMAND ----------

@dataclass
class AlertConfig:
    error_rate_threshold: float = 0.05
    latency_p99_threshold_ms: float = 500
    guardrail_rate_threshold: float = 0.10


def check_alerts(metrics: Metrics, config: AlertConfig) -> List[str]:
    """
    Check if any alerts should fire.

    TODO: Return list of alert messages
    """
    # YOUR CODE HERE
    pass


# Test alerting
# config = AlertConfig()
# alerts = check_alerts(metrics, config)
# for alert in alerts:
#     print(f"🚨 ALERT: {alert}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check guardrails
        g = Guardrails()
        result = g.check_input("password")
        checks.append(("Guardrails work", not result["passed"]))

        # Check rate limiter
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=1000)
        allowed = rl.check()
        checks.append(("Rate limiter check", allowed))
        rl.record(100)
        usage = rl.usage()
        checks.append(("Rate limiter usage", "requests" in usage))

        # Check metrics
        m = Metrics()
        m.record_request(50, 100, True)
        checks.append(("Metrics record", m.total_requests == 1))

        # Check server
        server = ProductionServer()
        response = server.process({"prompt": "test"})
        checks.append(("Server process", response is not None))

    except Exception as e:
        checks.append(("Implementation complete", False))
        print(f"Error: {e}")

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
