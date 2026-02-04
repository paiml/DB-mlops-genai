# Databricks notebook source
# MAGIC %md
# MAGIC # Foundation Models on Databricks
# MAGIC
# MAGIC **Course 4, Week 1: LLM Serving and Tokenization**
# MAGIC
# MAGIC This notebook demonstrates Databricks Foundation Model APIs.
# MAGIC Compare with realizar for self-hosted LLM serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json

print("Foundation Models Demo - Course 4 Week 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Foundation Model APIs
# MAGIC
# MAGIC Databricks provides access to foundation models via APIs.

# COMMAND ----------

# Available foundation models (Free tier compatible)
foundation_models = {
    "llama-2-70b-chat": {
        "provider": "Meta",
        "parameters": "70B",
        "context_length": 4096,
        "use_case": "Chat, instruction following"
    },
    "mixtral-8x7b-instruct": {
        "provider": "Mistral AI",
        "parameters": "47B (8x7B MoE)",
        "context_length": 32768,
        "use_case": "Instruction following, reasoning"
    },
    "dbrx-instruct": {
        "provider": "Databricks",
        "parameters": "132B",
        "context_length": 32768,
        "use_case": "General purpose, enterprise"
    }
}

print("Available Foundation Models:")
for name, info in foundation_models.items():
    print(f"\n  {name}:")
    for key, value in info.items():
        print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Completion API
# MAGIC
# MAGIC OpenAI-compatible completion endpoint.

# COMMAND ----------

def simulate_completion(prompt: str, model: str = "dbrx-instruct") -> dict:
    """
    Simulates Foundation Model API completion.
    In production, use: mlflow.deployments.get_deploy_client("databricks")
    """
    # Pattern-based responses for demo
    prompt_lower = prompt.lower()

    if "capital" in prompt_lower and "france" in prompt_lower:
        response_text = "The capital of France is Paris."
    elif "hello" in prompt_lower:
        response_text = "Hello! I'm an AI assistant. How can I help you today?"
    elif "machine learning" in prompt_lower:
        response_text = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    else:
        response_text = "I understand your query. Let me provide a helpful response based on my training."

    return {
        "id": "cmpl-demo-123",
        "object": "text_completion",
        "model": model,
        "choices": [{
            "index": 0,
            "text": response_text,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    }

# Test completions
prompts = [
    "What is the capital of France?",
    "Hello, how are you?",
    "Explain machine learning briefly."
]

print("Completion API Demo:\n")
for prompt in prompts:
    response = simulate_completion(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response['choices'][0]['text']}")
    print(f"Tokens: {response['usage']['total_tokens']}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Chat API
# MAGIC
# MAGIC Multi-turn conversation support.

# COMMAND ----------

def simulate_chat(messages: list, model: str = "dbrx-instruct") -> dict:
    """
    Simulates Foundation Model API chat completion.
    In production, use mlflow.deployments client.
    """
    # Get last user message
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )

    # Generate response
    completion = simulate_completion(last_user_msg, model)

    return {
        "id": "chatcmpl-demo-456",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion["choices"][0]["text"]
            },
            "finish_reason": "stop"
        }],
        "usage": completion["usage"]
    }

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

print("Chat API Demo:\n")
print("Conversation:")
for msg in conversation:
    print(f"  {msg['role']}: {msg['content']}")

response = simulate_chat(conversation)
print(f"\nAssistant: {response['choices'][0]['message']['content']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tokenization
# MAGIC
# MAGIC Understanding token counts for cost estimation.

# COMMAND ----------

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (actual tokenizers vary by model).
    Rule of thumb: ~4 characters per token for English.
    """
    return max(1, len(text) // 4)

def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """
    Estimate API cost based on token usage.
    Prices are illustrative (check current Databricks pricing).
    """
    pricing = {
        "dbrx-instruct": {"input": 0.75, "output": 2.25},  # per 1M tokens
        "llama-2-70b-chat": {"input": 0.50, "output": 1.50},
        "mixtral-8x7b-instruct": {"input": 0.50, "output": 1.00},
    }

    if model not in pricing:
        model = "dbrx-instruct"

    input_cost = (prompt_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing[model]["output"]

    return input_cost + output_cost

# Token estimation demo
texts = [
    "Hello, world!",
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
]

print("Token Estimation:\n")
for text in texts:
    tokens = estimate_tokens(text)
    print(f"  \"{text[:40]}...\"" if len(text) > 40 else f"  \"{text}\"")
    print(f"    Characters: {len(text)}, Estimated tokens: {tokens}\n")

# Cost estimation
print("\nCost Estimation (per 1000 requests):")
prompt_tokens = 50
completion_tokens = 100
for model in ["dbrx-instruct", "llama-2-70b-chat", "mixtral-8x7b-instruct"]:
    cost = estimate_cost(prompt_tokens * 1000, completion_tokens * 1000, model)
    print(f"  {model}: ${cost:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Serving Endpoints
# MAGIC
# MAGIC Deploy custom models alongside foundation models.

# COMMAND ----------

# Endpoint configuration (for reference)
endpoint_config = {
    "name": "my-llm-endpoint",
    "config": {
        "served_entities": [
            {
                "name": "foundation-model",
                "external_model": {
                    "name": "databricks-dbrx-instruct",
                    "provider": "databricks-model-serving",
                    "task": "llm/v1/chat"
                }
            }
        ],
        "auto_capture_config": {
            "catalog_name": "main",
            "schema_name": "inference_logs",
            "table_name_prefix": "llm_requests"
        },
        "traffic_config": {
            "routes": [
                {"served_model_name": "foundation-model", "traffic_percentage": 100}
            ]
        }
    }
}

print("Model Serving Endpoint Configuration:")
print(json.dumps(endpoint_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks | Sovereign AI (realizar) |
# MAGIC |---------|------------|------------------------|
# MAGIC | **Model Access** | Foundation Model APIs | Self-hosted GGUF |
# MAGIC | **Scaling** | Auto-scaling | Manual |
# MAGIC | **Latency** | ~100-500ms | <50ms (local) |
# MAGIC | **Cost Model** | Per-token | Infrastructure |
# MAGIC | **Customization** | Fine-tuning | Full control |
# MAGIC | **Privacy** | Managed | Sovereign |
# MAGIC
# MAGIC **Key Insight:** Databricks offers convenience; realizar offers control.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices

# COMMAND ----------

best_practices = """
LLM Serving Best Practices:

1. **Token Management**
   - Monitor token usage for cost control
   - Truncate long inputs appropriately
   - Set max_tokens limits

2. **Latency Optimization**
   - Use streaming for long responses
   - Cache common queries
   - Choose appropriate model size

3. **Error Handling**
   - Implement retries with backoff
   - Handle rate limits gracefully
   - Log all requests for debugging

4. **Security**
   - Never log sensitive prompts
   - Use inference tables for auditing
   - Implement input validation

5. **Cost Control**
   - Set spending limits
   - Use smaller models for simple tasks
   - Batch requests when possible
"""

print(best_practices)

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. Foundation Models provide easy access to LLMs")
print("2. OpenAI-compatible APIs enable portability")
print("3. Token counting is essential for cost management")
print("4. Model Serving enables custom deployments")
print("5. Choose between managed (Databricks) and self-hosted (realizar)")
