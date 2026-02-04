# Week 2: Prompt Engineering

**Course 4: GenAI Engineering on Databricks**

## Learning Objectives

1. Master prompt engineering techniques
2. Build reusable prompt templates
3. Implement structured output parsing
4. Use Databricks AI Playground for prompt development

## Demos

### 1. Prompt Engineering (`prompt-engineering/`)

Rust implementation demonstrating prompt engineering patterns.

**What it demonstrates:**
- Prompt templates with variable substitution
- Five prompting techniques
- Prompt builder pattern
- JSON output parsing

**Run locally:**
```bash
cd prompt-engineering
cargo run
```

### 2. Databricks Notebook (`databricks/`)

Prompt engineering with Databricks AI Playground.

**What it demonstrates:**
- Template management
- Technique comparison
- Prompt metrics analysis
- Best practices

**Run on Databricks:**
1. Import `prompt_engineering.py` into your workspace
2. Attach to a cluster
3. Run all cells

## Key Concepts

### Prompting Techniques

| Technique | When to Use | Example |
|-----------|-------------|---------|
| Zero-shot | Simple tasks | "Classify: positive/negative" |
| Few-shot | Complex patterns | Include 2-5 examples |
| Chain-of-thought | Reasoning tasks | "Think step by step" |
| Role-based | Domain expertise | "You are an expert in..." |
| Structured | Data extraction | "Return as JSON" |

### Prompt Template Format

```
{system_message}

{context}

{examples}

{instruction}

{output_format}
```

### Output Parsing

```python
# Extract JSON from LLM output
def parse_json(text: str) -> dict:
    start = text.find('{')
    end = text.rfind('}')
    return json.loads(text[start:end+1])
```

## Comparison

| Feature | Databricks | Sovereign AI |
|---------|------------|--------------|
| Interface | Web UI | CLI/Library |
| Templates | Custom | Code-based |
| Testing | Manual | Programmatic |
| Version Control | Workspace | Git |

## Lab Exercises

1. **Lab 2.1**: Create prompt template library
2. **Lab 2.2**: Implement few-shot classifier
3. **Lab 2.3**: Build chain-of-thought solver

## Resources

- [Databricks AI Playground](https://docs.databricks.com/en/large-language-models/ai-playground.html)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
