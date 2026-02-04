# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Engineering on Databricks
# MAGIC
# MAGIC **Course 4, Week 2: Prompt Engineering**
# MAGIC
# MAGIC This notebook demonstrates prompt engineering techniques with Databricks AI Playground.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
from typing import Dict, List, Optional

print("Prompt Engineering Demo - Course 4 Week 2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prompt Templates

# COMMAND ----------

class PromptTemplate:
    """Reusable prompt template with variable substitution."""

    def __init__(self, name: str, template: str, description: str = ""):
        self.name = name
        self.template = template
        self.description = description
        self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract {variable} placeholders from template."""
        import re
        return re.findall(r'\{(\w+)\}', self.template)

    def format(self, **kwargs) -> str:
        """Format template with provided values."""
        return self.template.format(**kwargs)


# Create template library
templates = {
    "sentiment": PromptTemplate(
        "sentiment",
        "Classify the sentiment of the following text as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:",
        "Zero-shot sentiment classification"
    ),
    "summarize": PromptTemplate(
        "summarize",
        "Summarize the following text in {length} sentences.\n\nText: {text}\n\nSummary:",
        "Controllable length summarization"
    ),
    "extract": PromptTemplate(
        "extract",
        """Extract entities from the text and return as JSON.

Text: {text}

Return a JSON object with:
- people: list of person names
- organizations: list of organization names
- locations: list of location names

JSON:""",
        "Named entity extraction"
    )
}

print("Available templates:")
for name, template in templates.items():
    print(f"  {name}: {template.description}")
    print(f"    Variables: {template.variables}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prompting Techniques

# COMMAND ----------

# Zero-shot: Direct instruction
zero_shot = """Classify the sentiment of this review:
"This product is amazing! Best purchase ever."

Sentiment:"""

# Few-shot: Include examples
few_shot = """Classify sentiment as positive, negative, or neutral.

Review: "I love this!"
Sentiment: positive

Review: "Terrible quality."
Sentiment: negative

Review: "It arrived on time."
Sentiment: neutral

Review: "This product is amazing! Best purchase ever."
Sentiment:"""

# Chain-of-thought: Request reasoning
chain_of_thought = """Solve this step by step:
If a train travels at 60 mph for 2.5 hours, how far does it travel?

Let me think through this:
1."""

# Role-based: Assign persona
role_based = """You are an expert code reviewer with 20 years of experience.
Review this Python code for bugs, security issues, and best practices:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return db.execute(query)
```

Review:"""

print("Prompting Techniques:\n")
print("1. ZERO-SHOT (direct instruction)")
print("-" * 40)
print(zero_shot[:100] + "...")
print()

print("2. FEW-SHOT (with examples)")
print("-" * 40)
print(few_shot[:150] + "...")
print()

print("3. CHAIN-OF-THOUGHT (step-by-step)")
print("-" * 40)
print(chain_of_thought[:100] + "...")
print()

print("4. ROLE-BASED (expert persona)")
print("-" * 40)
print(role_based[:100] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prompt Builder

# COMMAND ----------

class PromptBuilder:
    """Fluent interface for building prompts."""

    def __init__(self, instruction: str):
        self.instruction = instruction
        self._system = None
        self._context = []
        self._examples = []
        self._output_format = None

    def system(self, message: str) -> 'PromptBuilder':
        self._system = message
        return self

    def add_context(self, context: str) -> 'PromptBuilder':
        self._context.append(context)
        return self

    def add_example(self, input_text: str, output_text: str) -> 'PromptBuilder':
        self._examples.append((input_text, output_text))
        return self

    def output_format(self, format_desc: str) -> 'PromptBuilder':
        self._output_format = format_desc
        return self

    def build(self) -> str:
        parts = []

        if self._system:
            parts.append(f"System: {self._system}\n")

        if self._context:
            parts.append("Context:")
            for ctx in self._context:
                parts.append(f"- {ctx}")
            parts.append("")

        if self._examples:
            parts.append("Examples:")
            for inp, out in self._examples:
                parts.append(f"Input: {inp}")
                parts.append(f"Output: {out}")
                parts.append("")

        parts.append(f"Instruction: {self.instruction}")

        if self._output_format:
            parts.append(f"Output format: {self._output_format}")

        parts.append("\nResponse:")

        return "\n".join(parts)


# Build a complex prompt
prompt = (
    PromptBuilder("Classify the sentiment")
    .system("You are a sentiment analysis expert.")
    .add_context("Analyzing customer feedback")
    .add_example("Great service!", "positive")
    .add_example("Long wait times.", "negative")
    .output_format("One word: positive, negative, or neutral")
    .build()
)

print("Built prompt:")
print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Structured Output

# COMMAND ----------

import json

def parse_json_output(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM output."""
    # Find JSON in the text
    start = text.find('{')
    end = text.rfind('}')

    if start == -1 or end == -1:
        # Try array
        start = text.find('[')
        end = text.rfind(']')

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            return None
    return None


# Example: Extract structured data
sample_output = """
Based on the text, I extracted the following entities:

{
    "people": ["John Smith", "Jane Doe"],
    "organizations": ["Acme Corp", "TechStart"],
    "locations": ["New York", "Boston"]
}

The text mentions two people and two organizations.
"""

parsed = parse_json_output(sample_output)
print("Parsed JSON output:")
print(json.dumps(parsed, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prompt Metrics

# COMMAND ----------

def analyze_prompt(prompt: str) -> dict:
    """Analyze prompt characteristics."""
    words = prompt.split()

    # Detect technique
    has_examples = "Example" in prompt or "Input:" in prompt
    has_cot = "step by step" in prompt.lower() or "think through" in prompt.lower()
    has_system = "System:" in prompt or "You are" in prompt
    has_format = "JSON" in prompt or "format:" in prompt.lower()

    if has_cot:
        technique = "chain-of-thought"
    elif has_examples:
        technique = "few-shot"
    elif has_system:
        technique = "role-based"
    elif has_format:
        technique = "structured"
    else:
        technique = "zero-shot"

    return {
        "word_count": len(words),
        "estimated_tokens": len(words) * 1.3,  # Rough estimate
        "technique": technique,
        "has_examples": has_examples,
        "has_system_message": has_system,
        "has_output_format": has_format
    }


# Analyze our prompts
prompts_to_analyze = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "built_prompt": prompt
}

print("Prompt Analysis:")
print("-" * 60)
for name, p in prompts_to_analyze.items():
    metrics = analyze_prompt(p)
    print(f"\n{name}:")
    print(f"  Technique: {metrics['technique']}")
    print(f"  Estimated tokens: {metrics['estimated_tokens']:.0f}")
    print(f"  Has examples: {metrics['has_examples']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Best Practices

# COMMAND ----------

best_practices = """
Prompt Engineering Best Practices:

1. **Be Specific**
   - Clear, unambiguous instructions
   - Define expected output format
   - Specify constraints

2. **Use Examples (Few-Shot)**
   - 2-5 examples usually optimal
   - Show edge cases
   - Consistent formatting

3. **Chain of Thought**
   - Use for complex reasoning
   - "Let's think step by step"
   - Show intermediate steps

4. **Role Assignment**
   - "You are an expert in..."
   - Provides context and tone
   - Improves domain accuracy

5. **Iterate and Test**
   - A/B test prompts
   - Measure output quality
   - Track token usage
"""

print(best_practices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: Databricks vs Sovereign AI Stack
# MAGIC
# MAGIC | Feature | Databricks AI Playground | Sovereign AI |
# MAGIC |---------|-------------------------|--------------|
# MAGIC | **Interface** | Web UI | CLI/Library |
# MAGIC | **Templates** | Custom | Code-based |
# MAGIC | **A/B Testing** | Manual | Programmatic |
# MAGIC | **Version Control** | Workspace | Git |
# MAGIC | **Metrics** | Built-in | Custom |

# COMMAND ----------

print("Demo complete!")
print("\nKey takeaways:")
print("1. Templates enable reusable, parameterized prompts")
print("2. Five main techniques: zero-shot, few-shot, CoT, role, structured")
print("3. Prompt builder pattern enables composition")
print("4. Always parse and validate structured outputs")
print("5. Track metrics for optimization")
