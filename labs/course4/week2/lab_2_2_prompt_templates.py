# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2.2: Build a Prompt Template Engine
# MAGIC
# MAGIC **Course 4, Week 2: Prompt Engineering**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Create reusable prompt templates
# MAGIC - Implement variable substitution
# MAGIC - Build a template library

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

print("Prompt Template Lab - Course 4 Week 2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Basic Template
# MAGIC
# MAGIC Implement a simple template with variable substitution.

# COMMAND ----------

@dataclass
class PromptTemplate:
    """Reusable prompt template."""
    name: str
    template: str
    description: str = ""

    def get_variables(self) -> List[str]:
        """Extract variable names from template."""
        return re.findall(r'\{(\w+)\}', self.template)

    def format(self, **kwargs) -> str:
        """Format template with provided values."""
        return self.template.format(**kwargs)


# Example template
sentiment_template = PromptTemplate(
    name="sentiment",
    template="Classify the sentiment of this text as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:",
    description="Zero-shot sentiment classification"
)

print(f"Template: {sentiment_template.name}")
print(f"Variables: {sentiment_template.get_variables()}")
print(f"\nFormatted:")
print(sentiment_template.format(text="I love this product!"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Template Library
# MAGIC
# MAGIC TODO: Implement a TemplateLibrary class.

# COMMAND ----------

class TemplateLibrary:
    """Collection of prompt templates."""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def add(self, template: PromptTemplate) -> None:
        """Add template to library."""
        # TODO: Store template by name
        # YOUR CODE HERE
        pass

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        # TODO: Return template or None if not found
        # YOUR CODE HERE
        pass

    def list_templates(self) -> List[str]:
        """List all template names."""
        # TODO: Return list of template names
        # YOUR CODE HERE
        pass

    def render(self, name: str, **kwargs) -> str:
        """Render a template with variables."""
        # TODO: Get template and format it
        # YOUR CODE HERE
        pass


# Test your library
# library = TemplateLibrary()
# library.add(sentiment_template)
# result = library.render("sentiment", text="Great experience!")
# print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Create Template Collection
# MAGIC
# MAGIC TODO: Create templates for common tasks.

# COMMAND ----------

# TODO: Create templates for:
# 1. Summarization (with {text} and {length} variables)
# 2. Translation (with {text} and {target_language} variables)
# 3. Question answering (with {context} and {question} variables)
# 4. Code explanation (with {code} and {language} variables)

# YOUR CODE HERE
# summarize_template = PromptTemplate(...)
# translate_template = PromptTemplate(...)
# qa_template = PromptTemplate(...)
# code_template = PromptTemplate(...)

# Add all to library
# library.add(summarize_template)
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Template Validation
# MAGIC
# MAGIC TODO: Add validation to ensure all variables are provided.

# COMMAND ----------

def validate_template(template: PromptTemplate, **kwargs) -> Dict:
    """
    Validate that all required variables are provided.

    TODO: Return dict with:
    - valid: bool
    - missing: list of missing variable names
    - extra: list of extra variable names not in template
    """
    # YOUR CODE HERE
    pass


# Test validation
# result = validate_template(sentiment_template, text="test")  # Should be valid
# result = validate_template(sentiment_template)  # Should be invalid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Template Chaining
# MAGIC
# MAGIC TODO: Implement template chaining for multi-step prompts.

# COMMAND ----------

class PromptChain:
    """Chain multiple templates together."""

    def __init__(self):
        self.steps: List[tuple] = []  # (template, output_var)

    def add_step(self, template: PromptTemplate, output_var: str) -> 'PromptChain':
        """Add a step to the chain."""
        # TODO: Store template and output variable name
        # YOUR CODE HERE
        pass
        return self

    def run(self, initial_vars: Dict, executor=None) -> Dict:
        """
        Execute the chain.

        TODO:
        1. Start with initial variables
        2. For each step, render template and (optionally) execute
        3. Store result in output_var for next step
        4. Return final variables dict
        """
        # YOUR CODE HERE
        pass


# Example chain:
# chain = PromptChain()
# chain.add_step(summarize_template, "summary")
# chain.add_step(translate_template, "translation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        # Check library
        library = TemplateLibrary()
        library.add(sentiment_template)
        checks.append(("Library add works", library.get("sentiment") is not None))

        # Check render
        rendered = library.render("sentiment", text="test")
        checks.append(("Render works", rendered is not None and "test" in rendered))

        # Check list
        templates = library.list_templates()
        checks.append(("List works", "sentiment" in templates))

        # Check validation
        if 'validate_template' in dir():
            result = validate_template(sentiment_template, text="test")
            checks.append(("Validation works", result.get("valid", False)))

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
