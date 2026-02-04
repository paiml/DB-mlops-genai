# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Engineering on Databricks
# MAGIC
# MAGIC **Course 4, Week 1: Prompt Engineering**
# MAGIC
# MAGIC This notebook demonstrates prompt engineering techniques using Databricks AI Functions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. AI Functions in SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use AI functions for text analysis
# MAGIC SELECT
# MAGIC   ai_query(
# MAGIC     'databricks-dbrx-instruct',
# MAGIC     CONCAT('Classify the sentiment of this text as positive, negative, or neutral: "', text, '"')
# MAGIC   ) as sentiment
# MAGIC FROM (
# MAGIC   SELECT 'I love this product!' as text
# MAGIC   UNION ALL
# MAGIC   SELECT 'This is terrible.' as text
# MAGIC   UNION ALL
# MAGIC   SELECT 'It was okay.' as text
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Few-Shot Prompting

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

# Few-shot prompt for classification
few_shot_prompt = """Classify the sentiment of text as positive, negative, or neutral.

Text: I absolutely love this!
Sentiment: positive

Text: This is the worst experience ever.
Sentiment: negative

Text: It's fine, nothing special.
Sentiment: neutral

Text: The food was absolutely delicious!
Sentiment:"""

response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "prompt": few_shot_prompt,
        "max_tokens": 10,
        "temperature": 0.1
    }
)

print("Sentiment:", response["choices"][0]["text"].strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chain of Thought

# COMMAND ----------

cot_prompt = """Let's solve this step by step.

Question: If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?

Let me think through this:
Step 1: Identify the speed - 60 miles per hour
Step 2: Identify the time - 2.5 hours
Step 3: Calculate distance = speed × time = 60 × 2.5 = 150 miles

The train will travel 150 miles.

Question: If a car uses 5 gallons of gas to travel 150 miles, how many gallons does it need for 300 miles?

Let me think through this:"""

response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "prompt": cot_prompt,
        "max_tokens": 200,
        "temperature": 0.3
    }
)

print("Chain of Thought Response:")
print(response["choices"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Structured Output

# COMMAND ----------

json_prompt = """Extract information from the text and return as JSON.

Text: "John Smith is a 35-year-old software engineer from Seattle who loves hiking."

Return a JSON object with fields: name, age, occupation, location, hobby

JSON:"""

response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "prompt": json_prompt,
        "max_tokens": 100,
        "temperature": 0.1
    }
)

print("Structured Output:")
print(response["choices"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. System Prompts

# COMMAND ----------

response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "messages": [
            {
                "role": "system",
                "content": "You are a data science expert. Provide concise, technical answers. Always include a code example when relevant."
            },
            {
                "role": "user",
                "content": "How do I normalize a feature in Python?"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }
)

print("Expert Response:")
print(response["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prompt Templates

# COMMAND ----------

from string import Template

# Define reusable template
qa_template = Template("""Answer the following question about $topic.

Question: $question

Provide a clear, concise answer:""")

# Use template
prompt = qa_template.substitute(
    topic="machine learning",
    question="What is gradient descent?"
)

response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.5
    }
)

print("Templated Response:")
print(response["choices"][0]["text"])
