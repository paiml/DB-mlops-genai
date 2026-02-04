# Lab: Prompt Templates

Build type-safe prompt templates with variable substitution.

## Objectives

- Create reusable templates
- Implement variable validation
- Build a prompt library

## Demo Code

See [`demos/course4/week2/prompt-engineering/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course4/week2/prompt-engineering)

## Lab Exercise

See [`labs/course4/week2/lab_2_6_prompt_templates.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course4/week2)

## Key Implementation

```rust
pub struct PromptTemplate {
    template: String,
    variables: Vec<String>,
}

impl PromptTemplate {
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String, PromptError> {
        let mut result = self.template.clone();
        for var in &self.variables {
            let value = vars.get(var)
                .ok_or(PromptError::MissingVariable(var.clone()))?;
            result = result.replace(&format!("{{{}}}", var), value);
        }
        Ok(result)
    }
}
```
