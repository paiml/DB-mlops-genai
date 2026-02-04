//! Prompt Engineering Patterns Demo - Course 4 Week 1
//!
//! Demonstrates prompt engineering techniques that map to Databricks AI features.
//! Shows template patterns, few-shot learning, and chain-of-thought prompting.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum PromptError {
    #[error("Template error: {0}")]
    Template(String),
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Validation error: {0}")]
    Validation(String),
}

// ============================================================================
// Prompt Templates
// ============================================================================

/// A prompt template with variable substitution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    name: String,
    template: String,
    variables: Vec<String>,
    description: Option<String>,
}

impl PromptTemplate {
    /// Create a new prompt template
    pub fn new(name: &str, template: &str) -> Self {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let variables: Vec<String> = re
            .captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();

        Self {
            name: name.to_string(),
            template: template.to_string(),
            variables,
            description: None,
        }
    }

    /// Add description to template
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Format template with provided variables
    pub fn format(&self, vars: &HashMap<String, String>) -> Result<String, PromptError> {
        let mut result = self.template.clone();

        for var in &self.variables {
            let value = vars.get(var).ok_or_else(|| {
                PromptError::VariableNotFound(format!("Variable '{}' not provided", var))
            })?;
            result = result.replace(&format!("{{{}}}", var), value);
        }

        Ok(result)
    }

    /// Get list of required variables
    pub fn required_variables(&self) -> &[String] {
        &self.variables
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn template_str(&self) -> &str {
        &self.template
    }
}

// ============================================================================
// Few-Shot Examples
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub input: String,
    pub output: String,
    pub explanation: Option<String>,
}

impl FewShotExample {
    pub fn new(input: &str, output: &str) -> Self {
        Self {
            input: input.to_string(),
            output: output.to_string(),
            explanation: None,
        }
    }

    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = Some(explanation.to_string());
        self
    }
}

/// Few-shot prompt builder
#[derive(Debug, Clone)]
pub struct FewShotPrompt {
    task_description: String,
    examples: Vec<FewShotExample>,
    input_prefix: String,
    output_prefix: String,
}

impl FewShotPrompt {
    pub fn new(task_description: &str) -> Self {
        Self {
            task_description: task_description.to_string(),
            examples: Vec::new(),
            input_prefix: "Input".to_string(),
            output_prefix: "Output".to_string(),
        }
    }

    pub fn add_example(mut self, example: FewShotExample) -> Self {
        self.examples.push(example);
        self
    }

    pub fn with_prefixes(mut self, input: &str, output: &str) -> Self {
        self.input_prefix = input.to_string();
        self.output_prefix = output.to_string();
        self
    }

    /// Build the complete few-shot prompt
    pub fn build(&self, query: &str) -> String {
        let mut prompt = format!("{}\n\n", self.task_description);

        for example in &self.examples {
            prompt.push_str(&format!("{}: {}\n", self.input_prefix, example.input));
            prompt.push_str(&format!("{}: {}\n\n", self.output_prefix, example.output));
        }

        prompt.push_str(&format!("{}: {}\n", self.input_prefix, query));
        prompt.push_str(&format!("{}:", self.output_prefix));

        prompt
    }

    pub fn example_count(&self) -> usize {
        self.examples.len()
    }
}

// ============================================================================
// Chain of Thought Prompting
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfThought {
    steps: Vec<String>,
    final_answer: Option<String>,
}

impl ChainOfThought {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_answer: None,
        }
    }

    pub fn add_step(mut self, step: &str) -> Self {
        self.steps.push(step.to_string());
        self
    }

    pub fn with_answer(mut self, answer: &str) -> Self {
        self.final_answer = Some(answer.to_string());
        self
    }

    pub fn format(&self) -> String {
        let mut output = String::new();

        for (i, step) in self.steps.iter().enumerate() {
            output.push_str(&format!("Step {}: {}\n", i + 1, step));
        }

        if let Some(answer) = &self.final_answer {
            output.push_str(&format!("\nTherefore, the answer is: {}", answer));
        }

        output
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

impl Default for ChainOfThought {
    fn default() -> Self {
        Self::new()
    }
}

/// Chain-of-thought prompt builder
#[derive(Debug, Clone)]
pub struct CoTPrompt {
    instruction: String,
    examples: Vec<(String, ChainOfThought)>,
}

impl CoTPrompt {
    pub fn new(instruction: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            examples: Vec::new(),
        }
    }

    pub fn add_example(mut self, problem: &str, reasoning: ChainOfThought) -> Self {
        self.examples.push((problem.to_string(), reasoning));
        self
    }

    pub fn build(&self, query: &str) -> String {
        let mut prompt = format!("{}\n\n", self.instruction);

        for (problem, reasoning) in &self.examples {
            prompt.push_str(&format!("Problem: {}\n", problem));
            prompt.push_str(&format!("Reasoning:\n{}\n\n", reasoning.format()));
        }

        prompt.push_str(&format!("Problem: {}\n", query));
        prompt.push_str("Reasoning:\n");

        prompt
    }
}

// ============================================================================
// System Prompts
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    role: String,
    capabilities: Vec<String>,
    constraints: Vec<String>,
    style: Option<String>,
}

impl SystemPrompt {
    pub fn new(role: &str) -> Self {
        Self {
            role: role.to_string(),
            capabilities: Vec::new(),
            constraints: Vec::new(),
            style: None,
        }
    }

    pub fn add_capability(mut self, cap: &str) -> Self {
        self.capabilities.push(cap.to_string());
        self
    }

    pub fn add_constraint(mut self, constraint: &str) -> Self {
        self.constraints.push(constraint.to_string());
        self
    }

    pub fn with_style(mut self, style: &str) -> Self {
        self.style = Some(style.to_string());
        self
    }

    pub fn build(&self) -> String {
        let mut prompt = format!("You are {}.\n\n", self.role);

        if !self.capabilities.is_empty() {
            prompt.push_str("Capabilities:\n");
            for cap in &self.capabilities {
                prompt.push_str(&format!("- {}\n", cap));
            }
            prompt.push('\n');
        }

        if !self.constraints.is_empty() {
            prompt.push_str("Constraints:\n");
            for constraint in &self.constraints {
                prompt.push_str(&format!("- {}\n", constraint));
            }
            prompt.push('\n');
        }

        if let Some(style) = &self.style {
            prompt.push_str(&format!("Response style: {}\n", style));
        }

        prompt
    }
}

// ============================================================================
// Prompt Library
// ============================================================================

#[derive(Debug, Default)]
pub struct PromptLibrary {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn add(&mut self, template: PromptTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    pub fn count(&self) -> usize {
        self.templates.len()
    }
}

// ============================================================================
// Output Parsing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutput {
    pub format: OutputFormat,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Markdown,
    Csv,
    PlainText,
}

impl StructuredOutput {
    pub fn json(fields: Vec<&str>) -> Self {
        Self {
            format: OutputFormat::Json,
            fields: fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn markdown(fields: Vec<&str>) -> Self {
        Self {
            format: OutputFormat::Markdown,
            fields: fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn format_instruction(&self) -> String {
        match &self.format {
            OutputFormat::Json => {
                let fields_str = self
                    .fields
                    .iter()
                    .map(|f| format!("\"{}\": \"...\"", f))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Respond in JSON format: {{{}}}", fields_str)
            }
            OutputFormat::Markdown => {
                let fields_str = self
                    .fields
                    .iter()
                    .map(|f| format!("## {}", f))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("Respond in Markdown format with sections:\n{}", fields_str)
            }
            OutputFormat::Csv => {
                let headers = self.fields.join(",");
                format!("Respond in CSV format with headers: {}", headers)
            }
            OutputFormat::PlainText => "Respond in plain text.".to_string(),
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Prompt Engineering Demo - Course 4 Week 1                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Basic Templates
    println!("📝 Step 1: Prompt Templates");
    let template = PromptTemplate::new(
        "qa",
        "Answer the following question about {topic}:\n\nQuestion: {question}\n\nAnswer:",
    )
    .with_description("Basic Q&A template");

    println!("   Template: {}", template.name());
    println!(
        "   Variables: {:?}",
        template.required_variables()
    );

    let mut vars = HashMap::new();
    vars.insert("topic".to_string(), "machine learning".to_string());
    vars.insert(
        "question".to_string(),
        "What is gradient descent?".to_string(),
    );

    match template.format(&vars) {
        Ok(prompt) => println!("   Formatted:\n   {}\n", prompt.replace('\n', "\n   ")),
        Err(e) => println!("   Error: {}", e),
    }

    // Step 2: Few-Shot Prompting
    println!("🎯 Step 2: Few-Shot Prompting");
    let few_shot = FewShotPrompt::new("Classify the sentiment of the following text.")
        .add_example(FewShotExample::new("I love this product!", "Positive"))
        .add_example(FewShotExample::new("This is terrible.", "Negative"))
        .add_example(FewShotExample::new("It's okay, nothing special.", "Neutral"));

    let prompt = few_shot.build("The food was absolutely delicious!");
    println!("   Examples: {}", few_shot.example_count());
    println!("   Prompt:\n   {}\n", prompt.replace('\n', "\n   "));

    // Step 3: Chain-of-Thought
    println!("🔗 Step 3: Chain-of-Thought Prompting");
    let cot = ChainOfThought::new()
        .add_step("First, identify the key information")
        .add_step("Then, apply the relevant formula")
        .add_step("Finally, calculate the result")
        .with_answer("42");

    println!("   Steps: {}", cot.step_count());
    println!("   Reasoning:\n   {}\n", cot.format().replace('\n', "\n   "));

    // Step 4: System Prompts
    println!("🤖 Step 4: System Prompts");
    let system = SystemPrompt::new("a helpful AI assistant specialized in data science")
        .add_capability("Explain complex ML concepts simply")
        .add_capability("Provide code examples in Python")
        .add_constraint("Do not provide harmful advice")
        .add_constraint("Acknowledge limitations")
        .with_style("Concise and technical");

    println!(
        "   System Prompt:\n   {}\n",
        system.build().replace('\n', "\n   ")
    );

    // Step 5: Structured Output
    println!("📋 Step 5: Structured Output");
    let json_output = StructuredOutput::json(vec!["summary", "confidence", "entities"]);
    let md_output = StructuredOutput::markdown(vec!["Overview", "Details", "Conclusion"]);

    println!("   JSON instruction: {}", json_output.format_instruction());
    println!(
        "   Markdown instruction:\n   {}\n",
        md_output.format_instruction().replace('\n', "\n   ")
    );

    // Step 6: Prompt Library
    println!("📚 Step 6: Prompt Library");
    let mut library = PromptLibrary::new();
    library.add(
        PromptTemplate::new("summarize", "Summarize the following text:\n\n{text}\n\nSummary:")
            .with_description("Text summarization"),
    );
    library.add(
        PromptTemplate::new(
            "translate",
            "Translate the following from {source} to {target}:\n\n{text}\n\nTranslation:",
        )
        .with_description("Language translation"),
    );

    println!("   Templates: {:?}", library.list());
    println!("   Count: {}\n", library.count());

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key patterns demonstrated:");
    println!("  • Variable-based prompt templates");
    println!("  • Few-shot learning with examples");
    println!("  • Chain-of-thought reasoning");
    println!("  • System prompt configuration");
    println!("  • Structured output formatting");
    println!();
    println!("Databricks equivalent: AI Functions, Foundation Model APIs");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_basic() {
        let template = PromptTemplate::new("test", "Hello {name}!");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());
        let result = template.format(&vars).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_variables() {
        let template = PromptTemplate::new("test", "{a} and {b}");
        assert_eq!(template.required_variables().len(), 2);
    }

    #[test]
    fn test_template_missing_var() {
        let template = PromptTemplate::new("test", "Hello {name}!");
        let vars = HashMap::new();
        assert!(template.format(&vars).is_err());
    }

    #[test]
    fn test_few_shot_build() {
        let few_shot = FewShotPrompt::new("Classify:")
            .add_example(FewShotExample::new("good", "positive"));
        let prompt = few_shot.build("test");
        assert!(prompt.contains("Classify:"));
        assert!(prompt.contains("good"));
        assert!(prompt.contains("test"));
    }

    #[test]
    fn test_few_shot_count() {
        let few_shot = FewShotPrompt::new("Test")
            .add_example(FewShotExample::new("a", "b"))
            .add_example(FewShotExample::new("c", "d"));
        assert_eq!(few_shot.example_count(), 2);
    }

    #[test]
    fn test_chain_of_thought() {
        let cot = ChainOfThought::new()
            .add_step("Step one")
            .with_answer("42");
        let output = cot.format();
        assert!(output.contains("Step 1:"));
        assert!(output.contains("42"));
    }

    #[test]
    fn test_cot_step_count() {
        let cot = ChainOfThought::new()
            .add_step("a")
            .add_step("b")
            .add_step("c");
        assert_eq!(cot.step_count(), 3);
    }

    #[test]
    fn test_system_prompt() {
        let system = SystemPrompt::new("an assistant")
            .add_capability("help")
            .add_constraint("be safe");
        let prompt = system.build();
        assert!(prompt.contains("assistant"));
        assert!(prompt.contains("help"));
        assert!(prompt.contains("be safe"));
    }

    #[test]
    fn test_structured_output_json() {
        let output = StructuredOutput::json(vec!["a", "b"]);
        let instruction = output.format_instruction();
        assert!(instruction.contains("JSON"));
        assert!(instruction.contains("\"a\""));
    }

    #[test]
    fn test_structured_output_markdown() {
        let output = StructuredOutput::markdown(vec!["Overview"]);
        let instruction = output.format_instruction();
        assert!(instruction.contains("Markdown"));
    }

    #[test]
    fn test_prompt_library() {
        let mut lib = PromptLibrary::new();
        lib.add(PromptTemplate::new("test", "Hello"));
        assert_eq!(lib.count(), 1);
        assert!(lib.get("test").is_some());
    }

    #[test]
    fn test_prompt_error() {
        let err = PromptError::VariableNotFound("x".to_string());
        assert!(err.to_string().contains("x"));
    }
}
