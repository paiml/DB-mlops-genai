//! Prompt Engineering Demo - Course 4 Week 2
//!
//! Demonstrates prompt engineering techniques that map to Databricks AI Playground.
//! Shows templating, chain-of-thought, and structured outputs.

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
    #[error("Validation error: {0}")]
    Validation(String),
}

// ============================================================================
// Prompt Templates
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub template: String,
    pub variables: Vec<String>,
    pub description: String,
}

impl PromptTemplate {
    pub fn new(name: &str, template: &str) -> Self {
        let variables = Self::extract_variables(template);
        Self {
            name: name.to_string(),
            template: template.to_string(),
            variables,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                let mut var_name = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '}' {
                        chars.next();
                        if !var_name.is_empty() && !vars.contains(&var_name) {
                            vars.push(var_name);
                        }
                        break;
                    }
                    var_name.push(chars.next().unwrap());
                }
            }
        }
        vars
    }

    pub fn format(&self, values: &HashMap<String, String>) -> Result<String, PromptError> {
        let mut result = self.template.clone();

        for var in &self.variables {
            let placeholder = format!("{{{}}}", var);
            let value = values
                .get(var)
                .ok_or_else(|| PromptError::Template(format!("Missing variable: {}", var)))?;
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }
}

// ============================================================================
// Prompt Library
// ============================================================================

pub struct PromptLibrary {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    pub fn new() -> Self {
        let mut lib = Self {
            templates: HashMap::new(),
        };
        lib.load_defaults();
        lib
    }

    fn load_defaults(&mut self) {
        // Zero-shot classification
        self.add(PromptTemplate::new(
            "classify_sentiment",
            "Classify the sentiment of the following text as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"
        ).with_description("Zero-shot sentiment classification"));

        // Few-shot classification
        self.add(
            PromptTemplate::new(
                "classify_sentiment_few_shot",
                r#"Classify the sentiment of texts as positive, negative, or neutral.

Text: "I love this product! It's amazing."
Sentiment: positive

Text: "This is terrible. Complete waste of money."
Sentiment: negative

Text: "The product arrived on time."
Sentiment: neutral

Text: "{text}"
Sentiment:"#,
            )
            .with_description("Few-shot sentiment classification with examples"),
        );

        // Chain of thought
        self.add(
            PromptTemplate::new(
                "math_cot",
                r#"Solve the following problem step by step.

Problem: {problem}

Let me think through this step by step:
1."#,
            )
            .with_description("Chain-of-thought math reasoning"),
        );

        // Structured output
        self.add(
            PromptTemplate::new(
                "extract_entities",
                r#"Extract entities from the text and return as JSON.

Text: {text}

Return a JSON object with these fields:
- people: list of person names
- organizations: list of organization names
- locations: list of location names

JSON:"#,
            )
            .with_description("Named entity extraction with JSON output"),
        );

        // Role-based
        self.add(
            PromptTemplate::new(
                "code_review",
                r#"You are an expert code reviewer. Review the following code for:
1. Bugs and errors
2. Security vulnerabilities
3. Performance issues
4. Code style

Code:
```{language}
{code}
```

Review:"#,
            )
            .with_description("Code review with specific criteria"),
        );

        // Summarization
        self.add(
            PromptTemplate::new(
                "summarize",
                "Summarize the following text in {length} sentences.\n\nText: {text}\n\nSummary:",
            )
            .with_description("Controllable length summarization"),
        );
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
}

impl Default for PromptLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Prompt Techniques
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTechnique {
    ZeroShot,
    FewShot,
    ChainOfThought,
    RoleBased,
    Structured,
}

impl PromptTechnique {
    pub fn description(&self) -> &'static str {
        match self {
            Self::ZeroShot => "Direct instruction without examples",
            Self::FewShot => "Include examples in the prompt",
            Self::ChainOfThought => "Request step-by-step reasoning",
            Self::RoleBased => "Assign a specific role/persona",
            Self::Structured => "Request specific output format (JSON, etc.)",
        }
    }
}

// ============================================================================
// Prompt Builder
// ============================================================================

#[derive(Debug, Clone)]
pub struct PromptBuilder {
    system: Option<String>,
    context: Vec<String>,
    examples: Vec<(String, String)>,
    instruction: String,
    output_format: Option<String>,
}

impl PromptBuilder {
    pub fn new(instruction: &str) -> Self {
        Self {
            system: None,
            context: Vec::new(),
            examples: Vec::new(),
            instruction: instruction.to_string(),
            output_format: None,
        }
    }

    pub fn system(mut self, system: &str) -> Self {
        self.system = Some(system.to_string());
        self
    }

    pub fn add_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }

    pub fn add_example(mut self, input: &str, output: &str) -> Self {
        self.examples.push((input.to_string(), output.to_string()));
        self
    }

    pub fn output_format(mut self, format: &str) -> Self {
        self.output_format = Some(format.to_string());
        self
    }

    pub fn build(&self) -> String {
        let mut parts = Vec::new();

        // System message
        if let Some(ref system) = self.system {
            parts.push(format!("System: {}", system));
        }

        // Context
        if !self.context.is_empty() {
            parts.push("Context:".to_string());
            for ctx in &self.context {
                parts.push(format!("- {}", ctx));
            }
        }

        // Examples (few-shot)
        if !self.examples.is_empty() {
            parts.push("\nExamples:".to_string());
            for (input, output) in &self.examples {
                parts.push(format!("Input: {}", input));
                parts.push(format!("Output: {}", output));
                parts.push(String::new());
            }
        }

        // Instruction
        parts.push(format!("\nInstruction: {}", self.instruction));

        // Output format
        if let Some(ref format) = self.output_format {
            parts.push(format!("\nOutput format: {}", format));
        }

        parts.push("\nResponse:".to_string());

        parts.join("\n")
    }
}

// ============================================================================
// Output Parsers
// ============================================================================

#[derive(Debug, Clone)]
pub struct JsonOutputParser;

impl JsonOutputParser {
    pub fn parse<T: for<'de> Deserialize<'de>>(output: &str) -> Result<T, PromptError> {
        // Find JSON in the output
        let json_start = output.find('{').or_else(|| output.find('['));
        let json_end = output.rfind('}').or_else(|| output.rfind(']'));

        match (json_start, json_end) {
            (Some(start), Some(end)) if end >= start => {
                let json_str = &output[start..=end];
                serde_json::from_str(json_str)
                    .map_err(|e| PromptError::Validation(format!("JSON parse error: {}", e)))
            }
            _ => Err(PromptError::Validation(
                "No valid JSON found in output".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtraction {
    pub people: Vec<String>,
    pub organizations: Vec<String>,
    pub locations: Vec<String>,
}

// ============================================================================
// Prompt Evaluation
// ============================================================================

#[derive(Debug, Clone)]
pub struct PromptMetrics {
    pub token_count: usize,
    pub example_count: usize,
    pub has_system: bool,
    pub has_output_format: bool,
    pub technique: PromptTechnique,
}

impl PromptMetrics {
    pub fn analyze(prompt: &str, builder: &PromptBuilder) -> Self {
        let token_count = prompt.split_whitespace().count(); // Approximate

        let technique = if !builder.examples.is_empty() {
            if prompt.contains("step by step") || prompt.contains("think through") {
                PromptTechnique::ChainOfThought
            } else {
                PromptTechnique::FewShot
            }
        } else if builder.system.is_some() {
            PromptTechnique::RoleBased
        } else if builder.output_format.is_some() {
            PromptTechnique::Structured
        } else {
            PromptTechnique::ZeroShot
        };

        Self {
            token_count,
            example_count: builder.examples.len(),
            has_system: builder.system.is_some(),
            has_output_format: builder.output_format.is_some(),
            technique,
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Prompt Engineering Demo - Course 4 Week 2                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Prompt Templates
    println!("📝 Step 1: Prompt Templates");
    let library = PromptLibrary::new();

    println!("   Available templates:");
    for name in library.list() {
        if let Some(template) = library.get(name) {
            println!("     - {}: {}", name, template.description);
        }
    }
    println!();

    // Format a template
    if let Some(template) = library.get("classify_sentiment") {
        let mut values = HashMap::new();
        values.insert(
            "text".to_string(),
            "This product exceeded my expectations!".to_string(),
        );

        match template.format(&values) {
            Ok(prompt) => {
                println!("   Formatted prompt:");
                println!("   {}\n", prompt);
            }
            Err(e) => println!("   Error: {}\n", e),
        }
    }

    // Step 2: Prompt Techniques
    println!("🎯 Step 2: Prompt Techniques");
    let techniques = [
        PromptTechnique::ZeroShot,
        PromptTechnique::FewShot,
        PromptTechnique::ChainOfThought,
        PromptTechnique::RoleBased,
        PromptTechnique::Structured,
    ];

    for technique in &techniques {
        println!("   {:?}: {}", technique, technique.description());
    }
    println!();

    // Step 3: Prompt Builder
    println!("🔧 Step 3: Prompt Builder");

    let builder = PromptBuilder::new("Classify the sentiment of the given text")
        .system("You are a sentiment analysis expert.")
        .add_context("Analyze customer reviews")
        .add_example("Great product!", "positive")
        .add_example("Terrible experience.", "negative")
        .output_format("One word: positive, negative, or neutral");

    let prompt = builder.build();
    println!("   Built prompt:");
    for line in prompt.lines() {
        println!("   {}", line);
    }
    println!();

    // Analyze the prompt
    let metrics = PromptMetrics::analyze(&prompt, &builder);
    println!("   Metrics:");
    println!("     Token count: ~{}", metrics.token_count);
    println!("     Examples: {}", metrics.example_count);
    println!("     Technique: {:?}\n", metrics.technique);

    // Step 4: Chain of Thought
    println!("💭 Step 4: Chain of Thought");
    if let Some(template) = library.get("math_cot") {
        let mut values = HashMap::new();
        values.insert(
            "problem".to_string(),
            "If a train travels 60 mph for 2.5 hours, how far does it go?".to_string(),
        );

        match template.format(&values) {
            Ok(prompt) => {
                println!("   CoT Prompt:");
                for line in prompt.lines() {
                    println!("   {}", line);
                }
            }
            Err(e) => println!("   Error: {}", e),
        }
    }
    println!();

    // Step 5: Structured Output
    println!("📊 Step 5: Structured Output Parsing");
    let sample_output = r#"
Based on the text, here are the extracted entities:
{
    "people": ["John Smith", "Jane Doe"],
    "organizations": ["Acme Corp", "TechStart Inc"],
    "locations": ["New York", "San Francisco"]
}
"#;

    match JsonOutputParser::parse::<EntityExtraction>(sample_output) {
        Ok(entities) => {
            println!("   Parsed entities:");
            println!("     People: {:?}", entities.people);
            println!("     Organizations: {:?}", entities.organizations);
            println!("     Locations: {:?}", entities.locations);
        }
        Err(e) => println!("   Parse error: {}", e),
    }
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Prompt templates with variable substitution");
    println!("  • Five prompting techniques (zero-shot, few-shot, CoT, etc.)");
    println!("  • Prompt builder pattern for composition");
    println!("  • Structured output parsing (JSON)");
    println!();
    println!("Databricks equivalent: AI Playground, Prompt Engineering UI");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Template Tests
    // =========================================================================

    #[test]
    fn test_template_extraction() {
        let template = PromptTemplate::new("test", "Hello {name}, your order {order_id} is ready.");
        assert_eq!(template.variables, vec!["name", "order_id"]);
    }

    #[test]
    fn test_template_format() {
        let template = PromptTemplate::new("test", "Hello {name}!");
        let mut values = HashMap::new();
        values.insert("name".to_string(), "World".to_string());

        let result = template.format(&values).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_missing_var() {
        let template = PromptTemplate::new("test", "Hello {name}!");
        let values = HashMap::new();

        let result = template.format(&values);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_no_variables() {
        let template = PromptTemplate::new("static", "Hello World!");
        assert!(template.variables.is_empty());
        let result = template.format(&HashMap::new()).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_with_description() {
        let template =
            PromptTemplate::new("test", "Hello {name}!").with_description("A greeting template");
        assert_eq!(template.description, "A greeting template");
    }

    #[test]
    fn test_template_serialization() {
        let template = PromptTemplate::new("test", "Hello {name}!").with_description("Greeting");
        let json = serde_json::to_string(&template).unwrap();
        let restored: PromptTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(template.name, restored.name);
    }

    #[test]
    fn test_template_clone() {
        let template = PromptTemplate::new("test", "Hello {name}!");
        let cloned = template.clone();
        assert_eq!(template.name, cloned.name);
        assert_eq!(template.variables, cloned.variables);
    }

    #[test]
    fn test_template_multiple_same_var() {
        let template = PromptTemplate::new("test", "{name} says hi to {name}");
        assert_eq!(template.variables.len(), 1); // Should dedupe
    }

    // =========================================================================
    // Library Tests
    // =========================================================================

    #[test]
    fn test_prompt_library() {
        let library = PromptLibrary::new();
        assert!(library.get("classify_sentiment").is_some());
        assert!(library.get("nonexistent").is_none());
    }

    #[test]
    fn test_prompt_library_default() {
        let library = PromptLibrary::default();
        assert!(!library.list().is_empty());
    }

    #[test]
    fn test_prompt_library_list() {
        let library = PromptLibrary::new();
        let names = library.list();
        assert!(names.contains(&"classify_sentiment"));
        assert!(names.contains(&"math_cot"));
        assert!(names.contains(&"extract_entities"));
    }

    #[test]
    fn test_prompt_library_add() {
        let mut library = PromptLibrary::new();
        let initial_count = library.list().len();

        library.add(PromptTemplate::new("custom", "Custom template"));
        assert_eq!(library.list().len(), initial_count + 1);
        assert!(library.get("custom").is_some());
    }

    #[test]
    fn test_prompt_library_all_templates_valid() {
        let library = PromptLibrary::new();
        for name in library.list() {
            let template = library.get(name).unwrap();
            assert!(!template.template.is_empty());
        }
    }

    // =========================================================================
    // Builder Tests
    // =========================================================================

    #[test]
    fn test_prompt_builder() {
        let builder = PromptBuilder::new("Do something")
            .system("Be helpful")
            .add_example("input", "output");

        let prompt = builder.build();
        assert!(prompt.contains("System: Be helpful"));
        assert!(prompt.contains("Input: input"));
        assert!(prompt.contains("Output: output"));
    }

    #[test]
    fn test_prompt_builder_minimal() {
        let builder = PromptBuilder::new("Do this task");
        let prompt = builder.build();
        assert!(prompt.contains("Do this task"));
        assert!(prompt.contains("Response:"));
    }

    #[test]
    fn test_prompt_builder_with_context() {
        let builder = PromptBuilder::new("Analyze")
            .add_context("Context 1")
            .add_context("Context 2");
        let prompt = builder.build();
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("Context 1"));
        assert!(prompt.contains("Context 2"));
    }

    #[test]
    fn test_prompt_builder_with_output_format() {
        let builder =
            PromptBuilder::new("Classify").output_format("JSON object with 'label' field");
        let prompt = builder.build();
        assert!(prompt.contains("Output format:"));
    }

    #[test]
    fn test_prompt_builder_clone() {
        let builder = PromptBuilder::new("test").system("sys");
        let cloned = builder.clone();
        assert_eq!(builder.build(), cloned.build());
    }

    // =========================================================================
    // Technique Tests
    // =========================================================================

    #[test]
    fn test_prompt_technique_descriptions() {
        assert!(!PromptTechnique::ZeroShot.description().is_empty());
        assert!(!PromptTechnique::FewShot.description().is_empty());
        assert!(!PromptTechnique::ChainOfThought.description().is_empty());
        assert!(!PromptTechnique::RoleBased.description().is_empty());
        assert!(!PromptTechnique::Structured.description().is_empty());
    }

    #[test]
    fn test_prompt_technique_eq() {
        assert_eq!(PromptTechnique::ZeroShot, PromptTechnique::ZeroShot);
        assert_ne!(PromptTechnique::ZeroShot, PromptTechnique::FewShot);
    }

    #[test]
    fn test_prompt_technique_copy() {
        let t1 = PromptTechnique::FewShot;
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    // =========================================================================
    // Parser Tests
    // =========================================================================

    #[test]
    fn test_json_parser() {
        let output = r#"Here is the result: {"value": 42}"#;
        let parsed: HashMap<String, i32> = JsonOutputParser::parse(output).unwrap();
        assert_eq!(parsed.get("value"), Some(&42));
    }

    #[test]
    fn test_json_parser_no_json() {
        let output = "No JSON here";
        let result: Result<HashMap<String, i32>, _> = JsonOutputParser::parse(output);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_parser_array() {
        let output = "Result: [1, 2, 3]";
        let parsed: Vec<i32> = JsonOutputParser::parse(output).unwrap();
        assert_eq!(parsed, vec![1, 2, 3]);
    }

    #[test]
    fn test_json_parser_nested() {
        let output = r#"{"outer": {"inner": "value"}}"#;
        let parsed: HashMap<String, HashMap<String, String>> =
            JsonOutputParser::parse(output).unwrap();
        assert_eq!(
            parsed.get("outer").unwrap().get("inner"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_entity_extraction_serialization() {
        let entities = EntityExtraction {
            people: vec!["John".to_string()],
            organizations: vec!["Acme".to_string()],
            locations: vec!["NYC".to_string()],
        };
        let json = serde_json::to_string(&entities).unwrap();
        let restored: EntityExtraction = serde_json::from_str(&json).unwrap();
        assert_eq!(entities.people, restored.people);
    }

    // =========================================================================
    // Metrics Tests
    // =========================================================================

    #[test]
    fn test_prompt_metrics() {
        let builder = PromptBuilder::new("test")
            .add_example("a", "b")
            .add_example("c", "d");

        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);

        assert_eq!(metrics.example_count, 2);
        assert_eq!(metrics.technique, PromptTechnique::FewShot);
    }

    #[test]
    fn test_prompt_metrics_zero_shot() {
        let builder = PromptBuilder::new("Classify this text");
        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);

        assert_eq!(metrics.example_count, 0);
        assert_eq!(metrics.technique, PromptTechnique::ZeroShot);
    }

    #[test]
    fn test_prompt_metrics_role_based() {
        let builder = PromptBuilder::new("Classify").system("You are an expert classifier");
        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);

        assert!(metrics.has_system);
        assert_eq!(metrics.technique, PromptTechnique::RoleBased);
    }

    #[test]
    fn test_prompt_metrics_structured() {
        let builder =
            PromptBuilder::new("Extract entities").output_format("JSON with people, places, orgs");
        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);

        assert!(metrics.has_output_format);
        assert_eq!(metrics.technique, PromptTechnique::Structured);
    }

    #[test]
    fn test_prompt_metrics_token_count() {
        let builder = PromptBuilder::new("This is a test instruction");
        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);

        assert!(metrics.token_count > 0);
    }

    #[test]
    fn test_prompt_metrics_clone() {
        let builder = PromptBuilder::new("test");
        let prompt = builder.build();
        let metrics = PromptMetrics::analyze(&prompt, &builder);
        let cloned = metrics.clone();
        assert_eq!(metrics.token_count, cloned.token_count);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_prompt_error_template() {
        let err = PromptError::Template("missing variable".to_string());
        assert!(err.to_string().contains("missing variable"));
    }

    #[test]
    fn test_prompt_error_validation() {
        let err = PromptError::Validation("invalid JSON".to_string());
        assert!(err.to_string().contains("invalid JSON"));
    }

    #[test]
    fn test_prompt_error_debug() {
        let err = PromptError::Template("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Template"));
    }
}
