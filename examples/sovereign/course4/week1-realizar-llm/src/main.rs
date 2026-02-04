//! LLM Inference with realizar
//!
//! Demonstrates LLM inference patterns using realizar concepts.
//! This example shows GGUF model loading, tokenization, and text generation.
//!
//! # Course 4, Week 1: Foundation Models + Prompt Engineering

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),
}

// ============================================================================
// GGUF Model Metadata
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufMetadata {
    pub model_name: String,
    pub architecture: String,
    pub quantization: QuantizationType,
    pub context_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    F16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q4_K_M,
    Q4_K_S,
    Q3_K_M,
    Q2_K,
}

impl QuantizationType {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantizationType::F16 => 16.0,
            QuantizationType::Q8_0 => 8.0,
            QuantizationType::Q6_K => 6.5,
            QuantizationType::Q5_K_M => 5.5,
            QuantizationType::Q4_K_M => 4.5,
            QuantizationType::Q4_K_S => 4.5,
            QuantizationType::Q3_K_M => 3.5,
            QuantizationType::Q2_K => 2.5,
        }
    }

    pub fn memory_factor(&self) -> f32 {
        self.bits_per_weight() / 16.0
    }
}

impl GgufMetadata {
    pub fn new(name: &str, quant: QuantizationType) -> Self {
        Self {
            model_name: name.to_string(),
            architecture: "llama".to_string(),
            quantization: quant,
            context_length: 4096,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
        }
    }

    pub fn estimated_memory_gb(&self, params_billions: f64) -> f64 {
        params_billions * self.quantization.memory_factor() as f64 * 2.0
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bos_token: u32,
    eos_token: u32,
    pad_token: u32,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Special tokens
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 3);

        // Common tokens (simplified)
        let tokens = [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "be",
            "have",
            "has",
            "do",
            "does",
            "will",
            "can",
            "could",
            "would",
            "should",
            "hello",
            "world",
            "how",
            "what",
            "when",
            "where",
            "why",
            "who",
            "capital",
            "france",
            "paris",
            "of",
            "in",
            "on",
            "at",
            "to",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "for",
            "with",
            "AI",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "model",
            "data",
            " ",
            ".",
            ",",
            "?",
            "!",
            "\n",
        ];

        for (i, token) in tokens.iter().enumerate() {
            let id = (i + 4) as u32;
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        reverse_vocab.insert(1, "<s>".to_string());
        reverse_vocab.insert(2, "</s>".to_string());
        reverse_vocab.insert(0, "<pad>".to_string());
        reverse_vocab.insert(3, "<unk>".to_string());

        Self {
            vocab,
            reverse_vocab,
            bos_token: 1,
            eos_token: 2,
            pad_token: 0,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos_token];

        for word in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else {
                for c in word.chars() {
                    if let Some(&id) = self.vocab.get(&c.to_string()) {
                        tokens.push(id);
                    } else {
                        tokens.push(3); // UNK
                    }
                }
            }
            if let Some(&space_id) = self.vocab.get(" ") {
                tokens.push(space_id);
            }
        }

        if tokens.last() == self.vocab.get(" ") {
            tokens.pop();
        }

        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .filter(|t| !t.starts_with('<') || !t.ends_with('>'))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Generation Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_sequences: vec!["</s>".to_string()],
        }
    }
}

impl GenerationConfig {
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
}

// ============================================================================
// KV Cache
// ============================================================================

#[derive(Debug, Clone)]
pub struct KVCache {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub seq_len: usize,
    pub capacity: usize,
}

impl KVCache {
    pub fn new(capacity: usize, hidden_size: usize) -> Self {
        Self {
            keys: vec![vec![0.0; hidden_size]; capacity],
            values: vec![vec![0.0; hidden_size]; capacity],
            seq_len: 0,
            capacity,
        }
    }

    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) -> Result<(), LlmError> {
        if self.seq_len >= self.capacity {
            return Err(LlmError::ContextLengthExceeded(format!(
                "Cache full: {} >= {}",
                self.seq_len, self.capacity
            )));
        }
        self.keys[self.seq_len] = key;
        self.values[self.seq_len] = value;
        self.seq_len += 1;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

// ============================================================================
// LLM Model (Simulated)
// ============================================================================

pub struct LlmModel {
    metadata: GgufMetadata,
    tokenizer: Tokenizer,
    kv_cache: KVCache,
    loaded: bool,
}

impl LlmModel {
    pub fn new(metadata: GgufMetadata) -> Self {
        let cache = KVCache::new(metadata.context_length, metadata.hidden_size);
        Self {
            metadata,
            tokenizer: Tokenizer::new(),
            kv_cache: cache,
            loaded: false,
        }
    }

    pub fn load(&mut self) -> Result<(), LlmError> {
        // Simulate loading GGUF file
        self.loaded = true;
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationOutput, LlmError> {
        if !self.loaded {
            return Err(LlmError::ModelNotLoaded(self.metadata.model_name.clone()));
        }

        let input_tokens = self.tokenizer.encode(prompt);
        let prompt_len = input_tokens.len();

        // Check context length
        if prompt_len + config.max_new_tokens > self.metadata.context_length {
            return Err(LlmError::ContextLengthExceeded(format!(
                "{} + {} > {}",
                prompt_len, config.max_new_tokens, self.metadata.context_length
            )));
        }

        // Simulate generation
        let generated_text = self.simulate_generation(prompt);
        let output_tokens = self.tokenizer.encode(&generated_text);

        Ok(GenerationOutput {
            text: generated_text,
            prompt_tokens: prompt_len,
            completion_tokens: output_tokens.len(),
            finish_reason: FinishReason::Stop,
        })
    }

    fn simulate_generation(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();

        if prompt_lower.contains("capital") && prompt_lower.contains("france") {
            "The capital of France is Paris.".to_string()
        } else if prompt_lower.contains("hello") {
            "Hello! How can I help you today?".to_string()
        } else if prompt_lower.contains("what is ai")
            || prompt_lower.contains("artificial intelligence")
        {
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines."
                .to_string()
        } else if prompt_lower.contains("machine learning") {
            "Machine learning is a subset of AI that enables systems to learn from data."
                .to_string()
        } else {
            "I understand your question. Let me provide a helpful response.".to_string()
        }
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
}

// ============================================================================
// Prompt Templates
// ============================================================================

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: Vec<String>,
}

impl PromptTemplate {
    pub fn new(template: &str) -> Self {
        let mut variables = Vec::new();
        let mut in_var = false;
        let mut var_name = String::new();

        for c in template.chars() {
            if c == '{' {
                in_var = true;
                var_name.clear();
            } else if c == '}' && in_var {
                variables.push(var_name.clone());
                in_var = false;
            } else if in_var {
                var_name.push(c);
            }
        }

        Self {
            template: template.to_string(),
            variables,
        }
    }

    pub fn format(&self, values: &HashMap<String, String>) -> Result<String, LlmError> {
        let mut result = self.template.clone();
        for var in &self.variables {
            let value = values
                .get(var)
                .ok_or_else(|| LlmError::Generation(format!("Missing variable: {}", var)))?;
            result = result.replace(&format!("{{{}}}", var), value);
        }
        Ok(result)
    }

    pub fn variables(&self) -> &[String] {
        &self.variables
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     LLM Inference with realizar - Course 4, Week 1            ║");
    println!("║     GGUF Loading, Tokenization, Text Generation               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Step 1: Model Metadata
    println!("\n📋 Step 1: GGUF Model Metadata");
    let metadata = GgufMetadata::new("llama-7b", QuantizationType::Q4_K_M);

    println!("   Model: {}", metadata.model_name);
    println!("   Architecture: {}", metadata.architecture);
    println!(
        "   Quantization: {:?} ({:.1} bits/weight)",
        metadata.quantization,
        metadata.quantization.bits_per_weight()
    );
    println!("   Context length: {} tokens", metadata.context_length);
    println!("   Vocab size: {}", metadata.vocab_size);
    println!(
        "   Estimated memory: {:.1} GB",
        metadata.estimated_memory_gb(7.0)
    );

    // Step 2: Tokenization
    println!("\n📝 Step 2: Tokenization");
    let tokenizer = Tokenizer::new();

    let test_texts = [
        "Hello world",
        "What is the capital of France?",
        "Machine learning is AI",
    ];

    for text in &test_texts {
        let tokens = tokenizer.encode(text);
        println!("   \"{}\"", text);
        println!("     Tokens: {:?}", tokens);
        println!("     Count: {}", tokens.len());
    }

    // Step 3: Load Model
    println!("\n🔧 Step 3: Model Loading");
    let mut model = LlmModel::new(metadata);
    model.load().unwrap();
    println!("   Model loaded: {}", model.is_loaded());

    // Step 4: Generation
    println!("\n🚀 Step 4: Text Generation");
    let config = GenerationConfig::default()
        .with_max_tokens(100)
        .with_temperature(0.7);

    let prompts = [
        "What is the capital of France?",
        "Hello, how are you?",
        "Explain machine learning briefly.",
    ];

    for prompt in &prompts {
        match model.generate(prompt, &config) {
            Ok(output) => {
                println!("   Prompt: \"{}\"", prompt);
                println!("   Response: \"{}\"", output.text);
                println!(
                    "   Tokens: {} prompt + {} completion",
                    output.prompt_tokens, output.completion_tokens
                );
                println!("   Finish: {:?}\n", output.finish_reason);
            }
            Err(e) => println!("   Error: {}\n", e),
        }
    }

    // Step 5: Prompt Templates
    println!("📄 Step 5: Prompt Templates");
    let template = PromptTemplate::new(
        "You are a {role}. Answer the following question:\n{question}\n\nAnswer:",
    );

    println!("   Variables: {:?}", template.variables());

    let mut values = HashMap::new();
    values.insert("role".to_string(), "helpful assistant".to_string());
    values.insert("question".to_string(), "What is AI?".to_string());

    let formatted = template.format(&values).unwrap();
    println!(
        "   Formatted prompt:\n{}",
        formatted
            .lines()
            .map(|l| format!("     {}", l))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Step 6: Quantization Comparison
    println!("\n📊 Step 6: Quantization Comparison");
    let quants = [
        QuantizationType::F16,
        QuantizationType::Q8_0,
        QuantizationType::Q4_K_M,
        QuantizationType::Q2_K,
    ];

    for quant in &quants {
        let meta = GgufMetadata::new("llama-7b", *quant);
        println!(
            "   {:?}: {:.1} bits, ~{:.1} GB memory",
            quant,
            quant.bits_per_weight(),
            meta.estimated_memory_gb(7.0)
        );
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • GGUF model metadata and quantization");
    println!("  • BPE-style tokenization");
    println!("  • Text generation with sampling parameters");
    println!("  • KV cache for efficient inference");
    println!("  • Prompt templates");
    println!();
    println!("Sovereign AI Stack: realizar LLM inference");
    println!("Databricks equivalent: Foundation Model APIs");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // GgufMetadata Tests
    // ========================================================================

    #[test]
    fn test_gguf_metadata_new() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        assert_eq!(meta.model_name, "test");
        assert_eq!(meta.quantization, QuantizationType::Q4_K_M);
    }

    #[test]
    fn test_gguf_metadata_memory() {
        let meta = GgufMetadata::new("test", QuantizationType::F16);
        let mem = meta.estimated_memory_gb(7.0);
        assert!(mem > 0.0);
    }

    #[test]
    fn test_gguf_metadata_clone() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let cloned = meta.clone();
        assert_eq!(meta.model_name, cloned.model_name);
    }

    // ========================================================================
    // QuantizationType Tests
    // ========================================================================

    #[test]
    fn test_quantization_bits() {
        assert_eq!(QuantizationType::F16.bits_per_weight(), 16.0);
        assert_eq!(QuantizationType::Q8_0.bits_per_weight(), 8.0);
        assert!(QuantizationType::Q4_K_M.bits_per_weight() < 8.0);
    }

    #[test]
    fn test_quantization_memory_factor() {
        assert_eq!(QuantizationType::F16.memory_factor(), 1.0);
        assert!(QuantizationType::Q4_K_M.memory_factor() < 0.5);
    }

    // ========================================================================
    // Tokenizer Tests
    // ========================================================================

    #[test]
    fn test_tokenizer_encode() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], tokenizer.bos_token_id());
    }

    #[test]
    fn test_tokenizer_decode() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.bos_token_id(), 1);
        assert_eq!(tokenizer.eos_token_id(), 2);
    }

    #[test]
    fn test_tokenizer_default() {
        let tokenizer = Tokenizer::default();
        assert!(tokenizer.vocab_size() > 0);
    }

    // ========================================================================
    // GenerationConfig Tests
    // ========================================================================

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert!((config.temperature - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::default()
            .with_max_tokens(100)
            .with_temperature(0.5)
            .with_top_p(0.8);

        assert_eq!(config.max_new_tokens, 100);
        assert!((config.temperature - 0.5).abs() < 0.001);
        assert!((config.top_p - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_generation_config_clone() {
        let config = GenerationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_new_tokens, cloned.max_new_tokens);
    }

    // ========================================================================
    // KVCache Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(100, 512);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::new(10, 4);
        cache
            .append(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(10, 4);
        cache.append(vec![1.0; 4], vec![2.0; 4]).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KVCache::new(1, 4);
        cache.append(vec![1.0; 4], vec![2.0; 4]).unwrap();
        let result = cache.append(vec![1.0; 4], vec![2.0; 4]);
        assert!(result.is_err());
    }

    // ========================================================================
    // LlmModel Tests
    // ========================================================================

    #[test]
    fn test_llm_model_new() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let model = LlmModel::new(meta);
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_llm_model_load() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let mut model = LlmModel::new(meta);
        model.load().unwrap();
        assert!(model.is_loaded());
    }

    #[test]
    fn test_llm_model_generate() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let mut model = LlmModel::new(meta);
        model.load().unwrap();

        let config = GenerationConfig::default();
        let output = model.generate("Hello", &config).unwrap();
        assert!(!output.text.is_empty());
    }

    #[test]
    fn test_llm_model_generate_not_loaded() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let mut model = LlmModel::new(meta);

        let config = GenerationConfig::default();
        let result = model.generate("Hello", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_llm_model_metadata() {
        let meta = GgufMetadata::new("test", QuantizationType::Q4_K_M);
        let model = LlmModel::new(meta);
        assert_eq!(model.metadata().model_name, "test");
    }

    // ========================================================================
    // PromptTemplate Tests
    // ========================================================================

    #[test]
    fn test_prompt_template_new() {
        let template = PromptTemplate::new("Hello {name}!");
        assert_eq!(template.variables(), &["name"]);
    }

    #[test]
    fn test_prompt_template_format() {
        let template = PromptTemplate::new("Hello {name}!");
        let mut values = HashMap::new();
        values.insert("name".to_string(), "World".to_string());

        let result = template.format(&values).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_prompt_template_missing_variable() {
        let template = PromptTemplate::new("Hello {name}!");
        let values = HashMap::new();
        let result = template.format(&values);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_template_multiple_vars() {
        let template = PromptTemplate::new("{greeting} {name}!");
        let mut values = HashMap::new();
        values.insert("greeting".to_string(), "Hi".to_string());
        values.insert("name".to_string(), "Alice".to_string());

        let result = template.format(&values).unwrap();
        assert_eq!(result, "Hi Alice!");
    }

    // ========================================================================
    // GenerationOutput Tests
    // ========================================================================

    #[test]
    fn test_generation_output_serialization() {
        let output = GenerationOutput {
            text: "Hello".to_string(),
            prompt_tokens: 5,
            completion_tokens: 10,
            finish_reason: FinishReason::Stop,
        };
        let json = serde_json::to_string(&output).unwrap();
        let restored: GenerationOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output.text, restored.text);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_model_not_loaded() {
        let err = LlmError::ModelNotLoaded("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_error_generation() {
        let err = LlmError::Generation("failed".to_string());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_error_context_exceeded() {
        let err = LlmError::ContextLengthExceeded("4096".to_string());
        assert!(err.to_string().contains("4096"));
    }

    #[test]
    fn test_error_debug() {
        let err = LlmError::Tokenization("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Tokenization"));
    }
}
