//! LLM Serving Demo - Course 4 Week 1
//!
//! Demonstrates LLM serving concepts that map to Databricks Foundation Models.
//! Shows tokenization, model loading, and inference patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Tokenization error: {0}")]
    Tokenization(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Generation error: {0}")]
    Generation(String),
}

// ============================================================================
// Tokenizer (BPE-style demonstration)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    special_tokens: SpecialTokens,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
    pub unk_token: u32,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Special tokens
        vocab.insert("<s>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<pad>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 3);

        // Common tokens (simplified vocabulary)
        let common_tokens = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "hello", "world", "how", "what", "when", "where", "why", "who",
            "capital", "france", "paris", "of", "in", "on", "at", "to",
            "and", "or", "but", "not", "this", "that", "it", "for", "with",
            "AI", "machine", "learning", "model", "data", "train", "test",
            " ", ".", ",", "?", "!", ":", ";", "'", "\"", "-", "_",
        ];

        for (i, token) in common_tokens.iter().enumerate() {
            let id = (i + 4) as u32;
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        // Add special tokens to reverse vocab
        reverse_vocab.insert(0, "<s>".to_string());
        reverse_vocab.insert(1, "</s>".to_string());
        reverse_vocab.insert(2, "<pad>".to_string());
        reverse_vocab.insert(3, "<unk>".to_string());

        Self {
            vocab,
            reverse_vocab,
            special_tokens: SpecialTokens {
                bos_token: 0,
                eos_token: 1,
                pad_token: 2,
                unk_token: 3,
            },
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.special_tokens.bos_token];

        // Simple word-level tokenization
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else {
                // Character fallback for unknown words
                for c in word.chars() {
                    if let Some(&id) = self.vocab.get(&c.to_string()) {
                        tokens.push(id);
                    } else {
                        tokens.push(self.special_tokens.unk_token);
                    }
                }
            }
            // Add space token between words
            if let Some(&space_id) = self.vocab.get(" ") {
                tokens.push(space_id);
            }
        }

        // Remove trailing space and add EOS
        if tokens.last() == self.vocab.get(" ") {
            tokens.pop();
        }
        tokens.push(self.special_tokens.eos_token);

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
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Generation Parameters
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            stop_sequences: vec!["</s>".to_string()],
        }
    }
}

// ============================================================================
// Model Metadata
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub parameters: u64,
    pub context_length: usize,
    pub quantization: Option<String>,
}

impl ModelInfo {
    pub fn new(name: &str, params: u64) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            architecture: "transformer".to_string(),
            parameters: params,
            context_length: 4096,
            quantization: None,
        }
    }

    pub fn with_quantization(mut self, quant: &str) -> Self {
        self.quantization = Some(quant.to_string());
        self
    }
}

// ============================================================================
// LLM Server (simulated)
// ============================================================================

#[derive(Debug)]
pub struct LlmServer {
    model_info: ModelInfo,
    tokenizer: SimpleTokenizer,
    config: GenerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl LlmServer {
    pub fn new(model_info: ModelInfo) -> Self {
        Self {
            model_info,
            tokenizer: SimpleTokenizer::new(),
            config: GenerationConfig::default(),
        }
    }

    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Tokenize input
        let input_tokens = self.tokenizer.encode(&request.prompt);
        let prompt_tokens = input_tokens.len();

        // Simulate generation (in real impl, would run transformer forward pass)
        let generated_text = self.simulate_generation(&request.prompt);
        let output_tokens = self.tokenizer.encode(&generated_text);
        let completion_tokens = output_tokens.len();

        Ok(CompletionResponse {
            id: format!("cmpl-{}", uuid_simple()),
            model: self.model_info.name.clone(),
            choices: vec![Choice {
                index: 0,
                text: generated_text,
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }

    fn simulate_generation(&self, prompt: &str) -> String {
        // Pattern-based responses for demo
        let prompt_lower = prompt.to_lowercase();

        if prompt_lower.contains("capital") && prompt_lower.contains("france") {
            "Paris is the capital of France.".to_string()
        } else if prompt_lower.contains("hello") {
            "Hello! How can I assist you today?".to_string()
        } else if prompt_lower.contains("what is ai") || prompt_lower.contains("what is artificial intelligence") {
            "AI is the simulation of human intelligence by machines.".to_string()
        } else if prompt_lower.contains("machine learning") {
            "Machine learning is a subset of AI that enables systems to learn from data.".to_string()
        } else {
            "I understand your query. Let me provide a helpful response.".to_string()
        }
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    pub fn tokenizer(&self) -> &SimpleTokenizer {
        &self.tokenizer
    }
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", nanos)
}

// ============================================================================
// Chat Interface (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

impl LlmServer {
    pub fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        // Build prompt from messages
        let prompt = request
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Get completion
        let completion = self.complete(&CompletionRequest {
            prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop: None,
        })?;

        Ok(ChatResponse {
            id: completion.id,
            model: completion.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: completion.choices[0].text.clone(),
                },
                finish_reason: completion.choices[0].finish_reason.clone(),
            }],
            usage: completion.usage,
        })
    }
}

// ============================================================================
// Quantization Info
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    pub method: String,
    pub bits: u8,
    pub group_size: Option<usize>,
    pub size_reduction: f32,
}

impl QuantizationInfo {
    pub fn gguf_q4() -> Self {
        Self {
            method: "GGUF Q4_K_M".to_string(),
            bits: 4,
            group_size: Some(32),
            size_reduction: 0.25, // 4-bit is ~25% of fp16
        }
    }

    pub fn gguf_q8() -> Self {
        Self {
            method: "GGUF Q8_0".to_string(),
            bits: 8,
            group_size: None,
            size_reduction: 0.5, // 8-bit is ~50% of fp16
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     LLM Serving Demo - Course 4 Week 1                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Tokenization
    println!("📝 Step 1: Tokenization");
    println!("   Demonstrating BPE-style tokenization\n");

    let tokenizer = SimpleTokenizer::new();
    let test_texts = [
        "Hello world",
        "What is the capital of France?",
        "Machine learning is AI",
    ];

    for text in &test_texts {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        println!("   Input:   \"{}\"", text);
        println!("   Tokens:  {:?}", tokens);
        println!("   Decoded: \"{}\"\n", decoded);
    }

    println!("   Vocab size: {}\n", tokenizer.vocab_size());

    // Step 2: Model Info
    println!("🤖 Step 2: Model Configuration");
    let model_info = ModelInfo::new("llama-7b", 7_000_000_000)
        .with_quantization("Q4_K_M");

    println!("   Name: {}", model_info.name);
    println!("   Parameters: {}B", model_info.parameters / 1_000_000_000);
    println!("   Context: {} tokens", model_info.context_length);
    println!("   Quantization: {:?}\n", model_info.quantization);

    // Step 3: Completion API
    println!("🚀 Step 3: Completion API");
    let server = LlmServer::new(model_info);

    let prompts = [
        "What is the capital of France?",
        "Hello, how are you?",
        "Explain machine learning briefly.",
    ];

    for prompt in &prompts {
        let request = CompletionRequest {
            prompt: prompt.to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
            stop: None,
        };

        match server.complete(&request) {
            Ok(response) => {
                println!("   Prompt: \"{}\"", prompt);
                println!("   Response: \"{}\"", response.choices[0].text);
                println!("   Tokens: {} prompt + {} completion\n",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens);
            }
            Err(e) => println!("   Error: {}\n", e),
        }
    }

    // Step 4: Chat API
    println!("💬 Step 4: Chat API (OpenAI-compatible)");
    let chat_request = ChatRequest {
        model: "llama-7b".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is AI?".to_string(),
            },
        ],
        max_tokens: Some(100),
        temperature: Some(0.7),
    };

    match server.chat(&chat_request) {
        Ok(response) => {
            println!("   Messages:");
            for msg in &chat_request.messages {
                println!("     {}: {}", msg.role, msg.content);
            }
            println!("   Assistant: {}\n", response.choices[0].message.content);
        }
        Err(e) => println!("   Error: {}\n", e),
    }

    // Step 5: Quantization
    println!("📊 Step 5: Quantization Comparison");
    let q4 = QuantizationInfo::gguf_q4();
    let q8 = QuantizationInfo::gguf_q8();

    println!("   {} ({}-bit): {:.0}% of original size",
        q4.method, q4.bits, q4.size_reduction * 100.0);
    println!("   {} ({}-bit): {:.0}% of original size\n",
        q8.method, q8.bits, q8.size_reduction * 100.0);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • BPE-style tokenization with special tokens");
    println!("  • Completion API (text generation)");
    println!("  • Chat API (OpenAI-compatible)");
    println!("  • GGUF quantization for efficient inference");
    println!();
    println!("Databricks equivalent: Foundation Model APIs, Model Serving");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Tokenizer Tests
    // =========================================================================

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = SimpleTokenizer::new();
        let text = "hello world";
        let tokens = tokenizer.encode(text);

        assert!(tokens.len() > 2); // At least BOS + content + EOS
        assert_eq!(tokens[0], 0); // BOS
        assert_eq!(*tokens.last().unwrap(), 1); // EOS
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tokenizer = SimpleTokenizer::new();
        assert_eq!(tokenizer.special_tokens.bos_token, 0);
        assert_eq!(tokenizer.special_tokens.eos_token, 1);
        assert_eq!(tokenizer.special_tokens.pad_token, 2);
        assert_eq!(tokenizer.special_tokens.unk_token, 3);
    }

    #[test]
    fn test_tokenizer_default() {
        let tokenizer = SimpleTokenizer::default();
        assert!(tokenizer.vocab_size() > 4);
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokenizer = SimpleTokenizer::new();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_encode_unknown() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode("xyz123unknown");
        // Should contain UNK tokens for unknown chars
        assert!(tokens.contains(&3));
    }

    #[test]
    fn test_tokenizer_decode_filters_special() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = vec![0, 1, 2, 3]; // All special tokens
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.contains("<s>"));
        assert!(!decoded.contains("</s>"));
    }

    #[test]
    fn test_tokenizer_clone() {
        let tokenizer = SimpleTokenizer::new();
        let cloned = tokenizer.clone();
        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());
    }

    // =========================================================================
    // Model Info Tests
    // =========================================================================

    #[test]
    fn test_model_info() {
        let info = ModelInfo::new("test-model", 1_000_000);
        assert_eq!(info.name, "test-model");
        assert_eq!(info.parameters, 1_000_000);

        let info = info.with_quantization("Q4_K_M");
        assert_eq!(info.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn test_model_info_defaults() {
        let info = ModelInfo::new("llama", 7_000_000_000);
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.architecture, "transformer");
        assert_eq!(info.context_length, 4096);
        assert!(info.quantization.is_none());
    }

    #[test]
    fn test_model_info_serialization() {
        let info = ModelInfo::new("test", 1000).with_quantization("Q8_0");
        let json = serde_json::to_string(&info).unwrap();
        let restored: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.name, restored.name);
        assert_eq!(info.quantization, restored.quantization);
    }

    #[test]
    fn test_model_info_clone() {
        let info = ModelInfo::new("test", 1000);
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
    }

    // =========================================================================
    // Generation Config Tests
    // =========================================================================

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < 0.001);
        assert!((config.top_p - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_generation_config_serialization() {
        let config = GenerationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.max_tokens, restored.max_tokens);
    }

    #[test]
    fn test_generation_config_clone() {
        let config = GenerationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_tokens, cloned.max_tokens);
    }

    // =========================================================================
    // Completion Tests
    // =========================================================================

    #[test]
    fn test_completion() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = CompletionRequest {
            prompt: "What is the capital of France?".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
            stop: None,
        };

        let response = server.complete(&request).unwrap();
        assert!(!response.choices.is_empty());
        assert!(response.choices[0].text.contains("Paris"));
    }

    #[test]
    fn test_completion_hello() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = CompletionRequest {
            prompt: "Hello, how are you?".to_string(),
            max_tokens: Some(50),
            temperature: None,
            stop: None,
        };

        let response = server.complete(&request).unwrap();
        assert!(response.choices[0].text.contains("Hello"));
    }

    #[test]
    fn test_completion_ai_question() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = CompletionRequest {
            prompt: "What is AI?".to_string(),
            max_tokens: Some(50),
            temperature: None,
            stop: None,
        };

        let response = server.complete(&request).unwrap();
        assert!(!response.choices[0].text.is_empty());
    }

    #[test]
    fn test_completion_usage_tracking() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = CompletionRequest {
            prompt: "Hello".to_string(),
            max_tokens: Some(50),
            temperature: None,
            stop: None,
        };

        let response = server.complete(&request).unwrap();
        assert!(response.usage.prompt_tokens > 0);
        assert!(response.usage.completion_tokens > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.prompt_tokens + response.usage.completion_tokens
        );
    }

    #[test]
    fn test_completion_response_serialization() {
        let response = CompletionResponse {
            id: "test-123".to_string(),
            model: "llama".to_string(),
            choices: vec![Choice {
                index: 0,
                text: "Hello!".to_string(),
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let restored: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.id, restored.id);
    }

    // =========================================================================
    // Chat Tests
    // =========================================================================

    #[test]
    fn test_chat() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(50),
            temperature: Some(0.7),
        };

        let response = server.chat(&request).unwrap();
        assert!(!response.choices.is_empty());
        assert_eq!(response.choices[0].message.role, "assistant");
    }

    #[test]
    fn test_chat_multiple_messages() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);

        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                },
            ],
            max_tokens: Some(50),
            temperature: None,
        };

        let response = server.chat(&request).unwrap();
        assert!(!response.choices.is_empty());
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let restored: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role, restored.role);
    }

    #[test]
    fn test_chat_request_serialization() {
        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(100),
            temperature: Some(0.5),
        };
        let json = serde_json::to_string(&request).unwrap();
        let restored: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.model, restored.model);
    }

    // =========================================================================
    // Quantization Tests
    // =========================================================================

    #[test]
    fn test_quantization_info() {
        let q4 = QuantizationInfo::gguf_q4();
        assert_eq!(q4.bits, 4);
        assert!(q4.size_reduction < 0.5);

        let q8 = QuantizationInfo::gguf_q8();
        assert_eq!(q8.bits, 8);
        assert_eq!(q8.size_reduction, 0.5);
    }

    #[test]
    fn test_quantization_q4_details() {
        let q4 = QuantizationInfo::gguf_q4();
        assert!(q4.method.contains("Q4"));
        assert!(q4.group_size.is_some());
    }

    #[test]
    fn test_quantization_q8_details() {
        let q8 = QuantizationInfo::gguf_q8();
        assert!(q8.method.contains("Q8"));
        assert!(q8.group_size.is_none());
    }

    #[test]
    fn test_quantization_serialization() {
        let q4 = QuantizationInfo::gguf_q4();
        let json = serde_json::to_string(&q4).unwrap();
        let restored: QuantizationInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(q4.bits, restored.bits);
    }

    #[test]
    fn test_quantization_clone() {
        let q4 = QuantizationInfo::gguf_q4();
        let cloned = q4.clone();
        assert_eq!(q4.bits, cloned.bits);
    }

    // =========================================================================
    // LLM Server Tests
    // =========================================================================

    #[test]
    fn test_server_model_info() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);
        assert_eq!(server.model_info().name, "test");
    }

    #[test]
    fn test_server_tokenizer() {
        let model = ModelInfo::new("test", 1000);
        let server = LlmServer::new(model);
        assert!(server.tokenizer().vocab_size() > 0);
    }

    #[test]
    fn test_server_with_config() {
        let model = ModelInfo::new("test", 1000);
        let config = GenerationConfig {
            max_tokens: 128,
            ..Default::default()
        };
        let _server = LlmServer::new(model).with_config(config);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_llm_error_tokenization() {
        let err = LlmError::Tokenization("invalid token".to_string());
        assert!(err.to_string().contains("invalid token"));
    }

    #[test]
    fn test_llm_error_model() {
        let err = LlmError::Model("not loaded".to_string());
        assert!(err.to_string().contains("not loaded"));
    }

    #[test]
    fn test_llm_error_generation() {
        let err = LlmError::Generation("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_llm_error_debug() {
        let err = LlmError::Model("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Model"));
    }

    // =========================================================================
    // API Type Tests
    // =========================================================================

    #[test]
    fn test_choice_serialization() {
        let choice = Choice {
            index: 0,
            text: "Hello".to_string(),
            finish_reason: "stop".to_string(),
        };
        let json = serde_json::to_string(&choice).unwrap();
        let restored: Choice = serde_json::from_str(&json).unwrap();
        assert_eq!(choice.index, restored.index);
    }

    #[test]
    fn test_usage_serialization() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let restored: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage.total_tokens, restored.total_tokens);
    }

    #[test]
    fn test_chat_choice_serialization() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hi!".to_string(),
            },
            finish_reason: "stop".to_string(),
        };
        let json = serde_json::to_string(&choice).unwrap();
        let restored: ChatChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(choice.message.role, restored.message.role);
    }
}
