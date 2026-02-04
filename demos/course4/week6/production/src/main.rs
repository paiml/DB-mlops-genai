//! GenAI Production Demo - Course 4 Week 6
//!
//! Demonstrates production deployment patterns for GenAI systems.
//! Shows monitoring, guardrails, and A/B testing concepts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum ProductionError {
    #[error("Guardrail violation: {0}")]
    Guardrail(String),
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Model error: {0}")]
    Model(String),
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIRequest {
    pub id: String,
    pub prompt: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIResponse {
    pub id: String,
    pub request_id: String,
    pub model: String,
    pub content: String,
    pub usage: TokenUsage,
    pub latency_ms: u64,
    pub guardrails_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// Guardrails
// ============================================================================

#[derive(Debug, Clone)]
pub struct Guardrails {
    blocked_patterns: Vec<String>,
    max_prompt_length: usize,
    max_output_length: usize,
    pii_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub passed: bool,
    pub violations: Vec<String>,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            blocked_patterns: vec![
                "password".to_string(),
                "credit card".to_string(),
                "ssn".to_string(),
            ],
            max_prompt_length: 4096,
            max_output_length: 2048,
            pii_detection: true,
        }
    }
}

impl Guardrails {
    pub fn check_input(&self, text: &str) -> GuardrailResult {
        let mut violations = Vec::new();
        let text_lower = text.to_lowercase();

        // Check length
        if text.len() > self.max_prompt_length {
            violations.push(format!(
                "Prompt exceeds max length: {} > {}",
                text.len(),
                self.max_prompt_length
            ));
        }

        // Check blocked patterns
        for pattern in &self.blocked_patterns {
            if text_lower.contains(pattern) {
                violations.push(format!("Blocked pattern detected: {}", pattern));
            }
        }

        // Check for PII patterns
        if self.pii_detection {
            if Self::contains_email(&text_lower) {
                violations.push("PII detected: email address".to_string());
            }
            if Self::contains_phone(&text_lower) {
                violations.push("PII detected: phone number".to_string());
            }
        }

        GuardrailResult {
            passed: violations.is_empty(),
            violations,
        }
    }

    pub fn check_output(&self, text: &str) -> GuardrailResult {
        let mut violations = Vec::new();

        if text.len() > self.max_output_length {
            violations.push(format!(
                "Output exceeds max length: {} > {}",
                text.len(),
                self.max_output_length
            ));
        }

        GuardrailResult {
            passed: violations.is_empty(),
            violations,
        }
    }

    fn contains_email(text: &str) -> bool {
        text.contains('@') && text.contains('.')
    }

    fn contains_phone(text: &str) -> bool {
        // Simple pattern: 10+ consecutive digits
        let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
        digits.len() >= 10
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

#[derive(Debug, Clone)]
pub struct RateLimiter {
    requests_per_minute: usize,
    tokens_per_minute: usize,
    current_requests: usize,
    current_tokens: usize,
}

impl RateLimiter {
    pub fn new(requests_per_minute: usize, tokens_per_minute: usize) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
            current_requests: 0,
            current_tokens: 0,
        }
    }

    pub fn check(&self) -> bool {
        self.current_requests < self.requests_per_minute
            && self.current_tokens < self.tokens_per_minute
    }

    pub fn record(&mut self, tokens: usize) {
        self.current_requests += 1;
        self.current_tokens += tokens;
    }

    pub fn reset(&mut self) {
        self.current_requests = 0;
        self.current_tokens = 0;
    }

    pub fn usage(&self) -> (f32, f32) {
        (
            self.current_requests as f32 / self.requests_per_minute as f32,
            self.current_tokens as f32 / self.tokens_per_minute as f32,
        )
    }
}

// ============================================================================
// A/B Testing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub name: String,
    pub variants: Vec<Variant>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    pub model: String,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct ABRouter {
    experiments: HashMap<String, Experiment>,
}

impl ABRouter {
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
        }
    }

    pub fn add_experiment(&mut self, experiment: Experiment) {
        self.experiments.insert(experiment.name.clone(), experiment);
    }

    pub fn route(&self, experiment_name: &str, user_hash: u64) -> Option<&Variant> {
        let experiment = self.experiments.get(experiment_name)?;
        if !experiment.active {
            return None;
        }

        // Deterministic routing based on user hash
        let normalized = (user_hash % 100) as f32 / 100.0;
        let mut cumulative = 0.0;

        for variant in &experiment.variants {
            cumulative += variant.weight;
            if normalized < cumulative {
                return Some(variant);
            }
        }

        experiment.variants.last()
    }
}

impl Default for ABRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProductionMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub guardrail_violations: u64,
}

impl ProductionMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_request(&mut self, latency_ms: u64, tokens: u64, success: bool) {
        self.total_requests += 1;
        self.total_tokens += tokens;

        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        // Update average latency (simplified)
        let n = self.total_requests as f64;
        self.avg_latency_ms = self.avg_latency_ms * (n - 1.0) / n + latency_ms as f64 / n;

        // Track p99 (simplified - just max for demo)
        self.p99_latency_ms = self.p99_latency_ms.max(latency_ms as f64);
    }

    pub fn record_guardrail_violation(&mut self) {
        self.guardrail_violations += 1;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.successful_requests as f64 / self.total_requests as f64
    }
}

// ============================================================================
// Production Server
// ============================================================================

#[derive(Debug)]
pub struct ProductionServer {
    guardrails: Guardrails,
    rate_limiter: RateLimiter,
    ab_router: ABRouter,
    metrics: ProductionMetrics,
}

impl ProductionServer {
    pub fn new() -> Self {
        Self {
            guardrails: Guardrails::default(),
            rate_limiter: RateLimiter::new(60, 100000),
            ab_router: ABRouter::new(),
            metrics: ProductionMetrics::new(),
        }
    }

    pub fn process(&mut self, request: GenAIRequest) -> Result<GenAIResponse, ProductionError> {
        // 1. Rate limiting
        if !self.rate_limiter.check() {
            return Err(ProductionError::RateLimit);
        }

        // 2. Input guardrails
        let input_check = self.guardrails.check_input(&request.prompt);
        if !input_check.passed {
            self.metrics.record_guardrail_violation();
            return Err(ProductionError::Guardrail(input_check.violations.join("; ")));
        }

        // 3. Generate response (simulated)
        let content = self.simulate_generation(&request);
        let latency_ms = 50 + (request.prompt.len() as u64 / 10);

        // 4. Output guardrails
        let output_check = self.guardrails.check_output(&content);
        if !output_check.passed {
            self.metrics.record_guardrail_violation();
            return Err(ProductionError::Guardrail(output_check.violations.join("; ")));
        }

        // 5. Calculate usage
        let usage = TokenUsage {
            prompt_tokens: request.prompt.split_whitespace().count(),
            completion_tokens: content.split_whitespace().count(),
            total_tokens: request.prompt.split_whitespace().count()
                + content.split_whitespace().count(),
        };

        // 6. Record metrics
        self.rate_limiter.record(usage.total_tokens);
        self.metrics
            .record_request(latency_ms, usage.total_tokens as u64, true);

        Ok(GenAIResponse {
            id: format!("resp-{}", request.id),
            request_id: request.id,
            model: request.model,
            content,
            usage,
            latency_ms,
            guardrails_passed: true,
        })
    }

    fn simulate_generation(&self, request: &GenAIRequest) -> String {
        // Pattern-based for demo
        if request.prompt.to_lowercase().contains("hello") {
            "Hello! How can I assist you today?".to_string()
        } else {
            "I understand your request. Here is a helpful response based on your query.".to_string()
        }
    }

    pub fn metrics(&self) -> &ProductionMetrics {
        &self.metrics
    }

    pub fn ab_router(&self) -> &ABRouter {
        &self.ab_router
    }
}

impl Default for ProductionServer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     GenAI Production Demo - Course 4 Week 6                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Guardrails
    println!("🛡️  Step 1: Guardrails");
    let guardrails = Guardrails::default();

    let test_inputs = [
        "What is machine learning?",
        "My password is secret123",
        "Contact me at user@email.com",
    ];

    for input in &test_inputs {
        let result = guardrails.check_input(input);
        let status = if result.passed { "✓ PASS" } else { "✗ FAIL" };
        println!("   {} \"{}...\"", status, &input[..input.len().min(30)]);
        if !result.passed {
            for v in &result.violations {
                println!("      - {}", v);
            }
        }
    }
    println!();

    // Step 2: Rate Limiting
    println!("⏱️  Step 2: Rate Limiting");
    let mut rate_limiter = RateLimiter::new(60, 100000);

    for i in 0..5 {
        if rate_limiter.check() {
            rate_limiter.record(100);
            println!("   Request {}: allowed", i + 1);
        }
    }

    let (req_usage, token_usage) = rate_limiter.usage();
    println!("   Request usage: {:.1}%", req_usage * 100.0);
    println!("   Token usage: {:.3}%\n", token_usage * 100.0);

    // Step 3: A/B Testing
    println!("🔀 Step 3: A/B Testing");
    let mut router = ABRouter::new();

    router.add_experiment(Experiment {
        name: "model-comparison".to_string(),
        variants: vec![
            Variant {
                name: "control".to_string(),
                model: "llama-7b".to_string(),
                weight: 0.5,
            },
            Variant {
                name: "treatment".to_string(),
                model: "llama-13b".to_string(),
                weight: 0.5,
            },
        ],
        active: true,
    });

    for user_id in [100u64, 150, 200, 250, 300] {
        if let Some(variant) = router.route("model-comparison", user_id) {
            println!("   User {}: {} ({})", user_id, variant.name, variant.model);
        }
    }
    println!();

    // Step 4: Production Server
    println!("🚀 Step 4: Production Server");
    let mut server = ProductionServer::new();

    let requests = vec![
        GenAIRequest {
            id: "req-1".to_string(),
            prompt: "Hello, how are you?".to_string(),
            model: "llama-7b".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            user_id: Some("user-1".to_string()),
        },
        GenAIRequest {
            id: "req-2".to_string(),
            prompt: "Explain machine learning".to_string(),
            model: "llama-7b".to_string(),
            max_tokens: 200,
            temperature: 0.7,
            user_id: Some("user-2".to_string()),
        },
        GenAIRequest {
            id: "req-3".to_string(),
            prompt: "My password is secret".to_string(),
            model: "llama-7b".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            user_id: None,
        },
    ];

    for request in &requests {
        print!("   {} ", request.id);
        match server.process(request.clone()) {
            Ok(response) => {
                println!("✓ {}ms, {} tokens", response.latency_ms, response.usage.total_tokens);
            }
            Err(e) => println!("✗ {}", e),
        }
    }
    println!();

    // Step 5: Metrics
    println!("📊 Step 5: Production Metrics");
    let metrics = server.metrics();
    println!("   Total requests: {}", metrics.total_requests);
    println!("   Success rate: {:.1}%", metrics.success_rate() * 100.0);
    println!("   Avg latency: {:.1}ms", metrics.avg_latency_ms);
    println!("   Guardrail violations: {}\n", metrics.guardrail_violations);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Input/output guardrails (PII, blocked patterns)");
    println!("  • Rate limiting (requests, tokens)");
    println!("  • A/B testing with deterministic routing");
    println!("  • Production metrics tracking");
    println!();
    println!("Databricks equivalent: Model Serving, MLflow, Lakehouse Monitoring");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrails_pass() {
        let guardrails = Guardrails::default();
        let result = guardrails.check_input("What is machine learning?");
        assert!(result.passed);
    }

    #[test]
    fn test_guardrails_blocked_pattern() {
        let guardrails = Guardrails::default();
        let result = guardrails.check_input("My password is secret");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("password")));
    }

    #[test]
    fn test_guardrails_pii_email() {
        let guardrails = Guardrails::default();
        let result = guardrails.check_input("Contact me at test@example.com");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("email")));
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10, 1000);
        assert!(limiter.check());
        limiter.record(100);
        assert!(limiter.check());
    }

    #[test]
    fn test_ab_router() {
        let mut router = ABRouter::new();
        router.add_experiment(Experiment {
            name: "test".to_string(),
            variants: vec![
                Variant {
                    name: "a".to_string(),
                    model: "model-a".to_string(),
                    weight: 0.5,
                },
                Variant {
                    name: "b".to_string(),
                    model: "model-b".to_string(),
                    weight: 0.5,
                },
            ],
            active: true,
        });

        // Should get consistent routing for same user
        let v1 = router.route("test", 123);
        let v2 = router.route("test", 123);
        assert_eq!(v1.map(|v| &v.name), v2.map(|v| &v.name));
    }

    #[test]
    fn test_metrics() {
        let mut metrics = ProductionMetrics::new();
        metrics.record_request(50, 100, true);
        metrics.record_request(60, 150, true);
        metrics.record_request(70, 200, false);

        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_production_server() {
        let mut server = ProductionServer::new();
        let request = GenAIRequest {
            id: "test".to_string(),
            prompt: "Hello world".to_string(),
            model: "test-model".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            user_id: None,
        };

        let result = server.process(request);
        assert!(result.is_ok());
    }
}
