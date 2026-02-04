//! Production GenAI Demo - Course 4 Week 3
//!
//! Demonstrates production deployment patterns that map to Databricks
//! Model Serving. Shows monitoring, rate limiting, and quality gates.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum ProductionError {
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Quality gate failed: {0}")]
    QualityGateFailed(String),
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

// ============================================================================
// Rate Limiter
// ============================================================================

#[derive(Debug)]
pub struct RateLimiter {
    requests: VecDeque<Instant>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        Self {
            requests: VecDeque::new(),
            max_requests,
            window: Duration::from_secs(window_secs),
        }
    }

    /// Check if request is allowed
    pub fn check(&mut self) -> bool {
        let now = Instant::now();

        // Remove expired requests
        while let Some(front) = self.requests.front() {
            if now.duration_since(*front) > self.window {
                self.requests.pop_front();
            } else {
                break;
            }
        }

        // Check limit
        if self.requests.len() < self.max_requests {
            self.requests.push_back(now);
            true
        } else {
            false
        }
    }

    pub fn remaining(&self) -> usize {
        self.max_requests.saturating_sub(self.requests.len())
    }

    pub fn max_requests(&self) -> usize {
        self.max_requests
    }
}

// ============================================================================
// Request/Response Logging
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    pub request_id: String,
    pub timestamp_ms: u64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: u64,
    pub status: RequestStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    Success,
    Error(String),
    RateLimited,
    Filtered,
}

#[derive(Debug, Default)]
pub struct RequestLogger {
    logs: Vec<RequestLog>,
}

impl RequestLogger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log(&mut self, entry: RequestLog) {
        self.logs.push(entry);
    }

    pub fn success_rate(&self) -> f32 {
        if self.logs.is_empty() {
            return 0.0;
        }
        let successes = self
            .logs
            .iter()
            .filter(|l| matches!(l.status, RequestStatus::Success))
            .count();
        successes as f32 / self.logs.len() as f32
    }

    pub fn average_latency(&self) -> f32 {
        if self.logs.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.logs.iter().map(|l| l.latency_ms).sum();
        sum as f32 / self.logs.len() as f32
    }

    pub fn total_tokens(&self) -> usize {
        self.logs
            .iter()
            .map(|l| l.prompt_tokens + l.completion_tokens)
            .sum()
    }

    pub fn count(&self) -> usize {
        self.logs.len()
    }
}

// ============================================================================
// Content Filtering
// ============================================================================

#[derive(Debug, Clone)]
pub struct ContentFilter {
    blocked_patterns: Vec<String>,
    sensitive_topics: Vec<String>,
}

impl ContentFilter {
    pub fn new() -> Self {
        Self {
            blocked_patterns: Vec::new(),
            sensitive_topics: Vec::new(),
        }
    }

    pub fn add_blocked_pattern(mut self, pattern: &str) -> Self {
        self.blocked_patterns.push(pattern.to_lowercase());
        self
    }

    pub fn add_sensitive_topic(mut self, topic: &str) -> Self {
        self.sensitive_topics.push(topic.to_lowercase());
        self
    }

    /// Check if content should be filtered
    pub fn should_filter(&self, content: &str) -> FilterResult {
        let content_lower = content.to_lowercase();

        for pattern in &self.blocked_patterns {
            if content_lower.contains(pattern) {
                return FilterResult::Blocked(pattern.clone());
            }
        }

        for topic in &self.sensitive_topics {
            if content_lower.contains(topic) {
                return FilterResult::Flagged(topic.clone());
            }
        }

        FilterResult::Allowed
    }
}

impl Default for ContentFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum FilterResult {
    Allowed,
    Flagged(String),
    Blocked(String),
}

impl FilterResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, FilterResult::Allowed)
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, FilterResult::Blocked(_))
    }
}

// ============================================================================
// Quality Gates
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub relevance_score: f32,
    pub coherence_score: f32,
    pub safety_score: f32,
    pub latency_p95_ms: u64,
}

#[derive(Debug, Clone)]
pub struct QualityGate {
    min_relevance: f32,
    min_coherence: f32,
    min_safety: f32,
    max_latency_ms: u64,
}

impl QualityGate {
    pub fn new() -> Self {
        Self {
            min_relevance: 0.7,
            min_coherence: 0.7,
            min_safety: 0.9,
            max_latency_ms: 5000,
        }
    }

    pub fn with_min_relevance(mut self, score: f32) -> Self {
        self.min_relevance = score;
        self
    }

    pub fn with_min_safety(mut self, score: f32) -> Self {
        self.min_safety = score;
        self
    }

    pub fn with_max_latency(mut self, ms: u64) -> Self {
        self.max_latency_ms = ms;
        self
    }

    /// Check if metrics pass the gate
    pub fn check(&self, metrics: &QualityMetrics) -> QualityCheckResult {
        let mut issues = Vec::new();

        if metrics.relevance_score < self.min_relevance {
            issues.push(format!(
                "Relevance {:.2} < {:.2}",
                metrics.relevance_score, self.min_relevance
            ));
        }

        if metrics.coherence_score < self.min_coherence {
            issues.push(format!(
                "Coherence {:.2} < {:.2}",
                metrics.coherence_score, self.min_coherence
            ));
        }

        if metrics.safety_score < self.min_safety {
            issues.push(format!(
                "Safety {:.2} < {:.2}",
                metrics.safety_score, self.min_safety
            ));
        }

        if metrics.latency_p95_ms > self.max_latency_ms {
            issues.push(format!(
                "Latency {}ms > {}ms",
                metrics.latency_p95_ms, self.max_latency_ms
            ));
        }

        if issues.is_empty() {
            QualityCheckResult::Passed
        } else {
            QualityCheckResult::Failed(issues)
        }
    }
}

impl Default for QualityGate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum QualityCheckResult {
    Passed,
    Failed(Vec<String>),
}

impl QualityCheckResult {
    pub fn is_passed(&self) -> bool {
        matches!(self, QualityCheckResult::Passed)
    }
}

// ============================================================================
// Model Endpoint
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub name: String,
    pub model_name: String,
    pub version: String,
    pub min_replicas: usize,
    pub max_replicas: usize,
    pub target_concurrency: usize,
}

impl EndpointConfig {
    pub fn new(name: &str, model: &str) -> Self {
        Self {
            name: name.to_string(),
            model_name: model.to_string(),
            version: "1".to_string(),
            min_replicas: 1,
            max_replicas: 4,
            target_concurrency: 10,
        }
    }

    pub fn with_scaling(mut self, min: usize, max: usize) -> Self {
        self.min_replicas = min;
        self.max_replicas = max;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStatus {
    pub name: String,
    pub state: EndpointState,
    pub replicas: usize,
    pub pending_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndpointState {
    Ready,
    Updating,
    Failed,
    ScalingUp,
    ScalingDown,
}

// ============================================================================
// A/B Testing
// ============================================================================

#[derive(Debug, Clone)]
pub struct ABTest {
    name: String,
    variants: Vec<Variant>,
    traffic_split: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub model_version: String,
}

impl ABTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            variants: Vec::new(),
            traffic_split: Vec::new(),
        }
    }

    pub fn add_variant(mut self, name: &str, version: &str, traffic: f32) -> Self {
        self.variants.push(Variant {
            name: name.to_string(),
            model_version: version.to_string(),
        });
        self.traffic_split.push(traffic);
        self
    }

    /// Select variant based on random value (0-1)
    pub fn select_variant(&self, rand_val: f32) -> Option<&Variant> {
        let mut cumulative = 0.0;
        for (i, &split) in self.traffic_split.iter().enumerate() {
            cumulative += split;
            if rand_val < cumulative {
                return self.variants.get(i);
            }
        }
        self.variants.last()
    }

    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Cost Tracking
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    pub input_token_cost: f32,
    pub output_token_cost: f32,
    pub currency: String,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            input_token_cost: 0.00001,
            output_token_cost: 0.00003,
            currency: "USD".to_string(),
        }
    }
}

#[derive(Debug, Default)]
pub struct CostTracker {
    config: CostConfig,
    total_input_tokens: usize,
    total_output_tokens: usize,
}

impl CostTracker {
    pub fn new(config: CostConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    pub fn track(&mut self, input_tokens: usize, output_tokens: usize) {
        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
    }

    pub fn total_cost(&self) -> f32 {
        self.total_input_tokens as f32 * self.config.input_token_cost
            + self.total_output_tokens as f32 * self.config.output_token_cost
    }

    pub fn total_tokens(&self) -> usize {
        self.total_input_tokens + self.total_output_tokens
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Production GenAI Demo - Course 4 Week 3                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Rate Limiting
    println!("⏱️ Step 1: Rate Limiting");
    let mut limiter = RateLimiter::new(3, 60);

    for i in 1..=5 {
        let allowed = limiter.check();
        println!(
            "   Request {}: {} (remaining: {})",
            i,
            if allowed { "allowed" } else { "blocked" },
            limiter.remaining()
        );
    }
    println!();

    // Step 2: Request Logging
    println!("📊 Step 2: Request Logging");
    let mut logger = RequestLogger::new();

    logger.log(RequestLog {
        request_id: "req-1".to_string(),
        timestamp_ms: 1000,
        prompt_tokens: 100,
        completion_tokens: 50,
        latency_ms: 200,
        status: RequestStatus::Success,
    });
    logger.log(RequestLog {
        request_id: "req-2".to_string(),
        timestamp_ms: 2000,
        prompt_tokens: 150,
        completion_tokens: 75,
        latency_ms: 300,
        status: RequestStatus::Success,
    });
    logger.log(RequestLog {
        request_id: "req-3".to_string(),
        timestamp_ms: 3000,
        prompt_tokens: 0,
        completion_tokens: 0,
        latency_ms: 0,
        status: RequestStatus::Error("timeout".to_string()),
    });

    println!("   Total requests: {}", logger.count());
    println!("   Success rate: {:.1}%", logger.success_rate() * 100.0);
    println!("   Avg latency: {:.1}ms", logger.average_latency());
    println!("   Total tokens: {}\n", logger.total_tokens());

    // Step 3: Content Filtering
    println!("🛡️ Step 3: Content Filtering");
    let filter = ContentFilter::new()
        .add_blocked_pattern("harmful")
        .add_sensitive_topic("politics");

    let test_inputs = ["Hello world", "Harmful content here", "Let's discuss politics"];

    for input in &test_inputs {
        let result = filter.should_filter(input);
        println!(
            "   \"{}\" -> {:?}",
            &input[..input.len().min(20)],
            match &result {
                FilterResult::Allowed => "Allowed".to_string(),
                FilterResult::Flagged(t) => format!("Flagged: {}", t),
                FilterResult::Blocked(p) => format!("Blocked: {}", p),
            }
        );
    }
    println!();

    // Step 4: Quality Gates
    println!("✅ Step 4: Quality Gates");
    let gate = QualityGate::new()
        .with_min_relevance(0.7)
        .with_min_safety(0.9)
        .with_max_latency(3000);

    let good_metrics = QualityMetrics {
        relevance_score: 0.85,
        coherence_score: 0.9,
        safety_score: 0.95,
        latency_p95_ms: 1500,
    };

    let bad_metrics = QualityMetrics {
        relevance_score: 0.5,
        coherence_score: 0.6,
        safety_score: 0.8,
        latency_p95_ms: 5000,
    };

    println!("   Good metrics: {:?}", gate.check(&good_metrics));
    println!("   Bad metrics: {:?}\n", gate.check(&bad_metrics));

    // Step 5: A/B Testing
    println!("🔀 Step 5: A/B Testing");
    let ab_test = ABTest::new("model-comparison")
        .add_variant("control", "v1.0", 0.5)
        .add_variant("treatment", "v2.0", 0.5);

    println!("   Test: {}", ab_test.name());
    println!("   Variants: {}", ab_test.variant_count());

    // Simulate traffic split
    let mut control_count = 0;
    let mut treatment_count = 0;
    for i in 0..100 {
        let rand_val = (i as f32 * 0.01) % 1.0;
        if let Some(variant) = ab_test.select_variant(rand_val) {
            if variant.name == "control" {
                control_count += 1;
            } else {
                treatment_count += 1;
            }
        }
    }
    println!(
        "   Traffic split (100 requests): control={}, treatment={}\n",
        control_count, treatment_count
    );

    // Step 6: Cost Tracking
    println!("💰 Step 6: Cost Tracking");
    let mut cost_tracker = CostTracker::new(CostConfig::default());

    cost_tracker.track(1000, 500);
    cost_tracker.track(2000, 1000);

    println!("   Total tokens: {}", cost_tracker.total_tokens());
    println!("   Estimated cost: ${:.4}\n", cost_tracker.total_cost());

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Rate limiting and throttling");
    println!("  • Request logging and metrics");
    println!("  • Content filtering and safety");
    println!("  • Quality gates for deployment");
    println!("  • A/B testing for model comparison");
    println!("  • Cost tracking and optimization");
    println!();
    println!("Databricks equivalent: Model Serving, AI Gateway");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2, 60);
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(!limiter.check());
    }

    #[test]
    fn test_rate_limiter_remaining() {
        let mut limiter = RateLimiter::new(3, 60);
        assert_eq!(limiter.remaining(), 3);
        limiter.check();
        assert_eq!(limiter.remaining(), 2);
    }

    #[test]
    fn test_request_logger() {
        let mut logger = RequestLogger::new();
        logger.log(RequestLog {
            request_id: "1".to_string(),
            timestamp_ms: 0,
            prompt_tokens: 10,
            completion_tokens: 5,
            latency_ms: 100,
            status: RequestStatus::Success,
        });
        assert_eq!(logger.count(), 1);
        assert!((logger.success_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_content_filter_allowed() {
        let filter = ContentFilter::new();
        assert!(filter.should_filter("hello").is_allowed());
    }

    #[test]
    fn test_content_filter_blocked() {
        let filter = ContentFilter::new().add_blocked_pattern("bad");
        assert!(filter.should_filter("this is bad").is_blocked());
    }

    #[test]
    fn test_quality_gate_pass() {
        let gate = QualityGate::new();
        let metrics = QualityMetrics {
            relevance_score: 0.9,
            coherence_score: 0.9,
            safety_score: 0.95,
            latency_p95_ms: 1000,
        };
        assert!(gate.check(&metrics).is_passed());
    }

    #[test]
    fn test_quality_gate_fail() {
        let gate = QualityGate::new().with_min_safety(0.95);
        let metrics = QualityMetrics {
            relevance_score: 0.9,
            coherence_score: 0.9,
            safety_score: 0.8,
            latency_p95_ms: 1000,
        };
        assert!(!gate.check(&metrics).is_passed());
    }

    #[test]
    fn test_ab_test_select() {
        let ab_test = ABTest::new("test")
            .add_variant("a", "1", 0.5)
            .add_variant("b", "2", 0.5);

        let variant_a = ab_test.select_variant(0.25);
        let variant_b = ab_test.select_variant(0.75);

        assert!(variant_a.is_some());
        assert!(variant_b.is_some());
    }

    #[test]
    fn test_cost_tracker() {
        let mut tracker = CostTracker::new(CostConfig::default());
        tracker.track(1000, 500);
        assert_eq!(tracker.total_tokens(), 1500);
        assert!(tracker.total_cost() > 0.0);
    }

    #[test]
    fn test_endpoint_config() {
        let config = EndpointConfig::new("my-endpoint", "llama")
            .with_scaling(2, 8);
        assert_eq!(config.min_replicas, 2);
        assert_eq!(config.max_replicas, 8);
    }

    #[test]
    fn test_production_error() {
        let err = ProductionError::RateLimitExceeded;
        assert!(err.to_string().contains("Rate limit"));
    }
}
