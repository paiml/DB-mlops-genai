//! Model Serving with realizar
//!
//! Demonstrates model serving patterns using realizar concepts.
//! This example shows model loading, inference, and circuit breaker patterns.
//!
//! # Course 3, Week 2: Model Training + Model Serving

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum ServingError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

// ============================================================================
// Model Metadata
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub framework: String,
    pub input_schema: Vec<FeatureSchema>,
    pub output_schema: OutputSchema,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub labels: Option<Vec<String>>,
}

impl ModelMetadata {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            framework: "aprender".to_string(),
            input_schema: Vec::new(),
            output_schema: OutputSchema {
                dtype: "float32".to_string(),
                shape: vec![1],
                labels: None,
            },
            created_at: 0,
        }
    }

    pub fn with_input(mut self, name: &str, dtype: &str, shape: Vec<usize>) -> Self {
        self.input_schema.push(FeatureSchema {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
        });
        self
    }

    pub fn with_output(
        mut self,
        dtype: &str,
        shape: Vec<usize>,
        labels: Option<Vec<String>>,
    ) -> Self {
        self.output_schema = OutputSchema {
            dtype: dtype.to_string(),
            shape,
            labels,
        };
        self
    }
}

// ============================================================================
// Inference Request/Response
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub inputs: HashMap<String, Vec<f64>>,
    pub parameters: Option<InferenceParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParameters {
    pub timeout_ms: Option<u64>,
    pub return_probabilities: bool,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            timeout_ms: Some(1000),
            return_probabilities: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub predictions: Vec<f64>,
    pub probabilities: Option<Vec<Vec<f64>>>,
    pub latency_ms: u64,
    pub model_version: String,
}

// ============================================================================
// Circuit Breaker
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: usize,
    failure_threshold: usize,
    success_count: usize,
    success_threshold: usize,
    last_failure_time: Option<Instant>,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, reset_timeout_ms: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            success_count: 0,
            success_threshold: 3,
            last_failure_time: None,
            reset_timeout: Duration::from_millis(reset_timeout_ms),
        }
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.reset_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&mut self) {
        self.last_failure_time = Some(Instant::now());
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
            }
            CircuitState::Open => {}
        }
    }

    pub fn failure_count(&self) -> usize {
        self.failure_count
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 30000)
    }
}

// ============================================================================
// Model Server
// ============================================================================

pub struct ModelServer {
    metadata: ModelMetadata,
    weights: Vec<f64>,
    circuit_breaker: CircuitBreaker,
    request_count: usize,
    error_count: usize,
}

impl ModelServer {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            weights: Vec::new(),
            circuit_breaker: CircuitBreaker::default(),
            request_count: 0,
            error_count: 0,
        }
    }

    pub fn with_circuit_breaker(mut self, cb: CircuitBreaker) -> Self {
        self.circuit_breaker = cb;
        self
    }

    /// Load model weights (simulated)
    pub fn load(&mut self, weights: Vec<f64>) -> Result<(), ServingError> {
        if weights.is_empty() {
            return Err(ServingError::ModelNotLoaded("Empty weights".to_string()));
        }
        self.weights = weights;
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        !self.weights.is_empty()
    }

    /// Run inference
    pub fn predict(
        &mut self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, ServingError> {
        let start = Instant::now();
        self.request_count += 1;

        // Check circuit breaker
        if !self.circuit_breaker.can_execute() {
            self.error_count += 1;
            return Err(ServingError::CircuitBreakerOpen(
                "Too many failures, circuit is open".to_string(),
            ));
        }

        // Check model is loaded
        if !self.is_loaded() {
            self.circuit_breaker.record_failure();
            self.error_count += 1;
            return Err(ServingError::ModelNotLoaded(self.metadata.name.clone()));
        }

        // Simulate inference (linear model)
        let predictions = self.run_inference(request)?;

        self.circuit_breaker.record_success();

        let latency = start.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            predictions,
            probabilities: None,
            latency_ms: latency,
            model_version: self.metadata.version.clone(),
        })
    }

    fn run_inference(&self, request: &InferenceRequest) -> Result<Vec<f64>, ServingError> {
        // Get input features
        let inputs: Vec<f64> = request
            .inputs
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect();

        if inputs.is_empty() {
            return Err(ServingError::Inference("No input features".to_string()));
        }

        // Simple linear prediction: y = sum(w * x)
        let prediction: f64 = inputs
            .iter()
            .zip(self.weights.iter().cycle())
            .map(|(x, w)| x * w)
            .sum();

        // Apply sigmoid for classification
        let prob = 1.0 / (1.0 + (-prediction).exp());

        Ok(vec![prob])
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn stats(&self) -> ServerStats {
        ServerStats {
            request_count: self.request_count,
            error_count: self.error_count,
            error_rate: if self.request_count > 0 {
                self.error_count as f64 / self.request_count as f64
            } else {
                0.0
            },
            circuit_state: self.circuit_breaker.state(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerStats {
    pub request_count: usize,
    pub error_count: usize,
    pub error_rate: f64,
    pub circuit_state: CircuitState,
}

// ============================================================================
// Batch Inference
// ============================================================================

pub struct BatchPredictor {
    server: ModelServer,
    batch_size: usize,
}

impl BatchPredictor {
    pub fn new(server: ModelServer, batch_size: usize) -> Self {
        Self { server, batch_size }
    }

    pub fn predict_batch(
        &mut self,
        requests: Vec<InferenceRequest>,
    ) -> Vec<Result<InferenceResponse, ServingError>> {
        requests
            .chunks(self.batch_size)
            .flat_map(|batch| {
                batch
                    .iter()
                    .map(|req| self.server.predict(req))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

// ============================================================================
// Health Check
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub model_loaded: bool,
    pub circuit_state: String,
    pub uptime_seconds: u64,
}

impl HealthStatus {
    pub fn from_server(server: &ModelServer, uptime: Duration) -> Self {
        let circuit_state = match server.circuit_breaker.state() {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half-open",
        };

        Self {
            status: if server.is_loaded() && server.circuit_breaker.state() != CircuitState::Open {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            model_loaded: server.is_loaded(),
            circuit_state: circuit_state.to_string(),
            uptime_seconds: uptime.as_secs(),
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Model Serving with realizar - Course 3, Week 2            ║");
    println!("║     Inference, Circuit Breaker, Health Checks                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let start_time = Instant::now();

    // Step 1: Create model metadata
    println!("\n📋 Step 1: Model Metadata");
    let metadata = ModelMetadata::new("fraud-detector", "1.0.0")
        .with_input("amount", "float64", vec![1])
        .with_input("velocity", "float64", vec![1])
        .with_output(
            "float64",
            vec![1],
            Some(vec!["normal".to_string(), "fraud".to_string()]),
        );

    println!("   Model: {} v{}", metadata.name, metadata.version);
    println!("   Framework: {}", metadata.framework);
    println!("   Inputs: {} features", metadata.input_schema.len());

    // Step 2: Load model
    println!("\n🔧 Step 2: Model Loading");
    let mut server = ModelServer::new(metadata).with_circuit_breaker(CircuitBreaker::new(3, 5000));

    // Simulate model weights
    let weights = vec![0.5, 0.3, -0.2, 0.8];
    server.load(weights).unwrap();
    println!("   Model loaded: {}", server.is_loaded());

    // Step 3: Run inference
    println!("\n🚀 Step 3: Inference");
    let requests = vec![
        ("normal_tx", vec![100.0, 1.0]),
        ("suspicious_tx", vec![5000.0, 10.0]),
        ("high_velocity", vec![200.0, 50.0]),
    ];

    for (name, features) in &requests {
        let mut inputs = HashMap::new();
        inputs.insert("amount".to_string(), vec![features[0]]);
        inputs.insert("velocity".to_string(), vec![features[1]]);

        let request = InferenceRequest {
            inputs,
            parameters: Some(InferenceParameters::default()),
        };

        match server.predict(&request) {
            Ok(response) => {
                let label = if response.predictions[0] > 0.5 {
                    "fraud"
                } else {
                    "normal"
                };
                println!(
                    "   {}: prob={:.4} -> {} ({}ms)",
                    name, response.predictions[0], label, response.latency_ms
                );
            }
            Err(e) => println!("   {}: Error - {}", name, e),
        }
    }

    // Step 4: Circuit Breaker Demo
    println!("\n⚡ Step 4: Circuit Breaker");
    println!("   Initial state: {:?}", server.circuit_breaker.state());

    // Simulate failures by unloading model
    let weights_backup = server.weights.clone();
    server.weights.clear();

    for i in 1..=4 {
        let mut inputs = HashMap::new();
        inputs.insert("amount".to_string(), vec![100.0]);
        let request = InferenceRequest {
            inputs,
            parameters: None,
        };

        match server.predict(&request) {
            Ok(_) => println!("   Request {}: Success", i),
            Err(e) => println!(
                "   Request {}: {} (failures: {})",
                i,
                e,
                server.circuit_breaker.failure_count()
            ),
        }
    }

    println!("   Final state: {:?}", server.circuit_breaker.state());

    // Restore model
    server.weights = weights_backup;

    // Step 5: Health Check
    println!("\n🏥 Step 5: Health Check");
    let health = HealthStatus::from_server(&server, start_time.elapsed());
    println!("   Status: {}", health.status);
    println!("   Model loaded: {}", health.model_loaded);
    println!("   Circuit state: {}", health.circuit_state);
    println!("   Uptime: {}s", health.uptime_seconds);

    // Step 6: Server Stats
    println!("\n📊 Step 6: Server Statistics");
    let stats = server.stats();
    println!("   Total requests: {}", stats.request_count);
    println!("   Errors: {}", stats.error_count);
    println!("   Error rate: {:.1}%", stats.error_rate * 100.0);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Model metadata and schema definition");
    println!("  • Inference request/response handling");
    println!("  • Circuit breaker pattern for fault tolerance");
    println!("  • Health checks and server statistics");
    println!();
    println!("Sovereign AI Stack: realizar inference server");
    println!("Databricks equivalent: Model Serving endpoints");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ModelMetadata Tests
    // ========================================================================

    #[test]
    fn test_model_metadata_new() {
        let meta = ModelMetadata::new("test", "1.0");
        assert_eq!(meta.name, "test");
        assert_eq!(meta.version, "1.0");
    }

    #[test]
    fn test_model_metadata_with_input() {
        let meta = ModelMetadata::new("test", "1.0").with_input("feature", "float64", vec![1]);
        assert_eq!(meta.input_schema.len(), 1);
        assert_eq!(meta.input_schema[0].name, "feature");
    }

    #[test]
    fn test_model_metadata_with_output() {
        let meta = ModelMetadata::new("test", "1.0").with_output(
            "float64",
            vec![2],
            Some(vec!["a".to_string(), "b".to_string()]),
        );
        assert_eq!(meta.output_schema.shape, vec![2]);
        assert!(meta.output_schema.labels.is_some());
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata::new("test", "1.0");
        let cloned = meta.clone();
        assert_eq!(meta.name, cloned.name);
    }

    // ========================================================================
    // CircuitBreaker Tests
    // ========================================================================

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 1000);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_success_resets_count() {
        let mut cb = CircuitBreaker::new(3, 1000);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_can_execute() {
        let mut cb = CircuitBreaker::new(1, 1000);
        assert!(cb.can_execute());
        cb.record_failure();
        assert!(!cb.can_execute());
    }

    #[test]
    fn test_circuit_breaker_default() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // ========================================================================
    // ModelServer Tests
    // ========================================================================

    #[test]
    fn test_model_server_load() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        assert!(!server.is_loaded());
        server.load(vec![1.0, 2.0]).unwrap();
        assert!(server.is_loaded());
    }

    #[test]
    fn test_model_server_load_empty() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        let result = server.load(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_server_predict() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        server.load(vec![0.5, 0.5]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0]);
        let request = InferenceRequest {
            inputs,
            parameters: None,
        };

        let response = server.predict(&request).unwrap();
        assert!(!response.predictions.is_empty());
    }

    #[test]
    fn test_model_server_predict_not_loaded() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0]);
        let request = InferenceRequest {
            inputs,
            parameters: None,
        };

        let result = server.predict(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_server_stats() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        server.load(vec![1.0]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0]);
        let request = InferenceRequest {
            inputs,
            parameters: None,
        };

        server.predict(&request).unwrap();
        let stats = server.stats();
        assert_eq!(stats.request_count, 1);
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_model_server_metadata() {
        let meta = ModelMetadata::new("test", "1.0");
        let server = ModelServer::new(meta);
        assert_eq!(server.metadata().name, "test");
    }

    // ========================================================================
    // InferenceParameters Tests
    // ========================================================================

    #[test]
    fn test_inference_parameters_default() {
        let params = InferenceParameters::default();
        assert_eq!(params.timeout_ms, Some(1000));
        assert!(!params.return_probabilities);
    }

    #[test]
    fn test_inference_parameters_serialization() {
        let params = InferenceParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: InferenceParameters = serde_json::from_str(&json).unwrap();
        assert_eq!(params.timeout_ms, restored.timeout_ms);
    }

    // ========================================================================
    // BatchPredictor Tests
    // ========================================================================

    #[test]
    fn test_batch_predictor() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        server.load(vec![0.5]).unwrap();

        let mut predictor = BatchPredictor::new(server, 2);

        let requests: Vec<InferenceRequest> = (0..3)
            .map(|_| {
                let mut inputs = HashMap::new();
                inputs.insert("x".to_string(), vec![1.0]);
                InferenceRequest {
                    inputs,
                    parameters: None,
                }
            })
            .collect();

        let results = predictor.predict_batch(requests);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_predictor_batch_size() {
        let meta = ModelMetadata::new("test", "1.0");
        let server = ModelServer::new(meta);
        let predictor = BatchPredictor::new(server, 10);
        assert_eq!(predictor.batch_size(), 10);
    }

    // ========================================================================
    // HealthStatus Tests
    // ========================================================================

    #[test]
    fn test_health_status_healthy() {
        let meta = ModelMetadata::new("test", "1.0");
        let mut server = ModelServer::new(meta);
        server.load(vec![1.0]).unwrap();

        let health = HealthStatus::from_server(&server, Duration::from_secs(100));
        assert_eq!(health.status, "healthy");
        assert!(health.model_loaded);
    }

    #[test]
    fn test_health_status_unhealthy() {
        let meta = ModelMetadata::new("test", "1.0");
        let server = ModelServer::new(meta);

        let health = HealthStatus::from_server(&server, Duration::from_secs(0));
        assert_eq!(health.status, "unhealthy");
        assert!(!health.model_loaded);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus {
            status: "healthy".to_string(),
            model_loaded: true,
            circuit_state: "closed".to_string(),
            uptime_seconds: 100,
        };
        let json = serde_json::to_string(&status).unwrap();
        let restored: HealthStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status.status, restored.status);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_model_not_loaded() {
        let err = ServingError::ModelNotLoaded("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_error_inference() {
        let err = ServingError::Inference("failed".to_string());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_error_circuit_breaker() {
        let err = ServingError::CircuitBreakerOpen("open".to_string());
        assert!(err.to_string().contains("open"));
    }

    #[test]
    fn test_error_timeout() {
        let err = ServingError::Timeout("1000ms".to_string());
        assert!(err.to_string().contains("1000ms"));
    }

    #[test]
    fn test_error_debug() {
        let err = ServingError::Inference("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Inference"));
    }
}
