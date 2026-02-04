//! Model Inference Server Demo
//!
//! Demonstrates building an inference server similar to realizar.
//! Compare with Databricks Model Serving to understand what platforms abstract.
//!
//! # Course 3, Week 4: Model Serving and Inference

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{info, instrument};

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
}

// ============================================================================
// Model Types
// ============================================================================

/// A simple linear model for demonstration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModel {
    pub name: String,
    pub version: String,
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(name: &str, weights: Vec<f64>, bias: f64) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            weights,
            bias,
        }
    }

    /// Make a prediction
    pub fn predict(&self, features: &[f64]) -> Result<f64, InferenceError> {
        if features.len() != self.weights.len() {
            return Err(InferenceError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.weights.len(),
                features.len()
            )));
        }

        let mut prediction = self.bias;
        for (w, x) in self.weights.iter().zip(features.iter()) {
            prediction += w * x;
        }
        Ok(prediction)
    }

    /// Batch prediction
    pub fn predict_batch(&self, batch: &[Vec<f64>]) -> Result<Vec<f64>, InferenceError> {
        batch
            .iter()
            .map(|features| self.predict(features))
            .collect()
    }
}

// ============================================================================
// API Types (OpenAI-compatible style)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictRequest {
    pub inputs: Vec<Vec<f64>>,
    #[serde(default)]
    pub parameters: PredictParameters,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PredictParameters {
    #[serde(default)]
    pub return_probabilities: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    pub predictions: Vec<f64>,
    pub model: String,
    pub version: String,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub model_name: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    pub name: String,
    pub version: String,
    pub n_features: usize,
    pub model_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub total_predictions: u64,
    pub avg_latency_ms: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

// ============================================================================
// Server State
// ============================================================================

pub struct ServerState {
    pub model: Option<LinearModel>,
    pub metrics: ServerMetrics,
    pub start_time: Instant,
}

#[derive(Default)]
pub struct ServerMetrics {
    pub total_requests: u64,
    pub total_predictions: u64,
    pub total_latency_ms: f64,
}

impl ServerMetrics {
    pub fn record_request(&mut self, n_predictions: usize, latency_ms: f64) {
        self.total_requests += 1;
        self.total_predictions += n_predictions as u64;
        self.total_latency_ms += latency_ms;
    }

    pub fn avg_latency(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_latency_ms / self.total_requests as f64
        }
    }
}

type AppState = Arc<RwLock<ServerState>>;

// ============================================================================
// Handlers
// ============================================================================

/// Health check endpoint
#[instrument(skip(state))]
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let state = state.read().await;
    Json(HealthResponse {
        status: "healthy".to_string(),
        model_loaded: state.model.is_some(),
        model_name: state.model.as_ref().map(|m| m.name.clone()),
        version: state.model.as_ref().map(|m| m.version.clone()),
    })
}

/// Model info endpoint
#[instrument(skip(state))]
async fn model_info(
    State(state): State<AppState>,
) -> Result<Json<ModelInfoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let state = state.read().await;
    match &state.model {
        Some(model) => Ok(Json(ModelInfoResponse {
            name: model.name.clone(),
            version: model.version.clone(),
            n_features: model.weights.len(),
            model_type: "LinearModel".to_string(),
        })),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No model loaded".to_string(),
                code: "MODEL_NOT_LOADED".to_string(),
            }),
        )),
    }
}

/// Prediction endpoint
#[instrument(skip(state, request))]
async fn predict(
    State(state): State<AppState>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // Get model
    let mut state = state.write().await;
    let model = match &state.model {
        Some(m) => m.clone(),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "No model loaded".to_string(),
                    code: "MODEL_NOT_LOADED".to_string(),
                }),
            ));
        }
    };

    // Make predictions
    let predictions = match model.predict_batch(&request.inputs) {
        Ok(preds) => preds,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: "PREDICTION_FAILED".to_string(),
                }),
            ));
        }
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Update metrics
    state.metrics.record_request(predictions.len(), latency_ms);

    info!(
        "Predicted {} samples in {:.2}ms",
        predictions.len(),
        latency_ms
    );

    Ok(Json(PredictResponse {
        predictions,
        model: model.name,
        version: model.version,
        latency_ms,
    }))
}

/// Metrics endpoint
#[instrument(skip(state))]
async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let state = state.read().await;
    Json(MetricsResponse {
        total_requests: state.metrics.total_requests,
        total_predictions: state.metrics.total_predictions,
        avg_latency_ms: state.metrics.avg_latency(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

// ============================================================================
// Server
// ============================================================================

/// Build the inference server router
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(model_info))
        .route("/v1/predict", post(predict))
        .route("/metrics", get(metrics))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Create a demo model
fn create_demo_model() -> LinearModel {
    // Simulated fraud detection model
    LinearModel::new(
        "fraud-detector",
        vec![0.5, -0.3, 0.8, -0.2, 0.4], // 5 features
        0.1,                             // bias
    )
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Inference Server Demo - Course 3, Week 4                  ║");
    println!("║     Building Model Serving with realizar concepts             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Create server state with demo model
    let model = create_demo_model();
    println!("\n📦 Loaded model: {} v{}", model.name, model.version);
    println!("   Features: {}", model.weights.len());

    let state = Arc::new(RwLock::new(ServerState {
        model: Some(model),
        metrics: ServerMetrics::default(),
        start_time: Instant::now(),
    }));

    // Build router
    let app = build_router(state);

    // Start server
    let addr = "0.0.0.0:8080";
    println!("\n🚀 Starting server on {}", addr);
    println!("\n📖 Endpoints:");
    println!("   GET  /health     - Health check");
    println!("   GET  /v1/models  - Model info");
    println!("   POST /v1/predict - Make predictions");
    println!("   GET  /metrics    - Server metrics");
    println!("\n📝 Example request:");
    println!("   curl -X POST http://localhost:8080/v1/predict \\");
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{\"inputs\": [[0.1, 0.2, 0.3, 0.4, 0.5]]}}'");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Linear Model Tests
    // =========================================================================

    #[test]
    fn test_linear_model_predict() {
        let model = LinearModel::new("test", vec![1.0, 2.0, 3.0], 0.5);
        let result = model.predict(&[1.0, 1.0, 1.0]).unwrap();
        // 1*1 + 2*1 + 3*1 + 0.5 = 6.5
        assert!((result - 6.5).abs() < 0.001);
    }

    #[test]
    fn test_linear_model_predict_wrong_features() {
        let model = LinearModel::new("test", vec![1.0, 2.0, 3.0], 0.5);
        let result = model.predict(&[1.0, 1.0]); // Wrong number of features
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_model_new() {
        let model = LinearModel::new("fraud-detector", vec![0.5, 0.3], 0.1);
        assert_eq!(model.name, "fraud-detector");
        assert_eq!(model.version, "1.0.0");
        assert_eq!(model.weights.len(), 2);
        assert_eq!(model.bias, 0.1);
    }

    #[test]
    fn test_linear_model_predict_zeros() {
        let model = LinearModel::new("test", vec![1.0, 2.0], 0.5);
        let result = model.predict(&[0.0, 0.0]).unwrap();
        assert!((result - 0.5).abs() < 0.001); // Just bias
    }

    #[test]
    fn test_linear_model_predict_negative() {
        let model = LinearModel::new("test", vec![1.0, -1.0], 0.0);
        let result = model.predict(&[1.0, 2.0]).unwrap();
        assert!((result - (-1.0)).abs() < 0.001); // 1*1 + (-1)*2 = -1
    }

    #[test]
    fn test_linear_model_clone() {
        let model = LinearModel::new("test", vec![1.0, 2.0], 0.5);
        let cloned = model.clone();
        assert_eq!(model.name, cloned.name);
        assert_eq!(model.weights, cloned.weights);
    }

    // =========================================================================
    // Batch Prediction Tests
    // =========================================================================

    #[test]
    fn test_batch_predict() {
        let model = LinearModel::new("test", vec![1.0, 2.0], 0.0);
        let batch = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let results = model.predict_batch(&batch).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 0.001);
        assert!((results[1] - 2.0).abs() < 0.001);
        assert!((results[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_predict_empty() {
        let model = LinearModel::new("test", vec![1.0, 2.0], 0.0);
        let results = model.predict_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_predict_error_propagates() {
        let model = LinearModel::new("test", vec![1.0, 2.0], 0.0);
        let batch = vec![vec![1.0], vec![1.0, 2.0]]; // First has wrong features
        let result = model.predict_batch(&batch);
        assert!(result.is_err());
    }

    // =========================================================================
    // Metrics Tests
    // =========================================================================

    #[test]
    fn test_metrics_tracking() {
        let mut metrics = ServerMetrics::default();
        metrics.record_request(10, 5.0);
        metrics.record_request(20, 10.0);
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.total_predictions, 30);
        assert!((metrics.avg_latency() - 7.5).abs() < 0.001);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = ServerMetrics::default();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.total_predictions, 0);
        assert_eq!(metrics.total_latency_ms, 0.0);
    }

    #[test]
    fn test_metrics_avg_latency_zero_requests() {
        let metrics = ServerMetrics::default();
        assert_eq!(metrics.avg_latency(), 0.0);
    }

    #[test]
    fn test_metrics_single_request() {
        let mut metrics = ServerMetrics::default();
        metrics.record_request(5, 2.5);
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.total_predictions, 5);
        assert!((metrics.avg_latency() - 2.5).abs() < 0.001);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_model_serialization() {
        let model = LinearModel::new("test", vec![1.0, 2.0, 3.0], 0.5);
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: LinearModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model.name, deserialized.name);
        assert_eq!(model.weights.len(), deserialized.weights.len());
    }

    #[test]
    fn test_model_serialization_round_trip() {
        let model = LinearModel::new("fraud", vec![0.5, -0.3, 0.8], 0.1);
        let json = serde_json::to_string(&model).unwrap();
        let restored: LinearModel = serde_json::from_str(&json).unwrap();

        // Should produce identical predictions
        let pred1 = model.predict(&[1.0, 2.0, 3.0]).unwrap();
        let pred2 = restored.predict(&[1.0, 2.0, 3.0]).unwrap();
        assert!((pred1 - pred2).abs() < 0.001);
    }

    // =========================================================================
    // Request/Response Types Tests
    // =========================================================================

    #[test]
    fn test_predict_request_deserialization() {
        let json = r#"{"inputs": [[1.0, 2.0, 3.0]]}"#;
        let request: PredictRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.inputs.len(), 1);
        assert_eq!(request.inputs[0].len(), 3);
    }

    #[test]
    fn test_predict_request_with_parameters() {
        let json = r#"{"inputs": [[1.0]], "parameters": {"return_probabilities": true}}"#;
        let request: PredictRequest = serde_json::from_str(json).unwrap();
        assert!(request.parameters.return_probabilities);
    }

    #[test]
    fn test_predict_parameters_default() {
        let params = PredictParameters::default();
        assert!(!params.return_probabilities);
    }

    #[test]
    fn test_predict_response_serialization() {
        let response = PredictResponse {
            predictions: vec![0.5, 0.7],
            model: "test".to_string(),
            version: "1.0.0".to_string(),
            latency_ms: 2.5,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("predictions"));
        assert!(json.contains("latency_ms"));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            model_loaded: true,
            model_name: Some("test".to_string()),
            version: Some("1.0.0".to_string()),
        };
        let json = serde_json::to_string(&response).unwrap();
        let restored: HealthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.status, restored.status);
        assert_eq!(response.model_loaded, restored.model_loaded);
    }

    #[test]
    fn test_model_info_response_serialization() {
        let response = ModelInfoResponse {
            name: "fraud-detector".to_string(),
            version: "1.0.0".to_string(),
            n_features: 5,
            model_type: "LinearModel".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("n_features"));
    }

    #[test]
    fn test_metrics_response_serialization() {
        let response = MetricsResponse {
            total_requests: 100,
            total_predictions: 500,
            avg_latency_ms: 2.5,
            uptime_seconds: 3600,
        };
        let json = serde_json::to_string(&response).unwrap();
        let restored: MetricsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.total_requests, restored.total_requests);
    }

    #[test]
    fn test_error_response_serialization() {
        let response = ErrorResponse {
            error: "Model not found".to_string(),
            code: "MODEL_NOT_LOADED".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("MODEL_NOT_LOADED"));
    }

    // =========================================================================
    // Error Type Tests
    // =========================================================================

    #[test]
    fn test_inference_error_model_not_loaded() {
        let err = InferenceError::ModelNotLoaded("fraud-detector".to_string());
        assert!(err.to_string().contains("fraud-detector"));
    }

    #[test]
    fn test_inference_error_invalid_input() {
        let err = InferenceError::InvalidInput("wrong shape".to_string());
        assert!(err.to_string().contains("wrong shape"));
    }

    #[test]
    fn test_inference_error_prediction_failed() {
        let err = InferenceError::PredictionFailed("NaN detected".to_string());
        assert!(err.to_string().contains("NaN detected"));
    }

    #[test]
    fn test_inference_error_debug() {
        let err = InferenceError::InvalidInput("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidInput"));
    }

    // =========================================================================
    // Demo Model Tests
    // =========================================================================

    #[test]
    fn test_create_demo_model() {
        let model = create_demo_model();
        assert_eq!(model.name, "fraud-detector");
        assert_eq!(model.weights.len(), 5);
    }
}
