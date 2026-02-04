//! MLflow REST Client in Rust
//!
//! Demonstrates the MLflow Tracking REST API by building a type-safe client.
//! This helps understand what platforms like Databricks abstract away.
//!
//! # Course 3, Week 1: Experiment Tracking with MLflow

use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum MlflowError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {message}")]
    Api { message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ============================================================================
// Data Types (matching MLflow REST API schema)
// ============================================================================

/// An MLflow experiment - a named container for runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub experiment_id: String,
    pub name: String,
    pub artifact_location: Option<String>,
    pub lifecycle_stage: String,
    #[serde(default)]
    pub tags: Vec<ExperimentTag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTag {
    pub key: String,
    pub value: String,
}

/// An MLflow run - a single execution of ML code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub info: RunInfo,
    pub data: RunData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub artifact_uri: Option<String>,
    pub lifecycle_stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunStatus {
    Running,
    Scheduled,
    Finished,
    Failed,
    Killed,
}

impl fmt::Display for RunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunStatus::Running => write!(f, "RUNNING"),
            RunStatus::Scheduled => write!(f, "SCHEDULED"),
            RunStatus::Finished => write!(f, "FINISHED"),
            RunStatus::Failed => write!(f, "FAILED"),
            RunStatus::Killed => write!(f, "KILLED"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunData {
    #[serde(default)]
    pub metrics: Vec<Metric>,
    #[serde(default)]
    pub params: Vec<Param>,
    #[serde(default)]
    pub tags: Vec<RunTag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub key: String,
    pub value: f64,
    pub timestamp: i64,
    #[serde(default)]
    pub step: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTag {
    pub key: String,
    pub value: String,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct CreateExperimentResponse {
    experiment_id: String,
}

#[derive(Debug, Deserialize)]
struct CreateRunResponse {
    run: Run,
}

#[derive(Debug, Deserialize)]
struct GetExperimentResponse {
    experiment: Experiment,
}

#[derive(Debug, Deserialize)]
struct SearchRunsResponse {
    runs: Option<Vec<Run>>,
}

// ============================================================================
// MLflow Client
// ============================================================================

/// A Rust client for the MLflow REST API
///
/// # Example
/// ```no_run
/// use mlflow_client::MlflowClient;
///
/// #[tokio::main]
/// async fn main() {
///     let client = MlflowClient::new("http://localhost:5000");
///     let exp_id = client.create_experiment("my-experiment").await.unwrap();
///     println!("Created experiment: {}", exp_id);
/// }
/// ```
pub struct MlflowClient {
    base_url: String,
    client: Client,
}

impl MlflowClient {
    /// Create a new MLflow client
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    /// Build API endpoint URL
    fn endpoint(&self, path: &str) -> String {
        format!("{}/api/2.0/mlflow/{}", self.base_url, path)
    }

    /// Execute POST request and deserialize response
    async fn post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<T, MlflowError> {
        self.client
            .post(self.endpoint(path))
            .json(body)
            .send()
            .await?
            .json()
            .await
            .map_err(Into::into)
    }

    /// Execute POST request without response body
    async fn post_void(&self, path: &str, body: &serde_json::Value) -> Result<(), MlflowError> {
        self.client
            .post(self.endpoint(path))
            .json(body)
            .send()
            .await?;
        Ok(())
    }

    /// Execute GET request and deserialize response
    async fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, MlflowError> {
        self.client
            .get(url)
            .send()
            .await?
            .json()
            .await
            .map_err(Into::into)
    }

    /// Create a new experiment
    ///
    /// Maps to: POST /api/2.0/mlflow/experiments/create
    pub async fn create_experiment(&self, name: &str) -> Result<String, MlflowError> {
        let body = serde_json::json!({ "name": name });
        let response: CreateExperimentResponse = self.post("experiments/create", &body).await?;
        Ok(response.experiment_id)
    }

    /// Get experiment by ID
    ///
    /// Maps to: GET /api/2.0/mlflow/experiments/get
    pub async fn get_experiment(&self, experiment_id: &str) -> Result<Experiment, MlflowError> {
        let url = format!(
            "{}?experiment_id={}",
            self.endpoint("experiments/get"),
            experiment_id
        );
        let response: GetExperimentResponse = self.get(&url).await?;
        Ok(response.experiment)
    }

    /// Create a new run within an experiment
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/create
    pub async fn create_run(&self, experiment_id: &str) -> Result<Run, MlflowError> {
        let body = serde_json::json!({
            "experiment_id": experiment_id,
            "start_time": Utc::now().timestamp_millis()
        });
        let response: CreateRunResponse = self.post("runs/create", &body).await?;
        Ok(response.run)
    }

    /// Log a parameter to a run
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/log-parameter
    pub async fn log_param(&self, run_id: &str, key: &str, value: &str) -> Result<(), MlflowError> {
        let body = serde_json::json!({ "run_id": run_id, "key": key, "value": value });
        self.post_void("runs/log-parameter", &body).await
    }

    /// Log a metric to a run
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/log-metric
    pub async fn log_metric(
        &self,
        run_id: &str,
        key: &str,
        value: f64,
        step: Option<i64>,
    ) -> Result<(), MlflowError> {
        let body = serde_json::json!({
            "run_id": run_id,
            "key": key,
            "value": value,
            "timestamp": Utc::now().timestamp_millis(),
            "step": step.unwrap_or(0)
        });
        self.post_void("runs/log-metric", &body).await
    }

    /// Log multiple metrics in a batch
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/log-batch
    pub async fn log_batch(
        &self,
        run_id: &str,
        metrics: Vec<Metric>,
        params: Vec<Param>,
    ) -> Result<(), MlflowError> {
        let body = serde_json::json!({ "run_id": run_id, "metrics": metrics, "params": params });
        self.post_void("runs/log-batch", &body).await
    }

    /// End a run
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/update
    pub async fn end_run(&self, run_id: &str, status: RunStatus) -> Result<(), MlflowError> {
        let body = serde_json::json!({
            "run_id": run_id,
            "status": status,
            "end_time": Utc::now().timestamp_millis()
        });
        self.post_void("runs/update", &body).await
    }

    /// Search runs in an experiment
    ///
    /// Maps to: POST /api/2.0/mlflow/runs/search
    pub async fn search_runs(
        &self,
        experiment_ids: &[&str],
        filter: Option<&str>,
    ) -> Result<Vec<Run>, MlflowError> {
        let body = serde_json::json!({
            "experiment_ids": experiment_ids,
            "filter": filter.unwrap_or("")
        });
        let response: SearchRunsResponse = self.post("runs/search", &body).await?;
        Ok(response.runs.unwrap_or_default())
    }
}

// ============================================================================
// Demo: Simulated ML Training Run
// ============================================================================

/// Simulate a training run with metrics
async fn demo_training_run(client: &MlflowClient, experiment_id: &str) -> Result<(), MlflowError> {
    println!("\n📊 Starting training run...");

    // Create a new run
    let run = client.create_run(experiment_id).await?;
    let run_id = &run.info.run_id;
    println!("   Run ID: {}", run_id);

    // Log hyperparameters
    println!("   Logging parameters...");
    client.log_param(run_id, "learning_rate", "0.001").await?;
    client.log_param(run_id, "batch_size", "32").await?;
    client.log_param(run_id, "epochs", "10").await?;
    client.log_param(run_id, "optimizer", "adam").await?;

    // Simulate training epochs with metrics
    println!("   Simulating training...");
    for epoch in 0..10 {
        // Simulate decreasing loss and increasing accuracy
        let loss = 1.0 / (epoch as f64 + 1.0) + 0.1;
        let accuracy = 1.0 - loss + 0.05;

        client.log_metric(run_id, "loss", loss, Some(epoch)).await?;
        client
            .log_metric(run_id, "accuracy", accuracy.min(0.99), Some(epoch))
            .await?;

        print!(".");
    }
    println!(" done!");

    // End the run
    client.end_run(run_id, RunStatus::Finished).await?;
    println!("   Run completed successfully!");

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     MLflow REST Client Demo - Course 3, Week 1                ║");
    println!("║     Understanding Experiment Tracking by Building             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Default to local MLflow server
    let mlflow_url = std::env::var("MLFLOW_TRACKING_URI")
        .unwrap_or_else(|_| "http://localhost:5000".to_string());

    println!("\n🔗 MLflow Tracking URI: {}", mlflow_url);

    let client = MlflowClient::new(&mlflow_url);

    // Create experiment
    let experiment_name = format!("rust-demo-{}", Utc::now().format("%Y%m%d-%H%M%S"));
    println!("\n📁 Creating experiment: {}", experiment_name);

    match client.create_experiment(&experiment_name).await {
        Ok(exp_id) => {
            println!("   Experiment ID: {}", exp_id);

            // Run demo training
            demo_training_run(&client, &exp_id).await?;

            // Search runs
            println!("\n🔍 Searching runs...");
            let runs = client.search_runs(&[&exp_id], None).await?;
            println!("   Found {} run(s)", runs.len());

            for run in &runs {
                println!("   - {} ({})", run.info.run_id, run.info.status);
            }
        }
        Err(e) => {
            println!("\n⚠️  Could not connect to MLflow server: {}", e);
            println!("\n   To run this demo, start MLflow first:");
            println!("   $ mlflow server --host 0.0.0.0 --port 5000");
            println!("\n   Or set MLFLOW_TRACKING_URI environment variable.");
            println!("\n📖 This demo shows the MLflow REST API structure:");
            println!("   - POST /api/2.0/mlflow/experiments/create");
            println!("   - POST /api/2.0/mlflow/runs/create");
            println!("   - POST /api/2.0/mlflow/runs/log-parameter");
            println!("   - POST /api/2.0/mlflow/runs/log-metric");
            println!("   - POST /api/2.0/mlflow/runs/update");
            println!("   - POST /api/2.0/mlflow/runs/search");
        }
    }

    println!("\n✅ Demo complete!");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Metric Tests
    // ========================================================================

    #[test]
    fn test_metric_serialization() {
        let metric = Metric {
            key: "accuracy".to_string(),
            value: 0.95,
            timestamp: 1234567890,
            step: 5,
        };

        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("accuracy"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_metric_deserialization() {
        let json = r#"{"key":"loss","value":0.05,"timestamp":123456,"step":10}"#;
        let metric: Metric = serde_json::from_str(json).unwrap();
        assert_eq!(metric.key, "loss");
        assert!((metric.value - 0.05).abs() < 0.001);
        assert_eq!(metric.step, 10);
    }

    #[test]
    fn test_metric_default_step() {
        let json = r#"{"key":"acc","value":0.9,"timestamp":123}"#;
        let metric: Metric = serde_json::from_str(json).unwrap();
        assert_eq!(metric.step, 0); // default
    }

    #[test]
    fn test_metric_clone() {
        let metric = Metric {
            key: "f1".to_string(),
            value: 0.88,
            timestamp: 999,
            step: 1,
        };
        let cloned = metric.clone();
        assert_eq!(metric.key, cloned.key);
        assert_eq!(metric.value, cloned.value);
    }

    // ========================================================================
    // RunStatus Tests
    // ========================================================================

    #[test]
    fn test_run_status_serialization() {
        let status = RunStatus::Finished;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"FINISHED\"");
    }

    #[test]
    fn test_run_status_all_variants() {
        let statuses = vec![
            (RunStatus::Running, "RUNNING"),
            (RunStatus::Scheduled, "SCHEDULED"),
            (RunStatus::Finished, "FINISHED"),
            (RunStatus::Failed, "FAILED"),
            (RunStatus::Killed, "KILLED"),
        ];

        for (status, expected) in statuses {
            assert_eq!(format!("{}", status), expected);
        }
    }

    #[test]
    fn test_run_status_deserialization() {
        let json = "\"RUNNING\"";
        let status: RunStatus = serde_json::from_str(json).unwrap();
        assert!(matches!(status, RunStatus::Running));
    }

    #[test]
    fn test_run_status_debug() {
        let status = RunStatus::Failed;
        let debug = format!("{:?}", status);
        assert!(debug.contains("Failed"));
    }

    // ========================================================================
    // Param Tests
    // ========================================================================

    #[test]
    fn test_param_creation() {
        let param = Param {
            key: "learning_rate".to_string(),
            value: "0.001".to_string(),
        };
        assert_eq!(param.key, "learning_rate");
        assert_eq!(param.value, "0.001");
    }

    #[test]
    fn test_param_serialization() {
        let param = Param {
            key: "epochs".to_string(),
            value: "100".to_string(),
        };
        let json = serde_json::to_string(&param).unwrap();
        assert!(json.contains("epochs"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_param_clone() {
        let param = Param {
            key: "batch_size".to_string(),
            value: "32".to_string(),
        };
        let cloned = param.clone();
        assert_eq!(param.key, cloned.key);
    }

    // ========================================================================
    // RunTag Tests
    // ========================================================================

    #[test]
    fn test_run_tag_creation() {
        let tag = RunTag {
            key: "mlflow.user".to_string(),
            value: "alice".to_string(),
        };
        assert_eq!(tag.key, "mlflow.user");
        assert_eq!(tag.value, "alice");
    }

    #[test]
    fn test_run_tag_serialization() {
        let tag = RunTag {
            key: "version".to_string(),
            value: "1.0.0".to_string(),
        };
        let json = serde_json::to_string(&tag).unwrap();
        let deserialized: RunTag = serde_json::from_str(&json).unwrap();
        assert_eq!(tag.key, deserialized.key);
    }

    // ========================================================================
    // Experiment Tests
    // ========================================================================

    #[test]
    fn test_experiment_serialization() {
        let exp = Experiment {
            experiment_id: "123".to_string(),
            name: "my-experiment".to_string(),
            artifact_location: Some("/artifacts".to_string()),
            lifecycle_stage: "active".to_string(),
            tags: vec![],
        };
        let json = serde_json::to_string(&exp).unwrap();
        assert!(json.contains("my-experiment"));
        assert!(json.contains("123"));
    }

    #[test]
    fn test_experiment_with_tags() {
        let exp = Experiment {
            experiment_id: "456".to_string(),
            name: "tagged-exp".to_string(),
            artifact_location: None,
            lifecycle_stage: "active".to_string(),
            tags: vec![ExperimentTag {
                key: "team".to_string(),
                value: "ml".to_string(),
            }],
        };
        assert_eq!(exp.tags.len(), 1);
        assert_eq!(exp.tags[0].key, "team");
    }

    #[test]
    fn test_experiment_clone() {
        let exp = Experiment {
            experiment_id: "789".to_string(),
            name: "clone-test".to_string(),
            artifact_location: None,
            lifecycle_stage: "deleted".to_string(),
            tags: vec![],
        };
        let cloned = exp.clone();
        assert_eq!(exp.experiment_id, cloned.experiment_id);
    }

    // ========================================================================
    // RunData Tests
    // ========================================================================

    #[test]
    fn test_run_data_default() {
        let data = RunData::default();
        assert!(data.metrics.is_empty());
        assert!(data.params.is_empty());
        assert!(data.tags.is_empty());
    }

    #[test]
    fn test_run_data_with_metrics() {
        let data = RunData {
            metrics: vec![Metric {
                key: "acc".to_string(),
                value: 0.9,
                timestamp: 123,
                step: 0,
            }],
            params: vec![],
            tags: vec![],
        };
        assert_eq!(data.metrics.len(), 1);
    }

    // ========================================================================
    // RunInfo Tests
    // ========================================================================

    #[test]
    fn test_run_info_serialization() {
        let info = RunInfo {
            run_id: "run-123".to_string(),
            experiment_id: "exp-456".to_string(),
            status: RunStatus::Running,
            start_time: 1000,
            end_time: None,
            artifact_uri: Some("/artifacts/run-123".to_string()),
            lifecycle_stage: "active".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("run-123"));
        assert!(json.contains("RUNNING"));
    }

    #[test]
    fn test_run_info_with_end_time() {
        let info = RunInfo {
            run_id: "completed-run".to_string(),
            experiment_id: "exp-1".to_string(),
            status: RunStatus::Finished,
            start_time: 1000,
            end_time: Some(2000),
            artifact_uri: None,
            lifecycle_stage: "active".to_string(),
        };
        assert_eq!(info.end_time, Some(2000));
    }

    // ========================================================================
    // Run Tests
    // ========================================================================

    #[test]
    fn test_run_creation() {
        let run = Run {
            info: RunInfo {
                run_id: "test-run".to_string(),
                experiment_id: "test-exp".to_string(),
                status: RunStatus::Running,
                start_time: 0,
                end_time: None,
                artifact_uri: None,
                lifecycle_stage: "active".to_string(),
            },
            data: RunData::default(),
        };
        assert_eq!(run.info.run_id, "test-run");
    }

    // ========================================================================
    // Client Tests
    // ========================================================================

    #[test]
    fn test_client_url_normalization() {
        let client = MlflowClient::new("http://localhost:5000/");
        assert_eq!(client.base_url, "http://localhost:5000");
    }

    #[test]
    fn test_client_url_no_trailing_slash() {
        let client = MlflowClient::new("http://mlflow.example.com");
        assert_eq!(client.base_url, "http://mlflow.example.com");
    }

    #[test]
    fn test_client_endpoint() {
        let client = MlflowClient::new("http://localhost:5000");
        let endpoint = client.endpoint("experiments/create");
        assert_eq!(
            endpoint,
            "http://localhost:5000/api/2.0/mlflow/experiments/create"
        );
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_display_api() {
        let err = MlflowError::Api {
            message: "Not found".to_string(),
        };
        assert!(err.to_string().contains("Not found"));
    }

    #[test]
    fn test_error_display_serialization() {
        let json_err = serde_json::from_str::<Metric>("invalid").unwrap_err();
        let err = MlflowError::Serialization(json_err);
        assert!(err.to_string().contains("Serialization"));
    }

    #[test]
    fn test_error_debug() {
        let err = MlflowError::Api {
            message: "test".to_string(),
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("Api"));
    }
}
