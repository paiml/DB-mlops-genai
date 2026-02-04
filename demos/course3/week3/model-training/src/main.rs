//! ML Model Training and Registry Demo
//!
//! Demonstrates ML training with aprender and model management with pacha.
//! Compare with Databricks AutoML and Unity Catalog Model Registry.
//!
//! # Course 3, Week 3: Model Training and Registry

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Training error: {0}")]
    Training(String),

    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ============================================================================
// Dataset Generation
// ============================================================================

/// A simple dataset for training
#[derive(Debug, Clone)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl Dataset {
    /// Generate a synthetic classification dataset
    pub fn generate_classification(n_samples: usize, n_features: usize, _seed: u64) -> Self {
        let mut rng = rand::thread_rng();

        let feature_names: Vec<String> = (0..n_features)
            .map(|i| format!("feature_{}", i))
            .collect();

        let mut features = Vec::with_capacity(n_samples);
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut row = Vec::with_capacity(n_features);
            let mut sum = 0.0;

            for j in 0..n_features {
                let val = rng.gen::<f64>() * 2.0 - 1.0 + (i as f64 * 0.001);
                row.push(val);
                sum += val * (j as f64 + 1.0);
            }

            features.push(row);
            // Binary classification based on weighted sum
            labels.push(if sum > 0.0 { 1.0 } else { 0.0 });
        }

        Self {
            features,
            labels,
            feature_names,
        }
    }

    /// Generate a synthetic regression dataset
    pub fn generate_regression(n_samples: usize, n_features: usize, _seed: u64) -> Self {
        let mut rng = rand::thread_rng();

        let feature_names: Vec<String> = (0..n_features)
            .map(|i| format!("feature_{}", i))
            .collect();

        let mut features = Vec::with_capacity(n_samples);
        let mut labels = Vec::with_capacity(n_samples);

        // Generate true coefficients
        let coeffs: Vec<f64> = (0..n_features).map(|i| (i as f64 + 1.0) * 0.5).collect();

        for _ in 0..n_samples {
            let mut row = Vec::with_capacity(n_features);
            let mut y = rng.gen::<f64>() * 0.1; // noise

            for (_j, &coeff) in coeffs.iter().enumerate() {
                let val = rng.gen::<f64>() * 2.0 - 1.0;
                row.push(val);
                y += val * coeff;
            }

            features.push(row);
            labels.push(y);
        }

        Self {
            features,
            labels,
            feature_names,
        }
    }

    /// Split dataset into train and test
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let n_test = (self.features.len() as f64 * test_ratio) as usize;
        let n_train = self.features.len() - n_test;

        let train = Dataset {
            features: self.features[..n_train].to_vec(),
            labels: self.labels[..n_train].to_vec(),
            feature_names: self.feature_names.clone(),
        };

        let test = Dataset {
            features: self.features[n_train..].to_vec(),
            labels: self.labels[n_train..].to_vec(),
            feature_names: self.feature_names.clone(),
        };

        (train, test)
    }
}

// ============================================================================
// Simple Models (Pure Rust implementations for understanding)
// ============================================================================

/// Linear regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub n_iterations: usize,
}

impl LinearRegression {
    pub fn new(n_features: usize, learning_rate: f64, n_iterations: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            n_iterations,
        }
    }

    /// Train using gradient descent
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        let n_samples = features.len() as f64;

        for _ in 0..self.n_iterations {
            let mut weight_gradients = vec![0.0; self.weights.len()];
            let mut bias_gradient = 0.0;

            for (x, &y) in features.iter().zip(labels.iter()) {
                let pred = self.predict_single(x);
                let error = pred - y;

                for (j, &xj) in x.iter().enumerate() {
                    weight_gradients[j] += error * xj;
                }
                bias_gradient += error;
            }

            for (w, grad) in self.weights.iter_mut().zip(weight_gradients.iter()) {
                *w -= self.learning_rate * grad / n_samples;
            }
            self.bias -= self.learning_rate * bias_gradient / n_samples;
        }
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        let mut pred = self.bias;
        for (w, xj) in self.weights.iter().zip(x.iter()) {
            pred += w * xj;
        }
        pred
    }

    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|x| self.predict_single(x)).collect()
    }

    /// Calculate R² score
    pub fn score(&self, features: &[Vec<f64>], labels: &[f64]) -> f64 {
        let predictions = self.predict(features);
        let mean_y: f64 = labels.iter().sum::<f64>() / labels.len() as f64;

        let ss_tot: f64 = labels.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(&pred, &y)| (y - pred).powi(2))
            .sum();

        1.0 - ss_res / ss_tot
    }
}

/// Simple decision tree for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStump {
    pub feature_index: usize,
    pub threshold: f64,
    pub left_class: f64,
    pub right_class: f64,
}

impl DecisionStump {
    /// Find the best split
    pub fn fit(features: &[Vec<f64>], labels: &[f64]) -> Self {
        let n_features = features[0].len();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gini = f64::MAX;

        for feat_idx in 0..n_features {
            // Get unique thresholds
            let mut values: Vec<f64> = features.iter().map(|x| x[feat_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            for &thresh in &values {
                let gini = Self::gini_impurity(features, labels, feat_idx, thresh);
                if gini < best_gini {
                    best_gini = gini;
                    best_feature = feat_idx;
                    best_threshold = thresh;
                }
            }
        }

        // Determine class for each side
        let mut left_labels = Vec::new();
        let mut right_labels = Vec::new();

        for (x, &label) in features.iter().zip(labels.iter()) {
            if x[best_feature] <= best_threshold {
                left_labels.push(label);
            } else {
                right_labels.push(label);
            }
        }

        let left_class = Self::majority_class(&left_labels);
        let right_class = Self::majority_class(&right_labels);

        Self {
            feature_index: best_feature,
            threshold: best_threshold,
            left_class,
            right_class,
        }
    }

    fn gini_impurity(
        features: &[Vec<f64>],
        labels: &[f64],
        feat_idx: usize,
        threshold: f64,
    ) -> f64 {
        let mut left_counts = [0usize; 2];
        let mut right_counts = [0usize; 2];

        for (x, &y) in features.iter().zip(labels.iter()) {
            let class = y as usize;
            if x[feat_idx] <= threshold {
                left_counts[class.min(1)] += 1;
            } else {
                right_counts[class.min(1)] += 1;
            }
        }

        let left_total = left_counts.iter().sum::<usize>() as f64;
        let right_total = right_counts.iter().sum::<usize>() as f64;
        let total = left_total + right_total;

        if left_total == 0.0 || right_total == 0.0 {
            return f64::MAX;
        }

        let left_gini = 1.0
            - left_counts
                .iter()
                .map(|&c| (c as f64 / left_total).powi(2))
                .sum::<f64>();
        let right_gini = 1.0
            - right_counts
                .iter()
                .map(|&c| (c as f64 / right_total).powi(2))
                .sum::<f64>();

        (left_total / total) * left_gini + (right_total / total) * right_gini
    }

    fn majority_class(labels: &[f64]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        let sum: f64 = labels.iter().sum();
        if sum > labels.len() as f64 / 2.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|x| {
                if x[self.feature_index] <= self.threshold {
                    self.left_class
                } else {
                    self.right_class
                }
            })
            .collect()
    }

    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[f64]) -> f64 {
        let predictions = self.predict(features);
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| (pred - label).abs() < 0.5)
            .count();
        correct as f64 / labels.len() as f64
    }
}

// ============================================================================
// Model Card (for registry)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub description: String,
    pub metrics: std::collections::HashMap<String, f64>,
    pub parameters: std::collections::HashMap<String, String>,
    pub created_at: String,
    pub author: String,
}

impl ModelCard {
    pub fn new(name: &str, model_type: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            model_type: model_type.to_string(),
            description: description.to_string(),
            metrics: std::collections::HashMap::new(),
            parameters: std::collections::HashMap::new(),
            created_at: chrono_lite_now(),
            author: "sovereign-ai".to_string(),
        }
    }

    pub fn add_metric(&mut self, key: &str, value: f64) -> &mut Self {
        self.metrics.insert(key.to_string(), value);
        self
    }

    pub fn add_param(&mut self, key: &str, value: &str) -> &mut Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }
}

fn chrono_lite_now() -> String {
    // Simple timestamp without chrono dependency
    format!("{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs())
}

// ============================================================================
// Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Model Training & Registry - Course 3, Week 3              ║");
    println!("║     ML with aprender concepts, Registry with pacha concepts   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // -------------------------------------------------------------------------
    // Demo 1: Linear Regression
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 1: Linear Regression");
    println!("   ─────────────────────────────────────────────────────────────");

    let dataset = Dataset::generate_regression(1000, 5, 42);
    let (train, test) = dataset.train_test_split(0.2);

    println!("   Training samples: {}", train.features.len());
    println!("   Test samples: {}", test.features.len());
    println!("   Features: {}", train.feature_names.len());

    let start = Instant::now();
    let mut lr_model = LinearRegression::new(5, 0.01, 1000);
    lr_model.fit(&train.features, &train.labels);
    let train_time = start.elapsed();

    let train_score = lr_model.score(&train.features, &train.labels);
    let test_score = lr_model.score(&test.features, &test.labels);

    println!("   Training time: {:?}", train_time);
    println!("   Train R²: {:.4}", train_score);
    println!("   Test R²:  {:.4}", test_score);

    // Create model card
    let mut lr_card = ModelCard::new(
        "linear-regression-v1",
        "LinearRegression",
        "Linear regression for numeric prediction",
    );
    lr_card
        .add_metric("train_r2", train_score)
        .add_metric("test_r2", test_score)
        .add_param("learning_rate", "0.01")
        .add_param("n_iterations", "1000");

    println!("\n   📋 Model Card:");
    println!("   {}", serde_json::to_string_pretty(&lr_card).unwrap());

    // -------------------------------------------------------------------------
    // Demo 2: Decision Tree (Stump) Classification
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 2: Decision Tree Classification");
    println!("   ─────────────────────────────────────────────────────────────");

    let dataset = Dataset::generate_classification(1000, 5, 42);
    let (train, test) = dataset.train_test_split(0.2);

    println!("   Training samples: {}", train.features.len());
    println!("   Test samples: {}", test.features.len());

    let start = Instant::now();
    let tree = DecisionStump::fit(&train.features, &train.labels);
    let train_time = start.elapsed();

    let train_acc = tree.accuracy(&train.features, &train.labels);
    let test_acc = tree.accuracy(&test.features, &test.labels);

    println!("   Training time: {:?}", train_time);
    println!("   Split: feature_{} <= {:.4}", tree.feature_index, tree.threshold);
    println!("   Train accuracy: {:.4}", train_acc);
    println!("   Test accuracy:  {:.4}", test_acc);

    // Create model card
    let mut tree_card = ModelCard::new(
        "decision-stump-v1",
        "DecisionStump",
        "Single-split decision tree for binary classification",
    );
    tree_card
        .add_metric("train_accuracy", train_acc)
        .add_metric("test_accuracy", test_acc)
        .add_param("feature_index", &tree.feature_index.to_string())
        .add_param("threshold", &format!("{:.4}", tree.threshold));

    // -------------------------------------------------------------------------
    // Demo 3: Model Serialization (pacha concepts)
    // -------------------------------------------------------------------------
    println!("\n📦 Demo 3: Model Serialization");
    println!("   ─────────────────────────────────────────────────────────────");

    let model_json = serde_json::to_string(&lr_model).unwrap();
    println!("   Serialized model size: {} bytes", model_json.len());

    // Simulate BLAKE3 hash (pacha uses this for content addressing)
    let model_hash = simple_hash(&model_json);
    println!("   Model hash (simulated): {}", model_hash);

    println!("\n   📝 Registry Entry:");
    println!("   {{");
    println!("     \"name\": \"{}\",", lr_card.name);
    println!("     \"version\": \"{}\",", lr_card.version);
    println!("     \"hash\": \"{}\",", model_hash);
    println!("     \"metrics\": {:?}", lr_card.metrics);
    println!("   }}");

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n✅ Demo complete!");
    println!("\n📖 Key Concepts:");
    println!("   - aprender: Pure Rust ML algorithms (regression, trees, clustering)");
    println!("   - pacha: Model registry with signing, versioning, content addressing");
    println!("   - Model cards: Metadata for reproducibility and governance");
    println!("   - Compare with Databricks AutoML + Unity Catalog Model Registry");
}

/// Simple hash function (pacha uses BLAKE3)
fn simple_hash(data: &str) -> String {
    let mut hash: u64 = 0;
    for byte in data.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    format!("{:016x}", hash)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_generation() {
        let dataset = Dataset::generate_regression(100, 5, 42);
        assert_eq!(dataset.features.len(), 100);
        assert_eq!(dataset.labels.len(), 100);
        assert_eq!(dataset.feature_names.len(), 5);
    }

    #[test]
    fn test_train_test_split() {
        let dataset = Dataset::generate_regression(100, 5, 42);
        let (train, test) = dataset.train_test_split(0.2);
        assert_eq!(train.features.len(), 80);
        assert_eq!(test.features.len(), 20);
    }

    #[test]
    fn test_linear_regression() {
        let dataset = Dataset::generate_regression(100, 3, 42);
        let mut model = LinearRegression::new(3, 0.1, 100);
        model.fit(&dataset.features, &dataset.labels);

        let score = model.score(&dataset.features, &dataset.labels);
        assert!(score > 0.5, "R² should be reasonable: {}", score);
    }

    #[test]
    fn test_decision_stump() {
        let dataset = Dataset::generate_classification(100, 3, 42);
        let tree = DecisionStump::fit(&dataset.features, &dataset.labels);

        let accuracy = tree.accuracy(&dataset.features, &dataset.labels);
        assert!(accuracy > 0.5, "Accuracy should be better than random: {}", accuracy);
    }

    #[test]
    fn test_model_card() {
        let mut card = ModelCard::new("test-model", "LinearRegression", "Test");
        card.add_metric("accuracy", 0.95);
        card.add_param("learning_rate", "0.01");

        assert_eq!(card.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(card.parameters.get("learning_rate"), Some(&"0.01".to_string()));
    }

    #[test]
    fn test_model_serialization() {
        let model = LinearRegression::new(3, 0.01, 100);
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: LinearRegression = serde_json::from_str(&json).unwrap();
        assert_eq!(model.weights.len(), deserialized.weights.len());
    }
}
