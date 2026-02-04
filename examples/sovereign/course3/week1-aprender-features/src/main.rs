//! Feature Engineering with aprender
//!
//! Demonstrates ML feature engineering patterns using aprender concepts.
//! This example shows StandardScaler, LabelEncoder, and train_test_split.
//!
//! # Course 3, Week 1: Experiment Tracking + Feature Engineering

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not fitted: {0}")]
    NotFitted(String),

    #[error("Transform error: {0}")]
    Transform(String),
}

// ============================================================================
// StandardScaler - Z-score normalization
// ============================================================================

/// StandardScaler normalizes features by removing the mean and scaling to unit variance.
/// Formula: z = (x - mean) / std
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    mean: Option<Vec<f64>>,
    std: Option<Vec<f64>>,
    n_features: usize,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            n_features: 0,
        }
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<(), FeatureError> {
        if data.is_empty() {
            return Err(FeatureError::InvalidInput("Empty data".to_string()));
        }

        self.n_features = data[0].len();
        let n_samples = data.len() as f64;

        // Calculate mean for each feature
        let mut mean = vec![0.0; self.n_features];
        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                mean[i] += val / n_samples;
            }
        }

        // Calculate std for each feature
        let mut std = vec![0.0; self.n_features];
        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                std[i] += (val - mean[i]).powi(2) / n_samples;
            }
        }
        for s in &mut std {
            *s = s.sqrt().max(1e-10); // Prevent division by zero
        }

        self.mean = Some(mean);
        self.std = Some(std);
        Ok(())
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FeatureError> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;
        let std = self
            .std
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;

        let result: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (val - mean[i]) / std[i])
                    .collect()
            })
            .collect();

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FeatureError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform to original scale
    pub fn inverse_transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FeatureError> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;
        let std = self
            .std
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;

        let result: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| val * std[i] + mean[i])
                    .collect()
            })
            .collect();

        Ok(result)
    }

    pub fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MinMaxScaler - Scale to [0, 1] range
// ============================================================================

/// MinMaxScaler scales features to a given range (default [0, 1]).
/// Formula: x_scaled = (x - min) / (max - min)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    min: Option<Vec<f64>>,
    max: Option<Vec<f64>>,
    n_features: usize,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            n_features: 0,
        }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<(), FeatureError> {
        if data.is_empty() {
            return Err(FeatureError::InvalidInput("Empty data".to_string()));
        }

        self.n_features = data[0].len();
        let mut min = vec![f64::MAX; self.n_features];
        let mut max = vec![f64::MIN; self.n_features];

        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                min[i] = min[i].min(val);
                max[i] = max[i].max(val);
            }
        }

        self.min = Some(min);
        self.max = Some(max);
        Ok(())
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FeatureError> {
        let min = self
            .min
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;
        let max = self
            .max
            .as_ref()
            .ok_or_else(|| FeatureError::NotFitted("Scaler not fitted".to_string()))?;

        let result: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let range = max[i] - min[i];
                        if range < 1e-10 {
                            0.0
                        } else {
                            (val - min[i]) / range
                        }
                    })
                    .collect()
            })
            .collect();

        Ok(result)
    }

    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FeatureError> {
        self.fit(data)?;
        self.transform(data)
    }

    pub fn is_fitted(&self) -> bool {
        self.min.is_some() && self.max.is_some()
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LabelEncoder - Encode categorical labels
// ============================================================================

/// LabelEncoder encodes categorical labels as integers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEncoder {
    classes: Vec<String>,
    class_to_idx: HashMap<String, usize>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            class_to_idx: HashMap::new(),
        }
    }

    pub fn fit(&mut self, labels: &[&str]) -> Result<(), FeatureError> {
        let mut classes: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        classes.sort();
        classes.dedup();

        self.class_to_idx = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        self.classes = classes;

        Ok(())
    }

    pub fn transform(&self, labels: &[&str]) -> Result<Vec<usize>, FeatureError> {
        labels
            .iter()
            .map(|label| {
                self.class_to_idx
                    .get(*label)
                    .copied()
                    .ok_or_else(|| FeatureError::Transform(format!("Unknown label: {}", label)))
            })
            .collect()
    }

    pub fn fit_transform(&mut self, labels: &[&str]) -> Result<Vec<usize>, FeatureError> {
        self.fit(labels)?;
        self.transform(labels)
    }

    pub fn inverse_transform(&self, indices: &[usize]) -> Result<Vec<String>, FeatureError> {
        indices
            .iter()
            .map(|&idx| {
                self.classes
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| FeatureError::Transform(format!("Unknown index: {}", idx)))
            })
            .collect()
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Train/Test Split
// ============================================================================

/// Split data into training and test sets
pub fn train_test_split<T: Clone>(data: &[T], test_size: f64, seed: u64) -> (Vec<T>, Vec<T>) {
    let n = data.len();
    let test_count = (n as f64 * test_size).round() as usize;

    // Simple deterministic shuffle based on seed
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = seed;
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

    let (test_indices, train_indices) = indices.split_at(test_count);

    let train: Vec<T> = train_indices.iter().map(|&i| data[i].clone()).collect();
    let test: Vec<T> = test_indices.iter().map(|&i| data[i].clone()).collect();

    (train, test)
}

// ============================================================================
// Feature Statistics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub missing: usize,
}

impl FeatureStats {
    pub fn compute(name: &str, values: &[f64]) -> Self {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::MAX, f64::min);
        let max = values.iter().cloned().fold(f64::MIN, f64::max);
        let missing = values.iter().filter(|v| v.is_nan()).count();

        Self {
            name: name.to_string(),
            mean,
            std,
            min,
            max,
            missing,
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Feature Engineering with aprender - Course 3, Week 1      ║");
    println!("║     StandardScaler, LabelEncoder, train_test_split            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Step 1: Create sample data
    println!("\n📊 Step 1: Sample Data");
    let data = vec![
        vec![100.0, 1.5, 200.0],
        vec![150.0, 2.0, 300.0],
        vec![200.0, 2.5, 400.0],
        vec![250.0, 3.0, 500.0],
        vec![300.0, 3.5, 600.0],
        vec![350.0, 4.0, 700.0],
        vec![400.0, 4.5, 800.0],
        vec![450.0, 5.0, 900.0],
    ];
    let labels = vec![
        "fraud", "normal", "normal", "fraud", "normal", "normal", "fraud", "normal",
    ];

    println!("   Samples: {}", data.len());
    println!("   Features: {}", data[0].len());
    println!("   Sample row: {:?}", data[0]);

    // Step 2: StandardScaler
    println!("\n🔧 Step 2: StandardScaler (Z-score normalization)");
    let mut scaler = StandardScaler::new();
    let scaled = scaler.fit_transform(&data).unwrap();

    println!("   Original: {:?}", data[0]);
    println!(
        "   Scaled:   [{:.4}, {:.4}, {:.4}]",
        scaled[0][0], scaled[0][1], scaled[0][2]
    );

    // Verify: mean should be ~0, std should be ~1
    let col0: Vec<f64> = scaled.iter().map(|r| r[0]).collect();
    let stats = FeatureStats::compute("scaled_feature_0", &col0);
    println!(
        "   Verification: mean={:.6}, std={:.4}",
        stats.mean, stats.std
    );

    // Step 3: MinMaxScaler
    println!("\n🔧 Step 3: MinMaxScaler (Scale to [0, 1])");
    let mut minmax = MinMaxScaler::new();
    let normalized = minmax.fit_transform(&data).unwrap();

    println!("   Original: {:?}", data[0]);
    println!(
        "   Normalized: [{:.4}, {:.4}, {:.4}]",
        normalized[0][0], normalized[0][1], normalized[0][2]
    );
    println!(
        "   Min sample: [{:.4}, {:.4}, {:.4}]",
        normalized[0][0], normalized[0][1], normalized[0][2]
    );
    println!(
        "   Max sample: [{:.4}, {:.4}, {:.4}]",
        normalized[7][0], normalized[7][1], normalized[7][2]
    );

    // Step 4: LabelEncoder
    println!("\n🏷️  Step 4: LabelEncoder");
    let mut encoder = LabelEncoder::new();
    let encoded = encoder.fit_transform(&labels).unwrap();

    println!("   Original: {:?}", labels);
    println!("   Encoded:  {:?}", encoded);
    println!("   Classes:  {:?}", encoder.classes());
    println!(
        "   Decoded:  {:?}",
        encoder.inverse_transform(&encoded).unwrap()
    );

    // Step 5: Train/Test Split
    println!("\n✂️  Step 5: Train/Test Split");
    let (train, test) = train_test_split(&data, 0.25, 42);

    println!("   Total samples: {}", data.len());
    println!("   Train samples: {}", train.len());
    println!("   Test samples:  {}", test.len());
    println!(
        "   Split ratio:   {:.0}% / {:.0}%",
        train.len() as f64 / data.len() as f64 * 100.0,
        test.len() as f64 / data.len() as f64 * 100.0
    );

    // Step 6: Feature Statistics
    println!("\n📈 Step 6: Feature Statistics");
    let feature_names = ["amount", "velocity", "distance"];
    for (i, name) in feature_names.iter().enumerate() {
        let values: Vec<f64> = data.iter().map(|r| r[i]).collect();
        let stats = FeatureStats::compute(name, &values);
        println!(
            "   {:10} | mean: {:8.2} | std: {:7.2} | range: [{:.1}, {:.1}]",
            stats.name, stats.mean, stats.std, stats.min, stats.max
        );
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • StandardScaler: Z-score normalization (mean=0, std=1)");
    println!("  • MinMaxScaler: Scale features to [0, 1] range");
    println!("  • LabelEncoder: Convert categorical labels to integers");
    println!("  • train_test_split: Randomly split data for evaluation");
    println!();
    println!("Sovereign AI Stack: aprender preprocessing module");
    println!("Databricks equivalent: Feature Store, MLflow preprocessing");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // StandardScaler Tests
    // ========================================================================

    #[test]
    fn test_standard_scaler_fit_transform() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Mean should be approximately 0
        let col0: Vec<f64> = scaled.iter().map(|r| r[0]).collect();
        let mean: f64 = col0.iter().sum::<f64>() / col0.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let result = scaler.transform(&[vec![1.0, 2.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_scaler_empty_data() {
        let mut scaler = StandardScaler::new();
        let result = scaler.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_scaler_is_fitted() {
        let mut scaler = StandardScaler::new();
        assert!(!scaler.is_fitted());
        scaler.fit(&[vec![1.0], vec![2.0]]).unwrap();
        assert!(scaler.is_fitted());
    }

    #[test]
    fn test_standard_scaler_inverse_transform() {
        let data = vec![vec![10.0, 20.0], vec![20.0, 40.0]];
        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        let reconstructed = scaler.inverse_transform(&scaled).unwrap();

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            for (o, r) in orig.iter().zip(recon.iter()) {
                assert!((o - r).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_standard_scaler_default() {
        let scaler = StandardScaler::default();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_standard_scaler_clone() {
        let mut scaler = StandardScaler::new();
        scaler.fit(&[vec![1.0], vec![2.0]]).unwrap();
        let cloned = scaler.clone();
        assert!(cloned.is_fitted());
    }

    // ========================================================================
    // MinMaxScaler Tests
    // ========================================================================

    #[test]
    fn test_minmax_scaler_fit_transform() {
        let data = vec![vec![0.0, 10.0], vec![50.0, 20.0], vec![100.0, 30.0]];
        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        assert!((scaled[0][0] - 0.0).abs() < 1e-10);
        assert!((scaled[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler_not_fitted() {
        let scaler = MinMaxScaler::new();
        let result = scaler.transform(&[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_minmax_scaler_constant_feature() {
        let data = vec![vec![5.0], vec![5.0], vec![5.0]];
        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        // Constant feature should map to 0
        assert_eq!(scaled[0][0], 0.0);
    }

    #[test]
    fn test_minmax_scaler_is_fitted() {
        let mut scaler = MinMaxScaler::new();
        assert!(!scaler.is_fitted());
        scaler.fit(&[vec![1.0]]).unwrap();
        assert!(scaler.is_fitted());
    }

    #[test]
    fn test_minmax_scaler_default() {
        let scaler = MinMaxScaler::default();
        assert!(!scaler.is_fitted());
    }

    // ========================================================================
    // LabelEncoder Tests
    // ========================================================================

    #[test]
    fn test_label_encoder_fit_transform() {
        let labels = vec!["cat", "dog", "cat", "bird"];
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(&labels).unwrap();

        assert_eq!(encoder.n_classes(), 3);
        assert_eq!(encoder.classes(), &["bird", "cat", "dog"]);
        assert_eq!(encoded[0], encoded[2]); // Both "cat"
    }

    #[test]
    fn test_label_encoder_inverse_transform() {
        let labels = vec!["a", "b", "c"];
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(&labels).unwrap();
        let decoded = encoder.inverse_transform(&encoded).unwrap();

        let labels_owned: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        assert_eq!(decoded, labels_owned);
    }

    #[test]
    fn test_label_encoder_unknown_label() {
        let mut encoder = LabelEncoder::new();
        encoder.fit(&["a", "b"]).unwrap();
        let result = encoder.transform(&["c"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_encoder_unknown_index() {
        let mut encoder = LabelEncoder::new();
        encoder.fit(&["a", "b"]).unwrap();
        let result = encoder.inverse_transform(&[999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_encoder_default() {
        let encoder = LabelEncoder::default();
        assert_eq!(encoder.n_classes(), 0);
    }

    #[test]
    fn test_label_encoder_clone() {
        let mut encoder = LabelEncoder::new();
        encoder.fit(&["x", "y"]).unwrap();
        let cloned = encoder.clone();
        assert_eq!(cloned.n_classes(), 2);
    }

    // ========================================================================
    // Train/Test Split Tests
    // ========================================================================

    #[test]
    fn test_train_test_split_sizes() {
        let data: Vec<i32> = (0..100).collect();
        let (train, test) = train_test_split(&data, 0.2, 42);

        assert_eq!(train.len() + test.len(), 100);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_train_test_split_deterministic() {
        let data: Vec<i32> = (0..50).collect();
        let (train1, test1) = train_test_split(&data, 0.3, 123);
        let (train2, test2) = train_test_split(&data, 0.3, 123);

        assert_eq!(train1, train2);
        assert_eq!(test1, test2);
    }

    #[test]
    fn test_train_test_split_different_seeds() {
        let data: Vec<i32> = (0..50).collect();
        let (train1, _) = train_test_split(&data, 0.3, 1);
        let (train2, _) = train_test_split(&data, 0.3, 2);

        assert_ne!(train1, train2);
    }

    // ========================================================================
    // FeatureStats Tests
    // ========================================================================

    #[test]
    fn test_feature_stats_compute() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::compute("test", &values);

        assert_eq!(stats.name, "test");
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.missing, 0);
    }

    #[test]
    fn test_feature_stats_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0];
        let stats = FeatureStats::compute("test", &values);
        assert_eq!(stats.missing, 1);
    }

    #[test]
    fn test_feature_stats_clone() {
        let stats = FeatureStats::compute("test", &[1.0, 2.0]);
        let cloned = stats.clone();
        assert_eq!(stats.name, cloned.name);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_display_invalid_input() {
        let err = FeatureError::InvalidInput("bad data".to_string());
        assert!(err.to_string().contains("bad data"));
    }

    #[test]
    fn test_error_display_not_fitted() {
        let err = FeatureError::NotFitted("scaler".to_string());
        assert!(err.to_string().contains("scaler"));
    }

    #[test]
    fn test_error_display_transform() {
        let err = FeatureError::Transform("failed".to_string());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_error_debug() {
        let err = FeatureError::InvalidInput("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidInput"));
    }
}
