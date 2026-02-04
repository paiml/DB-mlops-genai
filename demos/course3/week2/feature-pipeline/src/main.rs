//! SIMD-Accelerated Feature Engineering Pipeline
//!
//! Demonstrates feature computation using trueno's SIMD operations.
//! Compare with Databricks Feature Store to understand the performance layer.
//!
//! # Course 3, Week 2: Feature Engineering

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;
use trueno::Vector;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Computation error: {0}")]
    Computation(String),

    #[error("Trueno error: {0}")]
    Trueno(#[from] trueno::TruenoError),
}

// ============================================================================
// Feature Definitions
// ============================================================================

/// A computed feature with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub values: Vec<f32>,
    pub dtype: String,
    pub statistics: FeatureStatistics,
}

/// Statistics for a feature column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub null_count: usize,
}

// ============================================================================
// SIMD Feature Computations using trueno
// ============================================================================

/// Compute feature statistics using trueno SIMD operations
pub fn compute_statistics(values: &[f32]) -> Result<FeatureStatistics, FeatureError> {
    if values.is_empty() {
        return Ok(FeatureStatistics {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            null_count: 0,
        });
    }

    let vec = Vector::from_slice(values);
    let mean = vec.mean()?;
    let std = vec.stddev()?;
    let min = vec.min()?;
    let max = vec.max()?;
    let null_count = values.iter().filter(|v| v.is_nan()).count();

    Ok(FeatureStatistics {
        mean,
        std,
        min,
        max,
        null_count,
    })
}

/// Normalize features using z-score normalization (SIMD accelerated)
pub fn normalize_zscore(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let vec = Vector::from_slice(values);
    let normalized = vec.zscore()?;
    Ok(normalized.as_slice().to_vec())
}

/// Min-max normalization to [0, 1] range (SIMD accelerated)
pub fn normalize_minmax(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let vec = Vector::from_slice(values);
    let normalized = vec.minmax_normalize()?;
    Ok(normalized.as_slice().to_vec())
}

/// Compute rolling mean with window size
pub fn rolling_mean(values: &[f32], window: usize) -> Vec<f32> {
    if values.len() < window {
        return vec![f32::NAN; values.len()];
    }

    let mut result = vec![f32::NAN; window - 1];

    for i in (window - 1)..values.len() {
        let window_slice = &values[i + 1 - window..=i];
        let vec = Vector::from_slice(window_slice);
        result.push(vec.mean().unwrap_or(f32::NAN));
    }

    result
}

/// Compute log transform (handling zeros via shift)
pub fn log_transform(values: &[f32]) -> Vec<f32> {
    // Add small epsilon and compute log
    values.iter().map(|&v| (v + 1e-8).ln()).collect()
}

/// Bin continuous values into discrete buckets
pub fn bin_values(values: &[f32], n_bins: usize) -> Result<Vec<usize>, FeatureError> {
    let vec = Vector::from_slice(values);
    let min = vec.min()?;
    let max = vec.max()?;

    let range = max - min;
    if range == 0.0 {
        return Ok(vec![0; values.len()]);
    }

    let bin_width = range / n_bins as f32;

    Ok(values
        .iter()
        .map(|&v| {
            let bin = ((v - min) / bin_width).floor() as usize;
            bin.min(n_bins - 1)
        })
        .collect())
}

/// Apply ReLU activation (SIMD accelerated)
pub fn relu(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let vec = Vector::from_slice(values);
    let activated = vec.relu()?;
    Ok(activated.as_slice().to_vec())
}

/// Apply sigmoid activation (SIMD accelerated)
pub fn sigmoid(values: &[f32]) -> Result<Vec<f32>, FeatureError> {
    let vec = Vector::from_slice(values);
    let activated = vec.sigmoid()?;
    Ok(activated.as_slice().to_vec())
}

// ============================================================================
// Feature Pipeline
// ============================================================================

/// A feature transformation specification
#[derive(Debug, Clone)]
pub enum Transform {
    ZScoreNormalize,
    MinMaxNormalize,
    LogTransform,
    RollingMean { window: usize },
    Bin { n_bins: usize },
    ReLU,
    Sigmoid,
}

/// Feature pipeline that applies transformations
pub struct FeaturePipeline {
    transforms: Vec<(String, Transform)>,
}

impl FeaturePipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add_transform(&mut self, name: &str, transform: Transform) -> &mut Self {
        self.transforms.push((name.to_string(), transform));
        self
    }

    /// Apply all transforms to input data (parallel execution)
    pub fn transform(&self, input: &[f32]) -> Vec<Feature> {
        self.transforms
            .par_iter()
            .filter_map(|(name, transform)| {
                let values = match transform {
                    Transform::ZScoreNormalize => normalize_zscore(input).ok()?,
                    Transform::MinMaxNormalize => normalize_minmax(input).ok()?,
                    Transform::LogTransform => log_transform(input),
                    Transform::RollingMean { window } => rolling_mean(input, *window),
                    Transform::Bin { n_bins } => bin_values(input, *n_bins)
                        .ok()?
                        .iter()
                        .map(|&b| b as f32)
                        .collect(),
                    Transform::ReLU => relu(input).ok()?,
                    Transform::Sigmoid => sigmoid(input).ok()?,
                };

                let statistics = compute_statistics(&values).ok()?;

                Some(Feature {
                    name: name.clone(),
                    values,
                    dtype: "float32".to_string(),
                    statistics,
                })
            })
            .collect()
    }
}

impl Default for FeaturePipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Benchmark Utilities
// ============================================================================

/// Compare SIMD vs scalar performance
pub fn benchmark_comparison(data: &[f32]) {
    println!("\n📊 Performance Comparison: SIMD vs Scalar");
    println!("   Data size: {} elements", data.len());
    println!("   ─────────────────────────────────────");

    // SIMD normalization (trueno)
    let start = Instant::now();
    for _ in 0..100 {
        let _ = normalize_zscore(data);
    }
    let simd_time = start.elapsed();

    // Scalar normalization
    let start = Instant::now();
    for _ in 0..100 {
        let _ = scalar_normalize(data);
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

    println!("   SIMD time:   {:?} (100 iterations)", simd_time);
    println!("   Scalar time: {:?} (100 iterations)", scalar_time);
    println!("   Speedup:     {:.2}x", speedup);
}

/// Scalar implementation for comparison
fn scalar_normalize(values: &[f32]) -> Vec<f32> {
    let n = values.len() as f32;
    let mean: f32 = values.iter().sum::<f32>() / n;
    let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|x| (x - mean) / std).collect()
}

// ============================================================================
// Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Feature Engineering Pipeline - Course 3, Week 2           ║");
    println!("║     SIMD-Accelerated Feature Computation with trueno          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Show detected backend
    let backend = trueno::Backend::select_best();
    println!("\n🖥️  Detected SIMD backend: {:?}", backend);

    // Generate sample transaction data
    println!("\n📥 Generating sample data...");
    let n_samples = 100_000;
    let raw_amounts: Vec<f32> = (0..n_samples)
        .map(|i| {
            let base = (i as f32 * 0.01).sin() * 100.0 + 500.0;
            base + (i as f32 % 7.0) * 10.0
        })
        .collect();

    println!("   Generated {} transaction amounts", n_samples);

    // Build feature pipeline
    println!("\n🔧 Building feature pipeline...");
    let mut pipeline = FeaturePipeline::new();
    pipeline
        .add_transform("amount_zscore", Transform::ZScoreNormalize)
        .add_transform("amount_minmax", Transform::MinMaxNormalize)
        .add_transform("amount_log", Transform::LogTransform)
        .add_transform("amount_rolling_7", Transform::RollingMean { window: 7 })
        .add_transform("amount_bin_10", Transform::Bin { n_bins: 10 })
        .add_transform("amount_relu", Transform::ReLU)
        .add_transform("amount_sigmoid", Transform::Sigmoid);

    println!("   Added {} feature transforms", pipeline.transforms.len());

    // Execute pipeline
    println!("\n⚡ Executing feature pipeline...");
    let start = Instant::now();
    let features = pipeline.transform(&raw_amounts);
    let elapsed = start.elapsed();

    println!("   Computed {} features in {:?}", features.len(), elapsed);

    // Display feature statistics
    println!("\n📈 Feature Statistics:");
    println!("   ─────────────────────────────────────────────────────────────");
    for feature in &features {
        println!(
            "   {:20} | mean: {:8.3} | std: {:8.3} | range: [{:.3}, {:.3}]",
            feature.name,
            feature.statistics.mean,
            feature.statistics.std,
            feature.statistics.min,
            feature.statistics.max
        );
    }

    // Run benchmark
    benchmark_comparison(&raw_amounts);

    println!("\n✅ Demo complete!");
    println!("\n📖 Key Concepts:");
    println!("   - trueno auto-selects best SIMD backend (AVX-512/AVX2/NEON)");
    println!("   - Built-in zscore() and minmax_normalize() are SIMD-accelerated");
    println!("   - Parallel feature computation with rayon");
    println!("   - Compare with Databricks Feature Store for managed solution");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_statistics_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&data).unwrap();

        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
        assert_eq!(stats.null_count, 0);
    }

    #[test]
    fn test_statistics_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_statistics(&data).unwrap();

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.null_count, 0);
    }

    #[test]
    fn test_statistics_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&data).unwrap();
        assert_eq!(stats.null_count, 1);
    }

    #[test]
    fn test_statistics_single_value() {
        let data = vec![42.0];
        let stats = compute_statistics(&data).unwrap();
        assert!((stats.mean - 42.0).abs() < 0.001);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }

    // ========================================================================
    // Normalization Tests
    // ========================================================================

    #[test]
    fn test_zscore_normalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_zscore(&data).unwrap();

        let stats = compute_statistics(&normalized).unwrap();
        assert!((stats.mean).abs() < 0.01);
        assert!((stats.std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_zscore_constant_values() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        // trueno returns DivisionByZero error when std is 0
        let result = normalize_zscore(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_minmax_normalization() {
        let data = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let normalized = normalize_minmax(&data).unwrap();

        assert!((normalized[0] - 0.0).abs() < 0.001);
        assert!((normalized[4] - 1.0).abs() < 0.001);
        assert!((normalized[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_minmax_negative_values() {
        let data = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let normalized = normalize_minmax(&data).unwrap();

        assert!((normalized[0] - 0.0).abs() < 0.001);
        assert!((normalized[4] - 1.0).abs() < 0.001);
        assert!((normalized[2] - 0.5).abs() < 0.001);
    }

    // ========================================================================
    // Transform Tests
    // ========================================================================

    #[test]
    fn test_log_transform() {
        let data = vec![1.0, 10.0, 100.0];
        let transformed = log_transform(&data);

        // ln(1 + 1e-8) ≈ ln(1) = 0, ln(10) ≈ 2.3, ln(100) ≈ 4.6
        assert!(transformed[0].is_finite());
        assert!(transformed[1] > transformed[0]);
        assert!(transformed[2] > transformed[1]);
    }

    #[test]
    fn test_log_transform_zeros() {
        let data = vec![0.0, 0.0, 1.0];
        let transformed = log_transform(&data);
        // Should handle zeros gracefully with epsilon
        assert!(transformed[0].is_finite());
        assert!(transformed[1].is_finite());
    }

    #[test]
    fn test_binning() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let bins = bin_values(&data, 4).unwrap();

        assert_eq!(bins[0], 0);
        assert_eq!(bins[4], 3);
    }

    #[test]
    fn test_binning_constant_values() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let bins = bin_values(&data, 4).unwrap();
        // All should be bin 0 when range is 0
        for bin in bins {
            assert_eq!(bin, 0);
        }
    }

    #[test]
    fn test_binning_many_bins() {
        let data = vec![0.0, 0.5, 1.0];
        let bins = bin_values(&data, 10).unwrap();
        assert_eq!(bins[0], 0);
        assert_eq!(bins[2], 9);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = rolling_mean(&data, 3);

        assert!(rolling[0].is_nan());
        assert!(rolling[1].is_nan());
        assert!((rolling[2] - 2.0).abs() < 0.001);
        assert!((rolling[3] - 3.0).abs() < 0.001);
        assert!((rolling[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_mean_window_too_large() {
        let data = vec![1.0, 2.0, 3.0];
        let rolling = rolling_mean(&data, 5);
        // All should be NaN when window > data length
        for val in rolling {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_rolling_mean_window_one() {
        let data = vec![1.0, 2.0, 3.0];
        let rolling = rolling_mean(&data, 1);
        assert!((rolling[0] - 1.0).abs() < 0.001);
        assert!((rolling[1] - 2.0).abs() < 0.001);
        assert!((rolling[2] - 3.0).abs() < 0.001);
    }

    // ========================================================================
    // Activation Tests
    // ========================================================================

    #[test]
    fn test_relu() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let activated = relu(&data).unwrap();

        assert_eq!(activated[0], 0.0);
        assert_eq!(activated[1], 0.0);
        assert_eq!(activated[2], 0.0);
        assert_eq!(activated[3], 1.0);
        assert_eq!(activated[4], 2.0);
    }

    #[test]
    fn test_relu_all_positive() {
        let data = vec![1.0, 2.0, 3.0];
        let activated = relu(&data).unwrap();
        assert_eq!(activated, data);
    }

    #[test]
    fn test_relu_all_negative() {
        let data = vec![-1.0, -2.0, -3.0];
        let activated = relu(&data).unwrap();
        assert_eq!(activated, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sigmoid() {
        let data = vec![-10.0, 0.0, 10.0];
        let activated = sigmoid(&data).unwrap();

        assert!(activated[0] < 0.01); // ~0 for very negative
        assert!((activated[1] - 0.5).abs() < 0.01); // 0.5 at 0
        assert!(activated[2] > 0.99); // ~1 for very positive
    }

    #[test]
    fn test_sigmoid_bounds() {
        let data = vec![-100.0, 100.0];
        let activated = sigmoid(&data).unwrap();

        // Should be bounded between 0 and 1
        assert!(activated[0] >= 0.0 && activated[0] <= 1.0);
        assert!(activated[1] >= 0.0 && activated[1] <= 1.0);
    }

    // ========================================================================
    // Pipeline Tests
    // ========================================================================

    #[test]
    fn test_feature_pipeline_single_transform() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform("normalized", Transform::ZScoreNormalize);

        let features = pipeline.transform(&data);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].name, "normalized");
        assert_eq!(features[0].dtype, "float32");
    }

    #[test]
    fn test_feature_pipeline_multiple_transforms() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut pipeline = FeaturePipeline::new();
        pipeline
            .add_transform("zscore", Transform::ZScoreNormalize)
            .add_transform("minmax", Transform::MinMaxNormalize)
            .add_transform("log", Transform::LogTransform)
            .add_transform("relu", Transform::ReLU)
            .add_transform("sigmoid", Transform::Sigmoid);

        let features = pipeline.transform(&data);
        assert_eq!(features.len(), 5);
    }

    #[test]
    fn test_feature_pipeline_with_rolling() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform("rolling_3", Transform::RollingMean { window: 3 });

        let features = pipeline.transform(&data);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].values.len(), 10);
    }

    #[test]
    fn test_feature_pipeline_with_binning() {
        let data = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform("bins", Transform::Bin { n_bins: 5 });

        let features = pipeline.transform(&data);
        assert_eq!(features.len(), 1);
    }

    #[test]
    fn test_feature_pipeline_default() {
        let pipeline = FeaturePipeline::default();
        let data = vec![1.0, 2.0, 3.0];
        let features = pipeline.transform(&data);
        assert_eq!(features.len(), 0); // Empty pipeline
    }

    #[test]
    fn test_feature_pipeline_chaining() {
        let data = vec![1.0, 2.0, 3.0];
        let mut pipeline = FeaturePipeline::new();

        // Test method chaining returns &mut Self
        let result = pipeline
            .add_transform("a", Transform::ReLU)
            .add_transform("b", Transform::Sigmoid);

        let features = result.transform(&data);
        assert_eq!(features.len(), 2);
    }

    // ========================================================================
    // Feature Struct Tests
    // ========================================================================

    #[test]
    fn test_feature_serialization() {
        let feature = Feature {
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
            dtype: "float32".to_string(),
            statistics: FeatureStatistics {
                mean: 2.0,
                std: 1.0,
                min: 1.0,
                max: 3.0,
                null_count: 0,
            },
        };

        let json = serde_json::to_string(&feature).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("float32"));
    }

    #[test]
    fn test_feature_statistics_serialization() {
        let stats = FeatureStatistics {
            mean: 2.5,
            std: 1.5,
            min: 0.0,
            max: 5.0,
            null_count: 2,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: FeatureStatistics = serde_json::from_str(&json).unwrap();

        assert!((deserialized.mean - 2.5).abs() < 0.001);
        assert_eq!(deserialized.null_count, 2);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let err = FeatureError::InvalidInput("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = FeatureError::Computation("compute error".to_string());
        assert!(err.to_string().contains("compute error"));
    }

    // ========================================================================
    // Transform Enum Tests
    // ========================================================================

    #[test]
    fn test_transform_clone() {
        let transform = Transform::RollingMean { window: 5 };
        let cloned = transform.clone();

        match cloned {
            Transform::RollingMean { window } => assert_eq!(window, 5),
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_transform_debug() {
        let transform = Transform::Bin { n_bins: 10 };
        let debug_str = format!("{:?}", transform);
        assert!(debug_str.contains("Bin"));
        assert!(debug_str.contains("10"));
    }
}
