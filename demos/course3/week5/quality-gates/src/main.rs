//! Production Quality Gates and Orchestration Demo
//!
//! Demonstrates quality gates (pmat concepts) and pipeline orchestration (batuta concepts).
//! Compare with Databricks Workflows and Lakehouse Monitoring.
//!
//! # Course 3, Week 5: Production Quality and Orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum QualityError {
    #[error("Quality gate failed: {0}")]
    GateFailed(String),

    #[error("Threshold exceeded: {metric} = {value}, threshold = {threshold}")]
    ThresholdExceeded {
        metric: String,
        value: f64,
        threshold: f64,
    },

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

// ============================================================================
// Quality Metrics (pmat concepts)
// ============================================================================

/// TDG (Toyota Development Grade) Score components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdgScore {
    pub complexity: ComplexityMetrics,
    pub coverage: CoverageMetrics,
    pub documentation: DocumentationMetrics,
    pub security: SecurityMetrics,
    pub overall_grade: Grade,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Grade {
    A,
    B,
    C,
    D,
    F,
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 90.0 => Grade::A,
            s if s >= 80.0 => Grade::B,
            s if s >= 70.0 => Grade::C,
            s if s >= 60.0 => Grade::D,
            _ => Grade::F,
        }
    }

    pub fn passes_threshold(&self, threshold: Grade) -> bool {
        *self <= threshold
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_avg: f64,
    pub cyclomatic_max: u32,
    pub cognitive_avg: f64,
    pub cognitive_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    pub line_coverage: f64,
    pub branch_coverage: f64,
    pub mutation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationMetrics {
    pub doc_coverage: f64,
    pub readme_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub vulnerabilities: u32,
    pub advisories: u32,
}

impl TdgScore {
    /// Calculate TDG score from component metrics
    pub fn calculate(
        complexity: ComplexityMetrics,
        coverage: CoverageMetrics,
        documentation: DocumentationMetrics,
        security: SecurityMetrics,
    ) -> Self {
        // Weighted scoring (Toyota Way principles)
        let complexity_score = Self::score_complexity(&complexity);
        let coverage_score = Self::score_coverage(&coverage);
        let doc_score = Self::score_documentation(&documentation);
        let security_score = Self::score_security(&security);

        // Weighted average (security is critical)
        let overall = complexity_score * 0.25
            + coverage_score * 0.35
            + doc_score * 0.15
            + security_score * 0.25;

        Self {
            complexity,
            coverage,
            documentation,
            security,
            overall_grade: Grade::from_score(overall),
        }
    }

    fn score_complexity(m: &ComplexityMetrics) -> f64 {
        let cyclo_score = (30.0 - m.cyclomatic_avg.min(30.0)) / 30.0 * 100.0;
        let cognitive_score = (25.0 - m.cognitive_avg.min(25.0)) / 25.0 * 100.0;
        (cyclo_score + cognitive_score) / 2.0
    }

    fn score_coverage(m: &CoverageMetrics) -> f64 {
        (m.line_coverage + m.branch_coverage + m.mutation_score) / 3.0
    }

    fn score_documentation(m: &DocumentationMetrics) -> f64 {
        (m.doc_coverage + m.readme_score) / 2.0
    }

    fn score_security(m: &SecurityMetrics) -> f64 {
        if m.vulnerabilities > 0 {
            0.0
        } else if m.advisories > 0 {
            50.0
        } else {
            100.0
        }
    }
}

// ============================================================================
// Quality Gates
// ============================================================================

/// A quality gate check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub name: String,
    pub checks: Vec<GateCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheck {
    pub name: String,
    pub threshold: f64,
    pub comparator: Comparator,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Comparator {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
}

impl Comparator {
    pub fn check(&self, value: f64, threshold: f64) -> bool {
        match self {
            Comparator::GreaterThan => value > threshold,
            Comparator::LessThan => value < threshold,
            Comparator::GreaterOrEqual => value >= threshold,
            Comparator::LessOrEqual => value <= threshold,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub check_results: Vec<CheckResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub value: f64,
    pub threshold: f64,
    pub message: String,
}

impl QualityGate {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            checks: Vec::new(),
        }
    }

    pub fn add_check(&mut self, name: &str, threshold: f64, comparator: Comparator) -> &mut Self {
        self.checks.push(GateCheck {
            name: name.to_string(),
            threshold,
            comparator,
        });
        self
    }

    pub fn evaluate(&self, metrics: &HashMap<String, f64>) -> GateResult {
        let check_results: Vec<CheckResult> = self
            .checks
            .iter()
            .map(|check| {
                let value = metrics.get(&check.name).copied().unwrap_or(0.0);
                let passed = check.comparator.check(value, check.threshold);
                CheckResult {
                    name: check.name.clone(),
                    passed,
                    value,
                    threshold: check.threshold,
                    message: if passed {
                        format!("{} = {:.2} (OK)", check.name, value)
                    } else {
                        format!(
                            "{} = {:.2} (FAILED, threshold: {:.2})",
                            check.name, value, check.threshold
                        )
                    },
                }
            })
            .collect();

        let all_passed = check_results.iter().all(|r| r.passed);

        GateResult {
            name: self.name.clone(),
            passed: all_passed,
            check_results,
        }
    }
}

// ============================================================================
// Pipeline Orchestration (batuta concepts)
// ============================================================================

/// A pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub stage_type: StageType,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageType {
    DataIngestion,
    FeatureEngineering,
    Training,
    Validation,
    Deployment,
    Monitoring,
}

/// ML Pipeline definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub name: String,
    pub stages: Vec<PipelineStage>,
    pub quality_gates: Vec<QualityGate>,
}

impl Pipeline {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stages: Vec::new(),
            quality_gates: Vec::new(),
        }
    }

    pub fn add_stage(&mut self, name: &str, stage_type: StageType, deps: Vec<&str>) -> &mut Self {
        self.stages.push(PipelineStage {
            name: name.to_string(),
            stage_type,
            dependencies: deps.into_iter().map(String::from).collect(),
        });
        self
    }

    pub fn add_quality_gate(&mut self, gate: QualityGate) -> &mut Self {
        self.quality_gates.push(gate);
        self
    }

    /// Get execution order (topological sort)
    pub fn execution_order(&self) -> Vec<String> {
        // Simple topological sort
        let mut order = Vec::new();
        let mut remaining: Vec<_> = self.stages.iter().collect();

        while !remaining.is_empty() {
            let next = remaining
                .iter()
                .position(|s| s.dependencies.iter().all(|d| order.contains(d)))
                .expect("Circular dependency detected");

            order.push(remaining.remove(next).name.clone());
        }

        order
    }
}

// ============================================================================
// Drift Detection
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    pub feature_name: String,
    pub baseline_mean: f64,
    pub current_mean: f64,
    pub psi: f64, // Population Stability Index
    pub drift_detected: bool,
}

impl DriftMetrics {
    pub fn calculate(feature: &str, baseline: &[f64], current: &[f64]) -> Self {
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let current_mean = current.iter().sum::<f64>() / current.len() as f64;

        // Simplified PSI calculation
        let psi = Self::calculate_psi(baseline, current);
        let drift_detected = psi > 0.1; // PSI > 0.1 indicates drift

        Self {
            feature_name: feature.to_string(),
            baseline_mean,
            current_mean,
            psi,
            drift_detected,
        }
    }

    fn calculate_psi(baseline: &[f64], current: &[f64]) -> f64 {
        // Simplified PSI using mean comparison
        let base_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let curr_mean = current.iter().sum::<f64>() / current.len() as f64;

        if base_mean == 0.0 {
            return 0.0;
        }

        let ratio = curr_mean / base_mean;
        (ratio - 1.0).abs()
    }
}

// ============================================================================
// Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Production Quality Gates - Course 3, Week 5               ║");
    println!("║     Quality with pmat, Orchestration with batuta              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // -------------------------------------------------------------------------
    // Demo 1: TDG Score Calculation
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 1: TDG Score Calculation");
    println!("   ─────────────────────────────────────────────────────────────");

    let tdg = TdgScore::calculate(
        ComplexityMetrics {
            cyclomatic_avg: 5.2,
            cyclomatic_max: 15,
            cognitive_avg: 8.1,
            cognitive_max: 20,
        },
        CoverageMetrics {
            line_coverage: 85.0,
            branch_coverage: 78.0,
            mutation_score: 72.0,
        },
        DocumentationMetrics {
            doc_coverage: 90.0,
            readme_score: 85.0,
        },
        SecurityMetrics {
            vulnerabilities: 0,
            advisories: 0,
        },
    );

    println!("   TDG Score: {:?}", tdg.overall_grade);
    println!("   Complexity: avg={:.1}, max={}", tdg.complexity.cyclomatic_avg, tdg.complexity.cyclomatic_max);
    println!("   Coverage: line={:.1}%, branch={:.1}%, mutation={:.1}%",
        tdg.coverage.line_coverage, tdg.coverage.branch_coverage, tdg.coverage.mutation_score);
    println!("   Security: {} vulnerabilities, {} advisories",
        tdg.security.vulnerabilities, tdg.security.advisories);

    // -------------------------------------------------------------------------
    // Demo 2: Quality Gate Evaluation
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 2: Quality Gate Evaluation");
    println!("   ─────────────────────────────────────────────────────────────");

    let mut gate = QualityGate::new("production-readiness");
    gate.add_check("accuracy", 0.95, Comparator::GreaterOrEqual)
        .add_check("latency_p99_ms", 100.0, Comparator::LessOrEqual)
        .add_check("error_rate", 0.01, Comparator::LessOrEqual);

    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.96);
    metrics.insert("latency_p99_ms".to_string(), 85.0);
    metrics.insert("error_rate".to_string(), 0.005);

    let result = gate.evaluate(&metrics);
    println!("   Gate: {} - {}", result.name, if result.passed { "PASSED" } else { "FAILED" });
    for check in &result.check_results {
        let status = if check.passed { "✓" } else { "✗" };
        println!("   {} {}", status, check.message);
    }

    // -------------------------------------------------------------------------
    // Demo 3: Pipeline Orchestration
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 3: Pipeline Orchestration");
    println!("   ─────────────────────────────────────────────────────────────");

    let mut pipeline = Pipeline::new("fraud-detection-pipeline");
    pipeline
        .add_stage("ingest", StageType::DataIngestion, vec![])
        .add_stage("features", StageType::FeatureEngineering, vec!["ingest"])
        .add_stage("train", StageType::Training, vec!["features"])
        .add_stage("validate", StageType::Validation, vec!["train"])
        .add_stage("deploy", StageType::Deployment, vec!["validate"])
        .add_stage("monitor", StageType::Monitoring, vec!["deploy"]);

    let order = pipeline.execution_order();
    println!("   Pipeline: {}", pipeline.name);
    println!("   Execution order:");
    for (i, stage) in order.iter().enumerate() {
        println!("   {}. {}", i + 1, stage);
    }

    // -------------------------------------------------------------------------
    // Demo 4: Drift Detection
    // -------------------------------------------------------------------------
    println!("\n📊 Demo 4: Drift Detection");
    println!("   ─────────────────────────────────────────────────────────────");

    let baseline: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
    let current_ok: Vec<f64> = (0..100).map(|i| 101.0 + (i as f64) * 0.1).collect();
    let current_drift: Vec<f64> = (0..100).map(|i| 150.0 + (i as f64) * 0.5).collect();

    let drift_ok = DriftMetrics::calculate("amount", &baseline, &current_ok);
    let drift_bad = DriftMetrics::calculate("amount", &baseline, &current_drift);

    println!("   Feature: amount");
    println!("   No drift:   PSI={:.4}, drift={}", drift_ok.psi, drift_ok.drift_detected);
    println!("   With drift: PSI={:.4}, drift={}", drift_bad.psi, drift_bad.drift_detected);

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n✅ Demo complete!");
    println!("\n📖 Key Concepts:");
    println!("   - TDG (Toyota Development Grade): Holistic quality scoring");
    println!("   - Quality gates: Automated checks before deployment");
    println!("   - Pipeline orchestration: Dependency-aware execution");
    println!("   - Drift detection: Monitor feature distributions");
    println!("   - Compare with Databricks Workflows + Lakehouse Monitoring");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_from_score() {
        assert_eq!(Grade::from_score(95.0), Grade::A);
        assert_eq!(Grade::from_score(85.0), Grade::B);
        assert_eq!(Grade::from_score(75.0), Grade::C);
        assert_eq!(Grade::from_score(65.0), Grade::D);
        assert_eq!(Grade::from_score(55.0), Grade::F);
    }

    #[test]
    fn test_comparator() {
        assert!(Comparator::GreaterThan.check(10.0, 5.0));
        assert!(!Comparator::GreaterThan.check(5.0, 10.0));
        assert!(Comparator::LessOrEqual.check(5.0, 5.0));
    }

    #[test]
    fn test_quality_gate() {
        let mut gate = QualityGate::new("test");
        gate.add_check("accuracy", 0.9, Comparator::GreaterOrEqual);

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let result = gate.evaluate(&metrics);
        assert!(result.passed);
    }

    #[test]
    fn test_quality_gate_failure() {
        let mut gate = QualityGate::new("test");
        gate.add_check("accuracy", 0.9, Comparator::GreaterOrEqual);

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);

        let result = gate.evaluate(&metrics);
        assert!(!result.passed);
    }

    #[test]
    fn test_pipeline_order() {
        let mut pipeline = Pipeline::new("test");
        pipeline
            .add_stage("a", StageType::DataIngestion, vec![])
            .add_stage("b", StageType::Training, vec!["a"])
            .add_stage("c", StageType::Deployment, vec!["b"]);

        let order = pipeline.execution_order();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_drift_detection() {
        let baseline: Vec<f64> = vec![100.0; 10];
        let current: Vec<f64> = vec![200.0; 10];

        let drift = DriftMetrics::calculate("test", &baseline, &current);
        assert!(drift.drift_detected);
    }

    #[test]
    fn test_tdg_score() {
        let tdg = TdgScore::calculate(
            ComplexityMetrics {
                cyclomatic_avg: 5.0,
                cyclomatic_max: 10,
                cognitive_avg: 5.0,
                cognitive_max: 10,
            },
            CoverageMetrics {
                line_coverage: 90.0,
                branch_coverage: 85.0,
                mutation_score: 80.0,
            },
            DocumentationMetrics {
                doc_coverage: 90.0,
                readme_score: 90.0,
            },
            SecurityMetrics {
                vulnerabilities: 0,
                advisories: 0,
            },
        );

        assert!(tdg.overall_grade <= Grade::B);
    }
}
