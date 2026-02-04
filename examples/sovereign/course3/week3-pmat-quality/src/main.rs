//! Quality Gates with pmat
//!
//! Demonstrates quality gate patterns using pmat concepts.
//! This example shows TDG scoring, code metrics, and quality enforcement.
//!
//! # Course 3, Week 3: Production Quality + Capstone

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

    #[error("Metric error: {0}")]
    Metric(String),

    #[error("Threshold violation: {0}")]
    ThresholdViolation(String),
}

// ============================================================================
// Code Metrics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetrics {
    pub lines_of_code: usize,
    pub test_count: usize,
    pub test_coverage: f64,
    pub mutation_score: f64,
    pub cyclomatic_complexity: f64,
    pub documentation_coverage: f64,
}

impl CodeMetrics {
    pub fn new() -> Self {
        Self {
            lines_of_code: 0,
            test_count: 0,
            test_coverage: 0.0,
            mutation_score: 0.0,
            cyclomatic_complexity: 0.0,
            documentation_coverage: 0.0,
        }
    }

    pub fn with_loc(mut self, loc: usize) -> Self {
        self.lines_of_code = loc;
        self
    }

    pub fn with_tests(mut self, count: usize, coverage: f64) -> Self {
        self.test_count = count;
        self.test_coverage = coverage;
        self
    }

    pub fn with_mutation_score(mut self, score: f64) -> Self {
        self.mutation_score = score;
        self
    }

    pub fn with_complexity(mut self, complexity: f64) -> Self {
        self.cyclomatic_complexity = complexity;
        self
    }

    pub fn with_doc_coverage(mut self, coverage: f64) -> Self {
        self.documentation_coverage = coverage;
        self
    }

    /// Calculate test density (tests per 100 LOC)
    pub fn test_density(&self) -> f64 {
        if self.lines_of_code == 0 {
            return 0.0;
        }
        (self.test_count as f64 / self.lines_of_code as f64) * 100.0
    }
}

impl Default for CodeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TDG (Technical Debt Grade) Scoring
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    pub fn passes_threshold(&self, min_grade: Grade) -> bool {
        self.numeric_value() >= min_grade.numeric_value()
    }

    pub fn numeric_value(&self) -> u8 {
        match self {
            Grade::A => 4,
            Grade::B => 3,
            Grade::C => 2,
            Grade::D => 1,
            Grade::F => 0,
        }
    }
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Grade::A => write!(f, "A"),
            Grade::B => write!(f, "B"),
            Grade::C => write!(f, "C"),
            Grade::D => write!(f, "D"),
            Grade::F => write!(f, "F"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdgScore {
    pub overall: f64,
    pub grade: Grade,
    pub components: TdgComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdgComponents {
    pub test_coverage_score: f64,
    pub mutation_score: f64,
    pub complexity_score: f64,
    pub documentation_score: f64,
}

impl TdgScore {
    pub fn calculate(metrics: &CodeMetrics) -> Self {
        // Weighted scoring
        let test_coverage_score = metrics.test_coverage * 100.0;
        let mutation_score = metrics.mutation_score * 100.0;

        // Complexity: lower is better, normalize to 0-100
        let complexity_score = (100.0 - metrics.cyclomatic_complexity * 5.0)
            .max(0.0)
            .min(100.0);

        let documentation_score = metrics.documentation_coverage * 100.0;

        // Weighted average
        let overall = test_coverage_score * 0.35
            + mutation_score * 0.25
            + complexity_score * 0.20
            + documentation_score * 0.20;

        Self {
            overall,
            grade: Grade::from_score(overall),
            components: TdgComponents {
                test_coverage_score,
                mutation_score,
                complexity_score,
                documentation_score,
            },
        }
    }
}

// ============================================================================
// Quality Gates
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub name: String,
    pub thresholds: GateThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateThresholds {
    pub min_test_coverage: f64,
    pub min_mutation_score: f64,
    pub max_complexity: f64,
    pub min_doc_coverage: f64,
    pub min_grade: Grade,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            min_test_coverage: 0.80,
            min_mutation_score: 0.70,
            max_complexity: 10.0,
            min_doc_coverage: 0.50,
            min_grade: Grade::B,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub passed: bool,
    pub gate_name: String,
    pub checks: Vec<CheckResult>,
    pub tdg_score: TdgScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub actual: f64,
    pub threshold: f64,
    pub message: String,
}

impl QualityGate {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            thresholds: GateThresholds::default(),
        }
    }

    pub fn with_thresholds(mut self, thresholds: GateThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn evaluate(&self, metrics: &CodeMetrics) -> GateResult {
        let mut checks = Vec::new();

        // Test coverage check
        checks.push(CheckResult {
            name: "Test Coverage".to_string(),
            passed: metrics.test_coverage >= self.thresholds.min_test_coverage,
            actual: metrics.test_coverage * 100.0,
            threshold: self.thresholds.min_test_coverage * 100.0,
            message: format!(
                "Coverage {:.1}% {} {:.1}%",
                metrics.test_coverage * 100.0,
                if metrics.test_coverage >= self.thresholds.min_test_coverage {
                    ">="
                } else {
                    "<"
                },
                self.thresholds.min_test_coverage * 100.0
            ),
        });

        // Mutation score check
        checks.push(CheckResult {
            name: "Mutation Score".to_string(),
            passed: metrics.mutation_score >= self.thresholds.min_mutation_score,
            actual: metrics.mutation_score * 100.0,
            threshold: self.thresholds.min_mutation_score * 100.0,
            message: format!(
                "Mutation score {:.1}% {} {:.1}%",
                metrics.mutation_score * 100.0,
                if metrics.mutation_score >= self.thresholds.min_mutation_score {
                    ">="
                } else {
                    "<"
                },
                self.thresholds.min_mutation_score * 100.0
            ),
        });

        // Complexity check
        checks.push(CheckResult {
            name: "Cyclomatic Complexity".to_string(),
            passed: metrics.cyclomatic_complexity <= self.thresholds.max_complexity,
            actual: metrics.cyclomatic_complexity,
            threshold: self.thresholds.max_complexity,
            message: format!(
                "Complexity {:.1} {} {:.1}",
                metrics.cyclomatic_complexity,
                if metrics.cyclomatic_complexity <= self.thresholds.max_complexity {
                    "<="
                } else {
                    ">"
                },
                self.thresholds.max_complexity
            ),
        });

        // Documentation check
        checks.push(CheckResult {
            name: "Documentation".to_string(),
            passed: metrics.documentation_coverage >= self.thresholds.min_doc_coverage,
            actual: metrics.documentation_coverage * 100.0,
            threshold: self.thresholds.min_doc_coverage * 100.0,
            message: format!(
                "Doc coverage {:.1}% {} {:.1}%",
                metrics.documentation_coverage * 100.0,
                if metrics.documentation_coverage >= self.thresholds.min_doc_coverage {
                    ">="
                } else {
                    "<"
                },
                self.thresholds.min_doc_coverage * 100.0
            ),
        });

        // TDG grade check
        let tdg = TdgScore::calculate(metrics);
        checks.push(CheckResult {
            name: "TDG Grade".to_string(),
            passed: tdg.grade.passes_threshold(self.thresholds.min_grade),
            actual: tdg.overall,
            threshold: self.thresholds.min_grade.numeric_value() as f64 * 25.0,
            message: format!(
                "Grade {} {} {}",
                tdg.grade,
                if tdg.grade.passes_threshold(self.thresholds.min_grade) {
                    ">="
                } else {
                    "<"
                },
                self.thresholds.min_grade
            ),
        });

        let passed = checks.iter().all(|c| c.passed);

        GateResult {
            passed,
            gate_name: self.name.clone(),
            checks,
            tdg_score: tdg,
        }
    }
}

// ============================================================================
// Quality Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub project_name: String,
    pub metrics: CodeMetrics,
    pub tdg_score: TdgScore,
    pub gate_results: Vec<GateResult>,
    pub recommendations: Vec<String>,
}

impl QualityReport {
    pub fn generate(project_name: &str, metrics: &CodeMetrics, gates: &[QualityGate]) -> Self {
        let tdg_score = TdgScore::calculate(metrics);
        let gate_results: Vec<GateResult> = gates.iter().map(|g| g.evaluate(metrics)).collect();

        let mut recommendations = Vec::new();

        if metrics.test_coverage < 0.80 {
            recommendations.push(format!(
                "Increase test coverage from {:.1}% to at least 80%",
                metrics.test_coverage * 100.0
            ));
        }

        if metrics.mutation_score < 0.70 {
            recommendations.push(format!(
                "Improve mutation score from {:.1}% to at least 70%",
                metrics.mutation_score * 100.0
            ));
        }

        if metrics.cyclomatic_complexity > 10.0 {
            recommendations.push(format!(
                "Reduce cyclomatic complexity from {:.1} to under 10",
                metrics.cyclomatic_complexity
            ));
        }

        if metrics.documentation_coverage < 0.50 {
            recommendations.push(format!(
                "Add documentation to reach at least 50% coverage (currently {:.1}%)",
                metrics.documentation_coverage * 100.0
            ));
        }

        Self {
            project_name: project_name.to_string(),
            metrics: metrics.clone(),
            tdg_score,
            gate_results,
            recommendations,
        }
    }

    pub fn all_gates_passed(&self) -> bool {
        self.gate_results.iter().all(|r| r.passed)
    }
}

// ============================================================================
// Trend Analysis
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub metric_name: String,
    pub values: Vec<f64>,
    pub timestamps: Vec<u64>,
}

impl MetricTrend {
    pub fn new(name: &str) -> Self {
        Self {
            metric_name: name.to_string(),
            values: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn add_point(&mut self, value: f64, timestamp: u64) {
        self.values.push(value);
        self.timestamps.push(timestamp);
    }

    pub fn trend_direction(&self) -> TrendDirection {
        if self.values.len() < 2 {
            return TrendDirection::Stable;
        }

        let recent = &self.values[self.values.len().saturating_sub(5)..];
        if recent.len() < 2 {
            return TrendDirection::Stable;
        }

        let first = recent[0];
        let last = *recent.last().unwrap();
        let change = (last - first) / first.abs().max(0.001);

        if change > 0.05 {
            TrendDirection::Improving
        } else if change < -0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    pub fn latest(&self) -> Option<f64> {
        self.values.last().copied()
    }

    pub fn average(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Quality Gates with pmat - Course 3, Week 3                ║");
    println!("║     TDG Scoring, Metrics, Quality Enforcement                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Step 1: Define code metrics
    println!("\n📊 Step 1: Code Metrics");
    let metrics = CodeMetrics::new()
        .with_loc(5000)
        .with_tests(250, 0.87)
        .with_mutation_score(0.75)
        .with_complexity(8.5)
        .with_doc_coverage(0.65);

    println!("   Lines of code: {}", metrics.lines_of_code);
    println!("   Test count: {}", metrics.test_count);
    println!("   Test coverage: {:.1}%", metrics.test_coverage * 100.0);
    println!("   Mutation score: {:.1}%", metrics.mutation_score * 100.0);
    println!(
        "   Cyclomatic complexity: {:.1}",
        metrics.cyclomatic_complexity
    );
    println!(
        "   Doc coverage: {:.1}%",
        metrics.documentation_coverage * 100.0
    );
    println!(
        "   Test density: {:.2} tests/100 LOC",
        metrics.test_density()
    );

    // Step 2: TDG Scoring
    println!("\n🎯 Step 2: TDG (Technical Debt Grade) Score");
    let tdg = TdgScore::calculate(&metrics);

    println!("   Overall score: {:.1}", tdg.overall);
    println!("   Grade: {}", tdg.grade);
    println!("   Components:");
    println!(
        "     - Test coverage: {:.1}",
        tdg.components.test_coverage_score
    );
    println!("     - Mutation: {:.1}", tdg.components.mutation_score);
    println!("     - Complexity: {:.1}", tdg.components.complexity_score);
    println!(
        "     - Documentation: {:.1}",
        tdg.components.documentation_score
    );

    // Step 3: Quality Gate Evaluation
    println!("\n🚦 Step 3: Quality Gate Evaluation");
    let gate = QualityGate::new("Production").with_thresholds(GateThresholds {
        min_test_coverage: 0.85,
        min_mutation_score: 0.70,
        max_complexity: 10.0,
        min_doc_coverage: 0.50,
        min_grade: Grade::B,
    });

    let result = gate.evaluate(&metrics);
    println!("   Gate: {}", result.gate_name);
    println!(
        "   Status: {}",
        if result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!("   Checks:");
    for check in &result.checks {
        let icon = if check.passed { "✓" } else { "✗" };
        println!("     {} {}: {}", icon, check.name, check.message);
    }

    // Step 4: Generate Quality Report
    println!("\n📝 Step 4: Quality Report");
    let gates = vec![
        QualityGate::new("Development"),
        QualityGate::new("Staging").with_thresholds(GateThresholds {
            min_test_coverage: 0.80,
            ..Default::default()
        }),
        QualityGate::new("Production").with_thresholds(GateThresholds {
            min_test_coverage: 0.90,
            min_mutation_score: 0.80,
            ..Default::default()
        }),
    ];

    let report = QualityReport::generate("fraud-detection-service", &metrics, &gates);

    println!("   Project: {}", report.project_name);
    println!(
        "   TDG Grade: {} ({:.1})",
        report.tdg_score.grade, report.tdg_score.overall
    );
    println!("   Gate Results:");
    for gr in &report.gate_results {
        let icon = if gr.passed { "✅" } else { "❌" };
        println!(
            "     {} {} - {}",
            icon,
            gr.gate_name,
            if gr.passed { "Passed" } else { "Failed" }
        );
    }

    if !report.recommendations.is_empty() {
        println!("   Recommendations:");
        for rec in &report.recommendations {
            println!("     • {}", rec);
        }
    }

    // Step 5: Trend Analysis
    println!("\n📈 Step 5: Trend Analysis");
    let mut coverage_trend = MetricTrend::new("test_coverage");
    coverage_trend.add_point(0.75, 1000);
    coverage_trend.add_point(0.78, 2000);
    coverage_trend.add_point(0.82, 3000);
    coverage_trend.add_point(0.85, 4000);
    coverage_trend.add_point(0.87, 5000);

    println!("   Metric: {}", coverage_trend.metric_name);
    println!(
        "   Latest: {:.1}%",
        coverage_trend.latest().unwrap_or(0.0) * 100.0
    );
    println!("   Average: {:.1}%", coverage_trend.average() * 100.0);
    println!("   Trend: {:?}", coverage_trend.trend_direction());

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Code metrics collection and analysis");
    println!("  • TDG (Technical Debt Grade) scoring");
    println!("  • Quality gate definition and evaluation");
    println!("  • Automated quality reports with recommendations");
    println!("  • Metric trend analysis");
    println!();
    println!("Sovereign AI Stack: pmat quality gates");
    println!("Databricks equivalent: MLflow Model Registry, Workflows");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CodeMetrics Tests
    // ========================================================================

    #[test]
    fn test_code_metrics_new() {
        let metrics = CodeMetrics::new();
        assert_eq!(metrics.lines_of_code, 0);
        assert_eq!(metrics.test_count, 0);
    }

    #[test]
    fn test_code_metrics_builder() {
        let metrics = CodeMetrics::new()
            .with_loc(1000)
            .with_tests(50, 0.85)
            .with_mutation_score(0.70);

        assert_eq!(metrics.lines_of_code, 1000);
        assert_eq!(metrics.test_count, 50);
        assert_eq!(metrics.test_coverage, 0.85);
        assert_eq!(metrics.mutation_score, 0.70);
    }

    #[test]
    fn test_code_metrics_test_density() {
        let metrics = CodeMetrics::new().with_loc(1000).with_tests(50, 0.8);
        assert!((metrics.test_density() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_code_metrics_test_density_zero_loc() {
        let metrics = CodeMetrics::new();
        assert_eq!(metrics.test_density(), 0.0);
    }

    #[test]
    fn test_code_metrics_default() {
        let metrics = CodeMetrics::default();
        assert_eq!(metrics.lines_of_code, 0);
    }

    #[test]
    fn test_code_metrics_clone() {
        let metrics = CodeMetrics::new().with_loc(100);
        let cloned = metrics.clone();
        assert_eq!(metrics.lines_of_code, cloned.lines_of_code);
    }

    // ========================================================================
    // Grade Tests
    // ========================================================================

    #[test]
    fn test_grade_from_score() {
        assert_eq!(Grade::from_score(95.0), Grade::A);
        assert_eq!(Grade::from_score(85.0), Grade::B);
        assert_eq!(Grade::from_score(75.0), Grade::C);
        assert_eq!(Grade::from_score(65.0), Grade::D);
        assert_eq!(Grade::from_score(55.0), Grade::F);
    }

    #[test]
    fn test_grade_passes_threshold() {
        assert!(Grade::A.passes_threshold(Grade::B));
        assert!(Grade::B.passes_threshold(Grade::B));
        assert!(!Grade::C.passes_threshold(Grade::B));
    }

    #[test]
    fn test_grade_numeric_value() {
        assert_eq!(Grade::A.numeric_value(), 4);
        assert_eq!(Grade::F.numeric_value(), 0);
    }

    #[test]
    fn test_grade_display() {
        assert_eq!(format!("{}", Grade::A), "A");
        assert_eq!(format!("{}", Grade::F), "F");
    }

    // ========================================================================
    // TdgScore Tests
    // ========================================================================

    #[test]
    fn test_tdg_score_calculate() {
        let metrics = CodeMetrics::new()
            .with_tests(100, 0.90)
            .with_mutation_score(0.80)
            .with_complexity(5.0)
            .with_doc_coverage(0.70);

        let tdg = TdgScore::calculate(&metrics);
        assert!(tdg.overall > 0.0);
        assert!(tdg.grade.passes_threshold(Grade::B));
    }

    #[test]
    fn test_tdg_score_poor_metrics() {
        let metrics = CodeMetrics::new()
            .with_tests(10, 0.30)
            .with_mutation_score(0.20)
            .with_complexity(20.0)
            .with_doc_coverage(0.10);

        let tdg = TdgScore::calculate(&metrics);
        assert_eq!(tdg.grade, Grade::F);
    }

    #[test]
    fn test_tdg_score_clone() {
        let metrics = CodeMetrics::new().with_tests(50, 0.85);
        let tdg = TdgScore::calculate(&metrics);
        let cloned = tdg.clone();
        assert_eq!(tdg.overall, cloned.overall);
    }

    // ========================================================================
    // QualityGate Tests
    // ========================================================================

    #[test]
    fn test_quality_gate_new() {
        let gate = QualityGate::new("Test");
        assert_eq!(gate.name, "Test");
    }

    #[test]
    fn test_quality_gate_evaluate_pass() {
        let gate = QualityGate::new("Test").with_thresholds(GateThresholds {
            min_test_coverage: 0.80,
            min_mutation_score: 0.70,
            max_complexity: 10.0,
            min_doc_coverage: 0.50,
            min_grade: Grade::C,
        });

        let metrics = CodeMetrics::new()
            .with_tests(100, 0.85)
            .with_mutation_score(0.75)
            .with_complexity(8.0)
            .with_doc_coverage(0.60);

        let result = gate.evaluate(&metrics);
        assert!(result.passed);
    }

    #[test]
    fn test_quality_gate_evaluate_fail() {
        let gate = QualityGate::new("Strict").with_thresholds(GateThresholds {
            min_test_coverage: 0.95,
            ..Default::default()
        });

        let metrics = CodeMetrics::new().with_tests(50, 0.80);
        let result = gate.evaluate(&metrics);
        assert!(!result.passed);
    }

    #[test]
    fn test_gate_thresholds_default() {
        let thresholds = GateThresholds::default();
        assert_eq!(thresholds.min_test_coverage, 0.80);
        assert_eq!(thresholds.min_grade, Grade::B);
    }

    // ========================================================================
    // QualityReport Tests
    // ========================================================================

    #[test]
    fn test_quality_report_generate() {
        let metrics = CodeMetrics::new()
            .with_tests(100, 0.85)
            .with_mutation_score(0.75);

        let gates = vec![QualityGate::new("Test")];
        let report = QualityReport::generate("test-project", &metrics, &gates);

        assert_eq!(report.project_name, "test-project");
        assert_eq!(report.gate_results.len(), 1);
    }

    #[test]
    fn test_quality_report_all_gates_passed() {
        let metrics = CodeMetrics::new()
            .with_tests(100, 0.90)
            .with_mutation_score(0.80)
            .with_complexity(5.0)
            .with_doc_coverage(0.70);

        let gates = vec![QualityGate::new("Test")];
        let report = QualityReport::generate("test", &metrics, &gates);
        assert!(report.all_gates_passed());
    }

    #[test]
    fn test_quality_report_recommendations() {
        let metrics = CodeMetrics::new()
            .with_tests(10, 0.50)
            .with_mutation_score(0.40);

        let report = QualityReport::generate("test", &metrics, &[]);
        assert!(!report.recommendations.is_empty());
    }

    // ========================================================================
    // MetricTrend Tests
    // ========================================================================

    #[test]
    fn test_metric_trend_new() {
        let trend = MetricTrend::new("coverage");
        assert_eq!(trend.metric_name, "coverage");
        assert!(trend.values.is_empty());
    }

    #[test]
    fn test_metric_trend_add_point() {
        let mut trend = MetricTrend::new("test");
        trend.add_point(0.80, 1000);
        trend.add_point(0.85, 2000);
        assert_eq!(trend.values.len(), 2);
    }

    #[test]
    fn test_metric_trend_direction_improving() {
        let mut trend = MetricTrend::new("test");
        trend.add_point(0.70, 1);
        trend.add_point(0.75, 2);
        trend.add_point(0.80, 3);
        trend.add_point(0.85, 4);
        trend.add_point(0.90, 5);
        assert_eq!(trend.trend_direction(), TrendDirection::Improving);
    }

    #[test]
    fn test_metric_trend_direction_degrading() {
        let mut trend = MetricTrend::new("test");
        trend.add_point(0.90, 1);
        trend.add_point(0.85, 2);
        trend.add_point(0.80, 3);
        trend.add_point(0.75, 4);
        trend.add_point(0.70, 5);
        assert_eq!(trend.trend_direction(), TrendDirection::Degrading);
    }

    #[test]
    fn test_metric_trend_direction_stable() {
        let trend = MetricTrend::new("test");
        assert_eq!(trend.trend_direction(), TrendDirection::Stable);
    }

    #[test]
    fn test_metric_trend_latest() {
        let mut trend = MetricTrend::new("test");
        assert!(trend.latest().is_none());
        trend.add_point(0.85, 1000);
        assert_eq!(trend.latest(), Some(0.85));
    }

    #[test]
    fn test_metric_trend_average() {
        let mut trend = MetricTrend::new("test");
        trend.add_point(0.80, 1);
        trend.add_point(0.90, 2);
        assert!((trend.average() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_metric_trend_average_empty() {
        let trend = MetricTrend::new("test");
        assert_eq!(trend.average(), 0.0);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_gate_failed() {
        let err = QualityError::GateFailed("production".to_string());
        assert!(err.to_string().contains("production"));
    }

    #[test]
    fn test_error_metric() {
        let err = QualityError::Metric("invalid".to_string());
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn test_error_threshold_violation() {
        let err = QualityError::ThresholdViolation("coverage < 80%".to_string());
        assert!(err.to_string().contains("coverage"));
    }

    #[test]
    fn test_error_debug() {
        let err = QualityError::GateFailed("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("GateFailed"));
    }
}
