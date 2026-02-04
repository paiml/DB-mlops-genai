//! Fine-Tuning with entrenar
//!
//! Demonstrates LLM fine-tuning patterns using entrenar concepts.
//! This example shows LoRA, QLoRA, and training configuration.
//!
//! # Course 4, Week 3: Fine-Tuning + Production + Capstone

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
}

// ============================================================================
// LoRA Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: u32,
    pub alpha: u32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: BiasMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasMode {
    None,
    All,
    LoraOnly,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: BiasMode::None,
        }
    }
}

impl LoraConfig {
    pub fn new(rank: u32, alpha: u32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_targets(mut self, modules: &[&str]) -> Self {
        self.target_modules = modules.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_bias(mut self, bias: BiasMode) -> Self {
        self.bias = bias;
        self
    }

    /// Calculate scaling factor (alpha / rank)
    pub fn scaling_factor(&self) -> f32 {
        self.alpha as f32 / self.rank as f32
    }

    /// Estimate trainable parameters
    pub fn trainable_params(&self, hidden_dim: usize) -> usize {
        // LoRA adds (rank * hidden_dim + hidden_dim * rank) per module = 2 * rank * hidden_dim
        2 * self.rank as usize * hidden_dim * self.target_modules.len()
    }

    /// Calculate percentage of trainable parameters
    pub fn trainable_ratio(&self, total_params: u64, hidden_dim: usize) -> f64 {
        let trainable = self.trainable_params(hidden_dim) as f64;
        trainable / total_params as f64 * 100.0
    }
}

// ============================================================================
// QLoRA Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QloraConfig {
    pub lora: LoraConfig,
    pub bits: u8,
    pub double_quant: bool,
    pub quant_type: QuantType,
    pub compute_dtype: ComputeDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantType {
    NF4,
    FP4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDtype {
    Float16,
    BFloat16,
    Float32,
}

impl Default for QloraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            bits: 4,
            double_quant: true,
            quant_type: QuantType::NF4,
            compute_dtype: ComputeDtype::BFloat16,
        }
    }
}

impl QloraConfig {
    pub fn new(lora: LoraConfig, bits: u8) -> Self {
        Self {
            lora,
            bits,
            ..Default::default()
        }
    }

    pub fn with_double_quant(mut self, enabled: bool) -> Self {
        self.double_quant = enabled;
        self
    }

    pub fn with_compute_dtype(mut self, dtype: ComputeDtype) -> Self {
        self.compute_dtype = dtype;
        self
    }

    /// Calculate memory reduction factor
    pub fn memory_reduction(&self) -> f32 {
        let base = self.bits as f32 / 16.0;
        if self.double_quant {
            base * 0.9 // Additional ~10% reduction
        } else {
            base
        }
    }

    /// Estimate memory usage in GB
    pub fn estimated_memory_gb(&self, model_params_billions: f64) -> f64 {
        model_params_billions * 2.0 * self.memory_reduction() as f64
    }
}

// ============================================================================
// Training Data
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub instruction: String,
    pub input: String,
    pub output: String,
}

impl TrainingSample {
    pub fn new(instruction: &str, input: &str, output: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            input: input.to_string(),
            output: output.to_string(),
        }
    }

    pub fn format_alpaca(&self) -> String {
        if self.input.is_empty() {
            format!(
                "### Instruction:\n{}\n\n### Response:\n{}",
                self.instruction, self.output
            )
        } else {
            format!(
                "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}",
                self.instruction, self.input, self.output
            )
        }
    }

    pub fn format_chatml(&self) -> String {
        let user_content = if self.input.is_empty() {
            self.instruction.clone()
        } else {
            format!("{}\n\n{}", self.instruction, self.input)
        };

        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>",
            user_content, self.output
        )
    }

    pub fn total_length(&self) -> usize {
        self.instruction.len() + self.input.len() + self.output.len()
    }
}

#[derive(Debug, Clone)]
pub struct TrainingDataset {
    samples: Vec<TrainingSample>,
    format: DatasetFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    Alpaca,
    ChatML,
    ShareGPT,
}

impl TrainingDataset {
    pub fn new(format: DatasetFormat) -> Self {
        Self {
            samples: Vec::new(),
            format,
        }
    }

    pub fn add(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    pub fn add_many(&mut self, samples: Vec<TrainingSample>) {
        self.samples.extend(samples);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn format_all(&self) -> Vec<String> {
        self.samples
            .iter()
            .map(|s| match self.format {
                DatasetFormat::Alpaca => s.format_alpaca(),
                DatasetFormat::ChatML => s.format_chatml(),
                DatasetFormat::ShareGPT => s.format_chatml(),
            })
            .collect()
    }

    pub fn split(&self, train_ratio: f32) -> (TrainingDataset, TrainingDataset) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;

        let mut train = TrainingDataset::new(self.format);
        train.samples = self.samples[..split_idx].to_vec();

        let mut val = TrainingDataset::new(self.format);
        val.samples = self.samples[split_idx..].to_vec();

        (train, val)
    }

    pub fn average_length(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.total_length()).sum::<usize>() as f64
            / self.samples.len() as f64
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub epochs: usize,
    pub max_steps: Option<usize>,
    pub warmup_ratio: f32,
    pub weight_decay: f64,
    pub max_seq_length: usize,
    pub logging_steps: usize,
    pub save_steps: usize,
    pub eval_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            epochs: 3,
            max_steps: None,
            warmup_ratio: 0.03,
            weight_decay: 0.01,
            max_seq_length: 512,
            logging_steps: 10,
            save_steps: 100,
            eval_steps: 100,
        }
    }
}

impl TrainingConfig {
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size * self.gradient_accumulation_steps
    }

    pub fn steps_per_epoch(&self, dataset_size: usize) -> usize {
        (dataset_size + self.effective_batch_size() - 1) / self.effective_batch_size()
    }

    pub fn total_steps(&self, dataset_size: usize) -> usize {
        match self.max_steps {
            Some(max) => max,
            None => self.steps_per_epoch(dataset_size) * self.epochs,
        }
    }

    pub fn warmup_steps(&self, dataset_size: usize) -> usize {
        (self.total_steps(dataset_size) as f32 * self.warmup_ratio) as usize
    }
}

// ============================================================================
// Training Metrics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub step: usize,
    pub epoch: f32,
    pub loss: f32,
    pub learning_rate: f64,
    pub grad_norm: f32,
    pub samples_per_second: f32,
}

#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    metrics: Vec<TrainingMetrics>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add(&mut self, metrics: TrainingMetrics) {
        self.metrics.push(metrics);
    }

    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    pub fn latest_loss(&self) -> Option<f32> {
        self.metrics.last().map(|m| m.loss)
    }

    pub fn average_loss(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        self.metrics.iter().map(|m| m.loss).sum::<f32>() / self.metrics.len() as f32
    }

    pub fn min_loss(&self) -> Option<f32> {
        self.metrics.iter().map(|m| m.loss).fold(None, |min, loss| {
            Some(min.map_or(loss, |m: f32| m.min(loss)))
        })
    }
}

// ============================================================================
// Trainer
// ============================================================================

pub struct Trainer {
    config: TrainingConfig,
    lora_config: LoraConfig,
    history: TrainingHistory,
}

impl Trainer {
    pub fn new(config: TrainingConfig, lora_config: LoraConfig) -> Self {
        Self {
            config,
            lora_config,
            history: TrainingHistory::new(),
        }
    }

    pub fn train(&mut self, dataset: &TrainingDataset) -> Result<TrainingResult, TrainingError> {
        if dataset.is_empty() {
            return Err(TrainingError::Data("Empty dataset".to_string()));
        }

        let total_steps = self.config.total_steps(dataset.len());
        let warmup_steps = self.config.warmup_steps(dataset.len());

        let mut loss = 2.5f32;
        for step in 0..total_steps {
            // Simulate learning rate schedule
            let lr = self.get_learning_rate(step, total_steps, warmup_steps);

            // Simulate loss decrease
            loss *= 0.997;
            loss += ((step as f32 * 0.1).sin() * 0.05).abs();

            let epoch = step as f32 / self.config.steps_per_epoch(dataset.len()) as f32;

            self.history.add(TrainingMetrics {
                step,
                epoch,
                loss,
                learning_rate: lr,
                grad_norm: 1.0 + (step as f32 * 0.01).cos() * 0.3,
                samples_per_second: 10.0 + (step as f32 * 0.001).sin() * 2.0,
            });
        }

        Ok(TrainingResult {
            final_loss: self.history.latest_loss().unwrap_or(0.0),
            best_loss: self.history.min_loss().unwrap_or(0.0),
            total_steps,
            epochs_completed: self.config.epochs as f32,
        })
    }

    fn get_learning_rate(&self, step: usize, total_steps: usize, warmup_steps: usize) -> f64 {
        if step < warmup_steps {
            // Linear warmup
            self.config.learning_rate * (step as f64 / warmup_steps as f64)
        } else {
            // Cosine decay
            let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
            self.config.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }

    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub final_loss: f32,
    pub best_loss: f32,
    pub total_steps: usize,
    pub epochs_completed: f32,
}

// ============================================================================
// Model Merging
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMethod {
    Linear,
    Slerp,
    TaskArithmetic,
}

pub struct LoraWeight {
    pub name: String,
    pub weight: f32,
}

impl LoraWeight {
    pub fn new(name: &str, weight: f32) -> Self {
        Self {
            name: name.to_string(),
            weight,
        }
    }
}

pub fn merge_adapters(
    adapters: &[LoraWeight],
    method: MergeMethod,
) -> Result<String, TrainingError> {
    if adapters.is_empty() {
        return Err(TrainingError::Config("No adapters to merge".to_string()));
    }

    // Normalize weights
    let total_weight: f32 = adapters.iter().map(|a| a.weight).sum();
    if total_weight < 1e-6 {
        return Err(TrainingError::Config("Weights sum to zero".to_string()));
    }

    let method_name = match method {
        MergeMethod::Linear => "linear",
        MergeMethod::Slerp => "slerp",
        MergeMethod::TaskArithmetic => "task_arithmetic",
    };

    Ok(format!(
        "Merged {} adapters using {} method",
        adapters.len(),
        method_name
    ))
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Fine-Tuning with entrenar - Course 4, Week 3              ║");
    println!("║     LoRA, QLoRA, Training Configuration                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Step 1: LoRA Configuration
    println!("\n🔧 Step 1: LoRA Configuration");
    let lora = LoraConfig::new(8, 16)
        .with_dropout(0.05)
        .with_targets(&["q_proj", "v_proj", "k_proj", "o_proj"]);

    println!("   Rank (r): {}", lora.rank);
    println!("   Alpha: {}", lora.alpha);
    println!("   Scaling factor: {:.2}", lora.scaling_factor());
    println!("   Dropout: {}", lora.dropout);
    println!("   Target modules: {:?}", lora.target_modules);

    let hidden_dim = 4096;
    let total_params = 7_000_000_000u64;
    println!("   Trainable params: {}", lora.trainable_params(hidden_dim));
    println!(
        "   Trainable ratio: {:.4}%",
        lora.trainable_ratio(total_params, hidden_dim)
    );

    // Step 2: QLoRA Configuration
    println!("\n💾 Step 2: QLoRA Configuration");
    let qlora = QloraConfig::new(lora.clone(), 4)
        .with_double_quant(true)
        .with_compute_dtype(ComputeDtype::BFloat16);

    println!("   Bits: {}", qlora.bits);
    println!("   Quant type: {:?}", qlora.quant_type);
    println!("   Double quantization: {}", qlora.double_quant);
    println!("   Compute dtype: {:?}", qlora.compute_dtype);
    println!(
        "   Memory reduction: {:.0}%",
        (1.0 - qlora.memory_reduction()) * 100.0
    );
    println!(
        "   Estimated memory (7B): {:.1} GB",
        qlora.estimated_memory_gb(7.0)
    );

    // Step 3: Training Data
    println!("\n📊 Step 3: Training Data");
    let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);

    let samples = vec![
        TrainingSample::new(
            "Summarize the text.",
            "Machine learning is a field of AI...",
            "ML enables systems to learn from data.",
        ),
        TrainingSample::new(
            "Translate to French.",
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
        ),
        TrainingSample::new(
            "Answer the question.",
            "What is the capital of France?",
            "Paris is the capital of France.",
        ),
        TrainingSample::new(
            "Classify sentiment.",
            "This product is amazing!",
            "Positive",
        ),
    ];

    for sample in samples {
        dataset.add(sample);
    }

    // Add more samples for simulation
    for i in 0..96 {
        dataset.add(TrainingSample::new(
            &format!("Task {}", i),
            &format!("Input {}", i),
            &format!("Output {}", i),
        ));
    }

    println!("   Dataset size: {} samples", dataset.len());
    println!("   Format: Alpaca");
    println!("   Average length: {:.0} chars", dataset.average_length());

    let (train, val) = dataset.split(0.9);
    println!("   Train/Val split: {} / {}", train.len(), val.len());

    // Step 4: Training Configuration
    println!("\n⚙️  Step 4: Training Configuration");
    let config = TrainingConfig {
        learning_rate: 2e-4,
        batch_size: 4,
        gradient_accumulation_steps: 4,
        epochs: 3,
        max_seq_length: 512,
        ..Default::default()
    };

    println!("   Learning rate: {:.0e}", config.learning_rate);
    println!("   Batch size: {}", config.batch_size);
    println!(
        "   Gradient accumulation: {}",
        config.gradient_accumulation_steps
    );
    println!("   Effective batch size: {}", config.effective_batch_size());
    println!("   Epochs: {}", config.epochs);
    println!(
        "   Steps per epoch: {}",
        config.steps_per_epoch(train.len())
    );
    println!("   Total steps: {}", config.total_steps(train.len()));
    println!("   Warmup steps: {}", config.warmup_steps(train.len()));

    // Step 5: Training
    println!("\n🚀 Step 5: Training Simulation");
    let mut trainer = Trainer::new(config, lora);
    let result = trainer.train(&train).unwrap();

    println!("   Final loss: {:.4}", result.final_loss);
    println!("   Best loss: {:.4}", result.best_loss);
    println!("   Total steps: {}", result.total_steps);
    println!("   Epochs completed: {:.1}", result.epochs_completed);

    // Step 6: Model Merging
    println!("\n🔀 Step 6: Adapter Merging");
    let adapters = vec![
        LoraWeight::new("adapter_math", 0.5),
        LoraWeight::new("adapter_code", 0.3),
        LoraWeight::new("adapter_writing", 0.2),
    ];

    let merge_result = merge_adapters(&adapters, MergeMethod::Linear).unwrap();
    println!("   {}", merge_result);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • LoRA configuration (rank, alpha, target modules)");
    println!("  • QLoRA for memory-efficient training");
    println!("  • Training data formatting (Alpaca, ChatML)");
    println!("  • Learning rate scheduling (warmup + cosine decay)");
    println!("  • Adapter merging strategies");
    println!();
    println!("Sovereign AI Stack: entrenar fine-tuning");
    println!("Databricks equivalent: Model Training, Fine-tuning APIs");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LoraConfig Tests
    // ========================================================================

    #[test]
    fn test_lora_config_new() {
        let lora = LoraConfig::new(8, 16);
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16);
    }

    #[test]
    fn test_lora_config_default() {
        let lora = LoraConfig::default();
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16);
        assert_eq!(lora.bias, BiasMode::None);
    }

    #[test]
    fn test_lora_config_scaling() {
        let lora = LoraConfig::new(8, 16);
        assert!((lora.scaling_factor() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_config_trainable_params() {
        let lora = LoraConfig::new(8, 16).with_targets(&["q", "v"]);
        let params = lora.trainable_params(4096);
        assert!(params > 0);
    }

    #[test]
    fn test_lora_config_clone() {
        let lora = LoraConfig::new(8, 16);
        let cloned = lora.clone();
        assert_eq!(lora.rank, cloned.rank);
    }

    #[test]
    fn test_lora_config_with_bias() {
        let lora = LoraConfig::new(8, 16).with_bias(BiasMode::All);
        assert_eq!(lora.bias, BiasMode::All);
    }

    // ========================================================================
    // QloraConfig Tests
    // ========================================================================

    #[test]
    fn test_qlora_config_default() {
        let qlora = QloraConfig::default();
        assert_eq!(qlora.bits, 4);
        assert!(qlora.double_quant);
        assert_eq!(qlora.quant_type, QuantType::NF4);
    }

    #[test]
    fn test_qlora_config_memory_reduction() {
        let qlora = QloraConfig::default();
        let reduction = qlora.memory_reduction();
        assert!(reduction < 0.5);
    }

    #[test]
    fn test_qlora_config_estimated_memory() {
        let qlora = QloraConfig::default();
        let mem = qlora.estimated_memory_gb(7.0);
        assert!(mem > 0.0 && mem < 14.0);
    }

    #[test]
    fn test_qlora_config_clone() {
        let qlora = QloraConfig::default();
        let cloned = qlora.clone();
        assert_eq!(qlora.bits, cloned.bits);
    }

    // ========================================================================
    // TrainingSample Tests
    // ========================================================================

    #[test]
    fn test_training_sample_new() {
        let sample = TrainingSample::new("instr", "input", "output");
        assert_eq!(sample.instruction, "instr");
        assert_eq!(sample.input, "input");
        assert_eq!(sample.output, "output");
    }

    #[test]
    fn test_training_sample_format_alpaca() {
        let sample = TrainingSample::new("Summarize", "text", "summary");
        let formatted = sample.format_alpaca();
        assert!(formatted.contains("### Instruction:"));
        assert!(formatted.contains("Summarize"));
    }

    #[test]
    fn test_training_sample_format_alpaca_no_input() {
        let sample = TrainingSample::new("Say hello", "", "Hello!");
        let formatted = sample.format_alpaca();
        assert!(!formatted.contains("### Input:"));
    }

    #[test]
    fn test_training_sample_format_chatml() {
        let sample = TrainingSample::new("Summarize", "text", "summary");
        let formatted = sample.format_chatml();
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_training_sample_total_length() {
        let sample = TrainingSample::new("abc", "def", "ghi");
        assert_eq!(sample.total_length(), 9);
    }

    // ========================================================================
    // TrainingDataset Tests
    // ========================================================================

    #[test]
    fn test_training_dataset_new() {
        let dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_training_dataset_add() {
        let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        dataset.add(TrainingSample::new("a", "b", "c"));
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_training_dataset_split() {
        let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        for i in 0..10 {
            dataset.add(TrainingSample::new(&i.to_string(), "", "out"));
        }

        let (train, val) = dataset.split(0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_training_dataset_format_all() {
        let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        dataset.add(TrainingSample::new("a", "b", "c"));
        dataset.add(TrainingSample::new("x", "y", "z"));

        let formatted = dataset.format_all();
        assert_eq!(formatted.len(), 2);
    }

    #[test]
    fn test_training_dataset_average_length() {
        let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        dataset.add(TrainingSample::new("abc", "", "def"));
        assert!((dataset.average_length() - 6.0).abs() < 0.001);
    }

    // ========================================================================
    // TrainingConfig Tests
    // ========================================================================

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.epochs, 3);
    }

    #[test]
    fn test_training_config_effective_batch_size() {
        let config = TrainingConfig::default();
        assert_eq!(config.effective_batch_size(), 16);
    }

    #[test]
    fn test_training_config_steps_per_epoch() {
        let config = TrainingConfig::default();
        let steps = config.steps_per_epoch(160);
        assert_eq!(steps, 10);
    }

    #[test]
    fn test_training_config_total_steps() {
        let config = TrainingConfig::default();
        let total = config.total_steps(160);
        assert_eq!(total, 30);
    }

    #[test]
    fn test_training_config_warmup_steps() {
        let config = TrainingConfig::default();
        // Use a larger dataset to ensure warmup > 0 (total_steps * warmup_ratio)
        let warmup = config.warmup_steps(1600);
        assert!(warmup > 0);
        // With 1600 samples: 100 steps/epoch * 3 epochs = 300 total, 0.03 ratio = 9 warmup
        assert_eq!(warmup, 9);
    }

    // ========================================================================
    // TrainingHistory Tests
    // ========================================================================

    #[test]
    fn test_training_history_new() {
        let history = TrainingHistory::new();
        assert!(history.is_empty());
    }

    #[test]
    fn test_training_history_add() {
        let mut history = TrainingHistory::new();
        history.add(TrainingMetrics {
            step: 0,
            epoch: 0.0,
            loss: 2.0,
            learning_rate: 1e-4,
            grad_norm: 1.0,
            samples_per_second: 10.0,
        });
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_training_history_latest_loss() {
        let mut history = TrainingHistory::new();
        history.add(TrainingMetrics {
            step: 0,
            epoch: 0.0,
            loss: 2.0,
            learning_rate: 1e-4,
            grad_norm: 1.0,
            samples_per_second: 10.0,
        });
        assert_eq!(history.latest_loss(), Some(2.0));
    }

    #[test]
    fn test_training_history_min_loss() {
        let mut history = TrainingHistory::new();
        history.add(TrainingMetrics {
            step: 0,
            epoch: 0.0,
            loss: 2.0,
            learning_rate: 1e-4,
            grad_norm: 1.0,
            samples_per_second: 10.0,
        });
        history.add(TrainingMetrics {
            step: 1,
            epoch: 0.1,
            loss: 1.5,
            learning_rate: 1e-4,
            grad_norm: 1.0,
            samples_per_second: 10.0,
        });
        assert_eq!(history.min_loss(), Some(1.5));
    }

    // ========================================================================
    // Trainer Tests
    // ========================================================================

    #[test]
    fn test_trainer_new() {
        let config = TrainingConfig::default();
        let lora = LoraConfig::default();
        let trainer = Trainer::new(config, lora);
        assert!(trainer.history().is_empty());
    }

    #[test]
    fn test_trainer_train() {
        let config = TrainingConfig::default();
        let lora = LoraConfig::default();
        let mut trainer = Trainer::new(config, lora);

        let mut dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        for i in 0..32 {
            dataset.add(TrainingSample::new(&i.to_string(), "", "out"));
        }

        let result = trainer.train(&dataset).unwrap();
        assert!(result.total_steps > 0);
    }

    #[test]
    fn test_trainer_train_empty_dataset() {
        let config = TrainingConfig::default();
        let lora = LoraConfig::default();
        let mut trainer = Trainer::new(config, lora);

        let dataset = TrainingDataset::new(DatasetFormat::Alpaca);
        let result = trainer.train(&dataset);
        assert!(result.is_err());
    }

    // ========================================================================
    // Merge Tests
    // ========================================================================

    #[test]
    fn test_merge_adapters() {
        let adapters = vec![LoraWeight::new("a", 0.5), LoraWeight::new("b", 0.5)];
        let result = merge_adapters(&adapters, MergeMethod::Linear);
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_adapters_empty() {
        let adapters: Vec<LoraWeight> = vec![];
        let result = merge_adapters(&adapters, MergeMethod::Linear);
        assert!(result.is_err());
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_config() {
        let err = TrainingError::Config("invalid".to_string());
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn test_error_data() {
        let err = TrainingError::Data("empty".to_string());
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_error_training() {
        let err = TrainingError::Training("OOM".to_string());
        assert!(err.to_string().contains("OOM"));
    }

    #[test]
    fn test_error_checkpoint() {
        let err = TrainingError::Checkpoint("not found".to_string());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_error_debug() {
        let err = TrainingError::Config("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Config"));
    }
}
