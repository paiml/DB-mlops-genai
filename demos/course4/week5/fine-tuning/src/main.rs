//! Fine-Tuning Demo - Course 4 Week 5
//!
//! Demonstrates LLM fine-tuning concepts that map to Databricks training capabilities.
//! Shows LoRA/QLoRA patterns and training configuration.

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

    pub fn format_chat(&self) -> String {
        format!(
            "<|user|>\n{}{}\n<|assistant|>\n{}",
            self.instruction,
            if self.input.is_empty() {
                String::new()
            } else {
                format!("\n{}", self.input)
            },
            self.output
        )
    }
}

#[derive(Debug, Clone)]
pub struct TrainingDataset {
    samples: Vec<TrainingSample>,
    format: DataFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Alpaca,
    ChatML,
    ShareGPT,
}

impl TrainingDataset {
    pub fn new(format: DataFormat) -> Self {
        Self {
            samples: Vec::new(),
            format,
        }
    }

    pub fn add(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
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
                DataFormat::Alpaca => s.format_alpaca(),
                DataFormat::ChatML => s.format_chat(),
                DataFormat::ShareGPT => s.format_chat(), // Simplified
            })
            .collect()
    }

    pub fn split(&self, train_ratio: f32) -> (TrainingDataset, TrainingDataset) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;
        let (train_samples, val_samples) = self.samples.split_at(split_idx);

        let mut train = TrainingDataset::new(self.format);
        train.samples = train_samples.to_vec();

        let mut val = TrainingDataset::new(self.format);
        val.samples = val_samples.to_vec();

        (train, val)
    }
}

// ============================================================================
// LoRA Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub r: u32,
    pub alpha: u32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: String,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
            ],
            bias: "none".to_string(),
        }
    }
}

impl LoraConfig {
    pub fn new(r: u32, alpha: u32) -> Self {
        Self {
            r,
            alpha,
            ..Default::default()
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_targets(mut self, targets: Vec<&str>) -> Self {
        self.target_modules = targets.into_iter().map(String::from).collect();
        self
    }

    pub fn trainable_params(&self, model_params: u64) -> u64 {
        // Approximate: LoRA adds r * hidden_dim * 2 params per target module
        let hidden_dim = 4096u64; // Typical for 7B model
        let lora_params = self.r as u64 * hidden_dim * 2 * self.target_modules.len() as u64;
        lora_params.min(model_params / 100) // Cap at 1% of model
    }

    pub fn scaling_factor(&self) -> f32 {
        self.alpha as f32 / self.r as f32
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
    pub quant_type: String,
}

impl Default for QloraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            bits: 4,
            double_quant: true,
            quant_type: "nf4".to_string(),
        }
    }
}

impl QloraConfig {
    pub fn memory_reduction(&self) -> f32 {
        // 4-bit = 0.25 of fp16, with double quant ~0.22
        let base_reduction = self.bits as f32 / 16.0;
        if self.double_quant {
            base_reduction * 0.9
        } else {
            base_reduction
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: usize,
    pub max_seq_length: usize,
    pub fp16: bool,
    pub bf16: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 4,
            epochs: 3,
            warmup_steps: 100,
            weight_decay: 0.01,
            gradient_accumulation_steps: 4,
            max_seq_length: 512,
            fp16: false,
            bf16: true,
        }
    }
}

impl TrainingConfig {
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size * self.gradient_accumulation_steps
    }

    pub fn steps_per_epoch(&self, dataset_size: usize) -> usize {
        dataset_size / self.effective_batch_size()
    }

    pub fn total_steps(&self, dataset_size: usize) -> usize {
        self.steps_per_epoch(dataset_size) * self.epochs
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
}

#[derive(Debug, Clone)]
pub struct TrainingHistory {
    metrics: Vec<TrainingMetrics>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self { metrics: Vec::new() }
    }

    pub fn add(&mut self, metrics: TrainingMetrics) {
        self.metrics.push(metrics);
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

    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Training Simulation
// ============================================================================

pub struct TrainingSimulator {
    config: TrainingConfig,
    lora_config: LoraConfig,
    history: TrainingHistory,
}

impl TrainingSimulator {
    pub fn new(config: TrainingConfig, lora_config: LoraConfig) -> Self {
        Self {
            config,
            lora_config,
            history: TrainingHistory::new(),
        }
    }

    pub fn simulate_training(&mut self, dataset: &TrainingDataset) -> Result<(), TrainingError> {
        let total_steps = self.config.total_steps(dataset.len());

        // Simulate training loop
        let mut loss = 2.5f32; // Starting loss
        for step in 0..total_steps {
            // Simulate loss decrease
            loss *= 0.995;
            loss += (step as f32 * 0.001).sin() * 0.1; // Add some noise

            let epoch = step as f32 / self.config.steps_per_epoch(dataset.len()) as f32;
            let lr = self.get_learning_rate(step, total_steps);

            self.history.add(TrainingMetrics {
                step,
                epoch,
                loss,
                learning_rate: lr,
                grad_norm: 1.0 + (step as f32 * 0.01).cos() * 0.5,
            });
        }

        Ok(())
    }

    fn get_learning_rate(&self, step: usize, total_steps: usize) -> f64 {
        // Cosine schedule with warmup
        if step < self.config.warmup_steps {
            self.config.learning_rate * (step as f64 / self.config.warmup_steps as f64)
        } else {
            let progress = (step - self.config.warmup_steps) as f64
                / (total_steps - self.config.warmup_steps) as f64;
            self.config.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }

    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Fine-Tuning Demo - Course 4 Week 5                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Create training dataset
    println!("📊 Step 1: Training Data Preparation");

    let mut dataset = TrainingDataset::new(DataFormat::Alpaca);

    let samples = vec![
        TrainingSample::new(
            "Summarize the following text.",
            "Machine learning is a subset of AI...",
            "Machine learning enables systems to learn from data.",
        ),
        TrainingSample::new(
            "Translate to French.",
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
        ),
        TrainingSample::new(
            "Answer the question.",
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
        TrainingSample::new(
            "Classify the sentiment.",
            "This product is amazing!",
            "Positive",
        ),
    ];

    for sample in samples {
        dataset.add(sample);
    }

    println!("   Dataset size: {} samples", dataset.len());
    println!("   Format: Alpaca\n");

    println!("   Sample formatted:");
    for line in dataset.format_all()[0].lines().take(5) {
        println!("   {}", line);
    }
    println!("   ...\n");

    // Step 2: LoRA Configuration
    println!("🔧 Step 2: LoRA Configuration");

    let lora = LoraConfig::new(8, 16)
        .with_dropout(0.05)
        .with_targets(vec!["q_proj", "v_proj", "k_proj", "o_proj"]);

    println!("   Rank (r): {}", lora.r);
    println!("   Alpha: {}", lora.alpha);
    println!("   Scaling: {:.1}", lora.scaling_factor());
    println!("   Dropout: {}", lora.dropout);
    println!("   Target modules: {:?}", lora.target_modules);

    let model_params = 7_000_000_000u64;
    let trainable = lora.trainable_params(model_params);
    println!("   Trainable params: {} ({:.4}%)\n",
        trainable,
        trainable as f64 / model_params as f64 * 100.0);

    // Step 3: QLoRA Configuration
    println!("💾 Step 3: QLoRA (Quantized LoRA)");

    let qlora = QloraConfig::default();
    println!("   Bits: {}", qlora.bits);
    println!("   Double quantization: {}", qlora.double_quant);
    println!("   Quant type: {}", qlora.quant_type);
    println!("   Memory reduction: {:.0}%\n", (1.0 - qlora.memory_reduction()) * 100.0);

    // Step 4: Training Configuration
    println!("⚙️  Step 4: Training Configuration");

    let config = TrainingConfig::default();
    println!("   Learning rate: {:.0e}", config.learning_rate);
    println!("   Batch size: {}", config.batch_size);
    println!("   Gradient accumulation: {}", config.gradient_accumulation_steps);
    println!("   Effective batch size: {}", config.effective_batch_size());
    println!("   Epochs: {}", config.epochs);
    println!("   Max sequence length: {}\n", config.max_seq_length);

    // Step 5: Simulate Training
    println!("🚀 Step 5: Training Simulation");

    // Create larger dataset for simulation
    let mut train_dataset = TrainingDataset::new(DataFormat::Alpaca);
    for i in 0..100 {
        train_dataset.add(TrainingSample::new(
            &format!("Instruction {}", i),
            &format!("Input {}", i),
            &format!("Output {}", i),
        ));
    }

    let mut trainer = TrainingSimulator::new(config.clone(), lora.clone());
    trainer.simulate_training(&train_dataset).unwrap();

    let history = trainer.history();
    println!("   Total steps: {}", history.len());
    println!("   Final loss: {:.4}", history.latest_loss().unwrap_or(0.0));
    println!("   Average loss: {:.4}\n", history.average_loss());

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Training data formatting (Alpaca, ChatML)");
    println!("  • LoRA configuration (rank, alpha, targets)");
    println!("  • QLoRA for memory-efficient training");
    println!("  • Training configuration (LR, batch size, epochs)");
    println!();
    println!("Databricks equivalent: Model Training, Fine-tuning APIs");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_sample_format() {
        let sample = TrainingSample::new("Summarize", "input text", "summary");
        let alpaca = sample.format_alpaca();
        assert!(alpaca.contains("### Instruction:"));
        assert!(alpaca.contains("Summarize"));
    }

    #[test]
    fn test_training_dataset() {
        let mut dataset = TrainingDataset::new(DataFormat::Alpaca);
        dataset.add(TrainingSample::new("test", "", "output"));
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_dataset_split() {
        let mut dataset = TrainingDataset::new(DataFormat::Alpaca);
        for i in 0..10 {
            dataset.add(TrainingSample::new(&format!("{}", i), "", "out"));
        }

        let (train, val) = dataset.split(0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_lora_config() {
        let lora = LoraConfig::new(8, 16);
        assert_eq!(lora.r, 8);
        assert_eq!(lora.alpha, 16);
        assert!((lora.scaling_factor() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_qlora_memory_reduction() {
        let qlora = QloraConfig::default();
        let reduction = qlora.memory_reduction();
        assert!(reduction < 0.5); // 4-bit should be less than 50%
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.effective_batch_size(), 16); // 4 * 4

        let steps = config.steps_per_epoch(160);
        assert_eq!(steps, 10); // 160 / 16
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.add(TrainingMetrics {
            step: 0,
            epoch: 0.0,
            loss: 2.0,
            learning_rate: 1e-4,
            grad_norm: 1.0,
        });
        history.add(TrainingMetrics {
            step: 1,
            epoch: 0.1,
            loss: 1.8,
            learning_rate: 1e-4,
            grad_norm: 0.9,
        });

        assert_eq!(history.len(), 2);
        assert!((history.latest_loss().unwrap() - 1.8).abs() < 0.001);
        assert!((history.average_loss() - 1.9).abs() < 0.001);
    }
}
