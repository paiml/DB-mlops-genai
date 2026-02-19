//! Fine-Tuning Demo - Course 4 Week 3
//!
//! Demonstrates fine-tuning concepts that map to Databricks model customization.
//! Shows LoRA, QLoRA, and training configuration patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum FineTuneError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Training error: {0}")]
    Training(String),
    #[error("Data error: {0}")]
    Data(String),
}

// ============================================================================
// LoRA Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: BiasMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiasMode {
    None,
    LoraOnly,
    All,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: BiasMode::None,
        }
    }
}

impl LoraConfig {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_target_modules(mut self, modules: Vec<&str>) -> Self {
        self.target_modules = modules.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Calculate scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Estimate trainable parameters
    pub fn estimate_params(&self, hidden_size: usize) -> usize {
        // Each LoRA adapter has A (hidden x rank) and B (rank x hidden)
        let per_module = 2 * hidden_size * self.rank;
        per_module * self.target_modules.len()
    }
}

// ============================================================================
// QLoRA Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLoraConfig {
    pub lora: LoraConfig,
    pub quantization_bits: u8,
    pub double_quantization: bool,
    pub compute_dtype: ComputeDtype,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeDtype {
    Float16,
    BFloat16,
    Float32,
}

impl QLoraConfig {
    pub fn from_lora(lora: LoraConfig) -> Self {
        Self {
            lora,
            quantization_bits: 4,
            double_quantization: true,
            compute_dtype: ComputeDtype::BFloat16,
        }
    }

    pub fn with_bits(mut self, bits: u8) -> Self {
        self.quantization_bits = bits;
        self
    }

    /// Estimate memory savings
    pub fn memory_ratio(&self) -> f32 {
        match self.quantization_bits {
            4 => 0.25,
            8 => 0.5,
            _ => 1.0,
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub warmup_ratio: f32,
    pub weight_decay: f32,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f32,
    pub optimizer: OptimizerType,
    pub scheduler: SchedulerType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    AdamW,
    Adam,
    SGD,
    Adafactor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    Linear,
    Cosine,
    Constant,
    CosineWithRestarts,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 4,
            epochs: 3,
            warmup_ratio: 0.03,
            weight_decay: 0.01,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
            optimizer: OptimizerType::AdamW,
            scheduler: SchedulerType::Cosine,
        }
    }
}

impl TrainingConfig {
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Calculate effective batch size
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size * self.gradient_accumulation_steps
    }

    /// Calculate total training steps
    pub fn total_steps(&self, dataset_size: usize) -> usize {
        let steps_per_epoch = dataset_size / self.effective_batch_size();
        steps_per_epoch * self.epochs
    }

    /// Calculate warmup steps
    pub fn warmup_steps(&self, dataset_size: usize) -> usize {
        let total = self.total_steps(dataset_size);
        (total as f32 * self.warmup_ratio) as usize
    }
}

// ============================================================================
// Training Data
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub instruction: String,
    pub input: Option<String>,
    pub output: String,
}

impl TrainingSample {
    pub fn new(instruction: &str, output: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            input: None,
            output: output.to_string(),
        }
    }

    pub fn with_input(mut self, input: &str) -> Self {
        self.input = Some(input.to_string());
        self
    }

    /// Format as prompt-completion pair
    pub fn format(&self, template: &PromptTemplate) -> String {
        template.format(self)
    }
}

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn alpaca() -> Self {
        Self {
            template:
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
                    .to_string(),
        }
    }

    pub fn chatml() -> Self {
        Self {
            template: "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>".to_string(),
        }
    }

    pub fn format(&self, sample: &TrainingSample) -> String {
        self.template
            .replace("{instruction}", &sample.instruction)
            .replace("{input}", sample.input.as_deref().unwrap_or(""))
            .replace("{output}", &sample.output)
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
    pub learning_rate: f32,
    pub grad_norm: Option<f32>,
}

#[derive(Debug, Default)]
pub struct MetricsTracker {
    metrics: Vec<TrainingMetrics>,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log(&mut self, metrics: TrainingMetrics) {
        self.metrics.push(metrics);
    }

    pub fn latest(&self) -> Option<&TrainingMetrics> {
        self.metrics.last()
    }

    pub fn average_loss(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.metrics.iter().map(|m| m.loss).sum();
        sum / self.metrics.len() as f32
    }

    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

// ============================================================================
// Trainer (Simulated)
// ============================================================================

#[derive(Debug)]
pub struct Trainer {
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    metrics: MetricsTracker,
}

impl Trainer {
    pub fn new(lora_config: LoraConfig, training_config: TrainingConfig) -> Self {
        Self {
            lora_config,
            training_config,
            metrics: MetricsTracker::new(),
        }
    }

    /// Simulate training (for demo)
    pub fn train(&mut self, samples: &[TrainingSample]) -> Result<(), FineTuneError> {
        if samples.is_empty() {
            return Err(FineTuneError::Data("No training samples".to_string()));
        }

        let total_steps = self.training_config.total_steps(samples.len());
        let warmup_steps = self.training_config.warmup_steps(samples.len());

        for step in 0..total_steps {
            let epoch = step as f32 / (total_steps as f32 / self.training_config.epochs as f32);

            // Simulate loss decrease
            let loss = 2.0 / (1.0 + step as f32 / 10.0) + 0.1;

            // Learning rate with warmup
            let lr = if step < warmup_steps {
                self.training_config.learning_rate * (step as f32 / warmup_steps as f32)
            } else {
                self.training_config.learning_rate
            };

            self.metrics.log(TrainingMetrics {
                step,
                epoch,
                loss,
                learning_rate: lr,
                grad_norm: Some(0.5),
            });
        }

        Ok(())
    }

    pub fn metrics(&self) -> &MetricsTracker {
        &self.metrics
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }
}

// ============================================================================
// Adapter Management
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    pub name: String,
    pub base_model: String,
    pub lora_config: LoraConfig,
    pub training_loss: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct AdapterRegistry {
    adapters: HashMap<String, AdapterInfo>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, adapter: AdapterInfo) {
        self.adapters.insert(adapter.name.clone(), adapter);
    }

    pub fn get(&self, name: &str) -> Option<&AdapterInfo> {
        self.adapters.get(name)
    }

    pub fn list(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    pub fn count(&self) -> usize {
        self.adapters.len()
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Fine-Tuning Demo - Course 4 Week 3                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: LoRA Configuration
    println!("⚙️ Step 1: LoRA Configuration");
    let lora = LoraConfig::new(8)
        .with_alpha(16.0)
        .with_dropout(0.05)
        .with_target_modules(vec!["q_proj", "k_proj", "v_proj", "o_proj"]);

    println!("   Rank: {}", lora.rank);
    println!("   Alpha: {}", lora.alpha);
    println!("   Scaling: {:.2}", lora.scaling());
    println!("   Target modules: {:?}", lora.target_modules);
    println!(
        "   Est. trainable params (4096 hidden): {}",
        lora.estimate_params(4096)
    );
    println!();

    // Step 2: QLoRA Configuration
    println!("🔧 Step 2: QLoRA Configuration");
    let qlora = QLoraConfig::from_lora(lora.clone()).with_bits(4);

    println!("   Quantization: {}-bit", qlora.quantization_bits);
    println!("   Double quantization: {}", qlora.double_quantization);
    println!("   Memory ratio: {:.0}%", qlora.memory_ratio() * 100.0);
    println!();

    // Step 3: Training Configuration
    println!("📊 Step 3: Training Configuration");
    let training_config = TrainingConfig::default()
        .with_learning_rate(2e-4)
        .with_batch_size(4)
        .with_epochs(3);

    println!("   Learning rate: {}", training_config.learning_rate);
    println!("   Batch size: {}", training_config.batch_size);
    println!(
        "   Effective batch: {}",
        training_config.effective_batch_size()
    );
    println!("   Epochs: {}", training_config.epochs);

    let dataset_size = 1000;
    println!(
        "   Total steps (1000 samples): {}",
        training_config.total_steps(dataset_size)
    );
    println!(
        "   Warmup steps: {}",
        training_config.warmup_steps(dataset_size)
    );
    println!();

    // Step 4: Training Data
    println!("📝 Step 4: Training Data Format");
    let sample = TrainingSample::new(
        "Explain machine learning in simple terms.",
        "Machine learning is a way for computers to learn from examples.",
    );

    let alpaca_template = PromptTemplate::alpaca();
    let chatml_template = PromptTemplate::chatml();

    println!("   Alpaca format:\n   ---");
    for line in alpaca_template.format(&sample).lines() {
        println!("   {}", line);
    }
    println!("   ---\n");

    println!("   ChatML format:\n   ---");
    for line in chatml_template.format(&sample).lines() {
        println!("   {}", line);
    }
    println!("   ---\n");

    // Step 5: Simulated Training
    println!("🏋️ Step 5: Training Simulation");
    let mut trainer = Trainer::new(lora.clone(), training_config);

    let samples: Vec<TrainingSample> = (0..100)
        .map(|i| TrainingSample::new(&format!("Question {}", i), &format!("Answer {}", i)))
        .collect();

    trainer.train(&samples).unwrap();

    println!("   Steps completed: {}", trainer.metrics().len());
    println!("   Average loss: {:.4}", trainer.metrics().average_loss());
    if let Some(latest) = trainer.metrics().latest() {
        println!("   Final loss: {:.4}", latest.loss);
        println!("   Final LR: {:.6}", latest.learning_rate);
    }
    println!();

    // Step 6: Adapter Registry
    println!("📦 Step 6: Adapter Registry");
    let mut registry = AdapterRegistry::new();

    registry.register(AdapterInfo {
        name: "my-adapter-v1".to_string(),
        base_model: "llama-7b".to_string(),
        lora_config: lora,
        training_loss: 0.15,
        metadata: HashMap::new(),
    });

    println!("   Registered adapters: {:?}", registry.list());
    println!("   Total: {}\n", registry.count());

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • LoRA low-rank adaptation");
    println!("  • QLoRA quantized training");
    println!("  • Training configuration");
    println!("  • Prompt templates (Alpaca, ChatML)");
    println!("  • Adapter management");
    println!();
    println!("Databricks equivalent: Foundation Model Fine-tuning");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
    }

    #[test]
    fn test_lora_scaling() {
        let config = LoraConfig::new(8).with_alpha(16.0);
        assert!((config.scaling() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_params() {
        let config = LoraConfig::new(8).with_target_modules(vec!["q_proj"]);
        let params = config.estimate_params(256);
        assert!(params > 0);
    }

    #[test]
    fn test_qlora_memory_ratio() {
        let qlora = QLoraConfig::from_lora(LoraConfig::default()).with_bits(4);
        assert!((qlora.memory_ratio() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_training_config_effective_batch() {
        let config = TrainingConfig::default().with_batch_size(4);
        assert_eq!(config.effective_batch_size(), 16); // 4 * 4 accumulation
    }

    #[test]
    fn test_training_config_total_steps() {
        let config = TrainingConfig::default().with_batch_size(4).with_epochs(2);
        let steps = config.total_steps(160); // 160 / 16 = 10 steps per epoch
        assert_eq!(steps, 20);
    }

    #[test]
    fn test_training_sample() {
        let sample = TrainingSample::new("Q", "A");
        assert_eq!(sample.instruction, "Q");
        assert_eq!(sample.output, "A");
    }

    #[test]
    fn test_prompt_template_alpaca() {
        let sample = TrainingSample::new("test", "answer");
        let template = PromptTemplate::alpaca();
        let formatted = template.format(&sample);
        assert!(formatted.contains("### Instruction:"));
        assert!(formatted.contains("test"));
    }

    #[test]
    fn test_prompt_template_chatml() {
        let sample = TrainingSample::new("test", "answer");
        let template = PromptTemplate::chatml();
        let formatted = template.format(&sample);
        assert!(formatted.contains("<|im_start|>"));
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        tracker.log(TrainingMetrics {
            step: 0,
            epoch: 0.0,
            loss: 1.0,
            learning_rate: 0.001,
            grad_norm: None,
        });
        assert_eq!(tracker.len(), 1);
        assert!((tracker.average_loss() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_trainer() {
        let lora = LoraConfig::default();
        let config = TrainingConfig::default().with_epochs(1);
        let mut trainer = Trainer::new(lora, config);

        let samples: Vec<TrainingSample> = (0..50)
            .map(|i| TrainingSample::new(&format!("Q{}", i), &format!("A{}", i)))
            .collect();

        assert!(trainer.train(&samples).is_ok());
        assert!(trainer.metrics().len() > 0);
    }

    #[test]
    fn test_adapter_registry() {
        let mut registry = AdapterRegistry::new();
        registry.register(AdapterInfo {
            name: "test".to_string(),
            base_model: "llama".to_string(),
            lora_config: LoraConfig::default(),
            training_loss: 0.1,
            metadata: HashMap::new(),
        });

        assert_eq!(registry.count(), 1);
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_finetune_error() {
        let err = FineTuneError::Config("invalid".to_string());
        assert!(err.to_string().contains("invalid"));
    }
}
