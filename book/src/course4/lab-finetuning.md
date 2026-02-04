# Lab: Fine-Tuning

Configure LoRA fine-tuning for domain adaptation.

## Objectives

- Configure LoRA parameters
- Prepare training data
- Calculate training metrics

## Demo Code

See [`demos/course4/week5/fine-tuning/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course4/week5/fine-tuning)

## Lab Exercise

See [`labs/course4/week5/lab_5_3_fine_tuning.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course4/week5)

## Key Implementation

```rust
pub struct LoraConfig {
    pub r: usize,           // Rank (4, 8, 16)
    pub alpha: usize,       // Scaling (16, 32)
    pub dropout: f32,       // Dropout rate
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    pub fn scaling_factor(&self) -> f32 {
        self.alpha as f32 / self.r as f32
    }

    pub fn estimated_params(&self, hidden: usize, layers: usize) -> usize {
        self.r * hidden * 2 * self.target_modules.len() * layers
    }
}

// Example: 7B model with r=8
// Params: 8 * 4096 * 2 * 2 * 32 = 4.2M (0.06% of 7B)
```
