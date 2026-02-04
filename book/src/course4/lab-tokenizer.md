# Lab: Tokenizer

Build a BPE tokenizer to understand LLM text processing.

## Objectives

- Implement byte-pair encoding
- Handle special tokens
- Encode and decode text

## Demo Code

See [`demos/course4/week1/llm-serving/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course4/week1/llm-serving)

## Lab Exercise

See [`labs/course4/week1/lab_1_7_tokenizer.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course4/week1)

## Key Implementation

```rust
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    special_tokens: HashMap<String, u32>,
}

impl BpeTokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.chars()
            .map(|c| c.to_string())
            .collect();

        // Apply merge rules
        for (a, b) in &self.merges {
            tokens = self.apply_merge(&tokens, a, b);
        }

        tokens.iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect()
    }
}
```
