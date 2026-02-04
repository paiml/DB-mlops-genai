# Lab: Embeddings

Build a vector search index with SIMD-accelerated similarity.

## Objectives

- Generate text embeddings
- Implement similarity metrics
- Build a searchable index

## Demo Code

See [`demos/course4/week3/vector-search/`](https://github.com/paiml/DB-mlops-genai/tree/main/demos/course4/week3/vector-search)

## Lab Exercise

See [`labs/course4/week3/lab_3_5_embeddings.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course4/week3)

## Key Implementation

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

pub struct VectorIndex {
    embeddings: Vec<Embedding>,
}

impl VectorIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut results: Vec<_> = self.embeddings.iter()
            .map(|e| (e.id.clone(), cosine_similarity(query, &e.vector)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(k).collect()
    }
}
```
