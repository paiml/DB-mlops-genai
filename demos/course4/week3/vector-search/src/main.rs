//! Vector Search Demo - Course 4 Week 3
//!
//! Demonstrates vector embeddings and similarity search that map to Databricks Vector Search.
//! Shows embedding generation, indexing, and retrieval patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Index error: {0}")]
    Index(String),
    #[error("Embedding error: {0}")]
    Embedding(String),
}

// ============================================================================
// Vector Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub dimension: usize,
}

impl Embedding {
    pub fn new(values: Vec<f32>) -> Self {
        let dimension = values.len();
        Self { values, dimension }
    }

    pub fn zeros(dimension: usize) -> Self {
        Self {
            values: vec![0.0; dimension],
            dimension,
        }
    }

    pub fn normalize(&self) -> Self {
        let norm = self.l2_norm();
        if norm < 1e-10 {
            return self.clone();
        }
        let values = self.values.iter().map(|v| v / norm).collect();
        Self::new(values)
    }

    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    pub fn dot(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot = self.dot(other);
        let norm_self = self.l2_norm();
        let norm_other = other.l2_norm();

        if norm_self < 1e-10 || norm_other < 1e-10 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }

    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ============================================================================
// Simple Embedding Model (Hash-based for demo)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SimpleEmbedder {
    dimension: usize,
}

impl SimpleEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn embed(&self, text: &str) -> Embedding {
        // Simple hash-based embedding for demonstration
        // In production, use transformer models
        let mut values = vec![0.0; self.dimension];

        for (i, word) in text.split_whitespace().enumerate() {
            for (j, c) in word.chars().enumerate() {
                let idx = ((c as usize) * (i + 1) + j) % self.dimension;
                values[idx] += 0.1;
            }
        }

        Embedding::new(values).normalize()
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Embedding> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

// ============================================================================
// Document
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(id: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ============================================================================
// Vector Index
// ============================================================================

#[derive(Debug, Clone)]
pub struct VectorIndex {
    dimension: usize,
    documents: Vec<Document>,
    embeddings: Vec<Embedding>,
    embedder: SimpleEmbedder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: Document,
    pub score: f32,
    pub rank: usize,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            documents: Vec::new(),
            embeddings: Vec::new(),
            embedder: SimpleEmbedder::new(dimension),
        }
    }

    pub fn add(&mut self, document: Document) -> Result<(), VectorError> {
        let embedding = self.embedder.embed(&document.content);

        if embedding.dimension != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.dimension,
            });
        }

        self.documents.push(document);
        self.embeddings.push(embedding);
        Ok(())
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_embedding = self.embedder.embed(query);

        let mut scores: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, query_embedding.cosine_similarity(emb)))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (idx, score))| SearchResult {
                document: self.documents[idx].clone(),
                score,
                rank: rank + 1,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

// ============================================================================
// Index Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: SimilarityMetric,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    Hnsw,
    IvfFlat,
}

impl IndexConfig {
    pub fn new(name: &str, dimension: usize) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            metric: SimilarityMetric::Cosine,
            index_type: IndexType::Flat,
        }
    }

    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }
}

// ============================================================================
// Chunking Utilities
// ============================================================================

#[derive(Debug, Clone)]
pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TextChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    pub fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        if words.len() <= self.chunk_size {
            chunks.push(words.join(" "));
            return chunks;
        }

        let mut start = 0;
        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            chunks.push(words[start..end].join(" "));

            if end >= words.len() {
                break;
            }

            start += self.chunk_size - self.chunk_overlap;
        }

        chunks
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Vector Search Demo - Course 4 Week 3                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Embeddings
    println!("📊 Step 1: Vector Embeddings");
    let embedder = SimpleEmbedder::new(128);

    let texts = ["machine learning", "deep learning", "artificial intelligence"];
    let embeddings: Vec<_> = texts.iter().map(|t| embedder.embed(t)).collect();

    for (text, emb) in texts.iter().zip(embeddings.iter()) {
        println!("   \"{}\": dim={}, norm={:.4}",
            text, emb.dimension, emb.l2_norm());
    }
    println!();

    // Step 2: Similarity
    println!("📐 Step 2: Similarity Metrics");
    let emb1 = &embeddings[0];
    let emb2 = &embeddings[1];
    let emb3 = &embeddings[2];

    println!("   Cosine similarity:");
    println!("     ML vs DL: {:.4}", emb1.cosine_similarity(emb2));
    println!("     ML vs AI: {:.4}", emb1.cosine_similarity(emb3));
    println!("     DL vs AI: {:.4}", emb2.cosine_similarity(emb3));
    println!();

    // Step 3: Vector Index
    println!("🗃️  Step 3: Vector Index");
    let mut index = VectorIndex::new(128);

    let documents = vec![
        Document::new("doc1", "Machine learning algorithms learn from data")
            .with_metadata("category", "ml"),
        Document::new("doc2", "Deep learning uses neural networks")
            .with_metadata("category", "dl"),
        Document::new("doc3", "Natural language processing handles text")
            .with_metadata("category", "nlp"),
        Document::new("doc4", "Computer vision analyzes images and video")
            .with_metadata("category", "cv"),
        Document::new("doc5", "Reinforcement learning optimizes through rewards")
            .with_metadata("category", "rl"),
    ];

    for doc in &documents {
        index.add(doc.clone()).unwrap();
    }

    println!("   Indexed {} documents\n", index.len());

    // Step 4: Search
    println!("🔍 Step 4: Semantic Search");
    let queries = ["neural networks", "text processing", "image analysis"];

    for query in &queries {
        println!("   Query: \"{}\"", query);
        let results = index.search(query, 3);
        for result in &results {
            println!("     #{} [score={:.4}] {}: {}",
                result.rank,
                result.score,
                result.document.id,
                &result.document.content[..40.min(result.document.content.len())]);
        }
        println!();
    }

    // Step 5: Text Chunking
    println!("📄 Step 5: Text Chunking");
    let chunker = TextChunker::new(10, 3);
    let long_text = "This is a longer document that needs to be split into smaller chunks for embedding. Each chunk should have some overlap with adjacent chunks to preserve context.";

    let chunks = chunker.chunk(long_text);
    println!("   Original: {} words", long_text.split_whitespace().count());
    println!("   Chunks: {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("     [{}] {} words: \"{}...\"",
            i, chunk.split_whitespace().count(),
            &chunk[..30.min(chunk.len())]);
    }
    println!();

    // Step 6: Index Configuration
    println!("⚙️  Step 6: Index Configuration");
    let config = IndexConfig::new("my-index", 1536)
        .with_metric(SimilarityMetric::Cosine)
        .with_index_type(IndexType::Hnsw);

    println!("   Name: {}", config.name);
    println!("   Dimension: {}", config.dimension);
    println!("   Metric: {:?}", config.metric);
    println!("   Type: {:?}\n", config.index_type);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Vector embeddings (dimension, normalization)");
    println!("  • Similarity metrics (cosine, euclidean)");
    println!("  • Vector index (add, search)");
    println!("  • Text chunking (overlap for context)");
    println!();
    println!("Databricks equivalent: Vector Search, Delta Sync Index");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalize() {
        let emb = Embedding::new(vec![3.0, 4.0]);
        let norm = emb.normalize();
        assert!((norm.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);

        let c = Embedding::new(vec![0.0, 1.0]);
        assert!(a.cosine_similarity(&c).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![3.0, 4.0]);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simple_embedder() {
        let embedder = SimpleEmbedder::new(64);
        let emb = embedder.embed("hello world");
        assert_eq!(emb.dimension, 64);
        assert!((emb.l2_norm() - 1.0).abs() < 1e-6); // Normalized
    }

    #[test]
    fn test_vector_index() {
        let mut index = VectorIndex::new(64);
        let doc = Document::new("test", "hello world");
        index.add(doc).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_search() {
        let mut index = VectorIndex::new(64);
        index.add(Document::new("a", "machine learning")).unwrap();
        index.add(Document::new("b", "deep learning")).unwrap();
        index.add(Document::new("c", "cooking recipes")).unwrap();

        let results = index.search("neural networks", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].rank, 1);
    }

    #[test]
    fn test_text_chunker() {
        let chunker = TextChunker::new(5, 2);
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() > 1);
        // Each chunk should have at most 5 words
        for chunk in &chunks {
            assert!(chunk.split_whitespace().count() <= 5);
        }
    }

    #[test]
    fn test_index_config() {
        let config = IndexConfig::new("test", 512)
            .with_metric(SimilarityMetric::DotProduct);
        assert_eq!(config.dimension, 512);
        assert!(matches!(config.metric, SimilarityMetric::DotProduct));
    }
}
