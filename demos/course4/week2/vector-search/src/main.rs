//! Vector Search Demo - Course 4 Week 2
//!
//! Demonstrates vector search concepts that map to Databricks Vector Search.
//! Shows embedding generation, indexing, and similarity search.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Index error: {0}")]
    Index(String),
    #[error("Search error: {0}")]
    Search(String),
}

// ============================================================================
// Embeddings
// ============================================================================

pub type Embedding = Vec<f32>;

/// Simple embedding model (simulated)
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    dimension: usize,
    model_name: String,
}

impl EmbeddingModel {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_name: "text-embedding".to_string(),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.model_name = name.to_string();
        self
    }

    /// Generate embedding for text (simulated with hash-based approach)
    pub fn embed(&self, text: &str) -> Embedding {
        let mut embedding = vec![0.0f32; self.dimension];

        // Simple hash-based embedding (for demo purposes)
        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i) % self.dimension;
            embedding[idx] += 1.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Batch embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Embedding> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// ============================================================================
// Similarity Metrics
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl SimilarityMetric {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::DotProduct => dot_product(a, b),
            SimilarityMetric::Euclidean => -euclidean_distance(a, b), // Negative for ranking
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ============================================================================
// Vector Index
// ============================================================================

#[derive(Debug, Clone)]
pub struct VectorDocument {
    pub id: String,
    pub text: String,
    pub embedding: Embedding,
    pub metadata: HashMap<String, String>,
}

impl VectorDocument {
    pub fn new(id: &str, text: &str, embedding: Embedding) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            embedding,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

/// In-memory vector index
#[derive(Debug)]
pub struct VectorIndex {
    documents: Vec<VectorDocument>,
    metric: SimilarityMetric,
    dimension: usize,
}

impl VectorIndex {
    pub fn new(dimension: usize, metric: SimilarityMetric) -> Self {
        Self {
            documents: Vec::new(),
            metric,
            dimension,
        }
    }

    /// Add a document to the index
    pub fn add(&mut self, doc: VectorDocument) -> Result<(), VectorError> {
        if doc.embedding.len() != self.dimension {
            return Err(VectorError::Index(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                doc.embedding.len()
            )));
        }
        self.documents.push(doc);
        Ok(())
    }

    /// Search for similar documents
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<_> = self
            .documents
            .iter()
            .map(|doc| {
                let score = self.metric.compute(query_embedding, &doc.embedding);
                SearchResult {
                    id: doc.id.clone(),
                    text: doc.text.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Search with text query
    pub fn search_text(
        &self,
        query: &str,
        model: &EmbeddingModel,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let query_embedding = model.embed(query);
        self.search(&query_embedding, top_k)
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    Hnsw { m: usize, ef_construction: usize },
    Ivf { n_lists: usize },
}

impl IndexConfig {
    pub fn flat(name: &str, dimension: usize) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            metric: SimilarityMetric::Cosine,
            index_type: IndexType::Flat,
        }
    }

    pub fn hnsw(name: &str, dimension: usize) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            metric: SimilarityMetric::Cosine,
            index_type: IndexType::Hnsw {
                m: 16,
                ef_construction: 200,
            },
        }
    }

    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }
}

// ============================================================================
// Metadata Filtering
// ============================================================================

#[derive(Debug, Clone)]
pub enum MetadataFilter {
    Equals(String, String),
    NotEquals(String, String),
    In(String, Vec<String>),
}

impl MetadataFilter {
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        match self {
            MetadataFilter::Equals(key, value) => metadata.get(key) == Some(value),
            MetadataFilter::NotEquals(key, value) => metadata.get(key) != Some(value),
            MetadataFilter::In(key, values) => metadata
                .get(key)
                .map(|v| values.contains(v))
                .unwrap_or(false),
        }
    }
}

impl VectorIndex {
    /// Search with metadata filter
    pub fn search_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Vec<SearchResult> {
        let mut results: Vec<_> = self
            .documents
            .iter()
            .filter(|doc| filter.matches(&doc.metadata))
            .map(|doc| {
                let score = self.metric.compute(query_embedding, &doc.embedding);
                SearchResult {
                    id: doc.id.clone(),
                    text: doc.text.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Vector Search Demo - Course 4 Week 2                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Embedding Model
    println!("📊 Step 1: Embedding Generation");
    let model = EmbeddingModel::new(64).with_name("all-MiniLM-L6");
    println!("   Model: {}", model.model_name());
    println!("   Dimension: {}\n", model.dimension());

    let texts = ["Machine learning is great", "Deep learning neural networks"];
    for text in &texts {
        let emb = model.embed(text);
        println!("   Text: \"{}\"", text);
        println!("   Embedding (first 5): {:?}...\n", &emb[..5]);
    }

    // Step 2: Similarity Metrics
    println!("📏 Step 2: Similarity Metrics");
    let emb1 = model.embed("machine learning");
    let emb2 = model.embed("deep learning");
    let emb3 = model.embed("cooking recipes");

    println!("   Cosine(ml, dl): {:.4}", cosine_similarity(&emb1, &emb2));
    println!(
        "   Cosine(ml, cooking): {:.4}\n",
        cosine_similarity(&emb1, &emb3)
    );

    // Step 3: Vector Index
    println!("🗂️ Step 3: Vector Index");
    let mut index = VectorIndex::new(64, SimilarityMetric::Cosine);

    let documents = [
        ("doc1", "Machine learning algorithms", "ml"),
        ("doc2", "Deep learning with neural networks", "ml"),
        ("doc3", "Natural language processing", "nlp"),
        ("doc4", "Computer vision and image recognition", "cv"),
        ("doc5", "Reinforcement learning for games", "ml"),
    ];

    for (id, text, category) in &documents {
        let emb = model.embed(text);
        let doc = VectorDocument::new(id, text, emb).with_metadata("category", category);
        index.add(doc).unwrap();
    }

    println!("   Added {} documents", index.len());
    println!("   Dimension: {}\n", index.dimension());

    // Step 4: Search
    println!("🔍 Step 4: Similarity Search");
    let query = "deep neural network training";
    let results = index.search_text(query, &model, 3);

    println!("   Query: \"{}\"", query);
    println!("   Results:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "     {}. [{}] \"{}\" (score: {:.4})",
            i + 1,
            result.id,
            result.text,
            result.score
        );
    }
    println!();

    // Step 5: Filtered Search
    println!("🔎 Step 5: Filtered Search");
    let filter = MetadataFilter::Equals("category".to_string(), "ml".to_string());
    let query_emb = model.embed("learning algorithms");
    let filtered_results = index.search_filtered(&query_emb, 3, &filter);

    println!("   Filter: category = 'ml'");
    println!("   Results:");
    for (i, result) in filtered_results.iter().enumerate() {
        println!(
            "     {}. [{}] \"{}\" (score: {:.4})",
            i + 1,
            result.id,
            result.text,
            result.score
        );
    }
    println!();

    // Step 6: Index Config
    println!("⚙️ Step 6: Index Configuration");
    let flat_config = IndexConfig::flat("flat-index", 384);
    let hnsw_config = IndexConfig::hnsw("hnsw-index", 384);

    println!("   Flat Index: {:?}", flat_config.index_type);
    println!("   HNSW Index: {:?}\n", hnsw_config.index_type);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Text embeddings and normalization");
    println!("  • Similarity metrics (cosine, dot product, euclidean)");
    println!("  • Vector index operations");
    println!("  • Similarity search with top-k");
    println!("  • Metadata filtering");
    println!();
    println!("Databricks equivalent: Vector Search, Model Serving Embeddings");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model() {
        let model = EmbeddingModel::new(64);
        let emb = model.embed("test");
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn test_embedding_normalized() {
        let model = EmbeddingModel::new(64);
        let emb = model.embed("test text");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_batch() {
        let model = EmbeddingModel::new(32);
        let embeddings = model.embed_batch(&["a", "b", "c"]);
        assert_eq!(embeddings.len(), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_index_add() {
        let mut index = VectorIndex::new(3, SimilarityMetric::Cosine);
        let doc = VectorDocument::new("1", "test", vec![1.0, 0.0, 0.0]);
        assert!(index.add(doc).is_ok());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_index_dimension_mismatch() {
        let mut index = VectorIndex::new(3, SimilarityMetric::Cosine);
        let doc = VectorDocument::new("1", "test", vec![1.0, 0.0]); // Wrong dimension
        assert!(index.add(doc).is_err());
    }

    #[test]
    fn test_vector_search() {
        let mut index = VectorIndex::new(3, SimilarityMetric::Cosine);
        index
            .add(VectorDocument::new("1", "a", vec![1.0, 0.0, 0.0]))
            .unwrap();
        index
            .add(VectorDocument::new("2", "b", vec![0.0, 1.0, 0.0]))
            .unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_metadata_filter_equals() {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());

        let filter = MetadataFilter::Equals("type".to_string(), "article".to_string());
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_metadata_filter_not_equals() {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());

        let filter = MetadataFilter::NotEquals("type".to_string(), "book".to_string());
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_index_config_flat() {
        let config = IndexConfig::flat("test", 384);
        assert!(matches!(config.index_type, IndexType::Flat));
    }

    #[test]
    fn test_index_config_hnsw() {
        let config = IndexConfig::hnsw("test", 384);
        assert!(matches!(config.index_type, IndexType::Hnsw { .. }));
    }

    #[test]
    fn test_vector_error() {
        let err = VectorError::Embedding("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
