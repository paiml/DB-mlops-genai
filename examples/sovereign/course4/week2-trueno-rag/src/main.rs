//! RAG Pipeline with trueno-rag
//!
//! Demonstrates RAG (Retrieval-Augmented Generation) patterns using trueno-rag concepts.
//! This example shows chunking, embeddings, retrieval, and reranking.
//!
//! # Course 4, Week 2: Vector Search + RAG Pipelines

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Chunking error: {0}")]
    Chunking(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Retrieval error: {0}")]
    Retrieval(String),

    #[error("Reranking error: {0}")]
    Reranking(String),
}

// ============================================================================
// Document and Chunking
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub text: String,
    pub start_idx: usize,
    pub end_idx: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub separator: String,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            separator: " ".to_string(),
        }
    }
}

impl ChunkerConfig {
    pub fn with_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
}

pub struct Chunker {
    config: ChunkerConfig,
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    pub fn chunk(&self, document: &Document) -> Vec<Chunk> {
        let words: Vec<&str> = document.content.split(&self.config.separator).collect();
        let mut chunks = Vec::new();
        let mut word_idx = 0;
        let mut chunk_num = 0;

        while word_idx < words.len() {
            let end_idx = (word_idx + self.config.chunk_size).min(words.len());
            let chunk_words = &words[word_idx..end_idx];
            let text = chunk_words.join(&self.config.separator);

            chunks.push(Chunk {
                id: format!("{}-chunk-{}", document.id, chunk_num),
                doc_id: document.id.clone(),
                text,
                start_idx: word_idx,
                end_idx,
                metadata: document.metadata.clone(),
            });

            if end_idx >= words.len() {
                break;
            }

            word_idx += self.config.chunk_size - self.config.chunk_overlap;
            chunk_num += 1;
        }

        chunks
    }

    pub fn config(&self) -> &ChunkerConfig {
        &self.config
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkerConfig::default())
    }
}

// ============================================================================
// Embeddings
// ============================================================================

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub dimension: usize,
}

impl Embedding {
    pub fn new(vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self { vector, dimension }
    }

    pub fn zeros(dimension: usize) -> Self {
        Self {
            vector: vec![0.0; dimension],
            dimension,
        }
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut self.vector {
                *v /= norm;
            }
        }
    }
}

pub struct EmbeddingModel {
    dimension: usize,
    model_name: String,
}

impl EmbeddingModel {
    pub fn new(dimension: usize, model_name: &str) -> Self {
        Self {
            dimension,
            model_name: model_name.to_string(),
        }
    }

    pub fn embed(&self, text: &str) -> Embedding {
        // Simulate embedding generation
        let mut vector = vec![0.0f32; self.dimension];

        for (i, word) in text.split_whitespace().enumerate() {
            for (j, c) in word.chars().enumerate() {
                let idx = ((c as usize) * (i + 1) + j) % self.dimension;
                vector[idx] += 0.1;
            }
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        Embedding::new(vector)
    }

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
// Vector Index
// ============================================================================

#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub chunk: Chunk,
    pub embedding: Embedding,
}

pub struct VectorIndex {
    chunks: Vec<IndexedChunk>,
    dimension: usize,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            chunks: Vec::new(),
            dimension,
        }
    }

    pub fn add(&mut self, chunk: Chunk, embedding: Embedding) -> Result<(), RagError> {
        if embedding.dimension != self.dimension {
            return Err(RagError::Embedding(format!(
                "Dimension mismatch: {} vs {}",
                embedding.dimension, self.dimension
            )));
        }
        self.chunks.push(IndexedChunk { chunk, embedding });
        Ok(())
    }

    pub fn search(&self, query_embedding: &Embedding, top_k: usize) -> Vec<RetrievalResult> {
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, ic)| (i, query_embedding.cosine_similarity(&ic.embedding)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (idx, score))| RetrievalResult {
                chunk: self.chunks[idx].chunk.clone(),
                score,
                rank: rank + 1,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
    pub rank: usize,
}

// ============================================================================
// Reranker
// ============================================================================

pub struct Reranker {
    model_name: String,
}

impl Reranker {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }

    pub fn rerank(
        &self,
        query: &str,
        results: &[RetrievalResult],
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<String> =
            query_lower.split_whitespace().map(String::from).collect();

        let mut scored: Vec<(RetrievalResult, f32)> = results
            .iter()
            .map(|r| {
                let chunk_lower = r.chunk.text.to_lowercase();
                let chunk_words: std::collections::HashSet<String> =
                    chunk_lower.split_whitespace().map(String::from).collect();
                let overlap = query_words.intersection(&chunk_words).count() as f32;
                let score = r.score * 0.5 + (overlap / query_words.len().max(1) as f32) * 0.5;
                (r.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (mut result, score))| {
                result.score = score;
                result.rank = rank + 1;
                result
            })
            .collect()
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// ============================================================================
// RAG Pipeline
// ============================================================================

pub struct RagPipeline {
    chunker: Chunker,
    embedding_model: EmbeddingModel,
    index: VectorIndex,
    reranker: Option<Reranker>,
    retrieval_top_k: usize,
    rerank_top_k: usize,
}

impl RagPipeline {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            chunker: Chunker::default(),
            embedding_model: EmbeddingModel::new(embedding_dim, "all-MiniLM-L6-v2"),
            index: VectorIndex::new(embedding_dim),
            reranker: None,
            retrieval_top_k: 10,
            rerank_top_k: 5,
        }
    }

    pub fn with_chunker(mut self, chunker: Chunker) -> Self {
        self.chunker = chunker;
        self
    }

    pub fn with_reranker(mut self, reranker: Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn with_retrieval_k(mut self, k: usize) -> Self {
        self.retrieval_top_k = k;
        self
    }

    pub fn with_rerank_k(mut self, k: usize) -> Self {
        self.rerank_top_k = k;
        self
    }

    pub fn ingest(&mut self, documents: Vec<Document>) -> Result<usize, RagError> {
        let mut chunk_count = 0;

        for doc in documents {
            let chunks = self.chunker.chunk(&doc);
            for chunk in chunks {
                let embedding = self.embedding_model.embed(&chunk.text);
                self.index.add(chunk, embedding)?;
                chunk_count += 1;
            }
        }

        Ok(chunk_count)
    }

    pub fn retrieve(&self, query: &str) -> Vec<RetrievalResult> {
        let query_embedding = self.embedding_model.embed(query);
        let results = self.index.search(&query_embedding, self.retrieval_top_k);

        match &self.reranker {
            Some(reranker) => reranker.rerank(query, &results, self.rerank_top_k),
            None => results.into_iter().take(self.rerank_top_k).collect(),
        }
    }

    pub fn generate_context(&self, results: &[RetrievalResult]) -> String {
        results
            .iter()
            .map(|r| r.chunk.text.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn query(&self, question: &str) -> RagResponse {
        let results = self.retrieve(question);
        let context = self.generate_context(&results);

        // Simulate answer generation
        let answer = if context.to_lowercase().contains("machine learning") {
            "Based on the context, machine learning is a field of AI that enables systems to learn from data.".to_string()
        } else if context.to_lowercase().contains("neural") {
            "According to the retrieved documents, neural networks are computing systems inspired by biological neurons.".to_string()
        } else {
            format!(
                "Based on the retrieved context: {}",
                &context[..100.min(context.len())]
            )
        };

        RagResponse {
            question: question.to_string(),
            answer,
            sources: results.iter().map(|r| r.chunk.id.clone()).collect(),
            context_length: context.len(),
        }
    }

    pub fn index_size(&self) -> usize {
        self.index.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    pub question: String,
    pub answer: String,
    pub sources: Vec<String>,
    pub context_length: usize,
}

// ============================================================================
// Hybrid Search
// ============================================================================

pub struct HybridSearch {
    vector_index: VectorIndex,
    keyword_index: HashMap<String, Vec<usize>>,
    alpha: f32, // Weight for vector vs keyword
}

impl HybridSearch {
    pub fn new(dimension: usize, alpha: f32) -> Self {
        Self {
            vector_index: VectorIndex::new(dimension),
            keyword_index: HashMap::new(),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    pub fn add(&mut self, chunk: Chunk, embedding: Embedding) -> Result<(), RagError> {
        let idx = self.vector_index.len();

        // Add to keyword index
        for word in chunk.text.to_lowercase().split_whitespace() {
            self.keyword_index
                .entry(word.to_string())
                .or_insert_with(Vec::new)
                .push(idx);
        }

        self.vector_index.add(chunk, embedding)
    }

    pub fn search(
        &self,
        query: &str,
        query_embedding: &Embedding,
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        // Vector search
        let vector_results = self.vector_index.search(query_embedding, top_k * 2);

        // Keyword search
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let mut keyword_scores: HashMap<usize, f32> = HashMap::new();

        for word in &query_words {
            if let Some(indices) = self.keyword_index.get(*word) {
                for &idx in indices {
                    *keyword_scores.entry(idx).or_insert(0.0) += 1.0 / query_words.len() as f32;
                }
            }
        }

        // Combine scores
        let mut combined: HashMap<usize, (RetrievalResult, f32)> = HashMap::new();

        for result in vector_results {
            let chunk_idx = result.rank - 1;
            let vector_score = result.score * self.alpha;
            let keyword_score = keyword_scores.get(&chunk_idx).unwrap_or(&0.0) * (1.0 - self.alpha);
            combined.insert(chunk_idx, (result, vector_score + keyword_score));
        }

        let mut results: Vec<_> = combined.into_values().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (mut result, score))| {
                result.score = score;
                result.rank = rank + 1;
                result
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.vector_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector_index.is_empty()
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     RAG Pipeline with trueno-rag - Course 4, Week 2           ║");
    println!("║     Chunking, Embeddings, Retrieval, Reranking                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Step 1: Create Documents
    println!("\n📄 Step 1: Document Ingestion");
    let documents = vec![
        Document::new("doc1", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to identify patterns in data and make predictions without being explicitly programmed.")
            .with_metadata("source", "ml-guide"),
        Document::new("doc2", "Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can learn hierarchical representations of data, enabling them to solve complex problems.")
            .with_metadata("source", "dl-guide"),
        Document::new("doc3", "Natural language processing combines linguistics and machine learning to enable computers to understand human language. Applications include translation, sentiment analysis, and chatbots.")
            .with_metadata("source", "nlp-guide"),
    ];

    println!("   Documents to ingest: {}", documents.len());

    // Step 2: Chunking
    println!("\n✂️  Step 2: Chunking");
    let chunker = Chunker::new(ChunkerConfig::default().with_size(20).with_overlap(5));

    for doc in &documents {
        let chunks = chunker.chunk(doc);
        println!("   {} -> {} chunks", doc.id, chunks.len());
        if let Some(first) = chunks.first() {
            println!(
                "     First chunk: \"{}...\"",
                &first.text[..50.min(first.text.len())]
            );
        }
    }

    // Step 3: Build RAG Pipeline
    println!("\n🔧 Step 3: Build RAG Pipeline");
    let mut pipeline = RagPipeline::new(128)
        .with_chunker(chunker)
        .with_reranker(Reranker::new("cross-encoder"))
        .with_retrieval_k(5)
        .with_rerank_k(3);

    let chunk_count = pipeline.ingest(documents).unwrap();
    println!("   Indexed {} chunks", chunk_count);
    println!("   Embedding dimension: {}", 128);
    println!("   Retrieval top-k: 5");
    println!("   Rerank top-k: 3");

    // Step 4: Query
    println!("\n🔍 Step 4: RAG Queries");
    let queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are NLP applications?",
    ];

    for query in &queries {
        println!("\n   Q: {}", query);
        let response = pipeline.query(query);
        println!("   A: {}", response.answer);
        println!("   Sources: {:?}", response.sources);
        println!("   Context length: {} chars", response.context_length);
    }

    // Step 5: Hybrid Search
    println!("\n🔀 Step 5: Hybrid Search (Vector + Keyword)");
    let embedding_model = EmbeddingModel::new(128, "all-MiniLM-L6-v2");
    let mut hybrid = HybridSearch::new(128, 0.7);

    let doc = Document::new(
        "hybrid-doc",
        "Machine learning algorithms process data patterns",
    );
    let chunks = Chunker::default().chunk(&doc);
    for chunk in chunks {
        let embedding = embedding_model.embed(&chunk.text);
        hybrid.add(chunk, embedding).unwrap();
    }

    let query = "machine learning patterns";
    let query_emb = embedding_model.embed(query);
    let results = hybrid.search(query, &query_emb, 3);

    println!("   Query: \"{}\"", query);
    println!("   Alpha (vector weight): 0.7");
    for r in &results {
        println!("   #{}: score={:.3}, chunk={}", r.rank, r.score, r.chunk.id);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Document chunking with overlap");
    println!("  • Dense embeddings for semantic search");
    println!("  • Vector similarity retrieval");
    println!("  • Cross-encoder reranking");
    println!("  • Hybrid search (vector + keyword)");
    println!();
    println!("Sovereign AI Stack: trueno-rag pipeline");
    println!("Databricks equivalent: Vector Search + RAG");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Document Tests
    // ========================================================================

    #[test]
    fn test_document_new() {
        let doc = Document::new("id1", "content");
        assert_eq!(doc.id, "id1");
        assert_eq!(doc.content, "content");
    }

    #[test]
    fn test_document_with_metadata() {
        let doc = Document::new("id1", "content").with_metadata("key", "value");
        assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_document_clone() {
        let doc = Document::new("id1", "content");
        let cloned = doc.clone();
        assert_eq!(doc.id, cloned.id);
    }

    // ========================================================================
    // Chunker Tests
    // ========================================================================

    #[test]
    fn test_chunker_default() {
        let chunker = Chunker::default();
        assert_eq!(chunker.config().chunk_size, 512);
    }

    #[test]
    fn test_chunker_chunk() {
        let chunker = Chunker::new(ChunkerConfig::default().with_size(5).with_overlap(2));
        let doc = Document::new("test", "one two three four five six seven eight nine ten");
        let chunks = chunker.chunk(&doc);

        assert!(chunks.len() >= 2);
        assert!(chunks[0].text.split_whitespace().count() <= 5);
    }

    #[test]
    fn test_chunker_chunk_ids() {
        let chunker = Chunker::default();
        let doc = Document::new("doc1", "hello world this is a test");
        let chunks = chunker.chunk(&doc);

        assert!(chunks[0].id.contains("doc1-chunk-0"));
    }

    #[test]
    fn test_chunk_clone() {
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "text".to_string(),
            start_idx: 0,
            end_idx: 1,
            metadata: HashMap::new(),
        };
        let cloned = chunk.clone();
        assert_eq!(chunk.id, cloned.id);
    }

    // ========================================================================
    // Embedding Tests
    // ========================================================================

    #[test]
    fn test_embedding_new() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dimension, 3);
    }

    #[test]
    fn test_embedding_zeros() {
        let emb = Embedding::zeros(10);
        assert_eq!(emb.dimension, 10);
        assert!(emb.vector.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embedding_similarity_identical() {
        let emb1 = Embedding::new(vec![1.0, 0.0, 0.0]);
        let emb2 = Embedding::new(vec![1.0, 0.0, 0.0]);
        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let emb1 = Embedding::new(vec![1.0, 0.0]);
        let emb2 = Embedding::new(vec![0.0, 1.0]);
        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();
        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // EmbeddingModel Tests
    // ========================================================================

    #[test]
    fn test_embedding_model_embed() {
        let model = EmbeddingModel::new(64, "test");
        let emb = model.embed("hello world");
        assert_eq!(emb.dimension, 64);
    }

    #[test]
    fn test_embedding_model_batch() {
        let model = EmbeddingModel::new(32, "test");
        let embeddings = model.embed_batch(&["text1", "text2", "text3"]);
        assert_eq!(embeddings.len(), 3);
    }

    #[test]
    fn test_embedding_model_dimension() {
        let model = EmbeddingModel::new(128, "test");
        assert_eq!(model.dimension(), 128);
    }

    // ========================================================================
    // VectorIndex Tests
    // ========================================================================

    #[test]
    fn test_vector_index_new() {
        let index = VectorIndex::new(64);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_add() {
        let mut index = VectorIndex::new(4);
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "text".to_string(),
            start_idx: 0,
            end_idx: 1,
            metadata: HashMap::new(),
        };
        let emb = Embedding::new(vec![1.0, 2.0, 3.0, 4.0]);
        index.add(chunk, emb).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_index_dimension_mismatch() {
        let mut index = VectorIndex::new(4);
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "text".to_string(),
            start_idx: 0,
            end_idx: 1,
            metadata: HashMap::new(),
        };
        let emb = Embedding::new(vec![1.0, 2.0]); // Wrong dimension
        let result = index.add(chunk, emb);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_search() {
        let mut index = VectorIndex::new(4);

        for i in 0..3 {
            let chunk = Chunk {
                id: format!("c{}", i),
                doc_id: "d1".to_string(),
                text: format!("text {}", i),
                start_idx: 0,
                end_idx: 1,
                metadata: HashMap::new(),
            };
            let emb = Embedding::new(vec![i as f32, 0.0, 0.0, 1.0]);
            index.add(chunk, emb).unwrap();
        }

        let query = Embedding::new(vec![2.0, 0.0, 0.0, 1.0]);
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
    }

    // ========================================================================
    // Reranker Tests
    // ========================================================================

    #[test]
    fn test_reranker_new() {
        let reranker = Reranker::new("cross-encoder");
        assert_eq!(reranker.model_name(), "cross-encoder");
    }

    #[test]
    fn test_reranker_rerank() {
        let reranker = Reranker::new("test");
        let results = vec![RetrievalResult {
            chunk: Chunk {
                id: "c1".to_string(),
                doc_id: "d1".to_string(),
                text: "machine learning data".to_string(),
                start_idx: 0,
                end_idx: 1,
                metadata: HashMap::new(),
            },
            score: 0.8,
            rank: 1,
        }];

        let reranked = reranker.rerank("machine learning", &results, 1);
        assert_eq!(reranked.len(), 1);
    }

    // ========================================================================
    // RagPipeline Tests
    // ========================================================================

    #[test]
    fn test_rag_pipeline_new() {
        let pipeline = RagPipeline::new(64);
        assert_eq!(pipeline.index_size(), 0);
    }

    #[test]
    fn test_rag_pipeline_ingest() {
        let mut pipeline =
            RagPipeline::new(64).with_chunker(Chunker::new(ChunkerConfig::default().with_size(10)));

        let docs = vec![Document::new("d1", "hello world this is a test document")];
        let count = pipeline.ingest(docs).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_rag_pipeline_query() {
        let mut pipeline =
            RagPipeline::new(64).with_chunker(Chunker::new(ChunkerConfig::default().with_size(10)));

        let docs = vec![Document::new(
            "d1",
            "machine learning algorithms process data",
        )];
        pipeline.ingest(docs).unwrap();

        let response = pipeline.query("machine learning");
        assert!(!response.answer.is_empty());
    }

    // ========================================================================
    // HybridSearch Tests
    // ========================================================================

    #[test]
    fn test_hybrid_search_new() {
        let hybrid = HybridSearch::new(64, 0.7);
        assert!(hybrid.is_empty());
    }

    #[test]
    fn test_hybrid_search_add() {
        let mut hybrid = HybridSearch::new(4, 0.5);
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "hello world".to_string(),
            start_idx: 0,
            end_idx: 1,
            metadata: HashMap::new(),
        };
        let emb = Embedding::new(vec![1.0, 2.0, 3.0, 4.0]);
        hybrid.add(chunk, emb).unwrap();
        assert_eq!(hybrid.len(), 1);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_chunking() {
        let err = RagError::Chunking("failed".to_string());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_error_embedding() {
        let err = RagError::Embedding("dimension".to_string());
        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn test_error_retrieval() {
        let err = RagError::Retrieval("not found".to_string());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_error_debug() {
        let err = RagError::Reranking("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Reranking"));
    }
}
