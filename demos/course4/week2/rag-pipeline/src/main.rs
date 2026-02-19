//! RAG Pipeline Demo - Course 4 Week 2
//!
//! Demonstrates Retrieval-Augmented Generation concepts that map to
//! Databricks RAG features. Shows chunking, retrieval, and context injection.

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
    #[error("Retrieval error: {0}")]
    Retrieval(String),
    #[error("Generation error: {0}")]
    Generation(String),
}

// ============================================================================
// Document Chunking
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub source_id: String,
    pub start_idx: usize,
    pub end_idx: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub separator: String,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 50,
            separator: "\n".to_string(),
        }
    }
}

impl ChunkingConfig {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
            separator: "\n".to_string(),
        }
    }

    pub fn with_separator(mut self, sep: &str) -> Self {
        self.separator = sep.to_string();
        self
    }
}

/// Simple text chunker
#[derive(Debug)]
pub struct TextChunker {
    config: ChunkingConfig,
}

impl TextChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk text into overlapping segments
    pub fn chunk(&self, source_id: &str, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut chunk_num = 0;

        while start < chars.len() {
            let end = (start + self.config.chunk_size).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            chunks.push(Chunk {
                id: format!("{}-{}", source_id, chunk_num),
                content,
                source_id: source_id.to_string(),
                start_idx: start,
                end_idx: end,
                metadata: HashMap::new(),
            });

            chunk_num += 1;

            // Move start with overlap
            if end >= chars.len() {
                break;
            }
            start = end.saturating_sub(self.config.overlap);
            if start >= chars.len() {
                break;
            }
        }

        chunks
    }

    /// Chunk by sentences/paragraphs
    pub fn chunk_by_separator(&self, source_id: &str, text: &str) -> Vec<Chunk> {
        let paragraphs: Vec<&str> = text.split(&self.config.separator).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_num = 0;
        let mut start_idx = 0;

        for para in paragraphs {
            if current_chunk.len() + para.len() > self.config.chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(Chunk {
                    id: format!("{}-{}", source_id, chunk_num),
                    content: current_chunk.trim().to_string(),
                    source_id: source_id.to_string(),
                    start_idx,
                    end_idx: start_idx + current_chunk.len(),
                    metadata: HashMap::new(),
                });
                chunk_num += 1;
                start_idx += current_chunk.len();
                current_chunk.clear();
            }
            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(para);
        }

        if !current_chunk.is_empty() {
            chunks.push(Chunk {
                id: format!("{}-{}", source_id, chunk_num),
                content: current_chunk.trim().to_string(),
                source_id: source_id.to_string(),
                start_idx,
                end_idx: start_idx + current_chunk.len(),
                metadata: HashMap::new(),
            });
        }

        chunks
    }

    pub fn config(&self) -> &ChunkingConfig {
        &self.config
    }
}

// ============================================================================
// Simple Embedding (for demo)
// ============================================================================

fn simple_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    for (i, c) in text.chars().enumerate() {
        let idx = (c as usize + i) % dim;
        embedding[idx] += 1.0;
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
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

// ============================================================================
// Retriever
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
}

#[derive(Debug)]
pub struct Retriever {
    chunks: Vec<(Chunk, Vec<f32>)>,
    dimension: usize,
}

impl Retriever {
    pub fn new(dimension: usize) -> Self {
        Self {
            chunks: Vec::new(),
            dimension,
        }
    }

    /// Index a chunk with its embedding
    pub fn add(&mut self, chunk: Chunk) {
        let embedding = simple_embed(&chunk.content, self.dimension);
        self.chunks.push((chunk, embedding));
    }

    /// Index multiple chunks
    pub fn add_chunks(&mut self, chunks: Vec<Chunk>) {
        for chunk in chunks {
            self.add(chunk);
        }
    }

    /// Retrieve top-k relevant chunks
    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        let query_embedding = simple_embed(query, self.dimension);

        let mut results: Vec<_> = self
            .chunks
            .iter()
            .map(|(chunk, embedding)| RetrievalResult {
                chunk: chunk.clone(),
                score: cosine_similarity(&query_embedding, embedding),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

// ============================================================================
// RAG Pipeline
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub top_k: usize,
    pub min_score: f32,
    pub context_template: String,
    pub prompt_template: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 3,
            min_score: 0.0,
            context_template: "Context:\n{context}\n\n".to_string(),
            prompt_template: "{context}Question: {question}\n\nAnswer:".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct RagPipeline {
    retriever: Retriever,
    config: RagConfig,
}

impl RagPipeline {
    pub fn new(retriever: Retriever, config: RagConfig) -> Self {
        Self { retriever, config }
    }

    /// Build augmented prompt with retrieved context
    pub fn build_prompt(&self, query: &str) -> RagPrompt {
        let results = self.retriever.retrieve(query, self.config.top_k);

        let relevant: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        let context = relevant
            .iter()
            .map(|r| format!("- {}", r.chunk.content))
            .collect::<Vec<_>>()
            .join("\n");

        let formatted_context = self.config.context_template.replace("{context}", &context);
        let prompt = self
            .config
            .prompt_template
            .replace("{context}", &formatted_context)
            .replace("{question}", query);

        RagPrompt {
            prompt,
            retrieved_chunks: relevant,
            query: query.to_string(),
        }
    }

    pub fn retriever(&self) -> &Retriever {
        &self.retriever
    }
}

#[derive(Debug, Clone)]
pub struct RagPrompt {
    pub prompt: String,
    pub retrieved_chunks: Vec<RetrievalResult>,
    pub query: String,
}

impl RagPrompt {
    pub fn context_count(&self) -> usize {
        self.retrieved_chunks.len()
    }
}

// ============================================================================
// Reranker
// ============================================================================

#[derive(Debug)]
pub struct Reranker {
    boost_keywords: Vec<String>,
}

impl Reranker {
    pub fn new() -> Self {
        Self {
            boost_keywords: Vec::new(),
        }
    }

    pub fn with_keywords(mut self, keywords: Vec<&str>) -> Self {
        self.boost_keywords = keywords.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Rerank results based on additional signals
    pub fn rerank(&self, query: &str, mut results: Vec<RetrievalResult>) -> Vec<RetrievalResult> {
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();

        for result in &mut results {
            let content_lower = result.chunk.content.to_lowercase();

            // Boost exact phrase match
            if content_lower.contains(&query_lower) {
                result.score *= 1.5;
            }

            // Boost keyword matches
            for keyword in &self.boost_keywords {
                if content_lower.contains(&keyword.to_lowercase()) {
                    result.score *= 1.2;
                }
            }

            // Boost word overlap
            let content_words: std::collections::HashSet<&str> =
                content_lower.split_whitespace().collect();
            let overlap = query_words.intersection(&content_words).count();
            result.score *= 1.0 + (overlap as f32 * 0.1);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Hybrid Search
// ============================================================================

#[derive(Debug)]
pub struct HybridSearch {
    retriever: Retriever,
    keyword_weight: f32,
}

impl HybridSearch {
    pub fn new(retriever: Retriever) -> Self {
        Self {
            retriever,
            keyword_weight: 0.3,
        }
    }

    pub fn with_keyword_weight(mut self, weight: f32) -> Self {
        self.keyword_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Hybrid search combining semantic and keyword
    pub fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        let semantic_results = self.retriever.retrieve(query, top_k * 2);

        let query_words: Vec<&str> = query.split_whitespace().collect();

        let mut results: Vec<_> = semantic_results
            .into_iter()
            .map(|mut r| {
                // Add keyword score
                let keyword_score: f32 = query_words
                    .iter()
                    .map(|w| {
                        if r.chunk.content.to_lowercase().contains(&w.to_lowercase()) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
                    / query_words.len().max(1) as f32;

                r.score =
                    (1.0 - self.keyword_weight) * r.score + self.keyword_weight * keyword_score;
                r
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
    println!("║     RAG Pipeline Demo - Course 4 Week 2                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Document Chunking
    println!("📄 Step 1: Document Chunking");
    let config = ChunkingConfig::new(100, 20);
    let chunker = TextChunker::new(config);

    let document = "Machine learning is a subset of artificial intelligence. \
        It focuses on building systems that learn from data. \
        Deep learning uses neural networks with many layers. \
        These networks can learn complex patterns automatically.";

    let chunks = chunker.chunk("doc1", document);
    println!("   Original: {} chars", document.len());
    println!("   Chunks created: {}", chunks.len());
    for chunk in &chunks {
        println!(
            "     [{}] {} chars: \"{}...\"",
            chunk.id,
            chunk.content.len(),
            &chunk.content[..chunk.content.len().min(30)]
        );
    }
    println!();

    // Step 2: Indexing
    println!("🗂️ Step 2: Indexing Chunks");
    let mut retriever = Retriever::new(64);

    let documents = [
        (
            "MLFlow is an open-source platform for ML lifecycle management.",
            "mlops",
        ),
        (
            "Feature engineering transforms raw data into model inputs.",
            "mlops",
        ),
        (
            "RAG combines retrieval with generation for better answers.",
            "genai",
        ),
        (
            "Vector databases store embeddings for similarity search.",
            "genai",
        ),
        (
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "genai",
        ),
    ];

    for (text, category) in &documents {
        let mut chunk = Chunk {
            id: format!("chunk-{}", retriever.len()),
            content: text.to_string(),
            source_id: "corpus".to_string(),
            start_idx: 0,
            end_idx: text.len(),
            metadata: HashMap::new(),
        };
        chunk
            .metadata
            .insert("category".to_string(), category.to_string());
        retriever.add(chunk);
    }

    println!("   Indexed {} chunks\n", retriever.len());

    // Step 3: Retrieval
    println!("🔍 Step 3: Retrieval");
    let query = "How does RAG work with vectors?";
    let results = retriever.retrieve(query, 3);

    println!("   Query: \"{}\"", query);
    println!("   Retrieved:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "     {}. (score: {:.4}) \"{}\"",
            i + 1,
            result.score,
            result.chunk.content
        );
    }
    println!();

    // Step 4: RAG Pipeline
    println!("🔗 Step 4: RAG Pipeline");
    let rag = RagPipeline::new(retriever, RagConfig::default());
    let rag_prompt = rag.build_prompt(query);

    println!("   Context chunks: {}", rag_prompt.context_count());
    println!("   Augmented prompt:\n   ---");
    for line in rag_prompt.prompt.lines() {
        println!("   {}", line);
    }
    println!("   ---\n");

    // Step 5: Reranking
    println!("📊 Step 5: Reranking");
    let reranker = Reranker::new().with_keywords(vec!["RAG", "vector"]);

    let mut retriever2 = Retriever::new(64);
    for (text, _) in &documents {
        let chunk = Chunk {
            id: format!("c-{}", retriever2.len()),
            content: text.to_string(),
            source_id: "test".to_string(),
            start_idx: 0,
            end_idx: text.len(),
            metadata: HashMap::new(),
        };
        retriever2.add(chunk);
    }

    let initial_results = retriever2.retrieve(query, 5);
    let reranked = reranker.rerank(query, initial_results.clone());

    println!("   Before reranking:");
    for (i, r) in initial_results.iter().take(3).enumerate() {
        println!(
            "     {}. (score: {:.4}) \"{}...\"",
            i + 1,
            r.score,
            &r.chunk.content[..30.min(r.chunk.content.len())]
        );
    }
    println!("   After reranking:");
    for (i, r) in reranked.iter().take(3).enumerate() {
        println!(
            "     {}. (score: {:.4}) \"{}...\"",
            i + 1,
            r.score,
            &r.chunk.content[..30.min(r.chunk.content.len())]
        );
    }
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo Complete!");
    println!();
    println!("Key concepts demonstrated:");
    println!("  • Document chunking with overlap");
    println!("  • Vector-based retrieval");
    println!("  • RAG prompt construction");
    println!("  • Reranking for improved relevance");
    println!("  • Hybrid search (semantic + keyword)");
    println!();
    println!("Databricks equivalent: Vector Search, Foundation Model RAG");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_basic() {
        let chunker = TextChunker::new(ChunkingConfig::new(10, 2));
        let chunks = chunker.chunk("test", "hello world");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_overlap() {
        let chunker = TextChunker::new(ChunkingConfig::new(20, 5));
        let chunks = chunker.chunk("test", "a".repeat(50).as_str());
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_chunk_metadata() {
        let mut chunk = Chunk {
            id: "1".to_string(),
            content: "test".to_string(),
            source_id: "src".to_string(),
            start_idx: 0,
            end_idx: 4,
            metadata: HashMap::new(),
        };
        chunk
            .metadata
            .insert("key".to_string(), "value".to_string());
        assert_eq!(chunk.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_retriever_add() {
        let mut retriever = Retriever::new(32);
        let chunk = Chunk {
            id: "1".to_string(),
            content: "test".to_string(),
            source_id: "src".to_string(),
            start_idx: 0,
            end_idx: 4,
            metadata: HashMap::new(),
        };
        retriever.add(chunk);
        assert_eq!(retriever.len(), 1);
    }

    #[test]
    fn test_retriever_retrieve() {
        let mut retriever = Retriever::new(32);
        retriever.add(Chunk {
            id: "1".to_string(),
            content: "machine learning".to_string(),
            source_id: "src".to_string(),
            start_idx: 0,
            end_idx: 16,
            metadata: HashMap::new(),
        });

        let results = retriever.retrieve("learning", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_rag_pipeline_build_prompt() {
        let mut retriever = Retriever::new(32);
        retriever.add(Chunk {
            id: "1".to_string(),
            content: "RAG is retrieval augmented generation".to_string(),
            source_id: "src".to_string(),
            start_idx: 0,
            end_idx: 37,
            metadata: HashMap::new(),
        });

        let rag = RagPipeline::new(retriever, RagConfig::default());
        let prompt = rag.build_prompt("What is RAG?");

        assert!(prompt.prompt.contains("Question:"));
        assert!(prompt.context_count() > 0);
    }

    #[test]
    fn test_rag_config_default() {
        let config = RagConfig::default();
        assert_eq!(config.top_k, 3);
    }

    #[test]
    fn test_reranker_basic() {
        let reranker = Reranker::new();
        let results = vec![RetrievalResult {
            chunk: Chunk {
                id: "1".to_string(),
                content: "test content".to_string(),
                source_id: "src".to_string(),
                start_idx: 0,
                end_idx: 12,
                metadata: HashMap::new(),
            },
            score: 0.5,
        }];

        let reranked = reranker.rerank("test", results);
        assert!(!reranked.is_empty());
    }

    #[test]
    fn test_reranker_keyword_boost() {
        let reranker = Reranker::new().with_keywords(vec!["boost"]);
        let results = vec![
            RetrievalResult {
                chunk: Chunk {
                    id: "1".to_string(),
                    content: "no match here".to_string(),
                    source_id: "src".to_string(),
                    start_idx: 0,
                    end_idx: 13,
                    metadata: HashMap::new(),
                },
                score: 0.5,
            },
            RetrievalResult {
                chunk: Chunk {
                    id: "2".to_string(),
                    content: "boost this result".to_string(),
                    source_id: "src".to_string(),
                    start_idx: 0,
                    end_idx: 17,
                    metadata: HashMap::new(),
                },
                score: 0.5,
            },
        ];

        let reranked = reranker.rerank("query", results);
        assert!(reranked[0].chunk.content.contains("boost"));
    }

    #[test]
    fn test_hybrid_search() {
        let mut retriever = Retriever::new(32);
        retriever.add(Chunk {
            id: "1".to_string(),
            content: "machine learning algorithms".to_string(),
            source_id: "src".to_string(),
            start_idx: 0,
            end_idx: 27,
            metadata: HashMap::new(),
        });

        let hybrid = HybridSearch::new(retriever);
        let results = hybrid.search("machine", 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rag_error() {
        let err = RagError::Chunking("invalid".to_string());
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn test_simple_embed() {
        let emb = simple_embed("test", 32);
        assert_eq!(emb.len(), 32);
    }

    #[test]
    fn test_cosine_similarity_same() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }
}
