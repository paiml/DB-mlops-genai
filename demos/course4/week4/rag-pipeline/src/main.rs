//! RAG Pipeline Demo - Course 4 Week 4
//!
//! Demonstrates Retrieval-Augmented Generation that maps to Databricks RAG solutions.
//! Shows document processing, retrieval, and generation integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Retrieval error: {0}")]
    Retrieval(String),
    #[error("Generation error: {0}")]
    Generation(String),
    #[error("Document error: {0}")]
    Document(String),
}

// ============================================================================
// Document Processing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
}

impl Document {
    pub fn new(id: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            chunks: Vec::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn chunk(&mut self, chunk_size: usize, overlap: usize) {
        let words: Vec<&str> = self.content.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut char_pos = 0;
        let mut word_idx = 0;

        while word_idx < words.len() {
            let end_idx = (word_idx + chunk_size).min(words.len());
            let chunk_words = &words[word_idx..end_idx];
            let chunk_text = chunk_words.join(" ");

            let start_char = self.content[char_pos..]
                .find(chunk_words[0])
                .map(|i| char_pos + i)
                .unwrap_or(char_pos);

            let end_char = start_char + chunk_text.len();

            chunks.push(Chunk {
                id: format!("{}-chunk-{}", self.id, chunks.len()),
                text: chunk_text,
                start_char,
                end_char,
            });

            if end_idx >= words.len() {
                break;
            }

            word_idx += chunk_size.saturating_sub(overlap);
            char_pos = end_char;
        }

        self.chunks = chunks;
    }
}

// ============================================================================
// Simple Embeddings
// ============================================================================

#[derive(Debug, Clone)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    pub fn from_text(text: &str, dim: usize) -> Self {
        let mut values = vec![0.0; dim];
        for (i, word) in text.split_whitespace().enumerate() {
            for (j, c) in word.chars().enumerate() {
                let idx = ((c as usize) * (i + 1) + j) % dim;
                values[idx] += 0.1;
            }
        }
        // Normalize
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut values {
                *v /= norm;
            }
        }
        Self(values)
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let norm_self: f32 = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_other: f32 = other.0.iter().map(|v| v * v).sum::<f32>().sqrt();

        if norm_self < 1e-10 || norm_other < 1e-10 {
            return 0.0;
        }
        dot / (norm_self * norm_other)
    }
}

// ============================================================================
// Vector Store
// ============================================================================

#[derive(Debug)]
pub struct VectorStore {
    dimension: usize,
    chunks: Vec<Chunk>,
    embeddings: Vec<Embedding>,
}

impl VectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            chunks: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    pub fn add_document(&mut self, document: &Document) {
        for chunk in &document.chunks {
            let embedding = Embedding::from_text(&chunk.text, self.dimension);
            self.chunks.push(chunk.clone());
            self.embeddings.push(embedding);
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<RetrievedChunk> {
        let query_emb = Embedding::from_text(query, self.dimension);

        let mut scored: Vec<_> = self
            .chunks
            .iter()
            .zip(self.embeddings.iter())
            .map(|(chunk, emb)| {
                let score = query_emb.cosine_similarity(emb);
                (chunk.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, (chunk, score))| RetrievedChunk {
                chunk,
                score,
                rank: i + 1,
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
pub struct RetrievedChunk {
    pub chunk: Chunk,
    pub score: f32,
    pub rank: usize,
}

// ============================================================================
// RAG Pipeline
// ============================================================================

#[derive(Debug)]
pub struct RagPipeline {
    vector_store: VectorStore,
    top_k: usize,
    prompt_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    pub query: String,
    pub answer: String,
    pub sources: Vec<Source>,
    pub context_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub chunk_id: String,
    pub text_preview: String,
    pub score: f32,
}

impl RagPipeline {
    pub fn new(dimension: usize) -> Self {
        Self {
            vector_store: VectorStore::new(dimension),
            top_k: 3,
            prompt_template: Self::default_template(),
        }
    }

    fn default_template() -> String {
        r#"Answer the question based on the context below. If the context doesn't contain relevant information, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"#.to_string()
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_template(mut self, template: &str) -> Self {
        self.prompt_template = template.to_string();
        self
    }

    pub fn ingest(&mut self, document: Document) {
        self.vector_store.add_document(&document);
    }

    pub fn query(&self, question: &str) -> Result<RagResponse, RagError> {
        // 1. Retrieve relevant chunks
        let retrieved = self.vector_store.search(question, self.top_k);

        if retrieved.is_empty() {
            return Err(RagError::Retrieval(
                "No relevant documents found".to_string(),
            ));
        }

        // 2. Build context
        let context = retrieved
            .iter()
            .map(|r| r.chunk.text.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        // 3. Format prompt
        let _prompt = self
            .prompt_template
            .replace("{context}", &context)
            .replace("{question}", question);

        // 4. Generate answer (simulated)
        let answer = self.simulate_generation(question, &context);

        // 5. Build response
        let sources = retrieved
            .iter()
            .map(|r| Source {
                chunk_id: r.chunk.id.clone(),
                text_preview: r.chunk.text.chars().take(50).collect::<String>() + "...",
                score: r.score,
            })
            .collect();

        Ok(RagResponse {
            query: question.to_string(),
            answer,
            sources,
            context_used: context,
        })
    }

    fn simulate_generation(&self, question: &str, context: &str) -> String {
        // Pattern matching for demo purposes
        let q_lower = question.to_lowercase();
        let c_lower = context.to_lowercase();

        if q_lower.contains("what") && c_lower.contains("machine learning") {
            "Based on the context, machine learning is a field of AI that enables systems to learn from data.".to_string()
        } else if q_lower.contains("how") {
            "According to the retrieved documents, the process involves multiple steps as described in the context.".to_string()
        } else if c_lower.contains(&q_lower.split_whitespace().next().unwrap_or("")) {
            format!(
                "Based on the retrieved context, I found relevant information about your query: {}",
                &context[..100.min(context.len())]
            )
        } else {
            "Based on the provided context, I can answer your question using the retrieved information.".to_string()
        }
    }

    pub fn document_count(&self) -> usize {
        self.vector_store.len()
    }
}

// ============================================================================
// RAG Evaluation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagMetrics {
    pub retrieval_precision: f32,
    pub context_relevance: f32,
    pub answer_faithfulness: f32,
}

impl RagMetrics {
    pub fn evaluate(response: &RagResponse, ground_truth: Option<&str>) -> Self {
        // Simplified metrics for demo
        let retrieval_precision = if !response.sources.is_empty() {
            response.sources.iter().map(|s| s.score).sum::<f32>() / response.sources.len() as f32
        } else {
            0.0
        };

        let context_relevance = if response
            .context_used
            .contains(&response.query.split_whitespace().next().unwrap_or(""))
        {
            0.8
        } else {
            0.5
        };

        let answer_faithfulness = match ground_truth {
            Some(gt) => {
                let answer_lower = response.answer.to_lowercase();
                let gt_lower = gt.to_lowercase();
                let answer_words: std::collections::HashSet<_> =
                    answer_lower.split_whitespace().collect();
                let gt_words: std::collections::HashSet<_> = gt_lower.split_whitespace().collect();
                let overlap = answer_words.intersection(&gt_words).count();
                overlap as f32 / gt_words.len().max(1) as f32
            }
            None => 0.7, // Default if no ground truth
        };

        Self {
            retrieval_precision,
            context_relevance,
            answer_faithfulness,
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     RAG Pipeline Demo - Course 4 Week 4                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Create documents
    println!("📄 Step 1: Document Ingestion");

    let docs = vec![
        Document::new("doc1", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to identify patterns in data and make predictions or decisions without being explicitly programmed for each task.")
            .with_metadata("source", "ml-guide"),
        Document::new("doc2", "Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can learn hierarchical representations of data, enabling them to solve complex problems like image recognition and natural language processing.")
            .with_metadata("source", "dl-guide"),
        Document::new("doc3", "Natural language processing (NLP) is a field that combines linguistics and machine learning to enable computers to understand, interpret, and generate human language. Applications include translation, sentiment analysis, and chatbots.")
            .with_metadata("source", "nlp-guide"),
    ];

    let mut rag = RagPipeline::new(128).with_top_k(2);

    for mut doc in docs {
        println!("   Processing: {} ({} chars)", doc.id, doc.content.len());
        doc.chunk(30, 10);
        println!("     Created {} chunks", doc.chunks.len());
        rag.ingest(doc);
    }
    println!("   Total chunks indexed: {}\n", rag.document_count());

    // Step 2: Query the RAG system
    println!("🔍 Step 2: RAG Queries");

    let queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are NLP applications?",
    ];

    for query in &queries {
        println!("   Q: {}", query);

        match rag.query(query) {
            Ok(response) => {
                println!("   A: {}", response.answer);
                println!("   Sources:");
                for source in &response.sources {
                    println!("     - {} [score={:.3}]", source.chunk_id, source.score);
                }
            }
            Err(e) => println!("   Error: {}", e),
        }
        println!();
    }

    // Step 3: Evaluate
    println!("📊 Step 3: RAG Evaluation");

    if let Ok(response) = rag.query("What is machine learning?") {
        let ground_truth = "Machine learning enables systems to learn from data";
        let metrics = RagMetrics::evaluate(&response, Some(ground_truth));

        println!("   Metrics:");
        println!(
            "     Retrieval Precision: {:.2}",
            metrics.retrieval_precision
        );
        println!("     Context Relevance: {:.2}", metrics.context_relevance);
        println!(
            "     Answer Faithfulness: {:.2}",
            metrics.answer_faithfulness
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
    println!("  • Context-aware generation");
    println!("  • RAG evaluation metrics");
    println!();
    println!("Databricks equivalent: Vector Search + Foundation Models");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Document Tests
    // =========================================================================

    #[test]
    fn test_document_new() {
        let doc = Document::new("doc1", "Hello world");
        assert_eq!(doc.id, "doc1");
        assert_eq!(doc.content, "Hello world");
        assert!(doc.chunks.is_empty());
    }

    #[test]
    fn test_document_with_metadata() {
        let doc = Document::new("doc1", "content").with_metadata("key", "value");
        assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_document_chunking() {
        let mut doc = Document::new("test", "one two three four five six seven eight nine ten");
        doc.chunk(5, 2);

        assert!(doc.chunks.len() >= 2);
        for chunk in &doc.chunks {
            assert!(chunk.text.split_whitespace().count() <= 5);
        }
    }

    #[test]
    fn test_document_chunking_short() {
        let mut doc = Document::new("test", "one two three");
        doc.chunk(10, 2);
        assert_eq!(doc.chunks.len(), 1);
    }

    #[test]
    fn test_document_chunk_ids() {
        let mut doc = Document::new("doc1", "hello world this is a test document");
        doc.chunk(10, 0); // Single chunk
        assert!(doc.chunks[0].id.contains("doc1-chunk-0"));
    }

    #[test]
    fn test_chunk_serialization() {
        let chunk = Chunk {
            id: "c1".to_string(),
            text: "test".to_string(),
            start_char: 0,
            end_char: 4,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let restored: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk.id, restored.id);
    }

    #[test]
    fn test_document_clone() {
        let doc = Document::new("test", "content").with_metadata("k", "v");
        let cloned = doc.clone();
        assert_eq!(doc.id, cloned.id);
        assert_eq!(doc.metadata, cloned.metadata);
    }

    // =========================================================================
    // Embedding Tests
    // =========================================================================

    #[test]
    fn test_embedding_new() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.0.len(), 3);
    }

    #[test]
    fn test_embedding_from_text() {
        let emb = Embedding::from_text("hello world", 32);
        assert_eq!(emb.0.len(), 32);
        // Should be normalized
        let norm: f32 = emb.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_similarity() {
        let e1 = Embedding::from_text("machine learning", 64);
        let e2 = Embedding::from_text("machine learning", 64);
        let sim = e1.cosine_similarity(&e2);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_similarity_different() {
        let e1 = Embedding::from_text("machine learning", 64);
        let e2 = Embedding::from_text("cooking recipes", 64);
        let sim = e1.cosine_similarity(&e2);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_embedding_clone() {
        let emb = Embedding::new(vec![1.0, 2.0]);
        let cloned = emb.clone();
        assert_eq!(emb.0, cloned.0);
    }

    // =========================================================================
    // Vector Store Tests
    // =========================================================================

    #[test]
    fn test_vector_store_new() {
        let store = VectorStore::new(64);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_vector_store() {
        let mut store = VectorStore::new(64);
        let mut doc = Document::new("test", "machine learning is great");
        doc.chunk(10, 0);
        store.add_document(&doc);

        assert!(!store.is_empty());
    }

    #[test]
    fn test_vector_search() {
        let mut store = VectorStore::new(64);

        let mut doc1 = Document::new("ml", "machine learning algorithms");
        doc1.chunk(10, 0);
        store.add_document(&doc1);

        let mut doc2 = Document::new("cook", "cooking recipes food");
        doc2.chunk(10, 0);
        store.add_document(&doc2);

        let results = store.search("neural networks", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_search_top_k() {
        let mut store = VectorStore::new(32);

        for i in 0..5 {
            let mut doc = Document::new(&format!("doc{}", i), &format!("content {}", i));
            doc.chunk(10, 0);
            store.add_document(&doc);
        }

        let results = store.search("content", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_retrieved_chunk_rank() {
        let mut store = VectorStore::new(32);
        let mut doc = Document::new("test", "hello world foo bar");
        doc.chunk(2, 0);
        store.add_document(&doc);

        let results = store.search("hello", 2);
        assert_eq!(results[0].rank, 1);
    }

    // =========================================================================
    // RAG Pipeline Tests
    // =========================================================================

    #[test]
    fn test_rag_pipeline_new() {
        let rag = RagPipeline::new(64);
        assert_eq!(rag.document_count(), 0);
    }

    #[test]
    fn test_rag_pipeline() {
        let mut rag = RagPipeline::new(64).with_top_k(2);

        let mut doc = Document::new(
            "test",
            "Machine learning enables pattern recognition in data",
        );
        doc.chunk(10, 2);
        rag.ingest(doc);

        let response = rag.query("What is machine learning?");
        assert!(response.is_ok());
    }

    #[test]
    fn test_rag_pipeline_with_template() {
        let rag = RagPipeline::new(64).with_template("Custom template: {context} - {question}");
        // Template should be updated
        assert!(rag.document_count() == 0);
    }

    #[test]
    fn test_rag_pipeline_empty() {
        let rag = RagPipeline::new(64);
        let result = rag.query("test query");
        assert!(result.is_err());
    }

    #[test]
    fn test_rag_response_sources() {
        let mut rag = RagPipeline::new(64).with_top_k(2);
        let mut doc = Document::new("test", "Machine learning algorithms process data");
        doc.chunk(5, 1);
        rag.ingest(doc);

        let response = rag.query("machine learning").unwrap();
        assert!(!response.sources.is_empty());
    }

    #[test]
    fn test_rag_response_serialization() {
        let response = RagResponse {
            query: "test".to_string(),
            answer: "answer".to_string(),
            sources: vec![],
            context_used: "context".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let restored: RagResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.query, restored.query);
    }

    #[test]
    fn test_source_serialization() {
        let source = Source {
            chunk_id: "c1".to_string(),
            text_preview: "preview".to_string(),
            score: 0.9,
        };
        let json = serde_json::to_string(&source).unwrap();
        let restored: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(source.chunk_id, restored.chunk_id);
    }

    // =========================================================================
    // RAG Metrics Tests
    // =========================================================================

    #[test]
    fn test_rag_metrics() {
        let response = RagResponse {
            query: "test".to_string(),
            answer: "Machine learning is great".to_string(),
            sources: vec![Source {
                chunk_id: "1".to_string(),
                text_preview: "test".to_string(),
                score: 0.9,
            }],
            context_used: "test context".to_string(),
        };

        let metrics = RagMetrics::evaluate(&response, Some("Machine learning systems"));
        assert!(metrics.retrieval_precision > 0.0);
    }

    #[test]
    fn test_rag_metrics_no_ground_truth() {
        let response = RagResponse {
            query: "test".to_string(),
            answer: "answer".to_string(),
            sources: vec![Source {
                chunk_id: "1".to_string(),
                text_preview: "test".to_string(),
                score: 0.8,
            }],
            context_used: "context".to_string(),
        };

        let metrics = RagMetrics::evaluate(&response, None);
        assert_eq!(metrics.answer_faithfulness, 0.7);
    }

    #[test]
    fn test_rag_metrics_empty_sources() {
        let response = RagResponse {
            query: "test".to_string(),
            answer: "answer".to_string(),
            sources: vec![],
            context_used: "context".to_string(),
        };

        let metrics = RagMetrics::evaluate(&response, None);
        assert_eq!(metrics.retrieval_precision, 0.0);
    }

    #[test]
    fn test_rag_metrics_serialization() {
        let metrics = RagMetrics {
            retrieval_precision: 0.9,
            context_relevance: 0.8,
            answer_faithfulness: 0.7,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: RagMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.retrieval_precision, restored.retrieval_precision);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_rag_error_retrieval() {
        let err = RagError::Retrieval("not found".to_string());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_rag_error_generation() {
        let err = RagError::Generation("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_rag_error_document() {
        let err = RagError::Document("invalid format".to_string());
        assert!(err.to_string().contains("invalid format"));
    }

    #[test]
    fn test_rag_error_debug() {
        let err = RagError::Retrieval("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Retrieval"));
    }
}
