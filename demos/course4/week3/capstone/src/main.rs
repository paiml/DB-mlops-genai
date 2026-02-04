//! GenAI Capstone Demo - Course 4 Week 3
//!
//! Demonstrates an end-to-end GenAI application combining all course concepts:
//! RAG, fine-tuning, production deployment, and quality gates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum CapstoneError {
    #[error("Pipeline error: {0}")]
    Pipeline(String),
    #[error("Quality error: {0}")]
    Quality(String),
}

// ============================================================================
// Document Store
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct DocumentStore {
    documents: HashMap<String, Document>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, doc: Document) {
        self.documents.insert(doc.id.clone(), doc);
    }

    pub fn get(&self, id: &str) -> Option<&Document> {
        self.documents.get(id)
    }

    pub fn search(&self, query: &str) -> Vec<&Document> {
        let query_lower = query.to_lowercase();
        self.documents
            .values()
            .filter(|d| d.content.to_lowercase().contains(&query_lower))
            .collect()
    }

    pub fn count(&self) -> usize {
        self.documents.len()
    }
}

// ============================================================================
// Simple Embedding (for demo)
// ============================================================================

fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    for (i, c) in text.chars().enumerate() {
        embedding[(c as usize + i) % dim] += 1.0;
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
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
// Vector Index
// ============================================================================

#[derive(Debug)]
pub struct VectorIndex {
    entries: Vec<(String, Vec<f32>)>,
    dimension: usize,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
        }
    }

    pub fn add(&mut self, id: &str, embedding: Vec<f32>) {
        self.entries.push((id.to_string(), embedding));
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|(id, emb)| (id.clone(), cosine_sim(query_embedding, emb)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// LLM (Simulated)
// ============================================================================

#[derive(Debug)]
pub struct LlmEngine {
    model_name: String,
}

impl LlmEngine {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }

    pub fn generate(&self, prompt: &str) -> LlmResponse {
        // Pattern-based responses for demo
        let prompt_lower = prompt.to_lowercase();

        let response = if prompt_lower.contains("summarize") {
            "Here is a summary of the key points from the provided context.".to_string()
        } else if prompt_lower.contains("explain") {
            "Let me explain this concept based on the available information.".to_string()
        } else if prompt_lower.contains("question") || prompt_lower.contains("?") {
            "Based on the context provided, here is the answer to your question.".to_string()
        } else {
            "I've processed your request using the provided context.".to_string()
        };

        LlmResponse {
            text: response,
            prompt_tokens: prompt.split_whitespace().count(),
            completion_tokens: 15,
            model: self.model_name.clone(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub model: String,
}

// ============================================================================
// RAG Pipeline
// ============================================================================

#[derive(Debug)]
pub struct RagPipeline {
    store: DocumentStore,
    index: VectorIndex,
    llm: LlmEngine,
    dimension: usize,
}

impl RagPipeline {
    pub fn new(dimension: usize, model_name: &str) -> Self {
        Self {
            store: DocumentStore::new(),
            index: VectorIndex::new(dimension),
            llm: LlmEngine::new(model_name),
            dimension,
        }
    }

    pub fn ingest(&mut self, docs: Vec<Document>) {
        for doc in docs {
            let embedding = embed_text(&doc.content, self.dimension);
            self.index.add(&doc.id, embedding);
            self.store.add(doc);
        }
    }

    pub fn query(&self, question: &str, top_k: usize) -> RagResponse {
        // Retrieve relevant documents
        let query_embedding = embed_text(question, self.dimension);
        let results = self.index.search(&query_embedding, top_k);

        // Build context from retrieved docs
        let mut context_parts = Vec::new();
        let mut sources = Vec::new();

        for (id, score) in &results {
            if let Some(doc) = self.store.get(id) {
                context_parts.push(format!("- {}", doc.content));
                sources.push(SourceReference {
                    doc_id: id.clone(),
                    score: *score,
                });
            }
        }

        let context = context_parts.join("\n");

        // Generate response with context
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer based on the context:",
            context, question
        );

        let llm_response = self.llm.generate(&prompt);

        RagResponse {
            answer: llm_response.text,
            sources,
            prompt_tokens: llm_response.prompt_tokens,
            completion_tokens: llm_response.completion_tokens,
        }
    }

    pub fn document_count(&self) -> usize {
        self.store.count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub doc_id: String,
    pub score: f32,
}

// ============================================================================
// Quality Evaluation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    pub relevance: f32,
    pub groundedness: f32,
    pub coherence: f32,
    pub overall: f32,
}

impl QualityScore {
    pub fn evaluate(response: &RagResponse) -> Self {
        // Simulated quality evaluation
        let relevance = if response.sources.is_empty() {
            0.5
        } else {
            response.sources.iter().map(|s| s.score).sum::<f32>() / response.sources.len() as f32
        };

        let groundedness = if response.sources.len() >= 2 { 0.85 } else { 0.6 };
        let coherence = 0.9; // Simulated

        let overall = (relevance + groundedness + coherence) / 3.0;

        Self {
            relevance,
            groundedness,
            coherence,
            overall,
        }
    }

    pub fn passes_threshold(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }
}

// ============================================================================
// Application
// ============================================================================

#[derive(Debug)]
pub struct QaApplication {
    pipeline: RagPipeline,
    quality_threshold: f32,
    request_count: usize,
}

impl QaApplication {
    pub fn new(dimension: usize, model: &str) -> Self {
        Self {
            pipeline: RagPipeline::new(dimension, model),
            quality_threshold: 0.7,
            request_count: 0,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.quality_threshold = threshold;
        self
    }

    pub fn load_knowledge_base(&mut self, docs: Vec<Document>) {
        self.pipeline.ingest(docs);
    }

    pub fn ask(&mut self, question: &str) -> AppResponse {
        self.request_count += 1;

        let rag_response = self.pipeline.query(question, 3);
        let quality = QualityScore::evaluate(&rag_response);

        let status = if quality.passes_threshold(self.quality_threshold) {
            ResponseStatus::Success
        } else {
            ResponseStatus::LowQuality
        };

        AppResponse {
            answer: rag_response.answer,
            sources: rag_response.sources,
            quality,
            status,
            tokens_used: rag_response.prompt_tokens + rag_response.completion_tokens,
        }
    }

    pub fn stats(&self) -> AppStats {
        AppStats {
            documents: self.pipeline.document_count(),
            requests: self.request_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppResponse {
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub quality: QualityScore,
    pub status: ResponseStatus,
    pub tokens_used: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    LowQuality,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct AppStats {
    pub documents: usize,
    pub requests: usize,
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     GenAI Capstone Demo - Course 4 Week 3                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Building an end-to-end Q&A application with RAG\n");

    // Step 1: Initialize Application
    println!("📦 Step 1: Initialize Application");
    let mut app = QaApplication::new(64, "llama-7b").with_threshold(0.6);
    println!("   Model: llama-7b");
    println!("   Quality threshold: 0.6\n");

    // Step 2: Load Knowledge Base
    println!("📚 Step 2: Load Knowledge Base");
    let documents = vec![
        Document {
            id: "doc1".to_string(),
            content: "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.".to_string(),
            metadata: HashMap::from([("topic".to_string(), "mlops".to_string())]),
        },
        Document {
            id: "doc2".to_string(),
            content: "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to provide grounded, factual responses.".to_string(),
            metadata: HashMap::from([("topic".to_string(), "genai".to_string())]),
        },
        Document {
            id: "doc3".to_string(),
            content: "LoRA (Low-Rank Adaptation) enables efficient fine-tuning of large language models by training small adapter weights.".to_string(),
            metadata: HashMap::from([("topic".to_string(), "genai".to_string())]),
        },
        Document {
            id: "doc4".to_string(),
            content: "Vector search uses embeddings to find semantically similar content, enabling powerful search and retrieval capabilities.".to_string(),
            metadata: HashMap::from([("topic".to_string(), "genai".to_string())]),
        },
        Document {
            id: "doc5".to_string(),
            content: "Feature engineering transforms raw data into informative features that improve machine learning model performance.".to_string(),
            metadata: HashMap::from([("topic".to_string(), "mlops".to_string())]),
        },
    ];

    app.load_knowledge_base(documents);
    println!("   Loaded {} documents\n", app.stats().documents);

    // Step 3: Query the Application
    println!("❓ Step 3: Query Examples");

    let questions = [
        "What is RAG and how does it work?",
        "How can I fine-tune a large language model efficiently?",
        "What is MLflow used for?",
    ];

    for question in &questions {
        println!("   Q: {}", question);
        let response = app.ask(question);

        println!("   A: {}", response.answer);
        println!(
            "   Quality: {:.2} ({:?})",
            response.quality.overall, response.status
        );
        println!(
            "   Sources: {}",
            response
                .sources
                .iter()
                .map(|s| format!("{}({:.2})", s.doc_id, s.score))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("   Tokens: {}\n", response.tokens_used);
    }

    // Step 4: Application Stats
    println!("📊 Step 4: Application Statistics");
    let stats = app.stats();
    println!("   Documents indexed: {}", stats.documents);
    println!("   Total requests: {}\n", stats.requests);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Capstone Complete!");
    println!();
    println!("This demo integrated:");
    println!("  • Document ingestion and storage");
    println!("  • Vector embedding and indexing");
    println!("  • Semantic search and retrieval");
    println!("  • LLM-based response generation");
    println!("  • Quality evaluation and gating");
    println!();
    println!("Databricks equivalent: End-to-end GenAI application");
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_store() {
        let mut store = DocumentStore::new();
        store.add(Document {
            id: "1".to_string(),
            content: "test".to_string(),
            metadata: HashMap::new(),
        });
        assert_eq!(store.count(), 1);
        assert!(store.get("1").is_some());
    }

    #[test]
    fn test_document_store_search() {
        let mut store = DocumentStore::new();
        store.add(Document {
            id: "1".to_string(),
            content: "machine learning".to_string(),
            metadata: HashMap::new(),
        });
        let results = store.search("machine");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_index() {
        let mut index = VectorIndex::new(32);
        index.add("1", embed_text("test", 32));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_index_search() {
        let mut index = VectorIndex::new(32);
        index.add("1", embed_text("machine learning", 32));
        index.add("2", embed_text("cooking recipes", 32));

        let results = index.search(&embed_text("machine", 32), 1);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn test_llm_engine() {
        let llm = LlmEngine::new("test-model");
        let response = llm.generate("explain this concept");
        assert!(!response.text.is_empty());
    }

    #[test]
    fn test_rag_pipeline() {
        let mut pipeline = RagPipeline::new(32, "test");
        pipeline.ingest(vec![Document {
            id: "1".to_string(),
            content: "RAG is retrieval augmented generation".to_string(),
            metadata: HashMap::new(),
        }]);

        let response = pipeline.query("What is RAG?", 1);
        assert!(!response.answer.is_empty());
        assert!(!response.sources.is_empty());
    }

    #[test]
    fn test_quality_score() {
        let response = RagResponse {
            answer: "test".to_string(),
            sources: vec![SourceReference {
                doc_id: "1".to_string(),
                score: 0.9,
            }],
            prompt_tokens: 10,
            completion_tokens: 5,
        };

        let quality = QualityScore::evaluate(&response);
        assert!(quality.overall > 0.0);
    }

    #[test]
    fn test_quality_threshold() {
        let score = QualityScore {
            relevance: 0.8,
            groundedness: 0.85,
            coherence: 0.9,
            overall: 0.85,
        };
        assert!(score.passes_threshold(0.7));
        assert!(!score.passes_threshold(0.9));
    }

    #[test]
    fn test_application() {
        let mut app = QaApplication::new(32, "test");
        app.load_knowledge_base(vec![Document {
            id: "1".to_string(),
            content: "Test content".to_string(),
            metadata: HashMap::new(),
        }]);

        let response = app.ask("What is this?");
        assert!(!response.answer.is_empty());
    }

    #[test]
    fn test_application_stats() {
        let mut app = QaApplication::new(32, "test");
        assert_eq!(app.stats().requests, 0);
        app.load_knowledge_base(vec![Document {
            id: "1".to_string(),
            content: "Test".to_string(),
            metadata: HashMap::new(),
        }]);
        app.ask("test");
        assert_eq!(app.stats().requests, 1);
    }

    #[test]
    fn test_capstone_error() {
        let err = CapstoneError::Pipeline("failed".to_string());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_embed_text() {
        let emb = embed_text("test", 32);
        assert_eq!(emb.len(), 32);
    }

    #[test]
    fn test_cosine_sim() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_sim(&a, &b) - 1.0).abs() < 0.001);
    }
}
