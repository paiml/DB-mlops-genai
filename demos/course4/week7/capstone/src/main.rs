//! Enterprise Knowledge Assistant Capstone - Course 4 Week 7
//!
//! End-to-end GenAI system combining all concepts from Course 4.
//! Implements document ingestion, RAG, and production serving.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum AssistantError {
    #[error("Document error: {0}")]
    Document(String),
    #[error("Retrieval error: {0}")]
    Retrieval(String),
    #[error("Generation error: {0}")]
    Generation(String),
    #[error("Guardrail violation: {0}")]
    Guardrail(String),
}

// ============================================================================
// Document Processing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    pub category: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct DocumentProcessor {
    chunk_size: usize,
    overlap: usize,
    embedding_dim: usize,
}

impl DocumentProcessor {
    pub fn new(chunk_size: usize, overlap: usize, embedding_dim: usize) -> Self {
        Self {
            chunk_size,
            overlap,
            embedding_dim,
        }
    }

    pub fn process(&self, document: &Document) -> Vec<Chunk> {
        let words: Vec<&str> = document.content.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            let text = words[start..end].join(" ");
            let embedding = self.embed(&text);

            chunks.push(Chunk {
                id: format!("{}-chunk-{}", document.id, chunks.len()),
                doc_id: document.id.clone(),
                text,
                embedding,
            });

            if end >= words.len() {
                break;
            }
            start += self.chunk_size - self.overlap;
        }

        chunks
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.embedding_dim];
        for (i, word) in text.split_whitespace().enumerate() {
            for (j, c) in word.chars().enumerate() {
                let idx = ((c as usize) * (i + 1) + j) % self.embedding_dim;
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
        values
    }
}

// ============================================================================
// Vector Store
// ============================================================================

pub struct VectorStore {
    chunks: Vec<Chunk>,
    embedding_dim: usize,
}

impl VectorStore {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            chunks: Vec::new(),
            embedding_dim,
        }
    }

    pub fn add(&mut self, chunk: Chunk) {
        self.chunks.push(chunk);
    }

    pub fn add_all(&mut self, chunks: Vec<Chunk>) {
        self.chunks.extend(chunks);
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&Chunk, f32)> {
        let mut scored: Vec<_> = self
            .chunks
            .iter()
            .map(|chunk| {
                let score = Self::cosine_similarity(query_embedding, &chunk.embedding);
                (chunk, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).collect()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

// ============================================================================
// Guardrails
// ============================================================================

pub struct Guardrails {
    blocked_patterns: Vec<String>,
    max_input_length: usize,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            blocked_patterns: vec![
                "password".to_string(),
                "secret".to_string(),
                "confidential".to_string(),
            ],
            max_input_length: 4096,
        }
    }
}

impl Guardrails {
    pub fn check(&self, text: &str) -> Result<(), String> {
        if text.len() > self.max_input_length {
            return Err(format!(
                "Input too long: {} > {}",
                text.len(),
                self.max_input_length
            ));
        }

        let text_lower = text.to_lowercase();
        for pattern in &self.blocked_patterns {
            if text_lower.contains(pattern) {
                return Err(format!("Blocked pattern detected: {}", pattern));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Knowledge Assistant
// ============================================================================

pub struct KnowledgeAssistant {
    processor: DocumentProcessor,
    vector_store: VectorStore,
    guardrails: Guardrails,
    documents: HashMap<String, Document>,
    embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantQuery {
    pub question: String,
    pub top_k: usize,
    pub include_sources: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponse {
    pub query: String,
    pub answer: String,
    pub sources: Vec<Source>,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub doc_id: String,
    pub doc_title: String,
    pub chunk_preview: String,
    pub relevance_score: f32,
}

impl KnowledgeAssistant {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            processor: DocumentProcessor::new(50, 10, embedding_dim),
            vector_store: VectorStore::new(embedding_dim),
            guardrails: Guardrails::default(),
            documents: HashMap::new(),
            embedding_dim,
        }
    }

    pub fn ingest(&mut self, document: Document) {
        let chunks = self.processor.process(&document);
        self.vector_store.add_all(chunks);
        self.documents.insert(document.id.clone(), document);
    }

    pub fn query(&self, query: AssistantQuery) -> Result<AssistantResponse, AssistantError> {
        // 1. Guardrails check
        self.guardrails
            .check(&query.question)
            .map_err(AssistantError::Guardrail)?;

        // 2. Embed query
        let query_embedding = self.embed(&query.question);

        // 3. Retrieve relevant chunks
        let results = self.vector_store.search(&query_embedding, query.top_k);

        if results.is_empty() {
            return Err(AssistantError::Retrieval(
                "No relevant documents found".to_string(),
            ));
        }

        // 4. Build context
        let context: Vec<String> = results
            .iter()
            .map(|(chunk, _)| chunk.text.clone())
            .collect();
        let context_str = context.join("\n\n");

        // 5. Generate answer
        let answer = self.generate(&query.question, &context_str);

        // 6. Build sources
        let sources: Vec<Source> = results
            .iter()
            .map(|(chunk, score)| {
                let doc = self.documents.get(&chunk.doc_id);
                Source {
                    doc_id: chunk.doc_id.clone(),
                    doc_title: doc.map(|d| d.title.clone()).unwrap_or_default(),
                    chunk_preview: chunk.text.chars().take(100).collect::<String>() + "...",
                    relevance_score: *score,
                }
            })
            .collect();

        Ok(AssistantResponse {
            query: query.question,
            answer,
            sources,
            latency_ms: 50,
        })
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.embedding_dim];
        for (i, word) in text.split_whitespace().enumerate() {
            for (j, c) in word.chars().enumerate() {
                let idx = ((c as usize) * (i + 1) + j) % self.embedding_dim;
                values[idx] += 0.1;
            }
        }
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut values {
                *v /= norm;
            }
        }
        values
    }

    fn generate(&self, question: &str, context: &str) -> String {
        // Simulated generation based on context
        let q_lower = question.to_lowercase();
        let c_lower = context.to_lowercase();

        if q_lower.contains("what") && c_lower.contains("machine learning") {
            "Based on the knowledge base, machine learning is a method of data analysis that automates analytical model building, enabling systems to learn from data.".to_string()
        } else if q_lower.contains("how") {
            "According to the documentation, the process involves following the guidelines outlined in the relevant policy documents.".to_string()
        } else if !context.is_empty() {
            format!(
                "Based on the retrieved documents: {}",
                &context[..context.len().min(200)]
            )
        } else {
            "I couldn't find relevant information in the knowledge base.".to_string()
        }
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.vector_store.len()
    }
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub total_latency_ms: u64,
    pub guardrail_blocks: u64,
}

impl AssistantMetrics {
    pub fn record_query(&mut self, latency_ms: u64, success: bool) {
        self.total_queries += 1;
        self.total_latency_ms += latency_ms;
        if success {
            self.successful_queries += 1;
        }
    }

    pub fn record_guardrail_block(&mut self) {
        self.guardrail_blocks += 1;
    }

    pub fn avg_latency(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.total_queries as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.successful_queries as f64 / self.total_queries as f64
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Enterprise Knowledge Assistant - Course 4 Capstone        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Create knowledge assistant
    println!("🏗️  Step 1: Initialize Knowledge Assistant");
    let mut assistant = KnowledgeAssistant::new(128);
    println!("   Embedding dimension: 128\n");

    // Step 2: Ingest documents
    println!("📚 Step 2: Document Ingestion");
    let documents = vec![
        Document {
            id: "policy-001".to_string(),
            title: "Employee Handbook".to_string(),
            content: "This handbook outlines company policies and procedures. All employees must follow the code of conduct. Vacation time is accrued monthly. Remote work requires manager approval.".to_string(),
            category: "HR".to_string(),
            metadata: HashMap::new(),
        },
        Document {
            id: "tech-001".to_string(),
            title: "ML Platform Guide".to_string(),
            content: "Machine learning on our platform uses distributed training. Models are versioned in the registry. Feature engineering happens in the feature store. Deployments use containerized serving.".to_string(),
            category: "Engineering".to_string(),
            metadata: HashMap::new(),
        },
        Document {
            id: "proc-001".to_string(),
            title: "Data Processing Guidelines".to_string(),
            content: "Data pipelines follow the medallion architecture. Bronze layer holds raw data. Silver layer has cleaned data. Gold layer contains business aggregates.".to_string(),
            category: "Data".to_string(),
            metadata: HashMap::new(),
        },
    ];

    for doc in documents {
        println!("   Ingesting: {} ({})", doc.title, doc.id);
        assistant.ingest(doc);
    }
    println!("   Documents: {}", assistant.document_count());
    println!("   Chunks: {}\n", assistant.chunk_count());

    // Step 3: Query the assistant
    println!("❓ Step 3: Knowledge Queries");
    let queries = vec![
        AssistantQuery {
            question: "What is machine learning on the platform?".to_string(),
            top_k: 2,
            include_sources: true,
        },
        AssistantQuery {
            question: "How does vacation time work?".to_string(),
            top_k: 2,
            include_sources: true,
        },
        AssistantQuery {
            question: "Explain the medallion architecture".to_string(),
            top_k: 2,
            include_sources: true,
        },
    ];

    let mut metrics = AssistantMetrics::default();

    for query in &queries {
        println!("   Q: {}", query.question);

        match assistant.query(query.clone()) {
            Ok(response) => {
                metrics.record_query(response.latency_ms, true);
                println!(
                    "   A: {}",
                    &response.answer[..response.answer.len().min(80)]
                );
                println!("   Sources:");
                for source in &response.sources {
                    println!(
                        "     - {} [score={:.3}]",
                        source.doc_title, source.relevance_score
                    );
                }
            }
            Err(e) => {
                metrics.record_query(0, false);
                println!("   Error: {}", e);
            }
        }
        println!();
    }

    // Step 4: Guardrail test
    println!("🛡️  Step 4: Guardrail Test");
    let blocked_query = AssistantQuery {
        question: "What is the admin password?".to_string(),
        top_k: 2,
        include_sources: false,
    };

    match assistant.query(blocked_query) {
        Ok(_) => println!("   Unexpected: query allowed"),
        Err(e) => {
            metrics.record_guardrail_block();
            println!("   Blocked: {}", e);
        }
    }
    println!();

    // Step 5: Metrics
    println!("📊 Step 5: Performance Metrics");
    println!("   Total queries: {}", metrics.total_queries);
    println!("   Success rate: {:.1}%", metrics.success_rate() * 100.0);
    println!("   Avg latency: {:.1}ms", metrics.avg_latency());
    println!("   Guardrail blocks: {}\n", metrics.guardrail_blocks);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Capstone Complete!");
    println!();
    println!("Components demonstrated:");
    println!("  • Document ingestion and chunking");
    println!("  • Vector embedding and similarity search");
    println!("  • RAG-based question answering");
    println!("  • Input guardrails for safety");
    println!("  • Production metrics tracking");
    println!();
    println!("Databricks equivalent: Vector Search + Foundation Models + Model Serving");
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
    fn test_document_serialization() {
        let doc = Document {
            id: "doc1".to_string(),
            title: "Test Doc".to_string(),
            content: "Content here".to_string(),
            category: "test".to_string(),
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let restored: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(doc.id, restored.id);
    }

    #[test]
    fn test_document_clone() {
        let doc = Document {
            id: "doc1".to_string(),
            title: "Test".to_string(),
            content: "Content".to_string(),
            category: "cat".to_string(),
            metadata: HashMap::new(),
        };
        let cloned = doc.clone();
        assert_eq!(doc.id, cloned.id);
    }

    #[test]
    fn test_chunk_serialization() {
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "text".to_string(),
            embedding: vec![0.1, 0.2],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let restored: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk.id, restored.id);
    }

    #[test]
    fn test_chunk_clone() {
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "text".to_string(),
            embedding: vec![0.1],
        };
        let cloned = chunk.clone();
        assert_eq!(chunk.id, cloned.id);
    }

    // =========================================================================
    // Document Processor Tests
    // =========================================================================

    #[test]
    fn test_document_processor() {
        let processor = DocumentProcessor::new(10, 2, 64);
        let doc = Document {
            id: "test".to_string(),
            title: "Test".to_string(),
            content: "one two three four five six seven eight nine ten eleven twelve".to_string(),
            category: "test".to_string(),
            metadata: HashMap::new(),
        };

        let chunks = processor.process(&doc);
        assert!(chunks.len() >= 1);
        assert_eq!(chunks[0].doc_id, "test");
    }

    #[test]
    fn test_document_processor_short_doc() {
        let processor = DocumentProcessor::new(50, 10, 64);
        let doc = Document {
            id: "short".to_string(),
            title: "Short".to_string(),
            content: "Just a few words".to_string(),
            category: "test".to_string(),
            metadata: HashMap::new(),
        };

        let chunks = processor.process(&doc);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_document_processor_embeddings() {
        let processor = DocumentProcessor::new(10, 2, 32);
        let doc = Document {
            id: "test".to_string(),
            title: "Test".to_string(),
            content: "hello world test".to_string(),
            category: "test".to_string(),
            metadata: HashMap::new(),
        };

        let chunks = processor.process(&doc);
        assert_eq!(chunks[0].embedding.len(), 32);
    }

    // =========================================================================
    // Vector Store Tests
    // =========================================================================

    #[test]
    fn test_vector_store_new() {
        let store = VectorStore::new(64);
        assert!(store.is_empty());
        assert_eq!(store.embedding_dim(), 64);
    }

    #[test]
    fn test_vector_store() {
        let mut store = VectorStore::new(64);
        let chunk = Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "test text".to_string(),
            embedding: vec![1.0; 64],
        };

        store.add(chunk);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_vector_store_add_all() {
        let mut store = VectorStore::new(4);
        let chunks = vec![
            Chunk {
                id: "c1".to_string(),
                doc_id: "d1".to_string(),
                text: "text".to_string(),
                embedding: vec![1.0; 4],
            },
            Chunk {
                id: "c2".to_string(),
                doc_id: "d1".to_string(),
                text: "text2".to_string(),
                embedding: vec![0.5; 4],
            },
        ];
        store.add_all(chunks);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_vector_search() {
        let mut store = VectorStore::new(4);
        store.add(Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "machine learning".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        });
        store.add(Chunk {
            id: "c2".to_string(),
            doc_id: "d2".to_string(),
            text: "cooking recipes".to_string(),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
        });

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "c1");
    }

    #[test]
    fn test_vector_search_scores() {
        let mut store = VectorStore::new(4);
        store.add(Chunk {
            id: "c1".to_string(),
            doc_id: "d1".to_string(),
            text: "exact match".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        });

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert!((results[0].1 - 1.0).abs() < 0.001); // Perfect match
    }

    // =========================================================================
    // Guardrails Tests
    // =========================================================================

    #[test]
    fn test_guardrails_default() {
        let guardrails = Guardrails::default();
        assert_eq!(guardrails.max_input_length, 4096);
    }

    #[test]
    fn test_guardrails_pass() {
        let guardrails = Guardrails::default();
        assert!(guardrails.check("What is ML?").is_ok());
    }

    #[test]
    fn test_guardrails_block() {
        let guardrails = Guardrails::default();
        assert!(guardrails.check("What is the password?").is_err());
    }

    #[test]
    fn test_guardrails_block_secret() {
        let guardrails = Guardrails::default();
        let result = guardrails.check("This is a secret document");
        assert!(result.is_err());
    }

    #[test]
    fn test_guardrails_block_confidential() {
        let guardrails = Guardrails::default();
        let result = guardrails.check("Confidential information");
        assert!(result.is_err());
    }

    #[test]
    fn test_guardrails_length() {
        let guardrails = Guardrails::default();
        let long_text = "x".repeat(5000);
        let result = guardrails.check(&long_text);
        assert!(result.is_err());
    }

    // =========================================================================
    // Knowledge Assistant Tests
    // =========================================================================

    #[test]
    fn test_knowledge_assistant_new() {
        let assistant = KnowledgeAssistant::new(64);
        assert_eq!(assistant.document_count(), 0);
        assert_eq!(assistant.chunk_count(), 0);
    }

    #[test]
    fn test_knowledge_assistant() {
        let mut assistant = KnowledgeAssistant::new(64);
        assistant.ingest(Document {
            id: "test".to_string(),
            title: "Test Doc".to_string(),
            content: "Machine learning is great for automation".to_string(),
            category: "tech".to_string(),
            metadata: HashMap::new(),
        });

        assert_eq!(assistant.document_count(), 1);
        assert!(assistant.chunk_count() >= 1);
    }

    #[test]
    fn test_assistant_query() {
        let mut assistant = KnowledgeAssistant::new(64);
        assistant.ingest(Document {
            id: "test".to_string(),
            title: "Test".to_string(),
            content: "Machine learning enables automation".to_string(),
            category: "tech".to_string(),
            metadata: HashMap::new(),
        });

        let query = AssistantQuery {
            question: "What is machine learning?".to_string(),
            top_k: 1,
            include_sources: true,
        };

        let result = assistant.query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_assistant_query_guardrail_block() {
        let mut assistant = KnowledgeAssistant::new(64);
        assistant.ingest(Document {
            id: "test".to_string(),
            title: "Test".to_string(),
            content: "Some content".to_string(),
            category: "test".to_string(),
            metadata: HashMap::new(),
        });

        let query = AssistantQuery {
            question: "What is the admin password?".to_string(),
            top_k: 1,
            include_sources: false,
        };

        let result = assistant.query(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_assistant_query_no_documents() {
        let assistant = KnowledgeAssistant::new(64);
        let query = AssistantQuery {
            question: "What is ML?".to_string(),
            top_k: 1,
            include_sources: true,
        };

        let result = assistant.query(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_assistant_query_serialization() {
        let query = AssistantQuery {
            question: "What is ML?".to_string(),
            top_k: 3,
            include_sources: true,
        };
        let json = serde_json::to_string(&query).unwrap();
        let restored: AssistantQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(query.question, restored.question);
    }

    #[test]
    fn test_assistant_response_serialization() {
        let response = AssistantResponse {
            query: "test".to_string(),
            answer: "answer".to_string(),
            sources: vec![],
            latency_ms: 50,
        };
        let json = serde_json::to_string(&response).unwrap();
        let restored: AssistantResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.query, restored.query);
    }

    #[test]
    fn test_source_serialization() {
        let source = Source {
            doc_id: "d1".to_string(),
            doc_title: "Title".to_string(),
            chunk_preview: "preview...".to_string(),
            relevance_score: 0.9,
        };
        let json = serde_json::to_string(&source).unwrap();
        let restored: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(source.doc_id, restored.doc_id);
    }

    // =========================================================================
    // Metrics Tests
    // =========================================================================

    #[test]
    fn test_metrics_default() {
        let metrics = AssistantMetrics::default();
        assert_eq!(metrics.total_queries, 0);
    }

    #[test]
    fn test_metrics() {
        let mut metrics = AssistantMetrics::default();
        metrics.record_query(50, true);
        metrics.record_query(60, true);
        metrics.record_query(70, false);

        assert_eq!(metrics.total_queries, 3);
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
        assert!((metrics.avg_latency() - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_metrics_guardrail_block() {
        let mut metrics = AssistantMetrics::default();
        metrics.record_guardrail_block();
        metrics.record_guardrail_block();
        assert_eq!(metrics.guardrail_blocks, 2);
    }

    #[test]
    fn test_metrics_success_rate_zero() {
        let metrics = AssistantMetrics::default();
        assert_eq!(metrics.success_rate(), 0.0);
    }

    #[test]
    fn test_metrics_avg_latency_zero() {
        let metrics = AssistantMetrics::default();
        assert_eq!(metrics.avg_latency(), 0.0);
    }

    #[test]
    fn test_metrics_serialization() {
        let metrics = AssistantMetrics {
            total_queries: 100,
            successful_queries: 95,
            total_latency_ms: 5000,
            guardrail_blocks: 3,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: AssistantMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.total_queries, restored.total_queries);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_assistant_error_document() {
        let err = AssistantError::Document("invalid".to_string());
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn test_assistant_error_retrieval() {
        let err = AssistantError::Retrieval("not found".to_string());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_assistant_error_generation() {
        let err = AssistantError::Generation("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_assistant_error_guardrail() {
        let err = AssistantError::Guardrail("blocked".to_string());
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_assistant_error_debug() {
        let err = AssistantError::Document("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Document"));
    }
}
