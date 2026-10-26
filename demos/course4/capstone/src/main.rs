//! Code Intelligence RAG — quality-aware code retrieval pipeline
//!
//! Demonstrates a complete RAG pipeline for code search that fuses semantic
//! retrieval (trueno-rag) with code quality signals (pmat TDG grades,
//! complexity, fault patterns) to rank results by relevance × quality.
//!
//! # Architecture
//!
//! ```text
//! .rs files → chunk → enrich (pmat) → index (vector + BM25) → query → eval
//! ```

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use trueno_rag::{
    embed::MockEmbedder, fusion::FusionStrategy, metrics::RetrievalMetrics,
    pipeline::RagPipelineBuilder, rerank::NoOpReranker, ChunkId, Document,
};

// ── CLI ─────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "code-intelligence-rag", about = "Quality-aware code retrieval")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index Rust source files from a directory
    Index {
        /// Directory containing .rs files
        #[arg(short, long)]
        dir: PathBuf,
    },
    /// Query the index with natural language
    Query {
        /// Natural language query
        #[arg(short, long)]
        text: String,
        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        k: usize,
    },
    /// Evaluate retrieval quality against ground-truth Q&A pairs
    Eval,
}

// ── Chunk ───────────────────────────────────────────────────────────────

/// Read all .rs files from a directory and convert them to Documents.
fn load_rust_sources(dir: &Path) -> std::io::Result<Vec<Document>> {
    let mut docs = Vec::new();
    if !dir.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("not a directory: {}", dir.display()),
        ));
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let content = std::fs::read_to_string(&path)?;
            if !content.is_empty() {
                let doc = Document::new(content)
                    .with_title(path.file_name().unwrap_or_default().to_string_lossy())
                    .with_source(path.to_string_lossy());
                docs.push(doc);
            }
        }
    }
    Ok(docs)
}

// ── Enrich ──────────────────────────────────────────────────────────────

/// Quality metadata from pmat for a code chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityAnnotation {
    /// TDG letter grade (A-F)
    grade: String,
    /// Cyclomatic complexity
    complexity: u32,
    /// Fault patterns found (e.g., "unwrap", "panic")
    faults: Vec<String>,
}

impl QualityAnnotation {
    /// Numeric score from TDG grade (A=5 .. F=1).
    fn grade_score(&self) -> f32 {
        match self.grade.as_str() {
            "A" => 5.0,
            "B" => 4.0,
            "C" => 3.0,
            "D" => 2.0,
            _ => 1.0,
        }
    }

    /// Combined quality score: grade bonus minus complexity/fault penalty.
    fn quality_score(&self) -> f32 {
        let base = self.grade_score();
        let complexity_penalty = (self.complexity as f32 / 50.0).min(1.0);
        let fault_penalty = (self.faults.len() as f32 * 0.15).min(1.0);
        (base - complexity_penalty - fault_penalty).max(0.0) / 5.0
    }
}

/// Run `pmat query` on a code snippet and parse quality metadata.
///
/// In production this shells out to `pmat query --include-source`.
/// For portability (CI, offline), falls back to heuristic annotation.
fn enrich_chunk(content: &str) -> QualityAnnotation {
    // Try pmat if available
    if let Ok(annotation) = try_pmat_enrich(content) {
        return annotation;
    }
    // Fallback: heuristic annotation
    heuristic_annotate(content)
}

/// Attempt to enrich via `pmat query` subprocess.
fn try_pmat_enrich(content: &str) -> Result<QualityAnnotation, String> {
    let first_line = content.lines().next().unwrap_or("");
    // Extract function name from `fn name(` or `pub fn name(`
    let fn_name = first_line
        .split("fn ")
        .nth(1)
        .and_then(|s| s.split('(').next())
        .unwrap_or("unknown");

    let output = std::process::Command::new("pmat")
        .args(["query", fn_name, "--limit", "1", "--json"])
        .output()
        .map_err(|e| format!("pmat not found: {e}"))?;

    if !output.status.success() {
        return Err("pmat query failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_pmat_json(&stdout)
}

/// Parse pmat JSON output into a QualityAnnotation.
fn parse_pmat_json(json_str: &str) -> Result<QualityAnnotation, String> {
    #[derive(Deserialize)]
    struct PmatResult {
        #[serde(default)]
        grade: Option<String>,
        #[serde(default)]
        complexity: Option<u32>,
        #[serde(default)]
        faults: Option<Vec<String>>,
    }

    let results: Vec<PmatResult> =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let result = results.first().ok_or("empty pmat results")?;

    Ok(QualityAnnotation {
        grade: result.grade.clone().unwrap_or_else(|| "C".to_string()),
        complexity: result.complexity.unwrap_or(10),
        faults: result.faults.clone().unwrap_or_default(),
    })
}

/// Heuristic annotation when pmat is unavailable.
fn heuristic_annotate(content: &str) -> QualityAnnotation {
    let mut faults = Vec::new();
    if content.contains(".unwrap()") {
        faults.push("unwrap".to_string());
    }
    if content.contains("panic!") {
        faults.push("panic".to_string());
    }
    if content.contains("unsafe ") {
        faults.push("unsafe".to_string());
    }
    if content.contains(".expect(") {
        faults.push("expect".to_string());
    }

    // Estimate complexity from control flow keywords
    let complexity = ["if ", "match ", "for ", "while ", "loop "]
        .iter()
        .map(|kw| content.matches(kw).count() as u32)
        .sum::<u32>()
        + 1;

    let grade = match (faults.len(), complexity) {
        (0, c) if c <= 5 => "A",
        (0, c) if c <= 10 => "B",
        (f, c) if f <= 1 && c <= 15 => "C",
        (f, _) if f <= 2 => "D",
        _ => "F",
    };

    QualityAnnotation {
        grade: grade.to_string(),
        complexity,
        faults,
    }
}

// ── Pipeline ────────────────────────────────────────────────────────────

/// Combined retrieval + quality score for a result.
#[derive(Debug, Clone, Serialize)]
struct ScoredResult {
    /// Original chunk content
    content: String,
    /// Source file
    source: Option<String>,
    /// Retrieval score from RAG pipeline
    retrieval_score: f32,
    /// Quality score from pmat enrichment
    quality_score: f32,
    /// Fused score: retrieval × quality
    fused_score: f32,
    /// Quality annotation details
    annotation: QualityAnnotation,
}

/// Build the RAG pipeline, index documents, and return scored query results.
fn run_pipeline(
    docs: &[Document],
    query: &str,
    k: usize,
) -> Result<Vec<ScoredResult>, Box<dyn std::error::Error>> {
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(trueno_rag::RecursiveChunker::new(512, 50))
        .embedder(MockEmbedder::new(384))
        .reranker(NoOpReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .build()?;

    pipeline.index_documents(docs)?;

    let results = pipeline.query(query, k * 2)?;

    let mut scored: Vec<ScoredResult> = results
        .into_iter()
        .map(|r| {
            let annotation = enrich_chunk(&r.chunk.content);
            let retrieval_score = r.best_score();
            let quality_score = annotation.quality_score();
            let fused_score = retrieval_score * (0.6 + 0.4 * quality_score);

            ScoredResult {
                content: r.chunk.content.clone(),
                source: r.chunk.metadata.title.clone(),
                retrieval_score,
                quality_score,
                fused_score,
                annotation,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);

    Ok(scored)
}

// ── Eval ────────────────────────────────────────────────────────────────

/// A ground-truth Q&A pair for evaluation.
#[derive(Debug, Clone)]
struct EvalPair {
    query: &'static str,
    /// Expected content substrings that should appear in top results
    expected_terms: &'static [&'static str],
}

/// Built-in evaluation dataset for code intelligence queries.
fn eval_dataset() -> Vec<EvalPair> {
    vec![
        EvalPair {
            query: "error handling with Result type",
            expected_terms: &["Result", "Err", "Ok"],
        },
        EvalPair {
            query: "vector operations and SIMD",
            expected_terms: &["vec", "simd", "f32"],
        },
        EvalPair {
            query: "serialization and deserialization",
            expected_terms: &["Serialize", "Deserialize", "serde"],
        },
        EvalPair {
            query: "async runtime and concurrency",
            expected_terms: &["async", "await", "tokio"],
        },
        EvalPair {
            query: "test functions and assertions",
            expected_terms: &["test", "assert"],
        },
    ]
}

/// Run evaluation: index sample code, query with ground-truth pairs, compute metrics.
fn run_eval() -> Result<(), Box<dyn std::error::Error>> {
    let sample_docs = sample_code_documents();

    let mut pipeline = RagPipelineBuilder::new()
        .chunker(trueno_rag::RecursiveChunker::new(256, 30))
        .embedder(MockEmbedder::new(384))
        .reranker(NoOpReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .build()?;

    let total_chunks = pipeline.index_documents(&sample_docs)?;
    println!(
        "Indexed {total_chunks} chunks from {} documents\n",
        sample_docs.len()
    );

    let dataset = eval_dataset();
    let k_values = [1, 3, 5];

    for pair in &dataset {
        let results = pipeline.query(pair.query, 5)?;
        let retrieved_ids: Vec<ChunkId> = results.iter().map(|r| r.chunk.id).collect();

        // Mark chunks as relevant if they contain any expected term
        let relevant_ids: HashSet<ChunkId> = results
            .iter()
            .filter(|r| {
                let content_lower = r.chunk.content.to_lowercase();
                pair.expected_terms
                    .iter()
                    .any(|term| content_lower.contains(&term.to_lowercase()))
            })
            .map(|r| r.chunk.id)
            .collect();

        let metrics = RetrievalMetrics::compute(&retrieved_ids, &relevant_ids, &k_values);

        println!("Query: \"{}\"", pair.query);
        println!("  MRR:        {:.3}", metrics.mrr);
        for &k in &k_values {
            println!(
                "  Recall@{k}:   {:.3}  Precision@{k}: {:.3}  NDCG@{k}: {:.3}",
                metrics.recall.get(&k).unwrap_or(&0.0),
                metrics.precision.get(&k).unwrap_or(&0.0),
                metrics.ndcg.get(&k).unwrap_or(&0.0),
            );
        }
        println!();
    }

    Ok(())
}

/// Sample code documents for evaluation (self-contained, no external files needed).
fn sample_code_documents() -> Vec<Document> {
    vec![
        Document::new(
            r#"use std::io::Result;

/// Read configuration from a TOML file.
/// Returns Err if the file is missing or malformed.
fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(config)
}

#[derive(serde::Deserialize)]
struct Config {
    model_name: String,
    batch_size: usize,
}"#,
        )
        .with_title("config.rs"),
        Document::new(
            r#"use serde::{Serialize, Deserialize};

/// Embedding vector with metadata for RAG indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Embedding {
    values: Vec<f32>,
    dimension: usize,
}

impl Embedding {
    fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.values.iter().zip(&other.values).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a * norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}"#,
        )
        .with_title("embedding.rs"),
        Document::new(
            r#"/// Chunk text using recursive splitting at semantic boundaries.
fn recursive_chunk(text: &str, max_size: usize, overlap: usize) -> Vec<String> {
    let separators = ["\n\n", "\n", ". ", " "];
    split_recursive(text, &separators, max_size, overlap)
}

fn split_recursive(text: &str, seps: &[&str], max_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= max_size {
        return vec![text.to_string()];
    }
    for &sep in seps {
        let parts: Vec<&str> = text.split(sep).collect();
        if parts.len() > 1 {
            let mut chunks = Vec::new();
            let mut current = String::new();
            for part in parts {
                if current.len() + part.len() > max_size && !current.is_empty() {
                    chunks.push(current.clone());
                    let keep = current.len().saturating_sub(overlap);
                    current = current[keep..].to_string();
                }
                if !current.is_empty() { current.push_str(sep); }
                current.push_str(part);
            }
            if !current.is_empty() { chunks.push(current); }
            return chunks;
        }
    }
    // Character-level fallback
    text.as_bytes().chunks(max_size).map(|c| String::from_utf8_lossy(c).to_string()).collect()
}"#,
        )
        .with_title("chunker.rs"),
        Document::new(
            r#"use tokio::sync::mpsc;

/// Pipeline stage that processes items concurrently.
async fn process_batch<T: Send + 'static>(
    items: Vec<T>,
    concurrency: usize,
    handler: impl Fn(T) -> Result<(), Box<dyn std::error::Error>> + Send + Sync + Clone + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel(concurrency);
    for item in items {
        let handler = handler.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let result = handler(item);
            let _ = tx.send(result).await;
        });
    }
    drop(tx);
    while let Some(result) = rx.recv().await {
        result?;
    }
    Ok(())
}"#,
        )
        .with_title("pipeline.rs"),
        Document::new(
            r#"#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Embedding { values: vec![1.0, 0.0, 0.0], dimension: 3 };
        let b = Embedding { values: vec![1.0, 0.0, 0.0], dimension: 3 };
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Embedding { values: vec![1.0, 0.0], dimension: 2 };
        let b = Embedding { values: vec![0.0, 1.0], dimension: 2 };
        assert!(a.cosine_similarity(&b).abs() < 1e-6);
    }

    #[test]
    fn test_chunk_small_input() {
        let chunks = recursive_chunk("hello world", 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }
}"#,
        )
        .with_title("tests.rs"),
    ]
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index { dir } => {
            let docs = load_rust_sources(&dir)?;
            println!("Found {} .rs files in {}", docs.len(), dir.display());

            let mut pipeline = RagPipelineBuilder::new()
                .chunker(trueno_rag::RecursiveChunker::new(512, 50))
                .embedder(MockEmbedder::new(384))
                .reranker(NoOpReranker::new())
                .fusion(FusionStrategy::RRF { k: 60.0 })
                .build()?;

            let chunks = pipeline.index_documents(&docs)?;
            println!("Indexed {chunks} chunks across {} documents", docs.len());

            // Enrich and report quality distribution
            let mut grade_counts: HashMap<String, usize> = HashMap::new();
            for doc in &docs {
                let annotation = enrich_chunk(&doc.content);
                *grade_counts.entry(annotation.grade.clone()).or_default() += 1;
            }

            println!("\nQuality distribution:");
            for grade in &["A", "B", "C", "D", "F"] {
                let count = grade_counts.get(*grade).unwrap_or(&0);
                println!("  {grade}: {count}");
            }
        }

        Command::Query { text, k } => {
            // For demo purposes, use sample documents
            let docs = sample_code_documents();
            let results = run_pipeline(&docs, &text, k)?;

            println!("Top {k} results for: \"{text}\"\n");
            for (i, result) in results.iter().enumerate() {
                println!(
                    "#{} [score={:.3} retrieval={:.3} quality={:.3} grade={}]",
                    i + 1,
                    result.fused_score,
                    result.retrieval_score,
                    result.quality_score,
                    result.annotation.grade,
                );
                if let Some(src) = &result.source {
                    println!("   source: {src}");
                }
                if !result.annotation.faults.is_empty() {
                    println!("   faults: {}", result.annotation.faults.join(", "));
                }
                let preview: String = result.content.chars().take(120).collect();
                println!("   {preview}...\n");
            }
        }

        Command::Eval => {
            run_eval()?;
        }
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Chunk tests ─────────────────────────────────────────────────

    #[test]
    fn test_load_rust_sources_nonexistent() {
        let result = load_rust_sources(Path::new("/tmp/code_intel_rag_truly_nonexistent_dir_xyz"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_rust_sources_empty_dir() {
        let dir = std::env::temp_dir().join("code_intel_rag_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let docs = load_rust_sources(&dir).unwrap();
        assert!(docs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_rust_sources_with_files() {
        let dir = std::env::temp_dir().join("code_intel_rag_test_files");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hello.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("readme.md"), "# Not Rust").unwrap();

        let docs = load_rust_sources(&dir).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("hello.rs"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Enrich tests ────────────────────────────────────────────────

    #[test]
    fn test_heuristic_annotate_clean_code() {
        let ann = heuristic_annotate("fn add(a: i32, b: i32) -> i32 { a + b }");
        assert_eq!(ann.grade, "A");
        assert!(ann.faults.is_empty());
        assert_eq!(ann.complexity, 1);
    }

    #[test]
    fn test_heuristic_annotate_unwrap() {
        let ann = heuristic_annotate("let x = foo().unwrap();");
        assert!(ann.faults.contains(&"unwrap".to_string()));
    }

    #[test]
    fn test_heuristic_annotate_panic() {
        let ann = heuristic_annotate("if bad { panic!(\"oops\"); }");
        assert!(ann.faults.contains(&"panic".to_string()));
    }

    #[test]
    fn test_heuristic_annotate_unsafe() {
        let ann = heuristic_annotate("unsafe { *ptr }");
        assert!(ann.faults.contains(&"unsafe".to_string()));
    }

    #[test]
    fn test_heuristic_annotate_complex() {
        let code = "if a { if b { for x in y { match z { _ => {} } } } }";
        let ann = heuristic_annotate(code);
        assert!(ann.complexity >= 4);
    }

    #[test]
    fn test_heuristic_annotate_expect() {
        let ann = heuristic_annotate("let x = foo.expect(\"msg\");");
        assert!(ann.faults.contains(&"expect".to_string()));
    }

    #[test]
    fn test_quality_score_bounds() {
        let ann = QualityAnnotation {
            grade: "A".to_string(),
            complexity: 0,
            faults: vec![],
        };
        let score = ann.quality_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_quality_score_monotonic() {
        let a = QualityAnnotation {
            grade: "A".to_string(),
            complexity: 1,
            faults: vec![],
        };
        let f = QualityAnnotation {
            grade: "F".to_string(),
            complexity: 50,
            faults: vec!["unwrap".into(), "panic".into()],
        };
        assert!(a.quality_score() > f.quality_score());
    }

    #[test]
    fn test_grade_score_values() {
        assert_eq!(
            QualityAnnotation {
                grade: "A".into(),
                complexity: 0,
                faults: vec![]
            }
            .grade_score(),
            5.0
        );
        assert_eq!(
            QualityAnnotation {
                grade: "B".into(),
                complexity: 0,
                faults: vec![]
            }
            .grade_score(),
            4.0
        );
        assert_eq!(
            QualityAnnotation {
                grade: "C".into(),
                complexity: 0,
                faults: vec![]
            }
            .grade_score(),
            3.0
        );
        assert_eq!(
            QualityAnnotation {
                grade: "D".into(),
                complexity: 0,
                faults: vec![]
            }
            .grade_score(),
            2.0
        );
        assert_eq!(
            QualityAnnotation {
                grade: "F".into(),
                complexity: 0,
                faults: vec![]
            }
            .grade_score(),
            1.0
        );
    }

    // ── Pipeline tests ──────────────────────────────────────────────

    #[test]
    fn test_run_pipeline_returns_results() {
        let docs = sample_code_documents();
        let results = run_pipeline(&docs, "error handling", 3).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_run_pipeline_results_sorted_by_fused_score() {
        let docs = sample_code_documents();
        let results = run_pipeline(&docs, "serialization", 5).unwrap();
        for window in results.windows(2) {
            assert!(window[0].fused_score >= window[1].fused_score);
        }
    }

    #[test]
    fn test_run_pipeline_single_doc() {
        let docs =
            vec![Document::new("fn hello() { println!(\"world\"); }").with_title("hello.rs")];
        let results = run_pipeline(&docs, "hello function", 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    // ── Eval tests ──────────────────────────────────────────────────

    #[test]
    fn test_eval_dataset_not_empty() {
        let dataset = eval_dataset();
        assert!(!dataset.is_empty());
        for pair in &dataset {
            assert!(!pair.query.is_empty());
            assert!(!pair.expected_terms.is_empty());
        }
    }

    #[test]
    fn test_sample_documents_not_empty() {
        let docs = sample_code_documents();
        assert!(docs.len() >= 3);
        for doc in &docs {
            assert!(!doc.content.is_empty());
            assert!(doc.title.is_some());
        }
    }

    #[test]
    fn test_run_eval_completes() {
        // Smoke test: eval pipeline runs without panicking
        run_eval().unwrap();
    }

    // ── Parse tests ─────────────────────────────────────────────────

    #[test]
    fn test_parse_pmat_json_valid() {
        let json = r#"[{"grade": "B", "complexity": 8, "faults": ["unwrap"]}]"#;
        let ann = parse_pmat_json(json).unwrap();
        assert_eq!(ann.grade, "B");
        assert_eq!(ann.complexity, 8);
        assert_eq!(ann.faults, vec!["unwrap"]);
    }

    #[test]
    fn test_parse_pmat_json_minimal() {
        let json = r#"[{}]"#;
        let ann = parse_pmat_json(json).unwrap();
        assert_eq!(ann.grade, "C"); // default
        assert_eq!(ann.complexity, 10); // default
    }

    #[test]
    fn test_parse_pmat_json_empty_array() {
        let result = parse_pmat_json("[]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_pmat_json_invalid() {
        let result = parse_pmat_json("not json");
        assert!(result.is_err());
    }

    // ── Property tests ──────────────────────────────────────────────

    #[test]
    fn test_quality_score_always_bounded() {
        for grade in &["A", "B", "C", "D", "F"] {
            for complexity in [0, 1, 10, 50, 100] {
                for n_faults in 0..5 {
                    let ann = QualityAnnotation {
                        grade: grade.to_string(),
                        complexity,
                        faults: (0..n_faults).map(|i| format!("fault{i}")).collect(),
                    };
                    let score = ann.quality_score();
                    assert!(score >= 0.0, "score {score} < 0 for grade={grade} complexity={complexity} faults={n_faults}");
                    assert!(score <= 1.0, "score {score} > 1 for grade={grade} complexity={complexity} faults={n_faults}");
                }
            }
        }
    }

    #[test]
    fn test_fused_score_respects_quality() {
        // Higher quality should produce higher fused score for same retrieval score
        let high_q = QualityAnnotation {
            grade: "A".into(),
            complexity: 1,
            faults: vec![],
        };
        let low_q = QualityAnnotation {
            grade: "F".into(),
            complexity: 50,
            faults: vec!["unwrap".into()],
        };

        let retrieval = 0.8;
        let high_fused = retrieval * (0.6 + 0.4 * high_q.quality_score());
        let low_fused = retrieval * (0.6 + 0.4 * low_q.quality_score());
        assert!(high_fused > low_fused);
    }
}
