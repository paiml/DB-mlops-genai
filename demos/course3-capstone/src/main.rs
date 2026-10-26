//! C3 MLOps Capstone: Pull, inspect, serve, and query Qwen2.5-Coder-1.5B
//!
//! Demonstrates sovereign AI inference — a single pipeline that downloads
//! a 1.5B code model, starts an OpenAI-compatible API, and generates Rust code.
//!
//! All testable logic lives in `lib.rs`. This binary is a thin I/O shell.

use std::io::{BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use course3_capstone::{
    build_chat_body, build_health_request, build_post_request, check_health_status,
    fallback_model_path, parse_chat_content, parse_model_path, read_http_body,
};

const MODEL_REPO: &str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF";
const PORT: u16 = 18090;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

fn main() {
    println!("=== C3 MLOps Capstone: Sovereign AI Inference ===\n");

    // Stage 1: Pull model (idempotent — skips if already cached)
    println!("[1/4] Pulling {MODEL_REPO}...");
    let model_path = pull_model();
    println!("  Model cached at: {model_path}\n");

    // Stage 2: Inspect architecture
    println!("[2/4] Inspecting model architecture...");
    inspect_model(&model_path);
    println!();

    // Stage 3: Serve
    println!("[3/4] Starting API server on port {PORT}...");
    let mut server = start_server(&model_path);
    println!("  Server ready.\n");

    // Stage 4: Query
    println!("[4/4] Sending code generation request...");
    let response = query_server();
    println!("\n--- Generated Code ---");
    println!("{response}");
    println!("--- End ---\n");

    // Cleanup
    println!("Shutting down server...");
    let _ = server.kill();
    let _ = server.wait();
    println!("Done.");
}

/// Run `apr pull` and return the cached model path.
fn pull_model() -> String {
    let output = Command::new("apr")
        .args(["pull", MODEL_REPO])
        .output()
        .expect("failed to run `apr pull` — is apr in PATH?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("apr pull failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(path) = parse_model_path(&stdout) {
        return path;
    }

    if let Some(path) = fallback_model_path() {
        return path;
    }

    panic!("apr pull did not output a .gguf path:\n{stdout}");
}

/// Run `apr inspect` and print architecture details.
fn inspect_model(model_path: &str) {
    let status = Command::new("apr")
        .args(["inspect", model_path])
        .status()
        .expect("failed to run `apr inspect`");

    if !status.success() {
        eprintln!("  Warning: apr inspect exited with {status}");
    }
}

/// Spawn `apr serve` and poll /health until ready.
fn start_server(model_path: &str) -> Child {
    let mut child = Command::new("apr")
        .args(["serve", model_path, "--port", &PORT.to_string(), "--gpu"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn `apr serve`");

    let start = Instant::now();
    loop {
        if start.elapsed() > HEALTH_TIMEOUT {
            let _ = child.kill();
            panic!("server did not become healthy within {HEALTH_TIMEOUT:?}");
        }
        thread::sleep(Duration::from_millis(500));
        if health_check().is_ok() {
            break;
        }
    }

    child
}

/// GET /health — returns Ok(()) when server responds 200.
fn health_check() -> Result<(), String> {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{PORT}")).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

    let req = build_health_request(PORT);
    stream
        .write_all(req.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut status_line).map_err(|e| e.to_string())?;

    check_health_status(&status_line)
}

/// POST /v1/chat/completions with a code generation prompt.
fn query_server() -> String {
    let body_str = build_chat_body(
        "Write a Rust function that checks if a number is prime",
        100,
    );

    let mut stream =
        TcpStream::connect(format!("127.0.0.1:{PORT}")).expect("failed to connect to server");
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();

    let req = build_post_request(PORT, "/v1/chat/completions", &body_str);
    stream
        .write_all(req.as_bytes())
        .expect("failed to send request");

    let mut reader = BufReader::new(stream);
    let resp_body = read_http_body(&mut reader);

    parse_chat_content(resp_body.trim())
}
