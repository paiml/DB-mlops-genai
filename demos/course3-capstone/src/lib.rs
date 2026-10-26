//! Testable core logic for the C3 MLOps Capstone demo.
//!
//! All pure functions are extracted here so the binary (`main.rs`) is a thin
//! I/O shell. This split lets us hit 95%+ coverage without needing a live
//! `apr` server.

use std::io::{BufRead, Lines};

/// Parse the last `.gguf` path from `apr pull` stdout.
pub fn parse_model_path(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .rfind(|l| l.ends_with(".gguf"))
        .map(String::from)
}

/// Check whether a file path exists, returning it as `Some` if so.
pub fn check_model_exists(path: &str) -> Option<String> {
    if std::path::Path::new(path).exists() {
        Some(path.to_string())
    } else {
        None
    }
}

/// Return the known fallback cache path for Qwen2.5-Coder-1.5B-Instruct,
/// or `None` if the file doesn't exist.
pub fn fallback_model_path() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let known = format!("{home}/.cache/pacha/models/687674c6a817024c.gguf");
    check_model_exists(&known)
}

/// Build an HTTP/1.1 GET request for the health endpoint.
pub fn build_health_request(port: u16) -> String {
    format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
}

/// Check whether an HTTP status line indicates success (200).
pub fn check_health_status(status_line: &str) -> Result<(), String> {
    if status_line.contains("200") {
        Ok(())
    } else {
        Err(format!("health: {status_line}"))
    }
}

/// Build the JSON body for a chat completion request.
pub fn build_chat_body(prompt: &str, max_tokens: u32) -> String {
    serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    })
    .to_string()
}

/// Build an HTTP/1.1 POST request with a JSON body.
pub fn build_post_request(port: u16, path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

/// Read HTTP response body from a buffered reader, skipping the status line
/// and headers. Works with any `BufRead` — TCP streams, `Cursor<Vec<u8>>`, etc.
pub fn read_http_body(reader: &mut impl BufRead) -> String {
    // Skip status line
    let mut line = String::new();
    reader.read_line(&mut line).ok();

    // Skip headers
    loop {
        let mut hdr = String::new();
        reader.read_line(&mut hdr).ok();
        if hdr.trim().is_empty() {
            break;
        }
    }

    collect_body_lines(reader.lines())
}

/// Collect remaining lines into a single string.
fn collect_body_lines<B: BufRead>(lines: Lines<B>) -> String {
    let mut body = String::new();
    for l in lines.map_while(Result::ok) {
        body.push_str(&l);
        body.push('\n');
    }
    body
}

/// Extract `choices[0].message.content` from an OpenAI-compatible JSON response.
pub fn parse_chat_content(body: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(v) => v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("<no content in response>")
            .to_string(),
        Err(e) => format!("Failed to parse response: {e}\nRaw: {body}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── parse_model_path ──────────────────────────────────────────────

    #[test]
    fn parse_model_path_finds_last_gguf() {
        let stdout = "Downloading...\n/tmp/foo.gguf\n/home/user/.cache/bar.gguf\n";
        assert_eq!(
            parse_model_path(stdout),
            Some("/home/user/.cache/bar.gguf".to_string())
        );
    }

    #[test]
    fn parse_model_path_single_line() {
        assert_eq!(
            parse_model_path("/models/qwen.gguf"),
            Some("/models/qwen.gguf".to_string())
        );
    }

    #[test]
    fn parse_model_path_no_gguf() {
        assert_eq!(parse_model_path("Downloading model...\nDone.\n"), None);
    }

    #[test]
    fn parse_model_path_empty() {
        assert_eq!(parse_model_path(""), None);
    }

    #[test]
    fn parse_model_path_gguf_mid_line_ignored() {
        // ".gguf" must be at end of line
        assert_eq!(parse_model_path("file.gguf.bak\n"), None);
    }

    #[test]
    fn parse_model_path_mixed_output() {
        let stdout = "info: checking cache\nHit: abc123\n/cache/model.gguf\ninfo: done\n";
        assert_eq!(
            parse_model_path(stdout),
            Some("/cache/model.gguf".to_string())
        );
    }

    // ── check_model_exists / fallback_model_path ─────────────────────

    #[test]
    fn check_model_exists_returns_none_for_missing() {
        assert_eq!(check_model_exists("/nonexistent/path/model.gguf"), None);
    }

    #[test]
    fn check_model_exists_returns_some_for_existing_file() {
        // Cargo.toml always exists in this crate
        let result = check_model_exists("Cargo.toml");
        assert_eq!(result, Some("Cargo.toml".to_string()));
    }

    #[test]
    fn fallback_model_path_returns_some_or_none() {
        let result = fallback_model_path();
        if let Some(ref path) = result {
            assert!(path.ends_with(".gguf"));
        }
    }

    // ── build_health_request ──────────────────────────────────────────

    #[test]
    fn health_request_format() {
        let req = build_health_request(8080);
        assert!(req.starts_with("GET /health HTTP/1.1\r\n"));
        assert!(req.contains("Host: 127.0.0.1:8080"));
        assert!(req.contains("Connection: close"));
        assert!(req.ends_with("\r\n\r\n"));
    }

    #[test]
    fn health_request_different_port() {
        let req = build_health_request(18090);
        assert!(req.contains("Host: 127.0.0.1:18090"));
    }

    // ── check_health_status ───────────────────────────────────────────

    #[test]
    fn health_status_200_ok() {
        assert!(check_health_status("HTTP/1.1 200 OK\r\n").is_ok());
    }

    #[test]
    fn health_status_503_err() {
        let err = check_health_status("HTTP/1.1 503 Service Unavailable\r\n");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("503"));
    }

    #[test]
    fn health_status_empty_err() {
        assert!(check_health_status("").is_err());
    }

    #[test]
    fn health_status_partial_200() {
        // "200" anywhere in line counts
        assert!(check_health_status("200").is_ok());
    }

    // ── build_chat_body ───────────────────────────────────────────────

    #[test]
    fn chat_body_valid_json() {
        let body = build_chat_body("hello", 50);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "default");
        assert_eq!(v["max_tokens"], 50);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hello");
    }

    #[test]
    fn chat_body_escapes_special_chars() {
        let body = build_chat_body("say \"hello\" & <world>", 10);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["messages"][0]["content"], "say \"hello\" & <world>");
    }

    // ── build_post_request ────────────────────────────────────────────

    #[test]
    fn post_request_format() {
        let req = build_post_request(18090, "/v1/chat/completions", "{\"a\":1}");
        assert!(req.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
        assert!(req.contains("Host: 127.0.0.1:18090"));
        assert!(req.contains("Content-Type: application/json"));
        assert!(req.contains("Content-Length: 7"));
        assert!(req.contains("Connection: close"));
        assert!(req.contains("{\"a\":1}"));
    }

    #[test]
    fn post_request_content_length_matches_body() {
        let body = r#"{"messages":[]}"#;
        let req = build_post_request(8080, "/api", body);
        let expected = format!("Content-Length: {}", body.len());
        assert!(req.contains(&expected));
    }

    // ── read_http_body ────────────────────────────────────────────────

    #[test]
    fn read_body_skips_status_and_headers() {
        let raw = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}\n";
        let mut cursor = Cursor::new(raw.as_bytes());
        let body = read_http_body(&mut cursor);
        assert_eq!(body.trim(), "{\"ok\":true}");
    }

    #[test]
    fn read_body_multiple_headers() {
        let raw = "HTTP/1.1 200 OK\r\n\
                    Content-Type: text/plain\r\n\
                    X-Custom: value\r\n\
                    Server: test\r\n\
                    \r\n\
                    hello world\n";
        let mut cursor = Cursor::new(raw.as_bytes());
        let body = read_http_body(&mut cursor);
        assert_eq!(body.trim(), "hello world");
    }

    #[test]
    fn read_body_empty_body() {
        let raw = "HTTP/1.1 204 No Content\r\n\r\n";
        let mut cursor = Cursor::new(raw.as_bytes());
        let body = read_http_body(&mut cursor);
        assert_eq!(body, "");
    }

    #[test]
    fn read_body_multiline_body() {
        let raw = "HTTP/1.1 200 OK\r\n\r\nline1\nline2\nline3\n";
        let mut cursor = Cursor::new(raw.as_bytes());
        let body = read_http_body(&mut cursor);
        assert_eq!(body, "line1\nline2\nline3\n");
    }

    // ── collect_body_lines ────────────────────────────────────────────

    #[test]
    fn collect_lines_empty() {
        let cursor = Cursor::new(b"" as &[u8]);
        assert_eq!(collect_body_lines(cursor.lines()), "");
    }

    #[test]
    fn collect_lines_multiple() {
        let cursor = Cursor::new(b"a\nb\nc\n" as &[u8]);
        assert_eq!(collect_body_lines(cursor.lines()), "a\nb\nc\n");
    }

    // ── parse_chat_content ────────────────────────────────────────────

    #[test]
    fn parse_valid_openai_response() {
        let json =
            r#"{"choices":[{"message":{"content":"fn is_prime(n: u64) -> bool { true }"}}]}"#;
        assert_eq!(
            parse_chat_content(json),
            "fn is_prime(n: u64) -> bool { true }"
        );
    }

    #[test]
    fn parse_empty_content() {
        let json = r#"{"choices":[{"message":{"content":""}}]}"#;
        assert_eq!(parse_chat_content(json), "");
    }

    #[test]
    fn parse_null_content() {
        let json = r#"{"choices":[{"message":{"content":null}}]}"#;
        assert_eq!(parse_chat_content(json), "<no content in response>");
    }

    #[test]
    fn parse_missing_choices() {
        let json = r#"{"error":"internal"}"#;
        assert_eq!(parse_chat_content(json), "<no content in response>");
    }

    #[test]
    fn parse_invalid_json() {
        let result = parse_chat_content("not json at all");
        assert!(result.starts_with("Failed to parse response:"));
        assert!(result.contains("not json at all"));
    }

    #[test]
    fn parse_nested_content_with_escapes() {
        let json = r#"{"choices":[{"message":{"content":"line1\nline2"}}]}"#;
        let result = parse_chat_content(json);
        assert_eq!(result, "line1\nline2");
    }

    #[test]
    fn parse_multiple_choices_takes_first() {
        let json =
            r#"{"choices":[{"message":{"content":"first"}},{"message":{"content":"second"}}]}"#;
        assert_eq!(parse_chat_content(json), "first");
    }

    // ── integration: build + parse round-trip ─────────────────────────

    #[test]
    fn chat_body_round_trip() {
        let body = build_chat_body("test prompt", 100);
        // Wrap in a mock OpenAI response
        let response = format!(
            r#"{{"choices":[{{"message":{{"content":"generated code"}}}}],"usage":{{"total_tokens":5}}}}"#
        );
        let content = parse_chat_content(&response);
        assert_eq!(content, "generated code");
        // Verify the request body is valid JSON
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["messages"][0]["content"], "test prompt");
    }

    #[test]
    fn post_request_then_read_response() {
        let req = build_post_request(18090, "/v1/chat/completions", "{}");
        assert!(req.contains("POST /v1/chat/completions"));

        let raw_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}\n";
        let mut cursor = Cursor::new(raw_response.as_bytes());
        let body = read_http_body(&mut cursor);
        let content = parse_chat_content(body.trim());
        assert_eq!(content, "hello");
    }
}
