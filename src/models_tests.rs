use super::*;

fn serve_once(body: &'static str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = sock.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn fetch_parses_models_page() {
    let body = r#"{"data":[
        {"type":"model","id":"claude-haiku-9-9","display_name":"Claude Haiku 9.9",
         "created_at":"2027-01-01T00:00:00Z","line":"haiku","max_input_tokens":1000000,
         "capabilities":{"effort":{"supported":true,"low":{"supported":true},
                   "medium":{"supported":true},"high":{"supported":true},
                   "xhigh":{"supported":false},"max":{"supported":false}}}},
        {"type":"model","id":"claude-opus-4-5-20251101","created_at":"2025-11-24T00:00:00Z",
         "line":"opus","max_input_tokens":200000}
    ],"has_more":false,"first_id":"claude-haiku-9-9","last_id":"claude-opus-4-5-20251101"}"#;
    let base = serve_once(body);
    let s = fetch_models(&base, "test-token").await.unwrap();
    let haiku = &s.models[0];
    assert_eq!(haiku.id, "claude-haiku-9-9");
    assert_eq!(haiku.display_name.as_deref(), Some("Claude Haiku 9.9"));
    assert_eq!(haiku.line.as_deref(), Some("haiku"));
    assert_eq!(haiku.max_input_tokens, Some(1_000_000));
    assert_eq!(
        haiku.efforts,
        Some(vec![
            "low".to_string(),
            "medium".to_string(),
            "high".to_string()
        ])
    );
    let opus = &s.models[1];
    assert_eq!(opus.display_name, None);
    assert_eq!(opus.efforts, None);
}

#[tokio::test]
async fn fetch_failure_is_an_error_not_a_panic() {
    // Closed port: connection refused → Err, caller falls back to pinned.
    assert!(fetch_models("http://127.0.0.1:1", "t").await.is_err());
    // A 401 is an error too, never a partial catalog.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        if let Ok((mut sock, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
            );
        }
    });
    assert!(
        fetch_models(&format!("http://127.0.0.1:{port}"), "t")
            .await
            .is_err()
    );
}

#[test]
fn oauth_token_reads_claude_credentials() {
    let dir = std::env::temp_dir().join(format!("claude-sub-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("creds.json");
    std::fs::write(&path, r#"{"claudeAiOauth":{"accessToken":"tok-abc"}}"#).unwrap();
    assert_eq!(oauth_token_at(&path).as_deref(), Some("tok-abc"));
    std::fs::write(&path, r#"{"claudeAiOauth":{}}"#).unwrap();
    assert_eq!(oauth_token_at(&path), None);
    assert_eq!(oauth_token_at(&dir.join("missing.json")), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn catalog_lists_routes_with_windows_and_efforts() {
    let s = Snapshot::new(vec![
        ModelInfo {
            id: "claude-sonnet-5-5".into(),
            display_name: Some("Claude Sonnet 5.5".into()),
            created_at: "2026-09-28".into(),
            line: Some("sonnet".into()),
            max_input_tokens: Some(1_000_000),
            efforts: Some(vec!["high".to_string()]),
        },
        ModelInfo {
            id: "claude-haiku-9-9".into(),
            display_name: None,
            created_at: "2027-01-01".into(),
            line: Some("haiku".into()),
            max_input_tokens: Some(200_000),
            efforts: None,
        },
    ]);
    let got = catalog_from(&s);
    assert!(!got.models.is_empty());
    let sonnet = got
        .models
        .iter()
        .find(|m| m.id == "sonnet")
        .expect("sonnet in catalog");
    assert_eq!(sonnet.context_window, Some(1_000_000));
    assert_eq!(sonnet.reasoning_efforts, vec!["high".to_string()]);
    let haiku = got
        .models
        .iter()
        .find(|m| m.id == "haiku")
        .expect("haiku in catalog");
    assert_eq!(haiku.context_window, Some(200_000));
    // Unknown effort support → every level (status quo).
    assert!(haiku.reasoning_efforts.contains(&"xhigh".to_string()));
    // Fallback snapshot renders the pinned shape.
    let pinned = catalog_from(&catalog::current());
    assert!(pinned.models.iter().any(|m| m.id == "sonnet"));
}
