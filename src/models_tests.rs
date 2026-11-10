use super::*;

/// The saved overview page: a copy of the public URL's body.
const PAGE: &str = include_str!("../tests/fixtures/models-overview.md");

/// `(id, window)` pairs, the shape the assertions compare.
fn pairs(models: &[ModelInfo]) -> Vec<(&str, Option<u32>)> {
    models.iter().map(|m| (m.id.as_str(), m.window)).collect()
}

/// A one-shot local server: reads one request head, sends it on the channel,
/// and answers with `status` and `body`. Returns the page URL and the channel.
fn serve_once(
    status: &'static str,
    body: &'static str,
) -> (String, std::sync::mpsc::Receiver<String>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match sock.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                }
            }
            let _ = tx.send(String::from_utf8_lossy(&head).into_owned());
            let resp = format!(
                "HTTP/1.1 {status}\r\ncontent-type: text/markdown; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    (format!("http://127.0.0.1:{port}/models/overview.md"), rx)
}

#[test]
fn default_url_is_the_public_page_and_the_override_wins() {
    assert_eq!(
        models_url(None),
        "https://platform.claude.com/docs/en/models/overview.md"
    );
    assert_eq!(models_url(Some("  ")), DEFAULT_MODELS_URL);
    assert_eq!(
        models_url(Some(" http://127.0.0.1:9/x.md ")),
        "http://127.0.0.1:9/x.md"
    );
}

#[test]
fn parses_the_saved_page() {
    let models = parse_overview(PAGE).unwrap();
    // Every current column's `Default effort` cell names a tier, so each takes
    // all of them; the legacy models the page only links to state none.
    for m in &models[..4] {
        assert_eq!(m.efforts, EFFORTS, "{}", m.id);
    }
    for m in &models[4..] {
        assert!(m.efforts.is_empty(), "{}", m.id);
    }
    assert_eq!(
        pairs(&models),
        vec![
            ("claude-fable-5-1", Some(1_000_000)),
            ("claude-opus-5-5", Some(1_000_000)),
            ("claude-sonnet-5-5", Some(1_000_000)),
            ("claude-haiku-5-5", Some(1_000_000)),
            // Legacy: after the current models, in the order the page lists them.
            ("claude-fable-5", None),
            ("claude-opus-5", None),
            ("claude-opus-4-8", None),
            ("claude-opus-4-7", None),
            ("claude-opus-4-6", None),
            ("claude-opus-4-5", None),
            ("claude-sonnet-5", None),
            ("claude-sonnet-4-6", None),
            ("claude-haiku-4-5", None),
        ]
    );
}

#[test]
fn malformed_pages_are_errors_not_short_lists() {
    // Not the overview at all: an error page, or nothing.
    assert!(parse_overview("").is_err());
    // The same table without its `Default effort` row: an error, never a
    // silent "no effort" for every model.
    let no_effort_row = PAGE
        .lines()
        .filter(|l| !l.starts_with("| [Default effort]"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(parse_overview(&no_effort_row).is_err());
    // A torn `Default effort` cell is an error too, not "no effort".
    let torn = PAGE.replacen("`high`", "`hi", 1);
    assert!(parse_overview(&torn).is_err());
    assert!(parse_overview("<html><body>Service unavailable</body></html>").is_err());
    // A truncated download, cut before the API id row.
    let cut = PAGE.find("Claude API ID").unwrap();
    assert!(parse_overview(&PAGE[..cut]).is_err());
    // Cut mid-cell inside an id: the torn row is an error — never a bogus
    // `claude-h`-style short list (which would overwrite a good cache).
    let cut = PAGE.find("`claude-haiku-5-5`").unwrap() + 7;
    assert!(parse_overview(&PAGE[..cut]).is_err());
    // Cut mid-cell in a later row of the same table.
    let cut = PAGE.find("1M tokens").unwrap() + 3;
    assert!(parse_overview(&PAGE[..cut]).is_err());
    // An unterminated backtick is a torn cell, not a model id — even when
    // the row still spans the full table.
    assert!(
        parse_overview(
            "| Feature | A | B |
| --- | --- | --- |
| Claude API ID | `claude-opus-5-5` | `claude-son |"
        )
        .is_err()
    );
    assert_eq!(backticked("`claude-son"), None);
    assert_eq!(
        backticked("`claude-sonnet-5-5` and more"),
        Some("claude-sonnet-5-5")
    );
    // A table with no API id row.
    assert!(
        parse_overview("| Feature | Claude Opus 5.5 |\n| --- | --- |\n| Pricing | $4 / MTok |\n")
            .is_err()
    );
    // An API id row whose cells hold no model id.
    assert!(parse_overview("| Claude API ID | n/a | `not-a-model` |\n").is_err());
    // Legacy links alone are not the page.
    assert!(
        parse_overview(
            "Legacy models (still available): [X](https://platform.claude.com/docs/en/models/x-1/overview)\n"
        )
        .is_err()
    );
}

#[test]
fn window_cells_read_as_tokens_or_unknown() {
    assert_eq!(window_tokens("1M tokens"), Some(1_000_000));
    assert_eq!(window_tokens(" 200k tokens "), Some(200_000));
    assert_eq!(window_tokens("200K tokens"), Some(200_000));
    // Anything not stated as a whole count of k or M tokens is unknown.
    assert_eq!(window_tokens("1.5M tokens"), None);
    assert_eq!(window_tokens("1,000,000 tokens"), None);
    assert_eq!(window_tokens("—"), None);
    assert_eq!(window_tokens(""), None);
    assert_eq!(window_tokens("0 tokens"), None);
}

#[test]
fn window_row_of_another_shape_is_not_trusted() {
    let page = "| Feature | A | B |\n\
                | Claude API ID | `claude-x-1` | `claude-x-2` |\n\
                | Default effort | `high` | — |\n\
                | [Context window](https://example.com/w) | 1M tokens | 200k tokens | extra |\n";
    let got = parse_overview(page).unwrap();
    assert_eq!(
        pairs(&got),
        vec![("claude-x-1", None), ("claude-x-2", None)]
    );
    // The dash cell names no default tier: that column takes no effort.
    assert_eq!(got[0].efforts.len(), EFFORTS.len());
    assert!(got[1].efforts.is_empty());
}

#[test]
fn legacy_links_dedupe_and_skip_other_links() {
    let page = "| Feature | A |\n\
                | Claude API ID | `claude-x-1` |\n\
                | Default effort | `medium` |\n\
                | [Context window](https://example.com/w) | 1M tokens |\n\
                \n\
                Legacy models (still available): [X 1](https://platform.claude.com/docs/en/models/x-1/overview), \
                [X 0](https://platform.claude.com/docs/en/models/x-0/overview), \
                [Pricing](https://platform.claude.com/docs/en/about-claude/pricing), \
                [X 0 again](https://platform.claude.com/docs/en/models/x-0/overview).\n";
    let got = parse_overview(page).unwrap();
    assert_eq!(
        pairs(&got),
        vec![("claude-x-1", Some(1_000_000)), ("claude-x-0", None)]
    );
    assert_eq!(got[0].efforts, EFFORTS);
    assert!(got[1].efforts.is_empty());
}

#[test]
fn catalog_lists_routes_with_windows_and_their_efforts() {
    let s = Snapshot::new(vec![
        ModelInfo {
            id: "claude-sonnet-9-9".into(),
            window: Some(1_000_000),
            efforts: EFFORTS.iter().map(|e| e.to_string()).collect(),
        },
        ModelInfo {
            id: "claude-haiku-9-9".into(),
            window: Some(200_000),
            efforts: Vec::new(),
        },
    ]);
    let got = catalog_from(&s);
    let sonnet = got
        .models
        .iter()
        .find(|m| m.id == "sonnet")
        .expect("sonnet in catalog");
    assert_eq!(sonnet.name, "Claude Sonnet 9.9");
    assert_eq!(sonnet.context_window, Some(1_000_000));
    let haiku = got
        .models
        .iter()
        .find(|m| m.id == "haiku")
        .expect("haiku in catalog");
    assert_eq!(haiku.context_window, Some(200_000));
    assert!(got.models.iter().any(|m| m.id == "claude-sonnet-9-9[1m]"));
    // A route carries its model's tiers; a model the page names none for
    // offers none, and a `[1m]` route inherits its model's tiers.
    assert_eq!(sonnet.reasoning_efforts, EFFORTS);
    assert_eq!(haiku.reasoning_efforts, Vec::<String>::new());
    let sonnet_1m = got
        .models
        .iter()
        .find(|m| m.id == "claude-sonnet-9-9[1m]")
        .expect("1M route in catalog");
    assert_eq!(sonnet_1m.reasoning_efforts, EFFORTS);
}

#[tokio::test]
async fn fetch_reads_the_page_with_no_credentials() {
    let (url, seen) = serve_once("200 OK", PAGE);
    let page = fetch_page(&url, "models page").await.unwrap();
    assert_eq!(
        parse_overview(&page).unwrap(),
        parse_overview(PAGE).unwrap()
    );
    let request = seen.recv().unwrap().to_ascii_lowercase();
    assert!(request.starts_with("get /models/overview.md "), "{request}");
    for header in [
        "authorization:",
        "x-api-key:",
        "anthropic-beta:",
        "anthropic-version:",
        "cookie:",
    ] {
        assert!(
            !request.contains(header),
            "request carries {header}: {request}"
        );
    }
}

#[tokio::test]
async fn failed_fetches_are_errors() {
    // Nothing listens on this port: a connection error, never a partial page.
    assert!(
        fetch_page("http://127.0.0.1:1/models/overview.md", "models page")
            .await
            .is_err()
    );
    // An error status is an error, even with a body.
    let (url, _seen) = serve_once("503 Service Unavailable", PAGE);
    let err = fetch_page(&url, "models page").await.unwrap_err();
    assert!(err.contains("503"), "{err}");
}
