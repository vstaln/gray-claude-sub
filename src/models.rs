//! Model catalog: ids, windows, names and effort support come from a live
//! `GET /v1/models` on the user's OAuth login (`catalog::snapshot()`);
//! when discovery can't run — no credential file, offline, 401 — the
//! pinned fallback table in `catalog` answers instead.
//!
//! The token is read straight from `$CLAUDE_CONFIG_DIR/.credentials.json`
//! (default `~/.claude/.credentials.json`): external login means the host
//! never hands us credential material. It is a bearer only — never
//! logged, never echoed into an error.

use std::sync::Arc;

use gray_plugin::{ProviderModel, ProviderModelCatalog};

use crate::catalog::{self, ModelInfo, Snapshot};

/// Models API root. `CLAUDE_SUB_MODELS_BASE_URL` overrides it (tests,
/// gateways); nothing per-release lives in this value.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Discovery is a picker nicety, never worth a hang: cap the whole call.
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
/// Pages are 100 entries; 5 pages bounds a runaway `has_more`.
const MAX_PAGES: usize = 5;

/// Render a snapshot as the protocol catalog. Pure — tests drive it with
/// synthetic snapshots, never the network.
pub fn catalog_from(s: &Snapshot) -> ProviderModelCatalog {
    let models = catalog::all_ids_in(s)
        .into_iter()
        .map(|id| {
            let canon = id.strip_suffix("[1m]").unwrap_or(&id);
            let efforts = s
                .models
                .iter()
                .find(|m| canon == m.id || canon == catalog::family(m))
                .and_then(|m| m.efforts.clone())
                .unwrap_or_else(|| catalog::EFFORTS.iter().map(|e| e.to_string()).collect());
            ProviderModel {
                name: catalog::display_name_in(&id, s),
                context_window: catalog::context_window_in(&id, s),
                reasoning_efforts: efforts,
                variants: Vec::new(),
                slots: Vec::new(),
                id,
            }
        })
        .collect();
    ProviderModelCatalog { models }
}

/// The live catalog: fetch once per process, fall back to pinned.
pub async fn catalog() -> ProviderModelCatalog {
    let snap = catalog::snapshot().await;
    catalog_from(&snap)
}

/// Where the Claude CLI keeps credentials: `$CLAUDE_CONFIG_DIR` or
/// `~/.claude`, same root [`crate::session`] derives for `projects/`.
fn credentials_path() -> Option<std::path::PathBuf> {
    std::env::var_os("CLAUDE_SUB_CREDENTIALS_FILE")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("CLAUDE_CONFIG_DIR")
                .filter(|v| !v.is_empty())
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".claude"))
                })
                .map(|d| d.join(".credentials.json"))
        })
}

/// The OAuth access token out of a credential file. `None` when the file
/// is missing, unreadable or carries no token — the caller treats that
/// as "can't discover", not an error. Never logged.
fn oauth_token_at(path: &std::path::Path) -> Option<String> {
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    value
        .pointer("/claudeAiOauth/accessToken")
        .and_then(serde_json::Value::as_str)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

fn oauth_token() -> Option<String> {
    oauth_token_at(&credentials_path()?)
}

/// `GET {base}/v1/models` → a newest-first [`Snapshot`]. Errors are plain
/// strings with no token material — safe to log upstream.
async fn fetch_models(base: &str, token: &str) -> Result<Arc<Snapshot>, String> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let mut after: Option<String> = None;
    let mut models: Vec<ModelInfo> = Vec::new();
    for _ in 0..MAX_PAGES {
        let mut url = format!("{base}/v1/models?limit=100");
        if let Some(a) = &after {
            url.push_str("&after_id=");
            url.push_str(a);
        }
        let resp = client
            .get(&url)
            .bearer_auth(token)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "oauth-2025-04-20")
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("models list HTTP {}", resp.status()));
        }
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        for item in body
            .get("data")
            .and_then(|d| d.as_array())
            .into_iter()
            .flatten()
        {
            if let Some(m) = parse_model(item) {
                models.push(m);
            }
        }
        let has_more = body
            .get("has_more")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let last = body
            .get("last_id")
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty());
        match (has_more, last) {
            (true, Some(last)) => after = Some(last.to_string()),
            _ => break,
        }
    }
    if models.is_empty() {
        return Err("models list returned no models".into());
    }
    Ok(Arc::new(Snapshot::new(models)))
}

/// One `/v1/models` item → [`ModelInfo`]. Only the fields we use are
/// read; a missing one stays `None` and falls back later.
fn parse_model(item: &serde_json::Value) -> Option<ModelInfo> {
    let id = item.get("id").and_then(serde_json::Value::as_str)?;
    if id.is_empty() {
        return None;
    }
    // `effort` rides inside `capabilities`; a top-level key is tolerated.
    let efforts = item
        .pointer("/capabilities/effort")
        .or_else(|| item.get("effort"))
        .and_then(|e| {
            let supported = |level: &str| {
                e.get(level)
                    .and_then(|v| v.get("supported"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            };
            // `effort.supported` alone isn't enough — a model can support the
            // flag while some levels are off. Only a nonempty supported set
            // overrides the default report (status quo for unknown models).
            let levels: Vec<String> = catalog::EFFORTS
                .iter()
                .filter(|l| supported(l))
                .map(|l| l.to_string())
                .collect();
            (!levels.is_empty()).then_some(levels)
        });
    Some(ModelInfo {
        id: id.to_string(),
        display_name: item
            .get("display_name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        created_at: item
            .get("created_at")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        line: item
            .get("line")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        max_input_tokens: item
            .get("max_input_tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32),
        efforts,
    })
}

/// The fetch [`catalog::snapshot`] installs: credentials + base URL from
/// env, then the API. `Err` strings carry no secrets.
pub async fn fetch_snapshot() -> Result<Arc<Snapshot>, String> {
    let token = oauth_token().ok_or_else(|| "no Claude credential file".to_string())?;
    let base = std::env::var("CLAUDE_SUB_MODELS_BASE_URL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    fetch_models(base.trim_end_matches('/'), &token).await
}

#[path = "models_tests.rs"]
#[cfg(test)]
mod tests;
