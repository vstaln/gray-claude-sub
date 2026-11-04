//! Native route catalog: the models Anthropic's public models page lists,
//! with the last successful parse cached on disk. Port of the Hermes
//! DirectSDK `model_catalog.py` (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT).
//!
//! A loopback gateway needs explicit long-context selection: native runs a
//! plain id inside its 200K default and selects 1M only for an explicit
//! `[1m]` route. Routable ids and their windows come from the public page
//! ([`crate::models`]): nothing per-release is compiled in, and no credential
//! is read or sent. Each successful parse is cached at
//! `<gray home>/claude-sub/models-cache.json`, and that cache answers when a
//! fetch or parse fails. With no cache the catalog is empty: nothing is
//! invented, and unknown ids pass through to native.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use serde::{Deserialize, Serialize};

/// One model the models page lists: its API id and context window. A `None`
/// window means the page doesn't state one, so no `[1m]` route is offered.
/// The same shape is the disk cache's entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// Context window in tokens, from the page's `Context window` row.
    #[serde(default)]
    pub window: Option<u32>,
    /// Effort tiers the page's `Default effort` row earns this model (the
    /// tier list is [`crate::models::EFFORTS`]). Empty = the page names no
    /// effort for it, so the host shows no effort selector.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub efforts: Vec<String>,
}

/// A resolved view of the catalog: parsed, cached or empty. `models` keeps the
/// page's order (current models in table order, then legacy as listed), so a
/// "first match" pick is the newest release of a family or prefix.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub models: Vec<ModelInfo>,
}

impl Snapshot {
    pub(crate) fn new(models: Vec<ModelInfo>) -> Self {
        Self { models }
    }

    pub(crate) fn by_id(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// The cache's file name under `<gray home>/claude-sub/`.
const CACHE_FILE: &str = "models-cache.json";

/// Where the disk cache lives. `None` when no gray home resolves, which
/// means there is no cache at all.
fn cache_path() -> Option<PathBuf> {
    crate::settings::state_dir().map(|dir| dir.join(CACHE_FILE))
}

/// The cache's on-disk shape: the models of the last successful parse.
#[derive(Serialize, Deserialize)]
struct CacheFile {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

/// The cached list, or `None` when the file is missing, unreadable, or
/// holds no routes. Never fetches.
fn load_cache_at(path: &Path) -> Option<Snapshot> {
    let file: CacheFile = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let models: Vec<ModelInfo> = file
        .models
        .into_iter()
        .filter(|m| !m.id.is_empty())
        .collect();
    (!models.is_empty()).then(|| Snapshot::new(models))
}

/// Persist a successful parse. The body goes to a sibling temp file, is
/// synced, then renamed over the cache: a reader sees the old file or the
/// new one, never a torn write. Best-effort: a failure only costs the next
/// cold start its cache.
fn write_cache_at(path: &Path, models: &[ModelInfo]) -> std::io::Result<()> {
    let body = serde_json::to_vec(&CacheFile {
        models: models.to_vec(),
    })?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let written = std::fs::File::create(&tmp)
        .and_then(|mut file| {
            file.write_all(&body)?;
            file.sync_all()
        })
        .and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// The catalog before any fetch: the cache if it reads, else empty.
fn initial_snapshot_from(path: Option<&Path>) -> Snapshot {
    path.and_then(load_cache_at)
        .unwrap_or_else(|| Snapshot::new(Vec::new()))
}

/// The snapshot everyone reads: the cache (or empty) until the first
/// successful fetch swaps in live data. Written once per process under
/// [`FETCH_LOCK`].
static SNAPSHOT: LazyLock<RwLock<Arc<Snapshot>>> =
    LazyLock::new(|| RwLock::new(Arc::new(initial_snapshot_from(cache_path().as_deref()))));
/// Set once live data is installed; failures leave it clear so a later
/// `provider/models` retries (transient errors heal).
static FETCHED: AtomicBool = AtomicBool::new(false);
static FETCH_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Whatever the catalog currently knows — never blocks on the network.
/// Until the first successful [`snapshot`] that is the cache, or empty.
pub fn current() -> Arc<Snapshot> {
    SNAPSHOT.read().unwrap().clone()
}

/// Install a fetched snapshot (tests and the fetch path). Marks the
/// catalog live so no further fetches run this process.
pub fn install(s: Arc<Snapshot>) {
    *SNAPSHOT.write().unwrap() = s;
    FETCHED.store(true, Ordering::Relaxed);
}

/// The live catalog for a caller that needs routes: fetch once per process,
/// then install and cache what comes back. A failed fetch or parse keeps what
/// is known and errs, with the reason, only when there is nothing to show.
pub async fn snapshot() -> Result<Arc<Snapshot>, String> {
    if FETCHED.load(Ordering::Relaxed) {
        return Ok(current());
    }
    let _guard = FETCH_LOCK.lock().await;
    if FETCHED.load(Ordering::Relaxed) {
        return Ok(current());
    }
    match crate::models::fetch_snapshot().await {
        Ok(s) => {
            install(s.clone());
            if let Some(path) = cache_path() {
                // Best-effort: the parsed list is already in place.
                let _ = write_cache_at(&path, &s.models);
            }
            Ok(s)
        }
        Err(why) => settle_failed_fetch(current(), why),
    }
}

/// A failed fetch: what is already known (the cache) still answers. With
/// nothing known, the failure itself is the answer.
fn settle_failed_fetch(known: Arc<Snapshot>, why: String) -> Result<Arc<Snapshot>, String> {
    if known.models.is_empty() {
        Err(why)
    } else {
        Ok(known)
    }
}

/// The family an id belongs to: the word after `claude-` in `claude-<family>-…`.
pub(crate) fn family(m: &ModelInfo) -> &str {
    m.id.strip_prefix("claude-")
        .and_then(|rest| rest.split('-').next())
        .unwrap_or(&m.id)
}

/// Resolve a route to a canonical id inside `s`. Unknown input passes
/// through untouched.
fn canonical_in<'a>(model: &'a str, s: &'a Snapshot) -> &'a str {
    let base = model.strip_suffix("[1m]").unwrap_or(model);
    resolve(base, s).unwrap_or(base)
}

/// One resolution pass; the first match in list order wins. Exact id, then a
/// family alias (a bare family name, or `claude-<family>`, picks the family's
/// first listed id), then a prefix on a hyphen boundary (`claude-<x>` picks
/// the first id that continues `claude-<x>-…`). Last: a dated tail variant of
/// a known id. A tail of eight or more digits (a date) is stripped and the
/// stem resolved, so a dated id inherits its window and routes instead of
/// falling to the 200K default.
fn resolve<'a>(base: &'a str, s: &'a Snapshot) -> Option<&'a str> {
    if s.by_id(base).is_some() {
        return Some(base);
    }
    let name = base.strip_prefix("claude-").unwrap_or(base);
    if let Some(m) = s.models.iter().find(|m| family(m) == name) {
        return Some(&m.id);
    }
    // Prefix on a hyphen boundary: `claude-<x>` → the first id `claude-<x>-…`.
    if let Some(m) = s.models.iter().find(|m| {
        m.id.len() > base.len() && m.id.starts_with(base) && m.id.as_bytes()[base.len()] == b'-'
    }) {
        return Some(&m.id);
    }
    if let Some((stem, tail)) = base.rsplit_once('-')
        && tail.len() >= 8
        && tail.bytes().all(|b| b.is_ascii_digit())
    {
        return resolve(stem, s);
    }
    None
}

/// The window the snapshot reports for an id: the page's window for a known
/// id. `None` when the id is unknown or the page didn't state one.
fn known_window(id: &str, s: &Snapshot) -> Option<u32> {
    s.by_id(id).and_then(|m| m.window)
}

/// Context window for a route: the page's window for a known id. A plain id
/// without one runs at native's 200K default, so that is its window; a `[1m]`
/// route without one has no window (`None`): no guess reaches the 1M budget.
pub fn context_window_in(model: &str, s: &Snapshot) -> Option<u32> {
    let canon = canonical_in(model, s);
    if let Some(w) = known_window(canon, s) {
        return Some(w);
    }
    if model.ends_with("[1m]") {
        return None;
    }
    Some(200_000)
}

pub fn context_window(model: &str) -> Option<u32> {
    context_window_in(model, &current())
}

/// The effort tiers a route offers, resolved through canonical ids like
/// [`context_window_in`], so a `[1m]` route and a family alias carry their
/// model's tiers. Empty when the page names none for the model.
pub fn efforts_in(model: &str, s: &Snapshot) -> Vec<String> {
    s.by_id(canonical_in(model, s))
        .map(|m| m.efforts.clone())
        .unwrap_or_default()
}

/// Native `--model` selection: ids with a 1M window get `[1m]`, 200K ids
/// go bare, unknown ids pass through untouched.
pub fn native_model_in(model: &str, s: &Snapshot) -> Result<String, String> {
    let canon = canonical_in(model, s);
    match known_window(canon, s) {
        Some(1_000_000) => Ok(format!("{canon}[1m]")),
        Some(_) => {
            if model.ends_with("[1m]") {
                return Err(format!("{canon} does not support a 1M context window"));
            }
            Ok(canon.to_string())
        }
        None => Ok(model.to_string()),
    }
}

pub fn native_model(model: &str) -> Result<String, String> {
    native_model_in(model, &current())
}

/// Every routable id: canonical ids, `[1m]` variants of 1M routes, and
/// one bare alias per family present in the snapshot.
pub fn all_ids_in(s: &Snapshot) -> Vec<String> {
    let mut out = Vec::new();
    let mut families: Vec<&str> = Vec::new();
    for m in &s.models {
        out.push(m.id.clone());
        if m.window == Some(1_000_000) {
            out.push(format!("{}[1m]", m.id));
        }
        let f = family(m);
        if !families.contains(&f) {
            families.push(f);
        }
    }
    out.extend(families.into_iter().map(str::to_string));
    out.sort();
    out.dedup();
    out
}

pub fn all_ids() -> Vec<String> {
    all_ids_in(&current())
}

/// Derive a name from an id (`claude-<family>-<major>-<minor>` →
/// `Claude <Family> <major>.<minor>`): a one-digit segment folds onto a
/// digit-ending word; a longer digit run (a date) stays a word of its own.
fn derive_name(base: &str) -> String {
    let short = base.strip_prefix("claude-").unwrap_or(base);
    let mut words: Vec<String> = Vec::new();
    for w in short.split('-') {
        let lone_digit = w.len() == 1 && w.bytes().next().is_some_and(|b| b.is_ascii_digit());
        if lone_digit
            && words
                .last()
                .is_some_and(|p: &String| p.ends_with(|c: char| c.is_ascii_digit()))
        {
            words.last_mut().unwrap().push('.');
            words.last_mut().unwrap().push_str(w);
            continue;
        }
        let mut c = w.chars();
        words.push(match c.next() {
            Some(f) => f.to_uppercase().chain(c).collect(),
            None => String::new(),
        });
    }
    format!("Claude {}", words.join(" "))
}

/// Display name for a route, derived from its canonical id. `[1m]` routes
/// get a suffix.
pub fn display_name_in(id: &str, s: &Snapshot) -> String {
    let name = derive_name(canonical_in(id, s));
    if id.ends_with("[1m]") {
        format!("{name} 1M")
    } else {
        name
    }
}

pub fn display_name(id: &str) -> String {
    display_name_in(id, &current())
}

#[path = "catalog_tests.rs"]
#[cfg(test)]
mod tests;
