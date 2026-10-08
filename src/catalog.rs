//! Native route catalog: live `/v1/models` discovery with a pinned
//! fallback. Port of the Hermes DirectSDK `model_catalog.py`
//! (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT).
//!
//! A loopback gateway needs explicit long-context selection: native runs a
//! plain id inside its 200K default and selects 1M only for an explicit
//! `[1m]` route. Routable ids, windows, display names, families and effort
//! support come from `GET /v1/models` on the user's OAuth login — nothing
//! per-release is hardcoded. When the fetch can't run (no credential,
//! offline) the last-known-good table below answers instead: stale beats
//! dead, and unknown ids still pass through at the 200K native default
//! rather than being guessed up to 1M.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

/// One model as `/v1/models` reports it (fields we use; everything else is
/// ignored). A `None` field means the API didn't say — callers fall back
/// to the pinned table, then to defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// `display_name` upstream ("Claude Sonnet 5.5"); derived when absent.
    pub display_name: Option<String>,
    /// `created_at` upstream (RFC 3339): the sort key for "newest of a
    /// family" — ordering only, the value is never shown.
    pub created_at: String,
    /// `line` upstream ("sonnet"): the family a bare alias resolves to.
    /// Absent → derived from the id (`claude-<family>-…`).
    pub line: Option<String>,
    /// `max_input_tokens` upstream: the real window, 1M included.
    pub max_input_tokens: Option<u32>,
    /// Effort levels the API marks supported, in canonical order.
    /// `None` = unknown → the renderer lists every effort (status quo).
    pub efforts: Option<Vec<String>>,
}

/// A resolved view of the catalog: fetched or fallback. `models` is kept
/// sorted newest-first so "first match" picks are always the newest
/// release of a family or prefix.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub models: Vec<ModelInfo>,
}

impl Snapshot {
    pub(crate) fn new(mut models: Vec<ModelInfo>) -> Self {
        // Newest first; `created_at` is RFC 3339 so lexical = chronological.
        // `id` descending breaks ties deterministically.
        models.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        Self { models }
    }

    fn by_id(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// Effort levels native `--effort` accepts, in order (gray's `off` = flag
/// omitted). Also the report when the API omits per-model effort support.
pub const EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// Last-known-good routes, consulted only when live discovery hasn't run
/// or a fetched entry omits a fact. `created` is a family-ordering hint
/// (matches upstream `created_at` at pin time); `window` doubles as the
/// `[1m]`-route marker: 1M means the route exists, 200K means none.
const FALLBACK: &[(&str, &str, u32, &str)] = &[
    // (id, family, window, created ordering hint)
    ("claude-sonnet-5", "sonnet", 1_000_000, "2026-06-29"),
    ("claude-haiku-4-5-20251001", "haiku", 200_000, "2025-10-15"),
    ("claude-opus-5-5", "opus", 1_000_000, "2026-09-21"),
    ("claude-opus-5", "opus", 1_000_000, "2026-07-24"),
    ("claude-opus-4-8", "opus", 1_000_000, "2026-05-28"),
    ("claude-fable-5-1", "fable", 1_000_000, "2026-08-28"),
];

fn fallback_snapshot() -> Snapshot {
    Snapshot::new(
        FALLBACK
            .iter()
            .map(|(id, line, window, created)| ModelInfo {
                id: id.to_string(),
                display_name: None,
                created_at: created.to_string(),
                line: Some(line.to_string()),
                max_input_tokens: Some(*window),
                efforts: None,
            })
            .collect(),
    )
}

/// The snapshot everyone reads: fallback until the first successful fetch
/// swaps in live data. Written once per process under [`FETCH_LOCK`].
static SNAPSHOT: LazyLock<RwLock<Arc<Snapshot>>> =
    LazyLock::new(|| RwLock::new(Arc::new(fallback_snapshot())));
/// Set once live data is installed; failures leave it clear so a later
/// `provider/models` retries (transient errors heal).
static FETCHED: AtomicBool = AtomicBool::new(false);
static FETCH_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Whatever the catalog currently knows — never blocks on the network.
/// This is the fallback until the first [`snapshot`] call succeeds.
pub fn current() -> Arc<Snapshot> {
    SNAPSHOT.read().unwrap().clone()
}

/// Install a fetched snapshot (tests and the fetch path). Marks the
/// catalog live so no further fetches run this process.
pub fn install(s: Arc<Snapshot>) {
    *SNAPSHOT.write().unwrap() = s;
    FETCHED.store(true, Ordering::Relaxed);
}

/// Live-or-fallback snapshot: fetch once per process (OAuth bearer from
/// the Claude credential file), keep the last good answer on failure.
pub async fn snapshot() -> Arc<Snapshot> {
    if FETCHED.load(Ordering::Relaxed) {
        return current();
    }
    let _guard = FETCH_LOCK.lock().await;
    if FETCHED.load(Ordering::Relaxed) {
        return current();
    }
    match crate::models::fetch_snapshot().await {
        Ok(s) => {
            install(s.clone());
            s
        }
        Err(_) => current(),
    }
}

/// The family an id belongs to: the API's `line`, else the first segment
/// after `claude-` (`claude-sonnet-5-5` → `sonnet`).
pub(crate) fn family(m: &ModelInfo) -> &str {
    if let Some(line) = &m.line {
        return line;
    }
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

/// One resolution pass: exact id, then family alias ("sonnet" → newest
/// sonnet; "claude-sonnet" counts too), then a `claude-`-less or dated
/// prefix ("claude-haiku-4-5" → its dated id), newest match winning.
/// Last: a dated tail variant of a known id — `claude-opus-5-5-20260101`
/// strips its all-digit date segment and resolves the stem, so a dated
/// id inherits its family's window and routes instead of falling to the
/// 200K default.
fn resolve<'a>(base: &'a str, s: &'a Snapshot) -> Option<&'a str> {
    if s.by_id(base).is_some() {
        return Some(base);
    }
    let name = base.strip_prefix("claude-").unwrap_or(base);
    if let Some(m) = s.models.iter().find(|m| family(m) == name) {
        return Some(&m.id);
    }
    // Prefix on a hyphen boundary: "claude-opus-4" → newest claude-opus-4-*.
    // models is newest-first, so the first match is the newest.
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

/// The window a snapshot assigns an id: fetched `max_input_tokens`, else
/// the pinned table entry for a known id, else `None` (unknown).
fn window_of(m: &ModelInfo) -> Option<u32> {
    m.max_input_tokens.or_else(|| {
        FALLBACK
            .iter()
            .find(|(id, ..)| *id == m.id)
            .map(|(_, _, w, _)| *w)
    })
}

fn known_window(id: &str, s: &Snapshot) -> Option<u32> {
    s.by_id(id).and_then(window_of).or_else(|| {
        FALLBACK
            .iter()
            .find(|(fid, ..)| *fid == id)
            .map(|(_, _, w, _)| *w)
    })
}

/// Context window for a route: fetched/pinned window for a known id, the
/// 200K native default for an unknown plain id, `None` for an unknown
/// `[1m]` route (no guess exceeds the 1M native budget).
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
        if window_of(m) == Some(1_000_000) {
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

/// Derive a name from an id (`claude-opus-5-5` → `Claude Opus 5.5`):
/// a lone digit folds onto a digit-ending word (`haiku-4-5-20251001`
/// keeps its date — six-plus digits never merge).
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

/// Display name for a route id: the API's `display_name` for a fetched
/// id, else derived from the canonical id. `[1m]` routes get a suffix.
pub fn display_name_in(id: &str, s: &Snapshot) -> String {
    let long = id.ends_with("[1m]");
    let canon = canonical_in(id, s);
    let name = s
        .by_id(canon)
        .and_then(|m| m.display_name.clone())
        .unwrap_or_else(|| derive_name(canon));
    if long { format!("{name} 1M") } else { name }
}

pub fn display_name(id: &str) -> String {
    display_name_in(id, &current())
}

#[path = "catalog_tests.rs"]
#[cfg(test)]
mod tests;
