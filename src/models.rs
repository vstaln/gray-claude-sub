//! Model discovery from Anthropic's public models page.
//!
//! `catalog::snapshot()` makes one plain GET of the overview page
//! ([`DEFAULT_MODELS_URL`], or `CLAUDE_SUB_MODELS_URL`) and parses its model
//! table and legacy list with [`parse_overview`]. No credential is read or
//! sent: the page is public. The last successful parse is cached on disk and
//! answers when the fetch or the parse fails; with neither, the listing says
//! why.

use std::sync::Arc;
use std::time::Duration;

use gray_plugin::{ProviderModel, ProviderModelCatalog};

use crate::catalog::{self, ModelInfo, Snapshot};

/// Anthropic's public models overview, as markdown.
const DEFAULT_MODELS_URL: &str = "https://platform.claude.com/docs/en/models/overview.md";
/// Points discovery at another copy of the page (tests, mirrors). Unset or
/// empty means [`DEFAULT_MODELS_URL`].
pub const MODELS_URL_ENV: &str = "CLAUDE_SUB_MODELS_URL";
/// Discovery is a picker nicety, never worth a hang: cap the whole call.
const FETCH_TIMEOUT: Duration = Duration::from_secs(8);
/// The overview is about 20 KB; a body this much bigger is not that page.
const MAX_PAGE_BYTES: usize = 1 << 20;
/// Row label of the table row that holds each current model's API id.
const API_ID_ROW: &str = "Claude API ID";
/// Row label of the table row that holds each column's context window.
const WINDOW_ROW: &str = "Context window";
/// The line that opens the legacy-model links.
const LEGACY_LINE: &str = "Legacy models (still available):";
/// Row label of the table row that holds each current model's default effort.
/// The page names no effort levels anywhere, so this row is the only per-model
/// evidence that a model takes effort at all.
const DEFAULT_EFFORT_ROW: &str = "Default effort";
/// Effort tiers native `--effort` accepts (gray's `off` = flag omitted). The
/// page names no levels, so a model whose `Default effort` cell names a tier
/// offers all of them, as the pinned builds always did.
pub const EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// Render a snapshot as the protocol catalog. Pure — tests drive it with
/// synthetic snapshots, never the network.
pub fn catalog_from(s: &Snapshot) -> ProviderModelCatalog {
    let s = &Snapshot::new(
        s.models
            .iter()
            .filter(|m| m.id.starts_with("claude-"))
            .cloned()
            .collect(),
    );
    let models = catalog::all_ids_in(s)
        .into_iter()
        .map(|id| ProviderModel {
            name: catalog::display_name_in(&id, s),
            context_window: catalog::context_window_in(&id, s),
            // Subscription usage isn't billed per token, so API price tiers
            // (long-prompt surcharges) are no reason to compact early.
            price_cliff_tokens: None,
            // The tiers the page's `Default effort` cell earns this model. An
            // empty list is "no reasoning control" to the host, so it is only
            // ever empty for a model the page says nothing about.
            reasoning_efforts: catalog::efforts_in(&id, s),
            variants: Vec::new(),
            slots: Vec::new(),
            id,
        })
        .collect();
    ProviderModelCatalog { models }
}

/// The catalog: fetched once per process, else the cache. With neither, the
/// error says why nothing could be listed.
pub async fn catalog() -> Result<ProviderModelCatalog, String> {
    listing(catalog::snapshot().await)
}

/// The `provider/models` answer for a snapshot outcome: the rendered catalog,
/// or the reason nothing could be listed. Pure.
pub fn listing(snap: Result<Arc<Snapshot>, String>) -> Result<ProviderModelCatalog, String> {
    snap.map(|s| catalog_from(&s))
        .map_err(|why| format!("couldn't list Claude models: {why}"))
}

/// The page to read: the `CLAUDE_SUB_MODELS_URL` override when it is set and
/// non-empty, else the public default. Takes the value rather than reading
/// the environment, so tests need no environment changes.
fn models_url(override_url: Option<&str>) -> String {
    override_url
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .unwrap_or(DEFAULT_MODELS_URL)
        .to_string()
}

/// GET a page. Deliberately sends no credentials: no `Authorization`, no
/// API key, no `anthropic-*` header. Errors are plain strings; `what` names
/// the page in them.
pub(crate) async fn fetch_page(url: &str, what: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .user_agent(concat!("gray-claude-sub/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{what} HTTP {}", resp.status()));
    }
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if body.len() + chunk.len() > MAX_PAGE_BYTES {
            return Err(format!("{what} is larger than expected"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| format!("{what} is not UTF-8"))
}

/// Parse the models overview page. Pure: tests run it on saved pages.
///
/// Current models come from the `Claude API ID` row of the page's first
/// comparison table, in column order. Each column's `Context window` cell is
/// its window (`1M tokens` → 1,000,000, `200k tokens` → 200,000, anything else
/// → `None`). Legacy models come from the `Legacy models (still available):`
/// line: each link's model slug becomes `claude-<slug>`, with no window, since
/// the page states none for them. A page with no `Claude API ID` row, or with
/// no model id in it, is an error, never a short list.
pub fn parse_overview(page: &str) -> Result<Vec<ModelInfo>, String> {
    let mut models: Vec<ModelInfo> = comparison_models(page)?
        .into_iter()
        .map(|(id, window, efforts)| ModelInfo {
            id,
            window,
            efforts,
        })
        .collect();
    for id in legacy_ids(page) {
        if !models.iter().any(|m| m.id == id) {
            models.push(ModelInfo {
                id,
                window: None,
                efforts: Vec::new(),
            });
        }
    }
    Ok(models)
}

/// One comparison-table model column: `(id, context window, effort tiers)`.
type ComparisonColumn = (String, Option<u32>, Vec<String>);

/// `(id, window, efforts)` for each model column of the first comparison table
/// that has a `Claude API ID` row, in column order. The same table must carry a
/// `Default effort` row: its cell decides each model's efforts (see
/// [`effort_tiers`]).
fn comparison_models(page: &str) -> Result<Vec<ComparisonColumn>, String> {
    let tables = markdown_tables(page);
    let Some(table) = tables
        .iter()
        .find(|t| t.iter().any(|row| row_label(row) == Some(API_ID_ROW)))
    else {
        return Err(format!("the models page has no `{API_ID_ROW}` row"));
    };
    // A row SHORTER than its table's first row is a torn download, not a
    // sparse table: error, or a bogus short list overwrites the cache. A
    // wider row is ragged but complete — the extra cells are ignored.
    let width = table.first().map(|row| row.len()).unwrap_or(0);
    if table.iter().any(|row| row.len() < width) {
        return Err("the models table is truncated mid-row".to_string());
    }
    let ids = table
        .iter()
        .find(|row| row_label(row) == Some(API_ID_ROW))
        .expect("table was found by its API-id row");
    // A page without the effort row is an error, never "no effort": dropping
    // the row would otherwise silently remove every model's effort selector.
    // A row of another shape can't be trusted column-for-column.
    let defaults = table
        .iter()
        .find(|row| row_label(row) == Some(DEFAULT_EFFORT_ROW))
        .filter(|row| row.len() == ids.len())
        .ok_or_else(|| {
            format!("the models page has no `{DEFAULT_EFFORT_ROW}` row lining up with its models")
        })?;
    let windows = table.iter().find(|row| row_label(row) == Some(WINDOW_ROW));
    let models: Vec<ComparisonColumn> = ids
        .iter()
        .enumerate()
        .skip(1) // column 0 is the row label
        .map(|(col, cell)| {
            match (backticked(cell), cell.contains('`')) {
                (Some(id), _) if is_model_id(id) => {
                    // A window row of another shape can't be trusted
                    // column-for-column.
                    let window = windows
                        .filter(|w| w.len() == ids.len())
                        .and_then(|w| window_tokens(w[col]));
                    let efforts = effort_tiers(defaults[col])?;
                    Ok(Some((id.to_string(), window, efforts)))
                }
                // A cell that opens a backtick and never closes is torn
                // data (``claude-son`), not a model column to skip.
                (None, true) => Err(format!(
                    "the `{API_ID_ROW}` row holds an unterminated id cell"
                )),
                _ => Ok(None),
            }
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .flatten()
        .collect();
    if models.is_empty() {
        return Err(format!("the `{API_ID_ROW}` row lists no model ids"));
    }
    Ok(models)
}

/// The effort tiers a model's `Default effort` cell earns it: all of
/// [`EFFORTS`] when the cell names a default tier (a backticked value such as
/// `` `high` ``), none for a cell that doesn't (`—`). A torn cell, a backtick
/// opened and never closed, is an error.
fn effort_tiers(cell: &str) -> Result<Vec<String>, String> {
    match (backticked(cell), cell.contains('`')) {
        (Some(_), _) => Ok(EFFORTS.iter().map(|e| e.to_string()).collect()),
        (None, true) => Err(format!(
            "the `{DEFAULT_EFFORT_ROW}` row holds an unterminated cell"
        )),
        _ => Ok(Vec::new()),
    }
}

/// The page's markdown tables: each a run of consecutive `|` lines, split
/// into trimmed cells with the outer pipes dropped.
pub(crate) fn markdown_tables(page: &str) -> Vec<Vec<Vec<&str>>> {
    let mut tables: Vec<Vec<Vec<&str>>> = Vec::new();
    let mut current: Vec<Vec<&str>> = Vec::new();
    for line in page.lines().map(str::trim) {
        if let Some(inner) = line.strip_prefix('|') {
            let cells: Vec<&str> = inner
                .trim_end_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            current.push(cells);
        } else if !current.is_empty() {
            tables.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tables.push(current);
    }
    tables
}

/// A table row's label: its first cell, with emphasis and link markers
/// removed (`[Context window](…)` → `Context window`).
pub(crate) fn row_label<'a>(row: &[&'a str]) -> Option<&'a str> {
    let cell = row.first().copied()?.trim_matches('*').trim();
    Some(match cell.strip_prefix('[') {
        Some(link) => link.split(']').next().unwrap_or_default().trim(),
        None => cell,
    })
}

/// The first backticked span of a cell: `` `claude-…` `` → `claude-…`.
/// The closing tick is required: an unterminated tail (``claude-son`) is a
/// torn cell, not an id.
fn backticked(cell: &str) -> Option<&str> {
    let rest = cell.split_once('`')?.1;
    let (id, _) = rest.split_once('`')?;
    Some(id)
}

/// A model id as the page writes one: `claude-`, then lowercase words,
/// digits and hyphens. Anything else in a cell is not an id.
fn is_model_id(id: &str) -> bool {
    id.strip_prefix("claude-")
        .is_some_and(|rest| !rest.is_empty())
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A context-window cell as tokens: `1M tokens` → 1,000,000 and `200k tokens`
/// → 200,000. Anything not stated as a whole count of k or M tokens (a
/// fraction, a thousands separator, a footnote) is unknown: `None`.
fn window_tokens(cell: &str) -> Option<u32> {
    let text = cell.trim().to_ascii_lowercase();
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    let (count, unit) = text.split_at(digits);
    let count: u32 = count.parse().ok()?;
    let (scale, rest): (u32, &str) = match unit.as_bytes().first() {
        Some(b'k') => (1_000, &unit[1..]),
        Some(b'm') => (1_000_000, &unit[1..]),
        _ => (1, unit),
    };
    if rest.trim_start() != "tokens" {
        return None;
    }
    count.checked_mul(scale).filter(|w| *w > 0)
}

/// Legacy ids: each link on the `Legacy models (still available):` line
/// points at a model page, `…/models/<slug>/overview`, and its id is
/// `claude-<slug>`. Duplicates and links to other pages are dropped.
fn legacy_ids(page: &str) -> Vec<String> {
    let Some(line) = page
        .lines()
        .find(|l| l.trim_start().starts_with(LEGACY_LINE))
    else {
        return Vec::new();
    };
    let mut ids: Vec<String> = Vec::new();
    // Each `](` opens a link target, which ends at the next `)`.
    for target in line.split("](").skip(1) {
        let url = target.split(')').next().unwrap_or_default();
        let slug = url
            .split("/models/")
            .nth(1)
            .and_then(|rest| rest.split('/').next());
        let Some(slug) = slug else { continue };
        let id = format!("claude-{slug}");
        if is_model_id(&id) && !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// The fetch [`catalog::snapshot`] installs: the public page, parsed. The
/// request carries no credentials, so no secret can reach an error string.
pub async fn fetch_snapshot() -> Result<Arc<Snapshot>, String> {
    let url = models_url(std::env::var(MODELS_URL_ENV).ok().as_deref());
    let page = fetch_page(&url, "models page").await?;
    Ok(Arc::new(Snapshot::new(parse_overview(&page)?)))
}

#[path = "models_tests.rs"]
#[cfg(test)]
mod tests;
