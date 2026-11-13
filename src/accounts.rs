//! Multi-account pooling: several `claude` CLI credential homes behind
//! one provider.
//!
//! Claude Code partitions credentials by `CLAUDE_CONFIG_DIR`: `auth
//! login` writes `.credentials.json` into it, `auth status` reads it
//! back. An "account" here is one such home plus a label, persisted in
//! `<gray home>/claude-sub/accounts.json`. With no file (or an empty
//! `accounts` list) the plugin serves the ambient login exactly as
//! before — zero-config back-compat.
//!
//! Selection is per `provider/chat` turn: `fill_first` (the default)
//! takes the first non-cooled account in file order; `least_used` takes
//! the non-cooled account with the oldest `last_used_at`. `account use`
//! pins a label: a usable pinned account always serves, else the
//! strategy picks among the rest.
//!
//! Cooldowns: only account-scoped verdicts cool an account — a 401 (a
//! dead login) or a 429 (quota). An explicit `retry-after`/`resets …`
//! hint in the upstream message wins over the computed TTL. Request-
//! scoped failures (other 4xx, dead wires, malformed turns) never cool:
//! with one account they would mask the real error behind "all accounts
//! cooling". A lone configured account caps every cooldown at
//! [`SOLE_CAP`] for the same reason.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// 401 TTL: a dead login, retried occasionally in case it healed.
const COOL_401: u64 = 5 * 60;
/// 429 TTL when the upstream message carries no usable reset hint.
const COOL_429: u64 = 60 * 60;
/// A lone configured account never sits out longer than this — a longer
/// bench would mask the real upstream verdict.
const SOLE_CAP: u64 = 60;
/// Sanity bound on a parsed reset hint (weekly windows reset within a
/// week; a bogus parse must not bench an account forever).
const MAX_RESET: u64 = 8 * 24 * 3600;

/// One credential home in the pool. `home` is the `CLAUDE_CONFIG_DIR`
/// the spawned `claude` child gets (empty = the ambient login, spawned
/// with none); health is write-through.
#[derive(Serialize, Deserialize)]
pub struct Account {
    pub label: String,
    pub home: PathBuf,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub org: Option<String>,
    #[serde(default)]
    pub added_at: u64,
    #[serde(default)]
    pub health: Health,
    /// Unix seconds the cooldown ends; 0 = not cooling.
    #[serde(default, skip_serializing)]
    cooldown_until: u64,
    #[serde(default, skip_serializing)]
    last_error: Option<String>,
    #[serde(default)]
    pub request_count: u64,
    #[serde(default)]
    pub last_used_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Debug)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Health {
    Cooling {
        until: u64,
        reason: String,
    },
    AuthFailed {
        until: u64,
        msg: String,
    },
    #[default]
    #[serde(other)]
    Ready,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
enum Strategy {
    LeastUsed,
    #[default]
    #[serde(other)]
    FillFirst,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::FillFirst => "fill_first",
            Strategy::LeastUsed => "least_used",
        }
    }
}

enum Verdict {
    Unauthorized,
    RateLimited { retry_after: Option<Duration> },
}

enum Outcome {
    Served,
    Refused { verdict: Verdict, detail: String },
}

#[derive(Serialize, Deserialize, Default)]
struct Store {
    #[serde(default)]
    accounts: Vec<Account>,
    /// `account use` pin: a usable pinned account always serves.
    #[serde(default)]
    selected: Option<String>,
    /// `fill_first` (default) or `least_used`.
    #[serde(default)]
    strategy: Strategy,
}

impl Store {
    fn parse(text: &str) -> Result<Store, serde_json::Error> {
        let mut store: Store = serde_json::from_str(text)?;
        for a in &mut store.accounts {
            a.adopt_legacy();
        }
        Ok(store)
    }
}

impl Account {
    fn adopt_legacy(&mut self) {
        let until = std::mem::take(&mut self.cooldown_until);
        let msg = self.last_error.take().unwrap_or_default();
        if until == 0 || self.health != Health::Ready {
            return;
        }
        let auth = matches!(crate::chat::api_error(&msg), Some((401, _)));
        self.health = if auth {
            Health::AuthFailed { until, msg }
        } else {
            Health::Cooling { until, reason: msg }
        };
    }
}

pub enum Serving {
    Account(Pick),
    Ambient,
    AllCooling { until: u64, reason: String },
}

/// The account a turn is served by: label plus the credential home the
/// spawned `claude` children get as `CLAUDE_CONFIG_DIR` (`None` = the
/// ambient login, no variable set).
pub struct Pick {
    pub label: String,
    pub home: Option<PathBuf>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `~` expands at read so hand-edited configs and `--home ~` work.
pub fn expand_home(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if s == "~" {
        if let Some(h) = crate::session::home_dir() {
            return h;
        }
    } else if let Some(rest) = s.strip_prefix("~/")
        && let Some(h) = crate::session::home_dir()
    {
        return h.join(rest);
    }
    p.to_path_buf()
}

/// Canonicalized-or-expanded `home`, for comparing paths across spellings.
fn normalize(p: &Path) -> PathBuf {
    let expanded = expand_home(p);
    std::fs::canonicalize(&expanded).unwrap_or(expanded)
}

fn spawn_home(home: &Path) -> Option<&Path> {
    (!home.as_os_str().is_empty()).then_some(home)
}

fn owns(a: &Account, want: &Option<PathBuf>) -> bool {
    match (spawn_home(&a.home), want) {
        (None, None) => true,
        (Some(h), Some(w)) => normalize(h) == *w,
        _ => false,
    }
}

fn home_text(home: &Path) -> String {
    spawn_home(home).map_or("(ambient login)".to_string(), |h| h.display().to_string())
}

fn file_path() -> Option<PathBuf> {
    crate::settings::state_dir().map(|d| d.join("accounts.json"))
}

/// Managed homes live here: `<gray home>/claude-sub/homes/<label>`.
fn homes_dir() -> Option<PathBuf> {
    crate::settings::state_dir().map(|d| d.join("homes"))
}

fn load() -> Store {
    load_checked().unwrap_or_default()
}

fn load_checked() -> Result<Store, String> {
    let Some(path) = file_path() else {
        return Ok(Store::default());
    };
    match std::fs::read_to_string(&path) {
        Ok(text) if text.trim().is_empty() => Ok(Store::default()),
        Ok(text) => Store::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Store::default()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn mutate<T>(durable: bool, f: impl FnOnce(&mut Store) -> Result<T, String>) -> Result<T, String> {
    let path = file_path().ok_or("cannot resolve GRAY_HOME")?;
    let _lock =
        crate::session::lock_file(&path).map_err(|e| format!("lock {}: {e}", path.display()))?;
    let mut store = match std::fs::read_to_string(&path) {
        Ok(text) if text.trim().is_empty() => Store::default(),
        Ok(text) => Store::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Store::default(),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let before = serde_json::to_value(&store).map_err(|e| e.to_string())?;
    let out = f(&mut store)?;
    if serde_json::to_value(&store).map_err(|e| e.to_string())? != before {
        let text = serde_json::to_string_pretty(&store).map_err(|e| e.to_string())?;
        let bytes = text.as_bytes();
        let written = if durable {
            crate::session::write_durable(&path, bytes)
        } else {
            crate::session::write_atomic(&path, bytes)
        };
        written.map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(out)
}

fn current(a: &Account, now: u64) -> Health {
    match &a.health {
        Health::Cooling { until, .. } | Health::AuthFailed { until, .. } if *until > now => {
            a.health.clone()
        }
        _ => Health::Ready,
    }
}

fn until_of(h: &Health) -> u64 {
    match h {
        Health::Ready => 0,
        Health::Cooling { until, .. } | Health::AuthFailed { until, .. } => *until,
    }
}

fn reason_of(h: &Health) -> Option<String> {
    match h {
        Health::Ready => None,
        Health::Cooling { reason, .. } => Some(reason.clone()),
        Health::AuthFailed { msg, .. } => Some(msg.clone()),
    }
}

fn usable(a: &Account, now: u64) -> bool {
    matches!(current(a, now), Health::Ready)
}

fn transition(h: &Health, outcome: &Outcome, now: u64, lone: bool) -> Health {
    let (verdict, detail) = match outcome {
        Outcome::Served => return Health::Ready,
        Outcome::Refused { verdict, detail } => (verdict, detail),
    };
    let (ttl, auth) = match verdict {
        Verdict::Unauthorized => (COOL_401, true),
        Verdict::RateLimited { retry_after } => (
            match retry_after {
                Some(d) => d.as_secs().min(MAX_RESET),
                None => COOL_429,
            },
            false,
        ),
    };
    let cap = if lone { SOLE_CAP } else { MAX_RESET };
    let until = until_of(h).max(now + ttl.min(cap).max(1));
    let msg: String = detail.chars().take(200).collect();
    if auth {
        Health::AuthFailed { until, msg }
    } else {
        Health::Cooling { until, reason: msg }
    }
}

fn pick_of(a: &Account) -> Pick {
    Pick {
        label: a.label.clone(),
        home: spawn_home(&a.home).map(expand_home),
    }
}

fn serve(store: &Store) -> Serving {
    if store.accounts.is_empty() {
        return Serving::Ambient;
    }
    let now = now_secs();
    // A usable pinned account always serves.
    if let Some(sel) = store.selected.as_deref()
        && let Some(a) = store
            .accounts
            .iter()
            .find(|a| a.label == sel && usable(a, now))
    {
        return Serving::Account(pick_of(a));
    }
    let next = match store.strategy {
        Strategy::LeastUsed => store
            .accounts
            .iter()
            .filter(|a| usable(a, now))
            .min_by_key(|a| a.last_used_at),
        // fill_first: file order is the priority order.
        Strategy::FillFirst => store.accounts.iter().find(|a| usable(a, now)),
    };
    if let Some(a) = next {
        return Serving::Account(pick_of(a));
    }
    let soon = store
        .accounts
        .iter()
        .min_by_key(|a| until_of(&current(a, now)))
        .expect("nonempty");
    let until = until_of(&current(soon, now));
    Serving::AllCooling {
        until,
        reason: format!(
            "all {} claude accounts are cooling down; {} next at {}",
            store.accounts.len(),
            soon.label,
            crate::session::iso_timestamp(UNIX_EPOCH + Duration::from_secs(until)),
        ),
    }
}

/// The account a turn would use — read-only (no use recorded). Probes
/// and `provider/auth/*` use this; only `provider/chat` calls [`pick`].
pub fn serving() -> Serving {
    serve(&load())
}

/// Per-turn selection: resolve the serving account and record the use.
pub fn pick() -> Result<Option<Pick>, String> {
    if load_checked()?.accounts.is_empty() {
        return Ok(None);
    }
    mutate(false, |store| match serve(store) {
        Serving::Account(p) => {
            if let Some(a) = store.accounts.iter_mut().find(|a| a.label == p.label) {
                a.request_count += 1;
                a.last_used_at = now_secs();
            }
            Ok(Some(p))
        }
        Serving::Ambient => Ok(None),
        Serving::AllCooling { reason, .. } => Err(reason),
    })
}

/// The turn this account served succeeded: clear any recorded failure.
pub fn note_success(label: &str) {
    let store = load();
    let Some(a) = store.accounts.iter().find(|a| a.label == label) else {
        return;
    };
    if a.health == Health::Ready {
        return;
    }
    record(label, Outcome::Served);
}

/// The turn this account served failed with `detail`. Only account-
/// scoped upstream verdicts cool it; anything else leaves the pool
/// untouched.
pub fn note_failure(label: &str, detail: &str) {
    let Some((code, msg)) = crate::chat::api_error(detail) else {
        return;
    };
    let verdict = match code {
        401 => Verdict::Unauthorized,
        429 => Verdict::RateLimited {
            retry_after: parse_retry_after(msg),
        },
        // Request-scoped verdicts never cool an account.
        _ => return,
    };
    record(
        label,
        Outcome::Refused {
            verdict,
            detail: detail.to_string(),
        },
    );
}

/// [`note_failure`] keyed by credential home: a keepalive probe knows
/// the session's `CLAUDE_CONFIG_DIR`, not its label.
pub fn note_failure_home(home: Option<&Path>, detail: &str) {
    let want = home.and_then(spawn_home).map(normalize);
    let store = load();
    let Some(label) = store
        .accounts
        .iter()
        .find(|a| owns(a, &want))
        .map(|a| a.label.clone())
    else {
        return;
    };
    drop(store);
    note_failure(&label, detail);
}

/// The label of the account owning `home`, if any — `usage` keys its
/// cache per account this way.
pub fn label_for_home(home: Option<&Path>) -> Option<String> {
    let want = home.and_then(spawn_home).map(normalize);
    load()
        .accounts
        .iter()
        .find(|a| owns(a, &want))
        .map(|a| a.label.clone())
}

fn record(label: &str, outcome: Outcome) {
    let _ = mutate(false, |store| {
        let now = now_secs();
        let lone = store.accounts.len() == 1;
        if let Some(a) = store.accounts.iter_mut().find(|a| a.label == label) {
            a.health = transition(&current(a, now), &outcome, now, lone);
        }
        Ok(())
    });
}

// ---- retry-after / reset-hint parsing ------------------------------------

/// `unit` as seconds; `None` when the word isn't a time unit.
fn unit_secs(unit: &str) -> Option<u64> {
    match unit {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(1),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(60),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(3600),
        "d" | "day" | "days" => Some(86400),
        _ => None,
    }
}

/// Split text into digit/letter runs: `"2h 15m"` → `["2","h","15","m"]`.
fn atoms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tok in text.split_whitespace().take(10) {
        let mut cur = String::new();
        let mut digit = None;
        for c in tok.chars() {
            if !c.is_ascii_alphanumeric() {
                continue;
            }
            let d = c.is_ascii_digit();
            if digit.is_some_and(|x| x != d) {
                out.push(std::mem::take(&mut cur));
            }
            digit = Some(d);
            cur.push(c);
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// Parse a leading duration like `2h 15m`, `90 seconds`, `an hour`, or a
/// bare `90` (seconds). `None` when nothing duration-like leads.
fn dur_prefix(text: &str) -> Option<Duration> {
    let a = atoms(text);
    let (mut total, mut i, mut any) = (0u64, 0usize, false);
    while i < a.len() {
        let n: u64 = if a[i] == "a" || a[i] == "an" {
            1
        } else {
            match a[i].parse() {
                Ok(n) => n,
                Err(_) => break,
            }
        };
        i += 1;
        match a.get(i).and_then(|u| unit_secs(u)) {
            Some(u) => {
                total += n * u;
                i += 1;
            }
            // A bare number is seconds ("retry after 90").
            None => total += n,
        }
        any = true;
    }
    (any && total > 0).then(|| Duration::from_secs(total.min(MAX_RESET)))
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(146097);
    let yoe = y - era * 400;
    let doy = (153 * if m > 2 { m - 3 } else { m + 9 } + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

/// A leading ISO-8601-ish datetime (`YYYY-MM-DD[ T]HH:MM[:SS][Z|±HH:MM]`)
/// as unix seconds. Missing time/zone reads as midnight UTC — a wrong
/// hint costs a longer bench, never a wrong answer.
fn iso_epoch(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    let digit = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    for i in 0..b.len().saturating_sub(9) {
        if !(digit(i) && digit(i + 1) && digit(i + 2) && digit(i + 3))
            || b.get(i + 4) != Some(&b'-')
            || b.get(i + 7) != Some(&b'-')
        {
            continue;
        }
        let (Ok(y), Ok(mo), Ok(da)) = (
            text[i..i + 4].parse::<i64>(),
            text[i + 5..i + 7].parse::<u32>(),
            text[i + 8..i + 10].parse::<u32>(),
        ) else {
            continue;
        };
        if !(1..=12).contains(&mo) || !(1..=31).contains(&da) {
            continue;
        }
        let (mut h, mut mi, mut se, mut off) = (0i64, 0i64, 0i64, 0i64);
        let j = i + 10;
        // Optional time after 'T' or a space.
        if matches!(b.get(j), Some(b'T') | Some(b't') | Some(b' ')) {
            let r = &text[j + 1..];
            let rb = r.as_bytes();
            if rb.len() >= 5
                && rb[0].is_ascii_digit()
                && rb[1].is_ascii_digit()
                && rb[2] == b':'
                && rb[3].is_ascii_digit()
                && rb[4].is_ascii_digit()
            {
                h = r[0..2].parse().ok()?;
                mi = r[3..5].parse().ok()?;
                let mut k = 5;
                if rb.get(k) == Some(&b':')
                    && rb.len() >= k + 3
                    && rb[k + 1].is_ascii_digit()
                    && rb[k + 2].is_ascii_digit()
                {
                    se = r[k + 1..k + 3].parse().unwrap_or(0);
                    k += 3;
                }
                // Optional zone: Z or ±HH[:MM].
                match rb.get(k) {
                    Some(b'Z') | Some(b'z') | None => {}
                    Some(&sign @ (b'+' | b'-')) => {
                        let z = &r[k + 1..];
                        let zh_len = z.bytes().take_while(|c| c.is_ascii_digit()).count();
                        if let Ok(zh) = z[..zh_len].parse::<i64>() {
                            let rest = &z[zh_len..];
                            let rest = rest.strip_prefix(':').unwrap_or(rest);
                            let zd: String =
                                rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                            let zm: i64 = zd.parse().unwrap_or(0);
                            let zi = zh * 3600 + zm * 60;
                            off = if sign == b'-' { -zi } else { zi };
                        }
                    }
                    _ => {}
                }
            }
        }
        return Some(days_from_civil(y, mo, da) * 86400 + h * 3600 + mi * 60 + se - off);
    }
    None
}

/// Honor an upstream-provided reset hint: `Retry-After`/retry-in
/// seconds, `resets in <duration>`, or `resets at <datetime>`.
fn parse_retry_after(msg: &str) -> Option<Duration> {
    let low = msg.to_lowercase();
    for key in ["retry-after", "retry after"] {
        if let Some(i) = low.find(key)
            && let Some(d) = dur_prefix(low[i + key.len()..].trim_start_matches([':', '=']))
        {
            return Some(d);
        }
    }
    for pat in ["resets in ", "reset in ", "try again in ", "retry in "] {
        if let Some(i) = low.find(pat)
            && let Some(d) = dur_prefix(&low[i + pat.len()..])
        {
            return Some(d);
        }
    }
    for pat in ["resets at ", "reset at "] {
        if let Some(i) = low.find(pat)
            && let Some(t) = iso_epoch(&msg[i + pat.len()..])
        {
            let now = now_secs() as i64;
            if t > now {
                return Some(Duration::from_secs((t - now) as u64));
            }
        }
    }
    None
}

// ---- `account` command ----------------------------------------------------

const USAGE: &str = "\
claude-sub account — multiple Claude logins behind one provider.
  claude-sub account add [--home <dir>] <label>   run `claude auth login` into a
                                                  new credential home (or adopt
                                                  an existing CLAUDE_CONFIG_DIR)
  claude-sub account add --ambient <label>         adopt the plain `claude` login
                                                  (spawned without CLAUDE_CONFIG_DIR);
                                                  the first add does this on its own
  claude-sub account list                          accounts + health
  claude-sub account use <label>                   pin an account (use --clear to unpin)
  claude-sub account remove [--keep] <label>       drop an account (+its managed home)
  claude-sub account strategy <fill_first|least_used>
                                                  pick order among usable accounts
  claude-sub account status                        which account serves now
  claude-sub account usage [<label>]               quota windows per account, or one (--json)
  any verb accepts --json for the machine contract the host /accounts panel reads
The file is <gray home>/claude-sub/accounts.json; absent or empty
`accounts` keeps the single ambient `claude` login.";

/// `claude-sub account …` (interactive: a real terminal) and `/claude
/// account …` (inside the sidecar — `add` can't drive OAuth there).
///
/// `--json` anywhere in the args switches every verb to the machine
/// contract the host `/accounts` panel reads: `list`/`status` print the
/// pool document, mutations print `{"ok":true,…}` (errors still exit 2
/// with the message on stderr).
pub fn command(args: &[String], interactive: bool) -> Result<String, String> {
    let json = args.iter().any(|a| a == "--json");
    let args: Vec<String> = args
        .iter()
        .filter(|a| a.as_str() != "--json")
        .cloned()
        .collect();
    match args.first().map(String::as_str) {
        Some("add") => cmd_add(&args[1..], interactive, json),
        Some("use") => cmd_use(&args[1..], json),
        Some("remove") | Some("rm") => cmd_remove(&args[1..], json),
        Some("strategy") => cmd_strategy(&args[1..], json),
        Some("status") => cmd_status(json),
        Some("usage") => cmd_usage(&args[1..], json),
        Some("list") | None => cmd_list(json),
        Some(other) => Err(format!("unknown account verb {other:?}\n\n{USAGE}")),
    }
}

/// The machine-readable pool document. `ambient` is true when no pool is
/// configured and the ambient `claude` login serves; `serving.kind` is
/// `account`, `ambient`, `cooldown` (every account benched — `reason`
/// names the earliest reset).
fn store_json(store: &Store) -> Result<String, String> {
    let now = now_secs();
    let accounts: Vec<Value> = store
        .accounts
        .iter()
        .map(|a| {
            let health = current(a, now);
            json!({
                "label": a.label,
                "home": a.home.display().to_string(),
                "email": a.email,
                "org": a.org,
                "added_at": a.added_at,
                "cooldown_until": until_of(&health),
                "cooling": !matches!(health, Health::Ready),
                "last_error": reason_of(&health),
                "health": health.clone(),
                "request_count": a.request_count,
                "last_used_at": a.last_used_at,
                "pinned": store.selected.as_deref() == Some(a.label.as_str()),
            })
        })
        .collect();
    let serving = match serve(store) {
        Serving::Account(p) => json!({"kind": "account", "label": p.label}),
        Serving::Ambient => json!({"kind": "ambient"}),
        Serving::AllCooling { until, reason } => {
            json!({"kind": "cooldown", "until": until, "reason": reason})
        }
    };
    serde_json::to_string_pretty(&json!({
        "version": 1,
        "plugin": crate::manifest::PLUGIN_NAME,
        "provider": crate::manifest::PROVIDER_ID,
        "strategy": store.strategy.name(),
        "selected": store.selected,
        "ambient": store.accounts.is_empty(),
        "accounts": accounts,
        "serving": serving,
    }))
    .map_err(|e| e.to_string())
}

/// `{"ok":true,…}` for the mutating verbs under `--json`.
fn ok_json(extra: Value) -> Result<String, String> {
    let mut v = extra;
    v["ok"] = json!(true);
    serde_json::to_string(&v).map_err(|e| e.to_string())
}

fn valid_label(label: &str) -> Result<(), String> {
    let ok = !label.is_empty()
        && label.len() <= 40
        && label
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphanumeric())
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    ok.then_some(())
        .ok_or_else(|| format!("bad label {label:?}: 1-40 chars of [A-Za-z0-9_-]"))
}

enum Source {
    Adopt(PathBuf),
    Managed(PathBuf),
    Ambient,
}

impl Source {
    fn home(&self) -> &Path {
        match self {
            Source::Adopt(h) | Source::Managed(h) => h.as_path(),
            Source::Ambient => Path::new(""),
        }
    }
}

enum Insert {
    Added {
        selected: Option<String>,
        who: String,
    },
    Duplicate {
        label: String,
        identity: String,
    },
}

fn admit(label: &str, source: &Source) -> Result<Insert, String> {
    let home = source.home().to_path_buf();
    let st = crate::setup::auth_status(spawn_home(&home))
        .ok_or("`claude auth status` failed in the new home")?;
    if st.get("loggedIn").and_then(Value::as_bool) != Some(true) {
        return Err(match source {
            Source::Ambient => {
                "no ambient `claude` login — run `claude auth login`, then retry".to_string()
            }
            Source::Managed(_) => format!(
                "no login in {} — `claude auth login` did not complete",
                home.display()
            ),
            Source::Adopt(_) => format!(
                "no login in {} — adopt a directory that holds a real login",
                home.display()
            ),
        });
    }
    let email = st.get("email").and_then(Value::as_str).map(str::to_string);
    let org = st
        .get("orgName")
        .and_then(Value::as_str)
        .or_else(|| st.get("orgId").and_then(Value::as_str))
        .map(str::to_string);
    let who = email
        .clone()
        .unwrap_or_else(|| "identity pending".to_string());
    // Dedupe on a verified upstream identity only — `email`/`orgId` can
    // be null until the first online call, and a null never proves two
    // homes are the same account.
    mutate(true, |store| {
        if store.accounts.iter().any(|a| a.label == label) {
            return Err(format!("account {label:?} already exists"));
        }
        if matches!(source, Source::Ambient)
            && let Some(a) = store
                .accounts
                .iter()
                .find(|a| spawn_home(&a.home).is_none())
        {
            return Err(format!(
                "the ambient login is already account {:?}",
                a.label
            ));
        }
        if let (Some(e), Some(o)) = (&email, &org)
            && let Some(dup) = store
                .accounts
                .iter()
                .find(|a| a.email.as_ref() == Some(e) && a.org.as_ref() == Some(o))
        {
            return Ok(Insert::Duplicate {
                label: dup.label.clone(),
                identity: format!("{e} / {o}"),
            });
        }
        store.accounts.push(Account {
            label: label.to_string(),
            home: home.clone(),
            email: email.clone(),
            org: org.clone(),
            added_at: now_secs(),
            health: Health::Ready,
            cooldown_until: 0,
            last_error: None,
            request_count: 0,
            last_used_at: 0,
        });
        if store.selected.is_none() {
            store.selected = Some(label.to_string());
        }
        Ok(Insert::Added {
            selected: store.selected.clone(),
            who: who.clone(),
        })
    })
}

fn ambient_label(email: Option<&str>, taken: &str) -> String {
    let mut label: String = email
        .and_then(|e| e.split('@').next())
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .take(32)
        .collect();
    if valid_label(&label).is_err() {
        label = "default".to_string();
    }
    if label == taken {
        label.push_str("-ambient");
    }
    label
}

fn adopt_ambient_first(new_label: &str) -> Option<String> {
    if !load().accounts.is_empty() {
        return None;
    }
    let st = crate::setup::auth_status(None)?;
    if st.get("loggedIn").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let label = ambient_label(st.get("email").and_then(Value::as_str), new_label);
    matches!(admit(&label, &Source::Ambient), Ok(Insert::Added { .. })).then_some(label)
}

fn cmd_add(args: &[String], interactive: bool, json: bool) -> Result<String, String> {
    let mut home: Option<PathBuf> = None;
    let mut ambient = false;
    let mut label: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--home" {
            let Some(h) = it.next() else {
                return Err("--home needs a directory".into());
            };
            home = Some(expand_home(Path::new(h)));
        } else if a == "--ambient" {
            ambient = true;
        } else if a.starts_with('-') {
            return Err(format!("unknown flag {a:?}\n\n{USAGE}"));
        } else if label.is_none() {
            label = Some(a.clone());
        } else {
            return Err(USAGE.into());
        }
    }
    let label = label.ok_or_else(|| format!("account add needs a label\n\n{USAGE}"))?;
    valid_label(&label)?;
    if ambient && home.is_some() {
        return Err("--ambient and --home are mutually exclusive".into());
    }
    if load().accounts.iter().any(|a| a.label == label) {
        return Err(format!("account {label:?} already exists"));
    }
    let binary = crate::setup::resolve_command().ok_or(crate::setup::INSTALL_HINT)?;
    let source = match home {
        _ if ambient => Source::Ambient,
        Some(h) => Source::Adopt(h),
        None => Source::Managed(homes_dir().ok_or("cannot resolve GRAY_HOME")?.join(&label)),
    };
    if let Source::Managed(dest) = &source {
        if !interactive {
            return Err(
                "account add runs `claude auth login`; use `gray claude-sub account add <label>` in a terminal"
                    .into(),
            );
        }
        let root = dest.parent().ok_or("cannot resolve GRAY_HOME")?;
        std::fs::create_dir_all(root).map_err(|e| format!("create {}: {e}", root.display()))?;
        let tmp = root.join(format!(
            ".{label}.{}.{}.tmp",
            std::process::id(),
            crate::session::new_uuid()
        ));
        std::fs::create_dir(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o700));
        }
        eprintln!("logging {label:?} into {} …", dest.display());
        let status = std::process::Command::new(&binary)
            .args(["auth", "login"])
            .envs(crate::setup::child_env())
            .env("CLAUDE_CONFIG_DIR", &tmp)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|e| format!("spawn claude auth login: {e}"));
        let placed = status
            .and_then(|s| {
                if s.success() {
                    Ok(())
                } else {
                    Err(format!("`claude auth login` exited {s}"))
                }
            })
            .and_then(|()| {
                std::fs::rename(&tmp, dest).map_err(|e| format!("place {}: {e}", dest.display()))
            });
        if let Err(e) = placed {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }
    }
    let adopted = if ambient {
        None
    } else {
        adopt_ambient_first(&label)
    };
    let insert = match admit(&label, &source) {
        Ok(insert) => insert,
        Err(e) => {
            if let Source::Managed(dest) = &source {
                let _ = std::fs::remove_dir_all(dest);
            }
            return Err(e);
        }
    };
    match insert {
        Insert::Duplicate {
            label: dup,
            identity,
        } => {
            // The login we just made is a duplicate — drop the managed home
            // it wrote into; adopted `--home` dirs are never touched.
            if let Source::Managed(dest) = &source {
                let _ = std::fs::remove_dir_all(dest);
            }
            Err(format!(
                "that login is already registered as {dup:?} ({identity})"
            ))
        }
        Insert::Added { selected, who } => {
            if json {
                return ok_json(json!({
                    "message": format!("account {label:?} added"),
                    "label": label,
                    "email": who,
                    "selected": selected,
                    "adopted_ambient": adopted,
                }));
            }
            let mut msg = format!(
                "account {label:?} added ({who}); home {}",
                home_text(source.home())
            );
            if let Some(a) = adopted {
                msg.push_str(&format!("\nambient login adopted as {a:?}"));
            }
            Ok(msg)
        }
    }
}

fn cmd_use(args: &[String], json: bool) -> Result<String, String> {
    match args.first().map(String::as_str) {
        None => {
            let store = load();
            let sel = store.selected.as_deref().unwrap_or("(none)");
            if json {
                return ok_json(json!({"selected": store.selected}));
            }
            Ok(format!("pinned account: {sel}"))
        }
        Some("--clear") | Some("none") | Some("auto") => {
            let selected = mutate(true, |store| {
                store.selected = None;
                Ok(store.selected.clone())
            })?;
            if json {
                return ok_json(json!({"selected": selected}));
            }
            Ok("pin cleared — selection follows the strategy".to_string())
        }
        Some(label) => {
            let selected = mutate(true, |store| {
                if !store.accounts.iter().any(|a| a.label == label) {
                    return Err(format!("no account {label:?}"));
                }
                store.selected = Some(label.to_string());
                Ok(store.selected.clone())
            })?;
            if json {
                return ok_json(json!({"selected": selected}));
            }
            Ok(format!("serving account: {label}"))
        }
    }
}

/// `account strategy fill_first|least_used` — bare prints the current
/// strategy. Selection among non-cooled accounts only; the pin still wins.
fn cmd_strategy(args: &[String], json: bool) -> Result<String, String> {
    let next = match args.first().map(String::as_str) {
        None => {
            let cur = load().strategy.name();
            if json {
                return ok_json(json!({"strategy": cur}));
            }
            return Ok(format!("strategy: {cur}"));
        }
        Some("fill_first") => Strategy::FillFirst,
        Some("least_used") => Strategy::LeastUsed,
        Some(other) => {
            return Err(format!(
                "unknown strategy {other:?} — fill_first or least_used"
            ));
        }
    };
    let set = mutate(true, |store| {
        store.strategy = next;
        Ok(store.strategy.name())
    })?;
    if json {
        return ok_json(json!({"strategy": set}));
    }
    Ok(format!("strategy: {set}"))
}

fn cmd_remove(args: &[String], json: bool) -> Result<String, String> {
    let mut keep = false;
    let mut label: Option<String> = None;
    for a in args {
        if a == "--keep" {
            keep = true;
        } else if label.is_none() {
            label = Some(a.clone());
        } else {
            return Err(USAGE.into());
        }
    }
    let label = label.ok_or_else(|| format!("account remove needs a label\n\n{USAGE}"))?;
    let acc = mutate(true, |store| {
        let Some(i) = store.accounts.iter().position(|a| a.label == label) else {
            return Err(format!("no account {label:?}"));
        };
        let acc = store.accounts.remove(i);
        if store.selected.as_deref() == Some(label.as_str()) {
            store.selected = None;
        }
        Ok(acc)
    })?;
    // Only homes we created are ours to delete; adopted `--home` dirs
    // belong to the user.
    let mut msg = format!("account {label:?} removed");
    if !keep
        && let Some(dir) = homes_dir()
        && let Some(home) = spawn_home(&acc.home)
        && home.starts_with(&dir)
        && home.is_dir()
    {
        std::fs::remove_dir_all(home)
            .map_err(|e| format!("removed {label:?} but not {}: {e}", home.display()))?;
        msg.push_str(&format!(" and its credential home {}", home.display()));
    }
    if json {
        return ok_json(json!({"message": msg}));
    }
    Ok(msg)
}

fn account_line(a: &Account, now: u64, selected: &Option<String>) -> String {
    let mark = if selected.as_deref() == Some(a.label.as_str()) {
        '*'
    } else {
        ' '
    };
    let state = match current(a, now) {
        Health::Ready => "ok".to_string(),
        Health::Cooling { until, .. } => format!(
            "cooled until {}",
            crate::session::iso_timestamp(UNIX_EPOCH + Duration::from_secs(until))
        ),
        Health::AuthFailed { until, .. } => format!(
            "auth failed until {}",
            crate::session::iso_timestamp(UNIX_EPOCH + Duration::from_secs(until))
        ),
    };
    let who = a.email.clone().unwrap_or_else(|| "-".to_string());
    format!(
        "{mark} {:<16} {:<12} {:<30} {}",
        a.label,
        state,
        who,
        home_text(&a.home)
    )
}

fn cmd_list(json: bool) -> Result<String, String> {
    let store = load_checked()?;
    if json {
        return store_json(&store);
    }
    if store.accounts.is_empty() {
        return Ok("no accounts configured — turns use the ambient `claude` login".to_string());
    }
    let now = now_secs();
    let mut out = format!("{} account(s):\n", store.accounts.len());
    for a in &store.accounts {
        out.push_str(&account_line(a, now, &store.selected));
        out.push('\n');
    }
    out.push_str("`use` pins *; fill_first order is file order.");
    Ok(out)
}

fn cmd_status(json: bool) -> Result<String, String> {
    let store = load_checked()?;
    if json {
        return store_json(&store);
    }
    if store.accounts.is_empty() {
        return Ok("no accounts configured — serving the ambient `claude` login".to_string());
    }
    let now = now_secs();
    let mut out = match serve(&store) {
        Serving::Account(p) => {
            format!(
                "serving account: {} (home {})\n",
                p.label,
                home_text(p.home.as_deref().unwrap_or(Path::new("")))
            )
        }
        Serving::Ambient => "serving account: (none)".to_string(),
        Serving::AllCooling { reason, .. } => format!("serving account: {reason}\n"),
    };
    out.push_str(&format!("strategy: {}\n", store.strategy.name()));
    for a in &store.accounts {
        out.push_str(&account_line(a, now, &store.selected));
        if let Some(e) = reason_of(&current(a, now)) {
            out.push_str(&format!("  last error: {e}"));
        }
        out.push('\n');
    }
    Ok(out)
}

const USAGE_DEADLINE: Duration = Duration::from_secs(crate::usage::PROBE_TIMEOUT.as_secs() + 5);

fn usage_entry(label: Option<String>, outcome: Result<Value, String>) -> Value {
    match outcome {
        Ok(card) => json!({"label": label, "usage": card, "error": null}),
        Err(e) => json!({"label": label, "usage": null, "error": e}),
    }
}

type UsageTargets = Vec<(Option<String>, Result<Option<Pick>, String>)>;

fn usage_targets(store: &Store, want: Option<&str>) -> UsageTargets {
    if store.accounts.is_empty() {
        return match want {
            None => vec![(None, Ok(None))],
            Some(w) => vec![(
                Some(w.to_string()),
                Err(format!(
                    "no account labelled {w:?}: no pool configured, the ambient login has no label"
                )),
            )],
        };
    }
    let picked: UsageTargets = store
        .accounts
        .iter()
        .filter(|a| want.is_none_or(|w| a.label == w))
        .map(|a| (Some(a.label.clone()), Ok(Some(pick_of(a)))))
        .collect();
    match want {
        Some(w) if picked.is_empty() => vec![(
            Some(w.to_string()),
            Err(format!("no account labelled {w:?}")),
        )],
        _ => picked,
    }
}

fn usage_text(entry: &Value) -> String {
    let mut out = match entry.get("label").and_then(Value::as_str) {
        Some(l) => format!("{l}\n"),
        None => "ambient\n".to_string(),
    };
    if let Some(e) = entry.get("error").and_then(Value::as_str) {
        out.push_str(&format!("  error: {e}\n"));
    }
    let windows = entry.pointer("/usage/windows").and_then(Value::as_array);
    if entry.get("usage").is_some_and(|u| !u.is_null()) && windows.is_none_or(Vec::is_empty) {
        out.push_str("  no quota windows reported\n");
    }
    for w in windows.into_iter().flatten() {
        let name = w
            .get("label")
            .or(w.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let pct = w
            .get("used_percent")
            .and_then(Value::as_f64)
            .map_or("-".to_string(), |p| format!("{p:.0}%"));
        let reset = w.get("resets_at").and_then(Value::as_str).unwrap_or("-");
        out.push_str(&format!("  {name:<14} {pct:>5}  resets {reset}\n"));
    }
    out.trim_end().to_string()
}

enum Fetch {
    Pending,
    Loaded(Value),
    Failed(String),
}

impl Fetch {
    fn settle(self) -> Result<Value, String> {
        match self {
            Fetch::Pending => Err(format!("no reply within {}s", USAGE_DEADLINE.as_secs())),
            Fetch::Loaded(card) => Ok(card),
            Fetch::Failed(e) => Err(e),
        }
    }
}

fn cmd_usage(args: &[String], json: bool) -> Result<String, String> {
    let want = args.first().map(String::as_str);
    let targets = usage_targets(&load_checked()?, want);
    let labels: Vec<Option<String>> = targets.iter().map(|(l, _)| l.clone()).collect();
    let mut fetches: Vec<Fetch> = targets
        .iter()
        .map(|(_, target)| match target {
            Err(e) => Fetch::Failed(e.clone()),
            Ok(_) => Fetch::Pending,
        })
        .collect();
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, (_, target)) in targets.into_iter().enumerate() {
        if let Ok(pick) = target {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let outcome = crate::usage::handle(pick.as_ref())
                    .map(|mut card| {
                        if pick.as_ref().is_none_or(|p| p.home.is_none())
                            && let Some(email) = crate::setup::auth_status(None)
                                .and_then(|s| s.get("email")?.as_str().map(str::to_string))
                        {
                            card["email"] = json!(email);
                        }
                        card
                    })
                    .map_err(|e| e.to_string());
                let _ = tx.send((i, outcome));
            });
        }
    }
    drop(tx);
    let deadline = Instant::now() + USAGE_DEADLINE;
    while let Ok((i, outcome)) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
    {
        fetches[i] = match outcome {
            Ok(card) => Fetch::Loaded(card),
            Err(e) => Fetch::Failed(e),
        };
    }
    let entries: Vec<Value> = fetches
        .into_iter()
        .zip(labels)
        .map(|(fetch, label)| usage_entry(label, fetch.settle()))
        .collect();
    if json {
        return serde_json::to_string_pretty(&json!({
            "version": 1,
            "plugin": crate::manifest::PLUGIN_NAME,
            "provider": crate::manifest::PROVIDER_ID,
            "accounts": entries,
        }))
        .map_err(|e| e.to_string());
    }
    Ok(entries
        .iter()
        .map(usage_text)
        .collect::<Vec<_>>()
        .join("\n"))
}

/// The pick an auth or usage probe uses: the serving account, or — when
/// every account is cooling — the first configured one. Login state and
/// quota windows don't care about cooldowns. `None` means the ambient
/// `claude` login (no accounts configured).
pub fn probe_pick() -> Option<Pick> {
    let store = load();
    let first = store.accounts.first()?;
    Some(match serve(&store) {
        Serving::Account(p) => p,
        _ => pick_of(first),
    })
}

/// Just the home of [`probe_pick`].
pub fn probe_home() -> Option<PathBuf> {
    probe_pick().and_then(|p| p.home)
}
