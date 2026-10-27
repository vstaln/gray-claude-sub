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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
/// the spawned `claude` child gets; health fields are write-through.
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
    /// Unix seconds the cooldown ends; 0 = not cooling.
    #[serde(default)]
    pub cooldown_until: u64,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub request_count: u64,
    #[serde(default)]
    pub last_used_at: u64,
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
    strategy: String,
}

/// The account a turn is served by: label plus the credential home the
/// spawned `claude` children get as `CLAUDE_CONFIG_DIR`.
pub struct Pick {
    pub label: String,
    pub home: PathBuf,
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

fn file_path() -> Option<PathBuf> {
    crate::settings::state_dir().map(|d| d.join("accounts.json"))
}

/// Managed homes live here: `<gray home>/claude-sub/homes/<label>`.
fn homes_dir() -> Option<PathBuf> {
    crate::settings::state_dir().map(|d| d.join("homes"))
}

fn load() -> Store {
    let Some(path) = file_path() else {
        return Store::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Store::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save(store: &Store) -> Result<(), String> {
    let path = file_path().ok_or("cannot resolve GRAY_HOME")?;
    let text = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    crate::session::write_atomic(&path, text.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn cooled(a: &Account, now: u64) -> bool {
    a.cooldown_until > now
}

fn pick_of(a: &Account) -> Pick {
    Pick {
        label: a.label.clone(),
        home: expand_home(&a.home),
    }
}

fn pick_from(store: &Store) -> Result<Option<Pick>, String> {
    if store.accounts.is_empty() {
        return Ok(None);
    }
    let now = now_secs();
    // A usable pinned account always serves.
    if let Some(sel) = store.selected.as_deref()
        && let Some(a) = store
            .accounts
            .iter()
            .find(|a| a.label == sel && !cooled(a, now))
    {
        return Ok(Some(pick_of(a)));
    }
    let next = if store.strategy == "least_used" {
        store
            .accounts
            .iter()
            .filter(|a| !cooled(a, now))
            .min_by_key(|a| a.last_used_at)
    } else {
        // fill_first: file order is the priority order.
        store.accounts.iter().find(|a| !cooled(a, now))
    };
    if let Some(a) = next {
        return Ok(Some(pick_of(a)));
    }
    let soon = store
        .accounts
        .iter()
        .min_by_key(|a| a.cooldown_until)
        .expect("nonempty");
    Err(format!(
        "all {} claude accounts are cooling down; {} next at {}",
        store.accounts.len(),
        soon.label,
        crate::session::iso_timestamp(UNIX_EPOCH + Duration::from_secs(soon.cooldown_until)),
    ))
}

/// The account a turn would use — read-only (no use recorded). Probes
/// and `provider/auth/*` use this; only `provider/chat` calls [`pick`].
pub fn serving() -> Result<Option<Pick>, String> {
    pick_from(&load())
}

/// Per-turn selection: resolve the serving account and record the use.
pub fn pick() -> Result<Option<Pick>, String> {
    let mut store = load();
    let Some(p) = pick_from(&store)? else {
        return Ok(None);
    };
    if let Some(a) = store.accounts.iter_mut().find(|a| a.label == p.label) {
        a.request_count += 1;
        a.last_used_at = now_secs();
        let _ = save(&store);
    }
    Ok(Some(p))
}

/// The turn this account served succeeded: clear any recorded failure.
pub fn note_success(label: &str) {
    let mut store = load();
    let Some(a) = store.accounts.iter_mut().find(|a| a.label == label) else {
        return;
    };
    if a.cooldown_until != 0 || a.last_error.is_some() {
        a.cooldown_until = 0;
        a.last_error = None;
        let _ = save(&store);
    }
}

/// The turn this account served failed with `detail`. Only account-
/// scoped upstream verdicts cool it; anything else leaves the pool
/// untouched.
pub fn note_failure(label: &str, detail: &str) {
    let Some((code, msg)) = crate::chat::api_error(detail) else {
        return;
    };
    let ttl = match code {
        401 => COOL_401,
        429 => parse_retry_after(msg)
            .map(|d| d.as_secs().min(MAX_RESET))
            .unwrap_or(COOL_429),
        // Request-scoped verdicts never cool an account.
        _ => return,
    };
    cool(label, ttl, detail);
}

/// [`note_failure`] keyed by credential home: a keepalive probe knows
/// the session's `CLAUDE_CONFIG_DIR`, not its label.
pub fn note_failure_home(home: &Path, detail: &str) {
    let want = normalize(home);
    let store = load();
    let Some(label) = store
        .accounts
        .iter()
        .find(|a| normalize(&a.home) == want)
        .map(|a| a.label.clone())
    else {
        return;
    };
    drop(store);
    note_failure(&label, detail);
}

/// The label of the account owning `home`, if any — `usage` keys its
/// cache per account this way.
pub fn label_for_home(home: &Path) -> Option<String> {
    let want = normalize(home);
    load()
        .accounts
        .iter()
        .find(|a| normalize(&a.home) == want)
        .map(|a| a.label.clone())
}

fn cool(label: &str, ttl: u64, detail: &str) {
    let mut store = load();
    let ttl = if store.accounts.len() == 1 {
        ttl.min(SOLE_CAP)
    } else {
        ttl.min(MAX_RESET)
    };
    let Some(a) = store.accounts.iter_mut().find(|a| a.label == label) else {
        return;
    };
    a.cooldown_until = now_secs() + ttl.max(1);
    a.last_error = Some(detail.chars().take(200).collect());
    let _ = save(&store);
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
  claude-sub account list                          accounts + health
  claude-sub account use <label>                   pin an account (use --clear to unpin)
  claude-sub account remove [--keep] <label>       drop an account (+its managed home)
  claude-sub account strategy <fill_first|least_used>
                                                  pick order among usable accounts
  claude-sub account status                        which account serves now
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
        Some("list") | None => cmd_list(json),
        Some(other) => Err(format!("unknown account verb {other:?}\n\n{USAGE}")),
    }
}

/// The machine-readable pool document. `ambient` is true when no pool is
/// configured and the ambient `claude` login serves; `serving.kind` is
/// `account`, `ambient`, `cooldown` (every account benched — `reason`
/// names the earliest reset) or `none`.
fn store_json(store: &Store) -> Result<String, String> {
    let now = now_secs();
    let accounts: Vec<Value> = store
        .accounts
        .iter()
        .map(|a| {
            json!({
                "label": a.label,
                "home": a.home.display().to_string(),
                "email": a.email,
                "org": a.org,
                "added_at": a.added_at,
                "cooldown_until": a.cooldown_until,
                "cooling": cooled(a, now),
                "last_error": a.last_error,
                "request_count": a.request_count,
                "last_used_at": a.last_used_at,
                "pinned": store.selected.as_deref() == Some(a.label.as_str()),
            })
        })
        .collect();
    let serving = match pick_from(store) {
        Ok(Some(p)) => json!({"kind": "account", "label": p.label}),
        Ok(None) if store.accounts.is_empty() => json!({"kind": "ambient"}),
        Ok(None) => json!({"kind": "none"}),
        Err(e) => {
            let until = store
                .accounts
                .iter()
                .map(|a| a.cooldown_until)
                .filter(|u| *u > now)
                .min()
                .unwrap_or(0);
            json!({"kind": "cooldown", "until": until, "reason": e})
        }
    };
    serde_json::to_string_pretty(&json!({
        "version": 1,
        "plugin": crate::manifest::PLUGIN_NAME,
        "provider": crate::manifest::PROVIDER_ID,
        "strategy": if store.strategy.is_empty() { "fill_first" } else { &store.strategy },
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

fn cmd_add(args: &[String], interactive: bool, json: bool) -> Result<String, String> {
    let mut home: Option<PathBuf> = None;
    let mut label: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--home" {
            let Some(h) = it.next() else {
                return Err("--home needs a directory".into());
            };
            home = Some(expand_home(Path::new(h)));
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
    let mut store = load();
    if store.accounts.iter().any(|a| a.label == label) {
        return Err(format!("account {label:?} already exists"));
    }
    let binary = crate::setup::resolve_command().ok_or(crate::setup::INSTALL_HINT)?;
    let managed = home.is_none();
    let home = match home {
        Some(h) => h,
        None => homes_dir().ok_or("cannot resolve GRAY_HOME")?.join(&label),
    };
    if managed {
        if !interactive {
            return Err(
                "account add runs `claude auth login`; use `gray claude-sub account add <label>` in a terminal"
                    .into(),
            );
        }
        std::fs::create_dir_all(&home).map_err(|e| format!("create {}: {e}", home.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700));
        }
        eprintln!("logging {label:?} into {} …", home.display());
        let status = std::process::Command::new(&binary)
            .args(["auth", "login"])
            .envs(crate::setup::child_env())
            .env("CLAUDE_CONFIG_DIR", &home)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|e| format!("spawn claude auth login: {e}"))?;
        if !status.success() {
            return Err(format!("`claude auth login` exited {status}"));
        }
    }
    let st = crate::setup::auth_status(Some(&home))
        .ok_or("`claude auth status` failed in the new home")?;
    if st.get("loggedIn").and_then(Value::as_bool) != Some(true) {
        return Err(format!(
            "no login in {} — {}",
            home.display(),
            if managed {
                "`claude auth login` did not complete"
            } else {
                "adopt a directory that holds a real login"
            },
        ));
    }
    let email = st.get("email").and_then(Value::as_str).map(str::to_string);
    let org = st
        .get("orgName")
        .and_then(Value::as_str)
        .or_else(|| st.get("orgId").and_then(Value::as_str))
        .map(str::to_string);
    // Dedupe on a verified upstream identity only — `email`/`orgId` can
    // be null until the first online call, and a null never proves two
    // homes are the same account.
    if let (Some(e), Some(o)) = (&email, &org)
        && let Some(dup) = store
            .accounts
            .iter()
            .find(|a| a.email.as_ref() == Some(e) && a.org.as_ref() == Some(o))
    {
        // The login we just made is a duplicate — drop the managed home
        // it wrote into; adopted `--home` dirs are never touched.
        if managed
            && let Some(dir) = homes_dir()
            && home.starts_with(&dir)
        {
            let _ = std::fs::remove_dir_all(&home);
        }
        return Err(format!(
            "that login is already registered as {:?} ({e} / {o})",
            dup.label
        ));
    }
    store.accounts.push(Account {
        label: label.clone(),
        home: home.clone(),
        email,
        org,
        added_at: now_secs(),
        cooldown_until: 0,
        last_error: None,
        request_count: 0,
        last_used_at: 0,
    });
    if store.selected.is_none() {
        store.selected = Some(label.clone());
    }
    save(&store)?;
    let who = store
        .accounts
        .last()
        .and_then(|a| a.email.clone())
        .unwrap_or_else(|| "identity pending".to_string());
    if json {
        return ok_json(json!({
            "message": format!("account {label:?} added"),
            "label": label,
            "email": who,
            "selected": store.selected,
        }));
    }
    Ok(format!(
        "account {label:?} added ({who}); home {}",
        home.display()
    ))
}

fn cmd_use(args: &[String], json: bool) -> Result<String, String> {
    let mut store = load();
    match args.first().map(String::as_str) {
        None => {
            let sel = store.selected.as_deref().unwrap_or("(none)");
            if json {
                return ok_json(json!({"selected": store.selected}));
            }
            Ok(format!("pinned account: {sel}"))
        }
        Some("--clear") | Some("none") | Some("auto") => {
            store.selected = None;
            save(&store)?;
            if json {
                return ok_json(json!({"selected": store.selected}));
            }
            Ok("pin cleared — selection follows the strategy".to_string())
        }
        Some(label) => {
            if !store.accounts.iter().any(|a| a.label == label) {
                return Err(format!("no account {label:?}"));
            }
            store.selected = Some(label.to_string());
            save(&store)?;
            if json {
                return ok_json(json!({"selected": store.selected}));
            }
            Ok(format!("serving account: {label}"))
        }
    }
}

/// `account strategy fill_first|least_used` — bare prints the current
/// strategy. Selection among non-cooled accounts only; the pin still wins.
fn cmd_strategy(args: &[String], json: bool) -> Result<String, String> {
    let mut store = load();
    match args.first().map(String::as_str) {
        None => {
            let cur = if store.strategy.is_empty() {
                "fill_first"
            } else {
                &store.strategy
            };
            if json {
                return ok_json(json!({"strategy": cur}));
            }
            Ok(format!("strategy: {cur}"))
        }
        Some("fill_first") | Some("least_used") => {
            store.strategy = args[0].clone();
            save(&store)?;
            if json {
                return ok_json(json!({"strategy": store.strategy}));
            }
            Ok(format!("strategy: {}", store.strategy))
        }
        Some(other) => Err(format!(
            "unknown strategy {other:?} — fill_first or least_used"
        )),
    }
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
    let mut store = load();
    let Some(i) = store.accounts.iter().position(|a| a.label == label) else {
        return Err(format!("no account {label:?}"));
    };
    let acc = store.accounts.remove(i);
    if store.selected.as_deref() == Some(label.as_str()) {
        store.selected = None;
    }
    save(&store)?;
    // Only homes we created are ours to delete; adopted `--home` dirs
    // belong to the user.
    let mut msg = format!("account {label:?} removed");
    if !keep
        && let Some(dir) = homes_dir()
        && acc.home.starts_with(&dir)
        && acc.home.is_dir()
    {
        std::fs::remove_dir_all(&acc.home)
            .map_err(|e| format!("removed {label:?} but not {}: {e}", acc.home.display()))?;
        msg.push_str(&format!(" and its credential home {}", acc.home.display()));
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
    let state = if cooled(a, now) {
        format!(
            "cooled until {}",
            crate::session::iso_timestamp(UNIX_EPOCH + Duration::from_secs(a.cooldown_until))
        )
    } else {
        "ok".to_string()
    };
    let who = a.email.clone().unwrap_or_else(|| "-".to_string());
    format!(
        "{mark} {:<16} {:<12} {:<30} {}",
        a.label,
        state,
        who,
        a.home.display()
    )
}

fn cmd_list(json: bool) -> Result<String, String> {
    let store = load();
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
    let store = load();
    if json {
        return store_json(&store);
    }
    if store.accounts.is_empty() {
        return Ok("no accounts configured — serving the ambient `claude` login".to_string());
    }
    let mut out = match pick_from(&store) {
        Ok(Some(p)) => format!("serving account: {} (home {})\n", p.label, p.home.display()),
        Ok(None) => "serving account: (none)".to_string(),
        Err(e) => format!("serving account: {e}\n"),
    };
    out.push_str(&format!(
        "strategy: {}\n",
        if store.strategy.is_empty() {
            "fill_first"
        } else {
            &store.strategy
        }
    ));
    let now = now_secs();
    for a in &store.accounts {
        out.push_str(&account_line(a, now, &store.selected));
        if let Some(e) = &a.last_error {
            out.push_str(&format!("  last error: {e}"));
        }
        out.push('\n');
    }
    Ok(out)
}

/// The pick an auth or usage probe uses: the serving account, or — when
/// every account is cooling — the first configured one. Login state and
/// quota windows don't care about cooldowns. `None` means the ambient
/// `claude` login (no accounts configured).
pub fn probe_pick() -> Option<Pick> {
    let store = load();
    let first = store.accounts.first()?;
    Some(match pick_from(&store) {
        Ok(Some(p)) => p,
        _ => pick_of(first),
    })
}

/// Just the home of [`probe_pick`].
pub fn probe_home() -> Option<PathBuf> {
    probe_pick().map(|p| p.home)
}
