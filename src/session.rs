//! Native session files: where a conversation lives between turns.
//!
//! Each turn persists its native session (`~/.claude/projects/<cwd>/<sid>.jsonl`,
//! written by the CLI) and records a resume point for the conversation in
//! `~/.cache/claude-sub/resume/<key>`. When no resume point matches, the
//! history is written as a synthetic transcript and resumed instead of being
//! replayed over stdin (stdin replay detaches every historical tool_result
//! from its tool_use; see `chat::spawn_turn`).
//!
//! Everything here is best-effort: a lost or unreadable file only costs a
//! re-synthesis, never a failed turn. Files we create are swept once idle
//! for [`TTL`]; resume is a cache optimization (the prompt cache lives an
//! hour), so nothing of value outlives a day.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde_json::{Value, json};

/// Idle age past which our resume points and native session files go.
pub const TTL: Duration = Duration::from_secs(24 * 3600);
/// Minimum gap between sweeps (shared across processes via a marker file).
const SWEEP_EVERY: Duration = Duration::from_secs(3600);

/// Where a conversation lives: native session id + transcript uuid of the
/// assistant message to resume at.
pub type ResumePoint = (String, String);

/// `$XDG_CACHE_HOME/claude-sub` (or `~/.cache/claude-sub`).
pub fn cache_root() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("claude-sub")
}

/// The CLI's project store: `$CLAUDE_CONFIG_DIR/projects` or
/// `~/.claude/projects` (the child inherits `CLAUDE_CONFIG_DIR`).
pub fn projects_root() -> Option<PathBuf> {
    std::env::var_os("CLAUDE_CONFIG_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".claude")))
        .map(|d| d.join("projects"))
}

/// The CLI's project directory name for `cwd`: every UTF-16 unit outside
/// `[A-Za-z0-9]` becomes `-`. Past 200 units the CLI appends a hash we
/// don't reproduce, so those cwds get `None` (no synthetic transcript).
pub fn project_dir_name(cwd: &str) -> Option<String> {
    let mut out = String::with_capacity(cwd.len());
    for c in cwd.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            out.extend(std::iter::repeat_n('-', c.len_utf16()));
        }
    }
    (out.len() <= 200).then_some(out)
}

/// Canonical 8-4-4-4-12 hex uuid. Checked before an id becomes an argv or
/// path component.
pub fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.char_indices().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => c == '-',
            _ => c.is_ascii_hexdigit(),
        })
}

/// Random v4 uuid without a crate: std's per-instance random SipHash keys.
pub fn new_uuid() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let word = || {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(SEQ.fetch_add(1, Ordering::Relaxed));
        h.write_u128(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.finish()
    };
    let (a, b) = (word(), word());
    let n = (u128::from(a) << 64 | u128::from(b)) & !(0xf000u128 << 64) & !(0xc << 60)
        | (0x4000u128 << 64)
        | (0x8u128 << 60);
    let h = format!("{n:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// RFC 3339 UTC timestamp with milliseconds, as the CLI writes them.
pub fn iso_timestamp(t: SystemTime) -> String {
    let d = t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60,
        d.subsec_millis()
    )
}

/// Write `bytes` to `path` atomically (temp file + rename), mode 0600,
/// creating the parent directory. Concurrent writers never expose a torn
/// file: readers see the old content or the new, whole.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut mkdir = std::fs::DirBuilder::new();
    mkdir.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        mkdir.mode(0o700);
    }
    mkdir.create(dir)?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp = dir.join(format!(".{name}.{}.{}.tmp", std::process::id(), new_uuid()));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let res = opts
        .open(&tmp)
        .and_then(|mut f| f.write_all(bytes))
        .and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Native transcript lines for `frames` (history ending in an assistant
/// frame) under session `sid`, chained by parentUuid. Returns the lines and
/// the last entry's uuid (the `--resume-session-at` point). `uuid` mints
/// entry ids; assistant messages without an id get a stable synthetic one
/// so the CLI doesn't merge neighbours.
pub fn transcript_lines(
    frames: &[Value],
    sid: &str,
    cwd: &str,
    timestamp: &str,
    mut uuid: impl FnMut() -> String,
) -> (Vec<String>, String) {
    let mut lines = Vec::with_capacity(frames.len());
    let mut parent: Option<String> = None;
    for (i, f) in frames.iter().enumerate() {
        let kind = f.get("type").and_then(Value::as_str).unwrap_or("user");
        let mut message = f.get("message").cloned().unwrap_or(json!({}));
        if kind == "assistant" {
            for (k, v) in [
                ("id", json!(format!("msg_claude_sub_{i}"))),
                ("type", json!("message")),
                ("role", json!("assistant")),
            ] {
                if message.get(k).is_none_or(Value::is_null) {
                    message[k] = v;
                }
            }
        }
        let id = uuid();
        lines.push(
            json!({
                "parentUuid": parent,
                "isSidechain": false,
                "type": kind,
                "message": message,
                "uuid": id,
                "timestamp": timestamp,
                "userType": "external",
                "cwd": cwd,
                "sessionId": sid,
            })
            .to_string(),
        );
        parent = Some(id);
    }
    (lines, parent.unwrap_or_default())
}

/// Write `frames` as a fresh native session for the current cwd and return
/// where to resume it. `None` when the CLI's project dir can't be derived
/// or the file can't be written: the caller falls back to stdin replay.
pub fn synthesize(frames: &[Value]) -> Option<ResumePoint> {
    let cwd = std::env::current_dir().ok()?;
    let cwd = cwd.to_str()?;
    let dir = projects_root()?.join(project_dir_name(cwd)?);
    let sid = new_uuid();
    let (lines, at) = transcript_lines(
        frames,
        &sid,
        cwd,
        &iso_timestamp(SystemTime::now()),
        new_uuid,
    );
    if at.is_empty() {
        return None;
    }
    touch_ledger(&sid);
    write_atomic(
        &dir.join(format!("{sid}.jsonl")),
        (lines.join("\n") + "\n").as_bytes(),
    )
    .ok()?;
    Some((sid, at))
}

/// Session file the CLI keeps for `sid` under the current cwd.
fn session_file(sid: &str) -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let name = project_dir_name(cwd.to_str()?)?;
    Some(projects_root()?.join(name).join(format!("{sid}.jsonl")))
}

/// Delete a throwaway session we created (a keepalive probe's fork): its
/// transcript, its sidecar directory and its ledger entry. Best-effort.
pub fn discard(sid: &str) {
    if !is_uuid(sid) {
        return;
    }
    if let Some(path) = session_file(sid) {
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(path.with_extension(""));
    }
    let _ = std::fs::remove_file(cache_root().join("sessions").join(sid));
}

/// Make session `sid` safe to resume: drop native's own tool executions.
///
/// With `--max-turns 1` native still tries each tool_use before stopping —
/// gray's tools don't exist natively, so it records an `is_error` "No such
/// tool available" result (marked `sourceToolAssistantUUID`). The next turn
/// branches the real result off the same assistant entry, but on a later
/// resume native re-attaches its own error result to that tool_use, and the
/// model sees its earlier call fail. Returns whether the session file is
/// in place and clean (only then is a resume point worth recording).
pub fn prune_native_tool_results(sid: &str) -> bool {
    let Some(path) = session_file(sid) else {
        return false;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return false;
    };
    match prune_lines(&text) {
        None => true,
        Some(clean) => write_atomic(&path, clean.as_bytes()).is_ok(),
    }
}

/// [`prune_native_tool_results`] on transcript text: drop every `user`
/// entry carrying `sourceToolAssistantUUID` and re-parent its children onto
/// its parent. Parallel calls need the re-parenting: native chains the
/// second tool_use block's entry under the first block's error result.
/// `None` when nothing needs dropping. Lines that don't parse are kept
/// verbatim.
pub fn prune_lines(text: &str) -> Option<String> {
    // Dropped entry uuid -> its nearest kept ancestor (None at the root).
    let mut dropped: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    for line in text.lines() {
        let Ok(mut v) = serde_json::from_str::<Value>(line) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let parent = v.get("parentUuid").cloned().unwrap_or(Value::Null);
        let kept_parent = parent
            .as_str()
            .and_then(|p| dropped.get(p))
            .cloned()
            .unwrap_or_else(|| parent.clone());
        if v.get("type").and_then(Value::as_str) == Some("user")
            && v.get("sourceToolAssistantUUID").is_some()
        {
            changed = true;
            if let Some(u) = v.get("uuid").and_then(Value::as_str) {
                dropped.insert(u.to_string(), kept_parent);
            }
            continue;
        }
        if kept_parent != parent {
            changed = true;
            v["parentUuid"] = kept_parent;
            out.push_str(&v.to_string());
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    changed.then_some(out)
}

fn resume_file(key: u64) -> PathBuf {
    cache_root().join("resume").join(format!("{key:016x}"))
}

/// The resume point recorded for conversation `key`, if any and well-formed.
pub fn lookup(key: u64) -> Option<ResumePoint> {
    parse_point(&std::fs::read_to_string(resume_file(key)).ok()?)
}

/// Parse a resume file: `<sid>\n<uuid>\n`, both canonical uuids.
pub fn parse_point(s: &str) -> Option<ResumePoint> {
    let mut it = s.lines();
    let (sid, uuid) = (it.next()?.trim(), it.next()?.trim());
    (is_uuid(sid) && is_uuid(uuid)).then(|| (sid.to_string(), uuid.to_string()))
}

/// Record that conversation `key` now lives at (`sid`, `uuid`), mark the
/// session as ours and recently used, and sweep idle leftovers.
pub fn save(key: u64, sid: &str, uuid: &str) {
    if !is_uuid(sid) || !is_uuid(uuid) {
        return;
    }
    let _ = write_atomic(&resume_file(key), format!("{sid}\n{uuid}\n").as_bytes());
    touch_ledger(sid);
    if let Some(projects) = projects_root() {
        sweep_if_due(&cache_root(), &projects, SystemTime::now());
    }
}

/// Mark native session `sid` (from a run's stream-json) as ours, so the
/// sweep reclaims it even if the turn failed and no resume point names it.
pub fn adopt(sid: &str) {
    if is_uuid(sid) {
        touch_ledger(sid);
    }
}

/// Ledger entry `sessions/<sid>`: marks a native session as created or used
/// by us; its mtime is the last use.
fn touch_ledger(sid: &str) {
    let _ = write_atomic(&cache_root().join("sessions").join(sid), b"");
}

fn idle_past(path: &Path, now: SystemTime, ttl: Duration) -> bool {
    path.symlink_metadata()
        .and_then(|m| m.modified())
        .is_ok_and(|t| now.duration_since(t).is_ok_and(|age| age > ttl))
}

/// At most once per [`SWEEP_EVERY`] (marker `root/.swept`), run [`sweep`].
fn sweep_if_due(root: &Path, projects: &Path, now: SystemTime) {
    let marker = root.join(".swept");
    if marker.exists() && !idle_past(&marker, now, SWEEP_EVERY) {
        return;
    }
    if write_atomic(&marker, b"").is_ok() {
        sweep(root, projects, now, TTL);
    }
}

/// Delete resume points idle past `ttl`, and every native session in the
/// ledger idle past `ttl` (its `<sid>.jsonl` and `<sid>/` in any project
/// dir — only ids we created or resumed, never the user's own sessions).
pub fn sweep(root: &Path, projects: &Path, now: SystemTime, ttl: Duration) {
    let entries = |d: PathBuf| {
        std::fs::read_dir(d)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
    };
    for p in entries(root.join("resume")) {
        if idle_past(&p, now, ttl) {
            let _ = std::fs::remove_file(&p);
        }
    }
    let mut gone: Vec<String> = Vec::new();
    for p in entries(root.join("sessions")) {
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        if !idle_past(&p, now, ttl) {
            continue;
        }
        if is_uuid(&name) {
            gone.push(name);
        }
        let _ = std::fs::remove_file(&p);
    }
    if gone.is_empty() {
        return;
    }
    for proj in entries(projects.to_path_buf()) {
        for sid in &gone {
            let _ = std::fs::remove_file(proj.join(format!("{sid}.jsonl")));
            let sub = proj.join(sid);
            if sub.is_dir() {
                let _ = std::fs::remove_dir_all(sub);
            }
        }
    }
}

#[path = "session_tests.rs"]
#[cfg(test)]
mod tests;
