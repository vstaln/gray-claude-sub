//! `provider/usage`: Claude subscription quota windows.
//!
//! Two sources, one cache (`~/.cache/claude-sub/usage.json`):
//!
//! - **Live probe**: `claude -p --input-format stream-json` answers a
//!   `control_request {"subtype":"get_usage"}` with a `control_response`
//!   carrying `rate_limits` — `five_hour`, `seven_day`, model-scoped
//!   weeklies, monthly dollar buckets and `extra_usage`. The probe spawns a
//!   throwaway child (~10–15s), so results cache for 5 minutes.
//! - **`rate_limit_event`** (streamed during turns): `unifiedWindows`
//!   carries every window's utilization + reset at once — one event
//!   refreshes the whole cache without a probe.
//!
//! Window shape is shared with t3code's `claudeUsageLimits` layer: same
//! ids (`five_hour`, `seven_day`, `seven_day_*`), same percent+reset rows.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use gray_plugin::ProviderRpcError;

/// A cached probe stays fresh this long before `/usage` respawns a child.
const CACHE_TTL_SECS: u64 = 5 * 60;
/// Hard cap on the probe child's lifetime: init + control_response is
/// normally seconds; a hung spawn must not hang `/usage`.
const PROBE_TIMEOUT: Duration = Duration::from_secs(45);

const SESSION_MINS: u64 = 5 * 60;
const WEEK_MINS: u64 = 7 * 24 * 60;

fn cache_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    Some(home.join(".cache/claude-sub/usage.json"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `provider/usage` entry: serve the cache when fresh, else probe.
pub fn handle() -> Result<Value, ProviderRpcError> {
    if let Some((limits, fetched)) = read_cache()
        && now_secs().saturating_sub(fetched) < CACHE_TTL_SECS
    {
        return Ok(limits);
    }
    match probe() {
        Ok(limits) => {
            write_cache(&limits, now_secs());
            Ok(limits)
        }
        Err(e) => {
            // A stale cache beats nothing: report it with its real age.
            if let Some((limits, _)) = read_cache() {
                return Ok(limits);
            }
            Err(e)
        }
    }
}

fn read_cache() -> Option<(Value, u64)> {
    let s = std::fs::read_to_string(cache_path()?).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;
    let fetched = v.get("fetched_at").and_then(Value::as_u64)?;
    Some((v.get("limits").cloned()?, fetched))
}

fn write_cache(limits: &Value, fetched: u64) {
    let Some(path) = cache_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        path,
        json!({"fetched_at": fetched, "limits": limits}).to_string(),
    );
}

/// Merge one streamed `rate_limit_event` into the cache (write-through):
/// `unifiedWindows` names every window with a 0–1 utilization fraction and
/// epoch-seconds reset, so one event refreshes the whole panel. Unknown
/// windows (model-scoped names drift) update in place by id only.
pub fn note_event(line: &Value) {
    let Some(windows) = line
        .get("rate_limit_info")
        .and_then(|i| i.get("unifiedWindows"))
        .and_then(Value::as_object)
    else {
        return;
    };
    let (mut limits, _) = read_cache().unwrap_or_else(|| {
        (
            json!({"available": true, "title": "Claude", "windows": []}),
            0,
        )
    });
    let arr = limits.pointer_mut("/windows").and_then(Value::as_array_mut);
    let Some(arr) = arr else { return };
    for (id, w) in windows {
        let pct = w
            .get("utilization")
            .and_then(Value::as_f64)
            .map(|u| u * 100.0);
        let reset = w.get("resetsAt").and_then(Value::as_i64);
        if let Some(existing) = arr
            .iter_mut()
            .find(|e| e.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            if let Some(p) = pct {
                existing["used_percent"] = json!(p);
            }
            if let Some(r) = reset {
                existing["resets_at"] = json!(iso_from_epoch(r));
            }
        } else if let Some(p) = pct {
            arr.push(json!({
                "id": id,
                "label": label_for(id),
                "kind": kind_for(id),
                "used_percent": p,
                "resets_at": reset.map(iso_from_epoch),
            }));
        }
    }
    write_cache(&limits, now_secs());
}

/// The probe: minimal `claude` child (no MCP, no tools) — the control
/// channel answers `get_usage` without spending tokens.
fn probe() -> Result<Value, ProviderRpcError> {
    let exe = crate::setup::resolve_command()
        .ok_or_else(|| ProviderRpcError::Unavailable(crate::setup::INSTALL_HINT.into()))?;
    let mut child = Command::new(exe)
        .args([
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--setting-sources",
            "",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| ProviderRpcError::Unavailable(format!("claude spawn: {e}")))?;
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let result = probe_inner(&mut child, deadline);
    let _ = child.kill();
    let _ = child.wait();
    result
}

fn probe_inner(child: &mut Child, deadline: Instant) -> Result<Value, ProviderRpcError> {
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ProviderRpcError::Unavailable("claude stdin closed".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ProviderRpcError::Unavailable("claude stdout closed".into()))?;
    // Reader thread: the pipe blocks on read; the deadline lives here.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                let is_reply = v.get("type").and_then(Value::as_str) == Some("control_response");
                if is_reply && tx.send(v).is_err() {
                    break;
                }
            }
        }
    });
    let req = json!({"type": "control_request", "request_id": "u1",
        "request": {"subtype": "get_usage"}});
    writeln!(stdin, "{req}")
        .and_then(|_| stdin.flush())
        .map_err(|_| ProviderRpcError::Unavailable("claude stdin closed".into()))?;
    let remaining = deadline.saturating_duration_since(Instant::now());
    let reply = rx
        .recv_timeout(remaining)
        .map_err(|_| ProviderRpcError::Unavailable("get_usage timed out".into()))?;
    let response = reply
        .get("response")
        .and_then(|r| r.get("response"))
        .cloned()
        .ok_or_else(|| ProviderRpcError::Protocol("invalid get_usage response".into()))?;
    Ok(map_usage(&response))
}

/// Map the `get_usage` payload into `ProviderUsageLimits` windows — the
/// t3code layer's id scheme so rows reconcile across sources.
fn map_usage(r: &Value) -> Value {
    let mut windows = Vec::new();
    let rl = r.get("rate_limits").cloned().unwrap_or(json!({}));
    let push = |windows: &mut Vec<Value>, id: &str, w: &Value| {
        let Some(u) = w.get("utilization").and_then(Value::as_f64) else {
            return;
        };
        windows.push(json!({
            "id": id,
            "label": label_for(id),
            "kind": kind_for(id),
            "duration_mins": match kind_for(id) {
                "session" => Some(SESSION_MINS),
                "weekly" => Some(WEEK_MINS),
                _ => None,
            },
            "used_percent": u,
            "used": w.get("used_dollars").cloned().unwrap_or(Value::Null),
            "limit": w.get("limit_dollars").cloned().unwrap_or(Value::Null),
            "unit": if w.get("limit_dollars").is_some() { Some("USD") } else { None },
            "resets_at": w.get("resets_at").cloned().unwrap_or(Value::Null),
        }));
    };
    for (id, w) in rl.as_object().into_iter().flatten() {
        if !w.is_object()
            || id == "extra_usage"
            || id == "limits"
            || id == "spend"
            || id == "seven_day_breakdown"
            || id == "model_scoped"
            || id == "weekly_scoped_shares"
            || id == "iguana_necktie"
        {
            // iguana_necktie is the monthly overage dollar bucket — this
            // panel shows the time-boxed windows, not billing ceilings.
            continue;
        }
        push(&mut windows, id, w);
    }
    for entry in rl
        .get("model_scoped")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = entry
            .get("display_name")
            .and_then(Value::as_str)
            .unwrap_or("model");
        let Some(u) = entry.get("utilization").and_then(Value::as_f64) else {
            continue;
        };
        windows.push(json!({
            "id": format!("seven_day_{}", name.to_lowercase().replace(|c: char| !c.is_ascii_alphanumeric(), "_")),
            "label": format!("Weekly · {name}"),
            "kind": "weekly",
            "duration_mins": WEEK_MINS,
            "used_percent": u,
            "resets_at": entry.get("resets_at").cloned().unwrap_or(Value::Null),
        }));
    }
    if let Some(eu) = rl.get("extra_usage")
        && eu.get("is_enabled").and_then(Value::as_bool) == Some(true)
    {
        windows.push(json!({
            "id": "extra_usage",
            "label": "Extra usage",
            "kind": "credits",
            "used_percent": eu.get("utilization").cloned().unwrap_or(Value::Null),
            "used": eu.get("used_credits").cloned().unwrap_or(Value::Null),
            "limit": eu.get("monthly_limit").cloned().unwrap_or(Value::Null),
            "unit": eu.get("currency").cloned().unwrap_or(Value::Null),
        }));
    }
    // Weekly breakdown rows (Claude Code 93% · Chats 2% …) become the note.
    let note = rl
        .get("seven_day_breakdown")
        .and_then(|b| b.get("rows"))
        .and_then(Value::as_array)
        .and_then(|rows| {
            let parts: Vec<String> = rows
                .iter()
                .filter_map(|r| {
                    let n = r.get("display_name").and_then(Value::as_str)?;
                    let p = r.get("percent").and_then(Value::as_f64)?;
                    (p > 0.0).then(|| format!("{n} {p:.0}%"))
                })
                .collect();
            (!parts.is_empty()).then(|| format!("weekly split: {}", parts.join(" · ")))
        });
    let mut limits = json!({
        "available": true,
        "title": "Claude",
        "windows": windows,
        "checked_at": iso_now(),
    });
    if let Some(plan) = r.get("subscription_type").and_then(Value::as_str) {
        limits["plan"] = json!(title_case(plan));
    }
    if let Some(n) = note {
        limits["note"] = json!(n);
    }
    limits
}

fn kind_for(id: &str) -> &'static str {
    match id {
        "five_hour" => "session",
        _ if id.starts_with("seven_day") => "weekly",
        _ => "monthly",
    }
}

fn label_for(id: &str) -> String {
    match id {
        "five_hour" => "Session".into(),
        "seven_day" => "Weekly".into(),
        _ if id.starts_with("seven_day_") => {
            format!("Weekly · {}", title_case(&id["seven_day_".len()..]))
        }
        "iguana_necktie" => "Monthly overage".into(),
        _ => title_case(id),
    }
}

fn title_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// RFC-3339 (UTC, second precision) without a chrono dep — civil-from-days
/// is ~20 lines and beats pulling a calendar crate for one string.
fn iso_from_epoch(secs: i64) -> String {
    let secs = secs.max(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        tod % 3600 / 60,
        tod % 60
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → y/m/d.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn iso_now() -> String {
    iso_from_epoch(now_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `get_usage` shape (trimmed): windows as utilization %,
    /// monthly dollar buckets, extra_usage, breakdown rows.
    #[test]
    fn maps_live_get_usage_payload() {
        let response = json!({
            "subscription_type": "pro",
            "rate_limits_available": true,
            "rate_limits": {
                "five_hour": {"utilization": 86.0, "resets_at": "2026-10-09T20:59:59Z",
                    "limit_dollars": null, "used_dollars": null},
                "seven_day": {"utilization": 85.0, "resets_at": "2026-10-12T16:59:59Z",
                    "limit_dollars": null, "used_dollars": null},
                "seven_day_opus": null,
                "iguana_necktie": {"utilization": 100.0, "resets_at": "2026-11-05T07:59:00Z",
                    "limit_dollars": 100, "used_dollars": 100.09},
                "extra_usage": {"is_enabled": false},
                "limits": [],
                "spend": {},
                "seven_day_breakdown": {"rows": [
                    {"display_name": "Claude Code", "percent": 93},
                    {"display_name": "Chats", "percent": 2}]},
                "model_scoped": []
            }
        });
        let limits = map_usage(&response);
        assert_eq!(limits["available"], true);
        assert_eq!(limits["plan"], "Pro");
        let windows = limits["windows"].as_array().unwrap();
        let ids: Vec<&str> = windows.iter().map(|w| w["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"five_hour"), "{ids:?}");
        assert!(ids.contains(&"seven_day"), "{ids:?}");
        assert!(!ids.contains(&"iguana_necktie"), "{ids:?}");
        // Null-valued and envelope keys are skipped.
        assert!(!ids.contains(&"seven_day_opus"), "{ids:?}");
        assert!(!ids.contains(&"spend"), "{ids:?}");
        assert!(!ids.contains(&"extra_usage"), "{ids:?}");
        let five = windows.iter().find(|w| w["id"] == "five_hour").unwrap();
        assert_eq!(five["kind"], "session");
        assert_eq!(five["duration_mins"], 300);
        assert!(limits["note"].as_str().unwrap().contains("Claude Code 93%"));
    }

    /// `extra_usage` only draws a row when the account has it enabled.
    #[test]
    fn extra_usage_enabled_draws_row() {
        let response = json!({
            "rate_limits": {
                "extra_usage": {"is_enabled": true, "monthly_limit": 50,
                    "used_credits": 12.5, "utilization": 25.0, "currency": "USD"}
            }
        });
        let limits = map_usage(&response);
        let windows = limits["windows"].as_array().unwrap();
        let eu = windows.iter().find(|w| w["id"] == "extra_usage").unwrap();
        assert_eq!(eu["kind"], "credits");
        assert_eq!(eu["used"], 12.5);
    }

    #[test]
    fn epoch_to_iso() {
        assert_eq!(iso_from_epoch(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_from_epoch(1791565891), "2026-10-09T17:11:31Z");
    }
}
