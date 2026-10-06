//! Prompt-cache keepalive: re-warm a resumed session before its upstream
//! cache entry lapses.
//!
//! Every host turn spawns a fresh `claude` child that resumes the recorded
//! session (see [`crate::chat::spawn_turn`]). The resumed prefix rides the
//! Anthropic prompt cache, which lives about an hour; a turn that lands
//! after it lapses pays a full cache write of the whole history. When a
//! turn leaves a usable resume point, [`note_turn`] remembers it here, and
//! once the key has seen no upstream contact for [`REFRESH_AFTER`] a
//! background probe re-runs the same request — same model, system prompt,
//! effort and tool manifest, or the prefix wouldn't match the cache entry
//! — resumed at the same point with a throwaway prompt. The upstream read
//! renews the TTL.
//!
//! The probe's exchange appends under the resume point in the native
//! session file. That is safe: `--resume-session-at` truncates the loaded
//! transcript at the stored uuid, so the probe branch never reaches a real
//! turn's context (branching under an interior uuid is exactly what the
//! next real turn, and the rewind feature, do anyway). The resume point
//! itself is never rewritten — the probe does not call `session::save`.
//!
//! Bounds: a key is warmed only while its last real turn is younger than
//! [`WARM_WINDOW`] (each probe costs one cache-read of the full prefix plus
//! a spawn), the sweeper is a single thread that claims a key under the
//! lock before spawning so a key never has two probes in flight, and
//! [`MAX_FAILURES`] consecutive failures — or a refused resume, which means
//! the point is dead — stop the warming. A probe counts as upstream
//! contact win or lose, so a persistent failure retries after
//! [`REFRESH_AFTER`], not every poll.
//!
//! Probes go straight to `claude`: they never touch the relay or an
//! intent, so nothing reaches the host — no turn, no usage. Under
//! `CLAUDE_SUB_DEBUG` their spawns dump like any other, marked
//! `{"keepalive": true}` in the sent frames.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::chat::{self, PreparedTurn};
use crate::session::{self, ResumePoint};

/// Upstream contact this idle gets re-warmed: under the ~1h cache TTL,
/// far enough that [`POLL`] jitter can't overshoot the TTL.
pub const REFRESH_AFTER: Duration = Duration::from_secs(50 * 60);
/// Stop warming a conversation this long after its last real turn.
pub const WARM_WINDOW: Duration = Duration::from_secs(6 * 3600);
/// Sweeper cadence.
const POLL: Duration = Duration::from_secs(60);
/// A probe is a cache-read plus a short completion; don't hold a child
/// longer than this. Best-effort: a timeout just skips the cycle.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
/// Consecutive probe failures before a key stops being warmed.
const MAX_FAILURES: u32 = 3;
/// The throwaway prompt: minimal, and its branch is never resumed.
const PROMPT: &str = ".";

/// Everything a probe needs to rebuild the real turn's request. Model,
/// system prompt, effort and tool manifest are all part of the cached
/// prefix: any difference and the probe warms a new entry, not the one
/// the next real turn will hit.
#[derive(Clone)]
struct Probe {
    model: String,
    system: String,
    extra: Value,
    effort: Option<String>,
}

/// A conversation worth warming: where to resume it, what to resend, and
/// the two clocks that bound the warming.
struct Warm {
    point: ResumePoint,
    probe: Probe,
    /// The cwd the session file lives under (`projects/<cwd-derived>`):
    /// the probe is only meaningful while the sidecar still runs there.
    cwd: PathBuf,
    /// Last host turn — the idle clock [`WARM_WINDOW`] bounds.
    last_real: Instant,
    /// Last upstream request, real turn or probe — the [`REFRESH_AFTER`]
    /// clock. A probe resets it: the cache entry is renewed either way.
    last_contact: Instant,
    /// Consecutive failed probes.
    failures: u32,
}

fn registry() -> &'static Mutex<HashMap<u64, Warm>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Warm>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A real turn finished and left a resume point: (re)start warming the
/// conversation. Called from [`crate::chat::spawn_turn`] right after the
/// point is saved; the key is the same one the next turn looks up.
pub fn note_turn(
    key: u64,
    point: ResumePoint,
    turn: &PreparedTurn,
    system: &str,
    extra: &Value,
    effort: Option<&str>,
) {
    let now = Instant::now();
    let warm = Warm {
        point,
        probe: Probe {
            model: turn.native_model.clone(),
            system: system.to_string(),
            extra: extra.clone(),
            effort: effort.map(str::to_string),
        },
        cwd: std::env::current_dir().unwrap_or_default(),
        last_real: now,
        last_contact: now,
        failures: 0,
    };
    let _ = registry().lock().map(|mut m| m.insert(key, warm));
}

/// Whether a key wants a probe now: upstream contact old enough to need
/// one while the real turn is still inside the warm window.
fn needs_refresh(contact_idle: Duration, real_idle: Duration) -> bool {
    contact_idle >= REFRESH_AFTER && real_idle < WARM_WINDOW
}

/// One poll's decision: drop entries that aged out of the window (or
/// whose session's cwd no longer matches), return the keys now due.
fn plan(map: &mut HashMap<u64, Warm>, now: Instant, cwd: &Path) -> Vec<u64> {
    map.retain(|_, w| w.cwd == cwd && now.duration_since(w.last_real) < WARM_WINDOW);
    let mut due: Vec<u64> = map
        .iter()
        .filter(|(_, w)| {
            needs_refresh(
                now.duration_since(w.last_contact),
                now.duration_since(w.last_real),
            )
        })
        .map(|(k, _)| *k)
        .collect();
    due.sort_unstable();
    due
}

/// The throwaway prompt as one querying user frame (the only stdin frame,
/// so `run_native` does not mark it `shouldQuery: false`).
fn probe_frames() -> Vec<Value> {
    vec![json!({"type": "user", "message": {"role": "user",
        "content": [{"type": "text", "text": PROMPT}]}})]
}

/// Claim key's refresh: mark upstream contact under the lock and hand out
/// what the probe needs. `None` when the entry went away or stopped being
/// due — a real turn that landed since the scan resets the idle clock.
fn claim(key: u64) -> Option<(ResumePoint, Probe)> {
    let mut m = registry().lock().ok()?;
    let w = m.get_mut(&key)?;
    if !needs_refresh(w.last_contact.elapsed(), w.last_real.elapsed()) {
        return None;
    }
    w.last_contact = Instant::now();
    Some((w.point.clone(), w.probe.clone()))
}

/// Fold a finished probe back into the entry: success clears the failure
/// streak, a refused resume drops the entry (the point is dead), anything
/// else counts toward [`MAX_FAILURES`].
fn finish(key: u64, err: Option<&str>) {
    let Ok(mut m) = registry().lock() else {
        return;
    };
    let dead = match err {
        None => {
            if let Some(w) = m.get_mut(&key) {
                w.failures = 0;
            }
            false
        }
        Some(e) if chat::replay_after_failed_resume(e) => true,
        Some(_) => match m.get_mut(&key) {
            Some(w) => {
                w.failures += 1;
                w.failures >= MAX_FAILURES
            }
            None => false,
        },
    };
    if dead {
        m.remove(&key);
    }
}

/// One probe: resume the stored session at the stored point and re-issue
/// the real turn's request shape with the throwaway prompt. The single
/// sweeper thread runs this inline, so keys refresh serially and never
/// concurrently with themselves.
fn refresh(key: u64) {
    let Some((point, probe)) = claim(key) else {
        return;
    };
    let turn = PreparedTurn {
        system: probe.system.clone(),
        frames: probe_frames(),
        names: Vec::new(),
        native_model: probe.model.clone(),
    };
    match chat::run_native(
        &turn,
        &probe.extra,
        &probe.system,
        probe.effort.as_deref(),
        PROBE_TIMEOUT,
        &turn.frames,
        Some(&point),
        true,
    ) {
        Ok(_) => {
            // If the probe ended on the tool boundary, native recorded its
            // own error results; drop them like a real turn would.
            session::prune_native_tool_results(&point.0);
            finish(key, None);
        }
        Err(e) => finish(key, Some(&e)),
    }
}

fn run() {
    loop {
        std::thread::sleep(POLL);
        let cwd = std::env::current_dir().unwrap_or_default();
        let due = registry()
            .lock()
            .map(|mut m| plan(&mut m, Instant::now(), &cwd))
            .unwrap_or_default();
        for key in due {
            refresh(key);
        }
    }
}

/// Start the background sweeper. Idempotent; the thread lives for the
/// sidecar's lifetime and dies with it.
pub fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("claude-sub-keepalive".into())
            .spawn(run);
    });
}

#[path = "keepalive_tests.rs"]
#[cfg(test)]
mod tests;
