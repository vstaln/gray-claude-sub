//! Prompt-cache keepalive for the live-session pool, plus the idle
//! reaper that bounds it.
//!
//! A live `claude` child ([`crate::live`]) keeps its own prompt cache
//! warm by existing — until it doesn't: Anthropic's cache entry lapses
//! about an hour after the last upstream contact, and a turn landing
//! after that pays a full cache write of the whole conversation. So
//! every [`REFRESH_AFTER`] of upstream silence the sweeper writes a
//! tiny [`PROMPT`] user frame into the session's own stdin and drains
//! it to its `result`: one cheap cache-read, the TTL renews, the
//! session never leaves the pool for a probe and no throwaway child or
//! `--fork-session` branch exists to sweep afterwards.
//!
//! A refresh never touches the matching state: the probe doesn't move
//! `last_used` (the idle clock [`WARM_WINDOW`] and the reaper read), so
//! warming can't make a dead conversation look alive — it only bumps
//! `last_contact`, the [`REFRESH_AFTER`] clock. Skipped: rate-limited
//! sessions (the probe would burn its own limit), sessions parked on a
//! host tool call (mid-conversation — they owe answers, not probes),
//! dead children, and anything checked out mid-turn (the pool hands a
//! due session out of the pool while it refreshes, so a matching turn
//! waits on it rather than re-billing the transcript on a fresh child).
//!
//! The same sweep reaps: a session idle past [`IDLE_TTL`] whose cache
//! is no longer worth warming, a dead child or a dead `gray` bridge,
//! and a session whose failure streak hit [`MAX_FAILURES`]. A probe
//! counts as upstream contact win or lose, so a persistent failure
//! retries after [`REFRESH_AFTER`], not every [`POLL`].

use std::time::{Duration, Instant};

/// Upstream contact this idle gets re-warmed: under the ~1h cache TTL,
/// far enough that [`POLL`] jitter can't overshoot the TTL.
pub const REFRESH_AFTER: Duration = Duration::from_secs(50 * 60);
/// Stop warming a conversation this long after its last real turn.
pub const WARM_WINDOW: Duration = Duration::from_secs(6 * 3600);
/// A session idle past this is swept — unless still warmable, i.e. its
/// cache entry may yet be refreshed inside the warm window.
pub(crate) const IDLE_TTL: Duration = Duration::from_secs(30 * 60);
/// Sweeper cadence.
const POLL: Duration = Duration::from_secs(60);
/// A probe is a cache-read plus a short completion; don't hold the
/// session longer than this. Best-effort: a timeout kills the session —
/// a prompt in flight past the deadline is ambiguous state.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
/// Consecutive probe failures before a session stops being warmed.
pub(crate) const MAX_FAILURES: u32 = 3;
/// The probe prompt: one character — the cheapest cache-touching query.
pub(crate) const PROMPT: &str = ".";

/// Whether a session's cache entry is still worth renewing: inside the
/// warm window since its last real turn, and not sitting out a rate
/// limit (the probe would be rejected anyway).
pub(crate) fn warm_eligible(now: Instant, last_used: Instant, rate_limited: bool) -> bool {
    !rate_limited && now.duration_since(last_used) < WARM_WINDOW
}

/// Whether a live session wants a refresh probe now: contact old enough
/// to need one while the conversation is still inside the warm window,
/// the session alive, below the failure cap and not parked on host tool
/// calls (a parked front query is mid-conversation — it owes answers,
/// not probes).
pub(crate) fn refresh_due(
    now: Instant,
    last_used: Instant,
    last_contact: Option<Instant>,
    rate_limited: bool,
    failures: u32,
    alive: bool,
    parked_open: bool,
) -> bool {
    alive
        && !parked_open
        && failures < MAX_FAILURES
        && warm_eligible(now, last_used, rate_limited)
        && last_contact.is_some_and(|c| now.duration_since(c) >= REFRESH_AFTER)
}

/// The sweeper's kill rule: a dead wire (child exited or `gray` bridge
/// gone), a spent failure budget, or idleness past [`IDLE_TTL`] with
/// nothing left to warm — a warmable session survives the TTL because
/// its refresh probes keep the cache entry it was kept alive for.
pub(crate) fn reapable(
    now: Instant,
    last_used: Instant,
    rate_limited: bool,
    failures: u32,
    dead: bool,
) -> bool {
    dead || failures >= MAX_FAILURES || {
        now.duration_since(last_used) > IDLE_TTL && !warm_eligible(now, last_used, rate_limited)
    }
}

/// The sweeper: reap first, then refresh every due session serially —
/// a refresh checks the session out of the pool so a matching turn
/// waits on it rather than spawning cold.
fn run() {
    loop {
        std::thread::sleep(POLL);
        crate::live::reap_idle();
        while let Some(mut s) = crate::live::take_refresh_candidate() {
            let result = s.refresh(Instant::now() + PROBE_TIMEOUT);
            crate::live::finish_refresh(s, result.err().as_deref());
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
