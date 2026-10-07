use super::*;

const MIN: Duration = Duration::from_secs(60);

#[test]
fn refresh_window_is_between_refresh_after_and_warm_window() {
    let now = Instant::now();
    let used = now - MIN;
    // Contact younger than REFRESH_AFTER never refreshes.
    assert!(!refresh_due(
        now,
        used,
        Some(now - (REFRESH_AFTER - MIN)),
        false,
        0,
        true,
        false
    ));
    assert!(refresh_due(
        now,
        used,
        Some(now - REFRESH_AFTER),
        false,
        0,
        true,
        false
    ));
    // A probe refreshes contact but never the real-turn clock.
    assert!(refresh_due(
        now,
        now - (WARM_WINDOW - MIN),
        Some(now - REFRESH_AFTER),
        false,
        0,
        true,
        false
    ));
    // Past the window the entry isn't worth warming.
    assert!(!refresh_due(
        now,
        now - WARM_WINDOW,
        Some(now - REFRESH_AFTER),
        false,
        0,
        true,
        false
    ));
    // No contact yet (a session that never completed a turn) is not due.
    assert!(!refresh_due(now, used, None, false, 0, true, false));
}

#[test]
fn refresh_skips_dead_limited_parked_and_spent() {
    let now = Instant::now();
    let due = |rate_limited, failures, alive, parked_open| {
        refresh_due(
            now,
            now - MIN,
            Some(now - REFRESH_AFTER),
            rate_limited,
            failures,
            alive,
            parked_open,
        )
    };
    assert!(due(false, 0, true, false));
    assert!(!due(true, 0, true, false)); // rate-limited: the probe burns the same limit
    assert!(!due(false, 0, false, false)); // dead child
    assert!(!due(false, 0, true, true)); // parked on a host tool call
    assert!(!due(false, MAX_FAILURES, true, false)); // budget spent
    assert!(due(false, MAX_FAILURES - 1, true, false));
}

#[test]
fn reap_kills_dead_spent_and_cold_idle_sessions() {
    let now = Instant::now();
    assert!(reapable(now, now - MIN, false, 0, true)); // dead wire
    assert!(reapable(now, now - MIN, false, MAX_FAILURES, false)); // budget spent
    // Past TTL and past the warm window: nothing left to warm.
    assert!(reapable(now, now - WARM_WINDOW, false, 0, false));
    // Past TTL and rate-limited: the limit made it unwarmable.
    assert!(reapable(now, now - (IDLE_TTL + MIN), true, 0, false));
    // Inside the warm window a live session survives past IDLE_TTL —
    // its refresh probes are what keep the cache entry it lives for.
    assert!(!reapable(now, now - (IDLE_TTL + MIN), false, 0, false));
    assert!(!reapable(now, now - MIN, false, 0, false));
}

#[test]
fn probe_prompt_is_one_tiny_user_message() {
    assert_eq!(PROMPT, ".");
}
