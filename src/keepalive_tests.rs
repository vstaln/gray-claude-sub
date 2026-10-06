use super::*;

const MIN: Duration = Duration::from_secs(60);

fn probe() -> Probe {
    Probe {
        model: "claude-sonnet-5".into(),
        system: "sys".into(),
        extra: json!({}),
        effort: None,
    }
}

fn warm(real_ago: Duration, contact_ago: Duration, cwd: &Path) -> Warm {
    let now = Instant::now();
    Warm {
        point: (
            "6b3513ca-34f8-4571-a982-b90a863b3b02".into(),
            "47d3d750-7c40-421c-b6e3-308ff8ceec2e".into(),
        ),
        probe: probe(),
        cwd: cwd.to_path_buf(),
        last_real: now - real_ago,
        last_contact: now - contact_ago,
        failures: 0,
    }
}

#[test]
fn refresh_window_is_between_refresh_after_and_warm_window() {
    assert!(!needs_refresh(REFRESH_AFTER - MIN, MIN));
    assert!(needs_refresh(REFRESH_AFTER, MIN));
    // A probe refreshes contact but never the real-turn clock.
    assert!(needs_refresh(REFRESH_AFTER, WARM_WINDOW - MIN));
    assert!(!needs_refresh(REFRESH_AFTER, WARM_WINDOW));
    assert!(!needs_refresh(2 * WARM_WINDOW, 2 * WARM_WINDOW));
}

#[test]
fn plan_drops_dead_and_foreign_keys_and_returns_due_sorted() {
    let cwd = PathBuf::from("/w");
    let now = Instant::now();
    let mut map = HashMap::new();
    map.insert(5, warm(MIN, MIN, &cwd)); // fresh: not due
    map.insert(2, warm(MIN, REFRESH_AFTER + MIN, &cwd)); // due
    map.insert(9, warm(MIN, REFRESH_AFTER + 2 * MIN, &cwd)); // due
    map.insert(7, warm(2 * WARM_WINDOW, MIN, &cwd)); // dead: real turn too old
    let mut foreign = warm(MIN, REFRESH_AFTER + MIN, &cwd);
    foreign.cwd = PathBuf::from("/elsewhere");
    map.insert(4, foreign); // dead: session lives under another cwd

    assert_eq!(plan(&mut map, now, &cwd), vec![2, 9]);
    let mut keys: Vec<u64> = map.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, vec![2, 5, 9]);
}

#[test]
fn claim_is_single_shot_and_rechecks_under_lock() {
    let cwd = std::env::current_dir().unwrap();
    // Unique keys: the registry is shared across tests in this binary.
    let (due, fresh) = (u64::MAX - 1, u64::MAX - 2);
    {
        let mut m = registry().lock().unwrap();
        m.insert(due, warm(MIN, REFRESH_AFTER + MIN, &cwd));
        m.insert(fresh, warm(MIN, MIN, &cwd));
    }
    // A due key yields its point and probe exactly once: the claim itself
    // is the upstream contact, so a second claim right after is not due.
    let (point, p) = claim(due).unwrap();
    assert!(session::is_uuid(&point.0) && session::is_uuid(&point.1));
    assert_eq!(p.model, "claude-sonnet-5");
    assert!(claim(due).is_none());
    // A fresh key never claims; a missing key neither.
    assert!(claim(fresh).is_none());
    assert!(claim(u64::MAX).is_none());
    let mut m = registry().lock().unwrap();
    m.remove(&due);
    m.remove(&fresh);
}

#[test]
fn finish_counts_failures_and_drops_dead_points() {
    let cwd = std::env::current_dir().unwrap();
    let key = u64::MAX - 3;
    registry()
        .lock()
        .unwrap()
        .insert(key, warm(MIN, REFRESH_AFTER + MIN, &cwd));

    // A refused resume means the point is dead: dropped outright.
    finish(
        key,
        Some("incomplete native response: assistant and one result required"),
    );
    assert!(!registry().lock().unwrap().contains_key(&key));

    registry()
        .lock()
        .unwrap()
        .insert(key, warm(MIN, REFRESH_AFTER + MIN, &cwd));
    for i in 1..MAX_FAILURES {
        finish(key, Some("native request failed"));
        assert_eq!(registry().lock().unwrap()[&key].failures, i);
    }
    finish(key, Some("native request failed"));
    assert!(!registry().lock().unwrap().contains_key(&key));

    // Success clears the streak.
    registry()
        .lock()
        .unwrap()
        .insert(key, warm(MIN, REFRESH_AFTER + MIN, &cwd));
    finish(key, Some("native request failed"));
    finish(key, None);
    assert_eq!(registry().lock().unwrap()[&key].failures, 0);
    registry().lock().unwrap().remove(&key);
}

#[test]
fn probe_frame_is_one_querying_user_message() {
    let frames = probe_frames();
    assert_eq!(frames.len(), 1);
    let f = &frames[0];
    assert_eq!(f["type"], "user");
    assert_eq!(f["message"]["role"], "user");
    assert_eq!(f["message"]["content"][0]["type"], "text");
    // Single-frame stdin never carries shouldQuery: it is the query.
    assert!(f.get("shouldQuery").is_none());
}
