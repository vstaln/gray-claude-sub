use super::*;

fn model(id: &str, line: &str, window: u32, created: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        display_name: None,
        created_at: created.to_string(),
        line: Some(line.to_string()),
        max_input_tokens: Some(window),
        efforts: None,
    }
}

/// A snapshot as a fresh `/v1/models` fetch would produce it: a newer
/// haiku the pinned table doesn't know.
fn live() -> Snapshot {
    Snapshot::new(vec![
        model("claude-haiku-9-9", "haiku", 200_000, "2027-01-01"),
        model("claude-sonnet-5-5", "sonnet", 1_000_000, "2026-09-28"),
        model("claude-sonnet-5", "sonnet", 1_000_000, "2026-06-29"),
        model("claude-haiku-4-5-20251001", "haiku", 200_000, "2025-10-15"),
    ])
}

#[test]
fn fallback_routes_match_pinned_behavior() {
    let s = fallback_snapshot();
    assert_eq!(
        native_model_in("sonnet", &s).unwrap(),
        "claude-sonnet-5[1m]"
    );
    assert_eq!(native_model_in("opus", &s).unwrap(), "claude-opus-5-5[1m]");
    assert_eq!(
        native_model_in("haiku", &s).unwrap(),
        "claude-haiku-4-5-20251001"
    );
    assert_eq!(context_window_in("sonnet", &s), Some(1_000_000));
    assert_eq!(context_window_in("haiku", &s), Some(200_000));
    assert_eq!(context_window_in("some-future-model", &s), Some(200_000));
    assert_eq!(context_window_in("some-future-model[1m]", &s), None);
}

#[test]
fn live_aliases_resolve_to_newest_fetched() {
    let s = live();
    // The family alias tracks the newest fetched id, not a pinned one.
    assert_eq!(native_model_in("haiku", &s).unwrap(), "claude-haiku-9-9");
    assert_eq!(
        native_model_in("sonnet", &s).unwrap(),
        "claude-sonnet-5-5[1m]"
    );
    // A dated prefix resolves to the newest matching id.
    assert_eq!(
        canonical_in("claude-haiku-4-5", &s),
        "claude-haiku-4-5-20251001"
    );
    assert_eq!(canonical_in("claude-sonnet", &s), "claude-sonnet-5-5");
    // Unknown ids pass through untouched.
    assert_eq!(
        native_model_in("some-future-model", &s).unwrap(),
        "some-future-model"
    );
}

#[test]
fn dated_ids_inherit_their_family_window() {
    let s = live();
    // A dated variant of a known undated id resolves to the stem: the
    // session meter and `--model` route agree on 1M, not the 200K default.
    assert_eq!(
        canonical_in("claude-sonnet-5-5-20260928", &s),
        "claude-sonnet-5-5"
    );
    assert_eq!(
        context_window_in("claude-sonnet-5-5-20260928", &s),
        Some(1_000_000)
    );
    assert_eq!(
        native_model_in("claude-sonnet-5-5-20260928", &s).unwrap(),
        "claude-sonnet-5-5[1m]"
    );
    // A non-date suffix never resolves — typos stay unpinned, not guessed.
    assert_eq!(
        context_window_in("claude-sonnet-5-5-beta", &s),
        Some(200_000)
    );
}

#[test]
fn haiku_rejects_1m() {
    let s = live();
    assert!(native_model_in("haiku[1m]", &s).is_err());
    assert!(native_model_in("claude-haiku-9-9[1m]", &s).is_err());
}

#[test]
fn live_windows_come_from_max_input_tokens() {
    let s = live();
    assert_eq!(context_window_in("claude-sonnet-5-5", &s), Some(1_000_000));
    assert_eq!(context_window_in("claude-haiku-9-9", &s), Some(200_000));
    // Fetched entry omitting the window falls back to the pin.
    let mut s2 = live();
    s2.models
        .iter_mut()
        .find(|m| m.id == "claude-haiku-4-5-20251001")
        .unwrap()
        .max_input_tokens = None;
    assert_eq!(
        context_window_in("claude-haiku-4-5-20251001", &s2),
        Some(200_000)
    );
}

#[test]
fn api_display_name_wins_derived() {
    let mut s = live();
    s.models
        .iter_mut()
        .find(|m| m.id == "claude-sonnet-5-5")
        .unwrap()
        .display_name = Some("Claude Sonnet 5.5".into());
    assert_eq!(
        display_name_in("claude-sonnet-5-5", &s),
        "Claude Sonnet 5.5"
    );
    assert_eq!(display_name_in("sonnet", &s), "Claude Sonnet 5.5");
    assert_eq!(
        display_name_in("claude-sonnet-5-5[1m]", &s),
        "Claude Sonnet 5.5 1M"
    );
    // No display_name → derived.
    assert_eq!(display_name_in("claude-haiku-9-9", &s), "Claude Haiku 9.9");
}

#[test]
fn display_name_restores_version_dots() {
    let s = fallback_snapshot();
    assert_eq!(display_name_in("opus", &s), "Claude Opus 5.5");
    assert_eq!(
        display_name_in("claude-opus-5-5[1m]", &s),
        "Claude Opus 5.5 1M"
    );
    assert_eq!(display_name_in("haiku", &s), "Claude Haiku 4.5 20251001");
    assert_eq!(display_name_in("claude-sonnet-5", &s), "Claude Sonnet 5");
    assert_eq!(display_name_in("claude-fable-5-1", &s), "Claude Fable 5.1");
}

#[test]
fn catalog_covers_aliases_and_bare_ids() {
    let ids = all_ids_in(&live());
    for want in [
        "sonnet",
        "haiku",
        "claude-haiku-9-9",
        "claude-sonnet-5-5[1m]",
    ] {
        assert!(ids.contains(&want.to_string()), "missing {want}: {ids:?}");
    }
    // 200K routes don't get a [1m] variant.
    assert!(!ids.contains(&"claude-haiku-9-9[1m]".to_string()));
    let pinned = all_ids_in(&fallback_snapshot());
    for want in ["sonnet", "opus", "haiku", "fable", "claude-sonnet-5[1m]"] {
        assert!(pinned.contains(&want.to_string()), "missing {want}");
    }
}
