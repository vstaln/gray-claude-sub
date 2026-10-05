use super::*;

#[test]
fn pinned_routes_get_windows_and_1m_selection() {
    assert_eq!(native_model("sonnet").unwrap(), "claude-sonnet-5[1m]");
    assert_eq!(native_model("opus").unwrap(), "claude-opus-5-5[1m]");
    assert_eq!(native_model("haiku").unwrap(), "claude-haiku-4-5-20251001");
    assert_eq!(context_window("sonnet"), Some(1_000_000));
    assert_eq!(context_window("haiku"), Some(200_000));
    assert_eq!(context_window("some-future-model"), Some(200_000));
    assert_eq!(context_window("some-future-model[1m]"), None);
}

#[test]
fn haiku_rejects_1m() {
    assert!(native_model("haiku[1m]").is_err());
}

#[test]
fn display_name_restores_version_dots() {
    assert_eq!(display_name("opus"), "Claude Opus 5.5");
    assert_eq!(display_name("claude-opus-5-5[1m]"), "Claude Opus 5.5 1M");
    assert_eq!(display_name("haiku"), "Claude Haiku 4.5 20251001");
    assert_eq!(display_name("claude-sonnet-5"), "Claude Sonnet 5");
    assert_eq!(display_name("claude-fable-5-1"), "Claude Fable 5.1");
}

#[test]
fn catalog_covers_aliases_and_bare_ids() {
    let ids = all_ids();
    for want in ["sonnet", "opus", "haiku", "fable", "claude-sonnet-5[1m]"] {
        assert!(ids.contains(&want.to_string()), "missing {want}: {ids:?}");
    }
}
