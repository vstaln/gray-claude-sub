use super::*;

fn model(id: &str, window: u32) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        window: Some(window),
        efforts: Vec::new(),
    }
}

/// A snapshot as a fresh page parse would produce it, in page order: a
/// current haiku newer than any this repo has seen, then older entries.
fn live() -> Snapshot {
    Snapshot::new(vec![
        model("claude-haiku-9-9", 200_000),
        model("claude-sonnet-5-5", 1_000_000),
        model("claude-sonnet-5", 1_000_000),
        model("claude-haiku-4-5-20251001", 200_000),
    ])
}

/// A scratch directory unique to one test, so parallel tests never share
/// files. Removed before use, so it starts empty.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("claude-sub-catalog-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn aliases_resolve_to_the_first_listed_match() {
    let s = live();
    // A family alias tracks the first listed id of that family.
    assert_eq!(native_model_in("haiku", &s).unwrap(), "claude-haiku-9-9");
    assert_eq!(
        native_model_in("sonnet", &s).unwrap(),
        "claude-sonnet-5-5[1m]"
    );
    // A dated prefix resolves to the first matching id.
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
fn dated_ids_inherit_their_stem_window() {
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
    // A non-date suffix never resolves — typos stay unresolved, not guessed.
    assert_eq!(
        context_window_in("claude-sonnet-5-5-beta", &s),
        Some(200_000)
    );
}

#[test]
fn a_200k_window_rejects_the_1m_route() {
    // The page's window decides the route, whatever the family is called.
    let s = live();
    assert!(native_model_in("haiku[1m]", &s).is_err());
    assert!(native_model_in("claude-haiku-9-9[1m]", &s).is_err());
}

#[test]
fn windows_come_from_the_page() {
    let s = live();
    assert_eq!(context_window_in("claude-sonnet-5-5", &s), Some(1_000_000));
    assert_eq!(context_window_in("claude-haiku-9-9", &s), Some(200_000));
}

#[test]
fn omitted_window_gets_no_1m_route() {
    // An entry the page gave no window for is not guessed up to 1M: no [1m]
    // route is offered, and an explicit [1m] request passes through to native.
    let mut s = live();
    s.models
        .iter_mut()
        .find(|m| m.id == "claude-haiku-4-5-20251001")
        .unwrap()
        .window = None;
    let id = "claude-haiku-4-5-20251001";
    let long = format!("{id}[1m]");
    assert_eq!(context_window_in(id, &s), Some(200_000));
    assert_eq!(context_window_in(&long, &s), None);
    assert!(!all_ids_in(&s).contains(&long));
    assert_eq!(native_model_in(&long, &s).unwrap(), long);
}

#[test]
fn display_names_are_derived_from_the_id() {
    let s = live();
    assert_eq!(
        display_name_in("claude-sonnet-5-5", &s),
        "Claude Sonnet 5.5"
    );
    assert_eq!(display_name_in("sonnet", &s), "Claude Sonnet 5.5");
    assert_eq!(
        display_name_in("claude-sonnet-5-5[1m]", &s),
        "Claude Sonnet 5.5 1M"
    );
    assert_eq!(display_name_in("claude-haiku-9-9", &s), "Claude Haiku 9.9");
}

#[test]
fn display_name_restores_version_dots() {
    // Names come from the id alone: no snapshot needed.
    let s = Snapshot::new(Vec::new());
    assert_eq!(display_name_in("claude-opus-5-5", &s), "Claude Opus 5.5");
    assert_eq!(
        display_name_in("claude-opus-5-5[1m]", &s),
        "Claude Opus 5.5 1M"
    );
    assert_eq!(
        display_name_in("claude-haiku-4-5-20251001", &s),
        "Claude Haiku 4.5 20251001"
    );
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
}

#[test]
fn empty_catalog_invents_no_routes() {
    // Nothing parsed or cached: no alias, family or [1m] route is
    // synthesized. Explicit input passes through; a plain id keeps native's
    // 200K default; a [1m] route has no known window.
    let s = Snapshot::new(Vec::new());
    assert!(all_ids_in(&s).is_empty());
    assert_eq!(native_model_in("opus", &s).unwrap(), "opus");
    assert_eq!(context_window_in("opus", &s), Some(200_000));
    assert_eq!(context_window_in("opus[1m]", &s), None);
    assert_eq!(native_model_in("opus[1m]", &s).unwrap(), "opus[1m]");
}

#[test]
fn cache_roundtrip_keeps_every_field_and_order() {
    let dir = scratch("roundtrip");
    // The parent directory is created on write; a full and a sparse entry
    // both survive the trip, in the order they were written.
    let path = dir.join("nested").join(CACHE_FILE);
    let models = vec![
        ModelInfo {
            id: "claude-sonnet-5-5".into(),
            window: Some(1_000_000),
            efforts: vec!["low".into(), "high".into()],
        },
        ModelInfo {
            id: "claude-haiku-9-9".into(),
            window: None,
            efforts: Vec::new(),
        },
    ];
    write_cache_at(&path, &models).unwrap();
    assert_eq!(
        load_cache_at(&path).expect("cache reads back").models,
        models
    );
    assert_eq!(initial_snapshot_from(Some(path.as_path())).models, models);
    // The rename consumed the temp file: the directory holds only the cache.
    assert_eq!(
        std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
        1
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_cache_means_an_empty_catalog() {
    let dir = scratch("empty");
    // No gray home at all, and no file yet: both start empty.
    assert!(initial_snapshot_from(None).models.is_empty());
    let path = dir.join(CACHE_FILE);
    assert!(
        initial_snapshot_from(Some(path.as_path()))
            .models
            .is_empty()
    );
    // A corrupt file, or one holding no routes, is no cache either.
    std::fs::create_dir_all(&dir).unwrap();
    for junk in ["{not json", "{}", r#"{"models":[{"id":""}]}"#] {
        std::fs::write(&path, junk).unwrap();
        assert!(
            initial_snapshot_from(Some(path.as_path()))
                .models
                .is_empty(),
            "{junk}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_fetch_without_cache_says_why() {
    let got = crate::models::listing(settle_failed_fetch(
        Arc::new(Snapshot::new(Vec::new())),
        "models page HTTP 503 Service Unavailable".into(),
    ));
    let err = got.err().expect("nothing cached: the listing must fail");
    assert_eq!(
        err,
        "couldn't list Claude models: models page HTTP 503 Service Unavailable"
    );
}

#[test]
fn failed_fetch_with_cache_serves_the_cache() {
    let got = crate::models::listing(settle_failed_fetch(
        Arc::new(live()),
        "connection refused".into(),
    ))
    .expect("cached routes still list");
    assert!(got.models.iter().any(|m| m.id == "sonnet"));
}
