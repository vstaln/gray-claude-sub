use super::*;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("claude-sub-test-{name}-{}", new_uuid()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn age(path: &Path, secs: u64) {
    let f = std::fs::File::options().write(true).open(path).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(secs))
        .unwrap();
}

#[test]
fn uuids_are_canonical_and_distinct() {
    let a = new_uuid();
    let b = new_uuid();
    assert!(is_uuid(&a), "{a}");
    assert_ne!(a, b);
    assert_eq!(&a[14..15], "4", "version nibble: {a}");
    assert!("89ab".contains(&a[19..20]), "variant nibble: {a}");
    assert!(is_uuid("6b3513ca-34f8-4571-a982-b90a863b3b02"));
    assert!(!is_uuid("6b3513ca-34f8-4571-a982-b90a863b3b0"));
    assert!(!is_uuid("--resume-session-at-xxxxxxxxxxxxxxxx"));
    assert!(!is_uuid("../../../../etc/passwd/aaaaaaaaaaaaaa"));
    assert!(!is_uuid("6b3513ca_34f8_4571_a982_b90a863b3b02"));
}

#[test]
fn project_dir_name_matches_native_sanitizer() {
    assert_eq!(
        project_dir_name("/tmp/claude-1000/ccs").as_deref(),
        Some("-tmp-claude-1000-ccs")
    );
    assert_eq!(
        project_dir_name("/home/u/.cache/x_y").as_deref(),
        Some("-home-u--cache-x-y")
    );
    // One dash per UTF-16 unit, like the JS regex replace.
    assert_eq!(project_dir_name("/é").as_deref(), Some("--"));
    assert_eq!(project_dir_name("/😀").as_deref(), Some("---"));
    // Native appends a hash past 200 units; we don't guess it.
    assert_eq!(project_dir_name(&format!("/{}", "a".repeat(250))), None);
}

#[test]
fn iso_timestamp_formats_utc_millis() {
    let t = SystemTime::UNIX_EPOCH + Duration::from_millis(1_791_228_600_123);
    assert_eq!(iso_timestamp(t), "2026-10-05T19:30:00.123Z");
    assert_eq!(
        iso_timestamp(SystemTime::UNIX_EPOCH),
        "1970-01-01T00:00:00.000Z"
    );
    // Leap day.
    let t = SystemTime::UNIX_EPOCH + Duration::from_secs(951_782_400);
    assert_eq!(iso_timestamp(t), "2000-02-29T00:00:00.000Z");
}

#[test]
fn transcript_chains_frames_and_ids_bare_assistants() {
    let frames = vec![
        json!({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}}),
        json!({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "mcp__gray__bash", "input": {}}]}}),
        json!({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "out"}]}}),
        json!({"type": "assistant", "message": {"id": "msg_native", "role": "assistant",
            "content": [{"type": "text", "text": "done"}]}}),
    ];
    let mut n = 0;
    let (lines, at) = transcript_lines(&frames, "S", "/w", "T", || {
        n += 1;
        format!("u{n}")
    });
    let v: Vec<Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(at, "u4");
    assert_eq!(v[0]["parentUuid"], Value::Null);
    for (i, e) in v.iter().enumerate().skip(1) {
        assert_eq!(e["parentUuid"], v[i - 1]["uuid"]);
    }
    for e in &v {
        assert_eq!(e["sessionId"], "S");
        assert_eq!(e["timestamp"], "T");
        assert_eq!(e["cwd"], "/w");
    }
    assert_eq!(v[1]["type"], "assistant");
    assert_eq!(v[1]["message"]["id"], "msg_claude_sub_1");
    assert_eq!(v[1]["message"]["type"], "message");
    assert_eq!(v[3]["message"]["id"], "msg_native");
    // The tool_result stays paired with its tool_use, in order.
    assert_eq!(v[2]["message"]["content"][0]["tool_use_id"], "c1");
}

#[test]
fn resume_file_parse_rejects_torn_or_foreign_content() {
    let sid = "6b3513ca-34f8-4571-a982-b90a863b3b02";
    let at = "47d3d750-7c40-421c-b6e3-308ff8ceec2e";
    assert_eq!(
        parse_point(&format!("{sid}\n{at}\n")),
        Some((sid.to_string(), at.to_string()))
    );
    assert_eq!(parse_point(&format!("{sid}\n{}", &at[..20])), None);
    assert_eq!(parse_point(sid), None);
    assert_eq!(parse_point(&format!("{sid}\n--help\n")), None);
}

#[test]
fn write_atomic_replaces_whole_file() {
    let d = scratch("atomic");
    let p = d.join("sub").join("f");
    write_atomic(&p, b"one").unwrap();
    write_atomic(&p, b"two").unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"two");
    // No temp files left behind.
    assert_eq!(std::fs::read_dir(d.join("sub")).unwrap().count(), 1);
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn sweep_removes_only_idle_ledgered_sessions() {
    let root = scratch("sweep-root");
    let projects = scratch("sweep-projects");
    let proj = projects.join("-home-u-gray");
    std::fs::create_dir_all(&proj).unwrap();
    let old = "11111111-1111-4111-8111-111111111111";
    let fresh = "22222222-2222-4222-8222-222222222222";
    let users = "33333333-3333-4333-8333-333333333333";
    for sid in [old, fresh, users] {
        std::fs::write(proj.join(format!("{sid}.jsonl")), "{}").unwrap();
    }
    std::fs::create_dir_all(proj.join(old)).unwrap();
    std::fs::create_dir_all(root.join("sessions")).unwrap();
    std::fs::create_dir_all(root.join("resume")).unwrap();
    for sid in [old, fresh] {
        std::fs::write(root.join("sessions").join(sid), "").unwrap();
    }
    age(&root.join("sessions").join(old), 2 * TTL.as_secs());
    std::fs::write(root.join("resume").join("aaaa"), "x").unwrap();
    std::fs::write(root.join("resume").join("bbbb"), "x").unwrap();
    age(&root.join("resume").join("aaaa"), 2 * TTL.as_secs());

    sweep(&root, &projects, SystemTime::now(), TTL);

    assert!(!proj.join(format!("{old}.jsonl")).exists());
    assert!(!proj.join(old).exists());
    assert!(!root.join("sessions").join(old).exists());
    assert!(proj.join(format!("{fresh}.jsonl")).exists());
    assert!(root.join("sessions").join(fresh).exists());
    // Never ours: untouched even though it is not in the ledger.
    assert!(proj.join(format!("{users}.jsonl")).exists());
    assert!(!root.join("resume").join("aaaa").exists());
    assert!(root.join("resume").join("bbbb").exists());
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(projects).unwrap();
}

#[test]
fn prune_drops_native_tool_runs_and_reparents_their_children() {
    // Shape recorded by claude 2.1.285 for a --max-turns 1 turn with two
    // parallel calls (each block its own entry, the second chained under
    // the first's error result), then the next turn's real results
    // branched off the last block's entry.
    let text = [
        r#"{"type":"queue-operation","operation":"enqueue"}"#,
        r#"{"type":"user","uuid":"u1","parentUuid":null,"message":{"role":"user","content":"hi"}}"#,
        r#"{"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"id":"m","content":[{"type":"tool_use","id":"t1"}]}}"#,
        r#"{"type":"user","uuid":"e1","parentUuid":"a1","sourceToolAssistantUUID":"a1","toolUseResult":"Error: No such tool available","message":{"content":[{"type":"tool_result","tool_use_id":"t1","is_error":true}]}}"#,
        r#"{"type":"assistant","uuid":"a2","parentUuid":"e1","message":{"id":"m","content":[{"type":"tool_use","id":"t2"}]}}"#,
        r#"{"type":"user","uuid":"e2","parentUuid":"a2","sourceToolAssistantUUID":"a2","toolUseResult":"Error: No such tool available","message":{"content":[{"type":"tool_result","tool_use_id":"t2","is_error":true}]}}"#,
        r#"{"type":"attachment","uuid":"x1","parentUuid":"e2","attachment":{"type":"max_turns_reached"}}"#,
        r#"{"type":"last-prompt","lastPrompt":"hi"}"#,
        r#"{"type":"user","uuid":"r1","parentUuid":"a2","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"OUT"}]}}"#,
        "not json",
    ]
    .join("\n");
    let clean = prune_lines(&text).unwrap();
    let kept: Vec<Value> = clean
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let uuids: Vec<&str> = kept.iter().filter_map(|v| v["uuid"].as_str()).collect();
    assert_eq!(uuids, ["u1", "a1", "a2", "x1", "r1"]);
    let parent = |u: &str| {
        kept.iter()
            .find(|v| v["uuid"] == u)
            .map(|v| v["parentUuid"].clone())
            .unwrap()
    };
    assert_eq!(parent("a2"), "a1");
    assert_eq!(parent("x1"), "a2");
    assert_eq!(parent("r1"), "a2");
    assert_eq!(clean.lines().count(), 8);
    assert!(clean.lines().any(|l| l == "not json"));
    // Already clean: nothing to rewrite.
    assert_eq!(prune_lines(&clean), None);
}
