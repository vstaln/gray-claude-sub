use super::*;

fn body(items: Vec<Value>) -> Value {
    json!({"model": "sonnet", "instructions": "sys", "input": items, "stream": true})
}

#[test]
fn user_message_becomes_native_frame() {
    let b = body(vec![json!({"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "hi"}]})]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    assert_eq!(turn.system, "sys");
    assert_eq!(turn.frames.len(), 1);
    assert_eq!(turn.frames[0]["type"], "user");
    assert_eq!(turn.native_model, "claude-sonnet-5[1m]");
}

#[test]
fn short_form_user_message_becomes_native_frame() {
    // EasyInputMessage short form: the host emits {"role","content"} with
    // no "type" field; it must still be read as a message.
    let b = body(vec![json!({"role": "user", "content": "hi"})]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    assert_eq!(turn.frames.len(), 1);
    assert_eq!(turn.frames[0]["type"], "user");
    assert_eq!(turn.frames[0]["message"]["content"][0]["text"], "hi");
}

#[test]
fn short_form_assistant_then_user_replay() {
    let b = body(vec![
        json!({"role": "assistant", "content": "earlier"}),
        json!({"role": "user", "content": "next"}),
    ]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    assert_eq!(turn.frames.len(), 2);
    assert_eq!(turn.frames[0]["type"], "assistant");
    assert_eq!(turn.frames[0]["message"]["content"][0]["text"], "earlier");
    assert_eq!(turn.frames[1]["type"], "user");
    assert_eq!(turn.frames[1]["message"]["content"][0]["text"], "next");
}

#[test]
fn empty_history_is_rejected() {
    let b = body(vec![]);
    assert!(prepare_turn(&b, "sonnet").is_err());
}

#[test]
fn carrier_restores_thinking_on_tool_use_replay() {
    // Anthropic requires the signed thinking block that accompanied a
    // tool_use to be replayed with it. The carrier saves the native
    // assistant message; on replay it must replace the re-derived bare
    // tool_use frame (joined on the host-owned call id).
    let native = json!({"type": "claude-subscription-native", "version": 1,
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_x", "name": "mcp__gray__bash",
             "input": {"command": "echo hi"}}]}],
        "projection": {"content": "",
            "tool_calls": [{"id": "toolu_x", "name": "mcp__gray__bash",
                "input": {"command": "echo hi"}}]}});
    let b = body(vec![
        json!({"role": "user", "content": "run it"}),
        json!({"type": "function_call", "call_id": "toolu_x",
            "name": "bash", "arguments": "{\"command\":\"echo hi\"}"}),
        json!({"type": "reasoning", "summary": [],
            "encrypted_content": native.to_string()}),
        json!({"type": "function_call_output", "call_id": "toolu_x",
            "output": "hi"}),
    ]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    let assistant = turn
        .frames
        .iter()
        .find(|f| f["type"] == "assistant")
        .expect("assistant frame");
    let kinds: Vec<&str> = assistant["message"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|b| b["type"].as_str())
        .collect();
    assert!(kinds.contains(&"thinking"), "{kinds:?}");
    assert!(kinds.contains(&"tool_use"), "{kinds:?}");
    let tool_use = assistant["message"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap();
    assert_eq!(tool_use["id"], "toolu_x");
}

#[test]
fn usage_reports_cache_reads_and_writes() {
    let lines = vec![
        json!({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "hello"}]}}),
        json!({"type": "result", "subtype": "success", "is_error": false,
            "usage": {"input_tokens": 4, "output_tokens": 3,
                "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50}}),
    ];
    let say: Arc<dyn Fn(String) + Send + Sync> = Arc::new(|_| {});
    let (sse, _, _, _, usage, _) = fold_lines(&lines, &[], &say).unwrap();
    assert_eq!(usage.input_tokens, 154);
    assert_eq!(usage.cached_tokens, 100);
    assert_eq!(usage.cache_write_tokens, 50);
    let s = String::from_utf8(sse).unwrap();
    assert!(s.contains("\"cached_tokens\":100"));
    assert!(s.contains("\"cache_creation_tokens\":50"));
}

#[test]
fn fold_emits_valid_responses_sse() {
    let lines = vec![
        json!({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "hello"}]}}),
        json!({"type": "result", "subtype": "success", "is_error": false,
            "usage": {"input_tokens": 10, "output_tokens": 3}}),
    ];
    let say: Arc<dyn Fn(String) + Send + Sync> = Arc::new(|_| {});
    let (sse, natives, text, calls, usage, stop) = fold_lines(&lines, &[], &say).unwrap();
    assert_eq!(text, "hello");
    assert!(calls.is_empty());
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(stop, "completed");
    assert_eq!(natives.len(), 1);
    let s = String::from_utf8(sse).unwrap();
    assert!(s.contains("response.created"));
    assert!(s.contains("response.output_text.done"));
    assert!(s.contains("response.output_item.done"));
    assert!(s.contains("response.completed"));
    assert!(s.ends_with("data: [DONE]\n\n"));
}

#[test]
fn fold_treats_max_turns_with_calls_as_tool_boundary() {
    // --max-turns 1 ends the run on tool_use: subtype error_max_turns with
    // emitted calls is the tool boundary, not a failure.
    let lines = vec![
        json!({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "tool_use", "id": "t1",
                "name": "mcp__gray__bash", "input": {"command": "ls"}}]}}),
        json!({"type": "result", "subtype": "error_max_turns",
            "is_error": true, "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ];
    let say: Arc<dyn Fn(String) + Send + Sync> = Arc::new(|_| {});
    let (sse, _, _, calls, _, stop) = fold_lines(&lines, &["bash".to_string()], &say).unwrap();
    assert_eq!(stop, "tool_use");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, "bash");
    assert_eq!(calls[0].2, "{\"command\":\"ls\"}");
    let s = String::from_utf8(sse).unwrap();
    assert!(s.contains("response.output_item.done"));
}

#[test]
fn fold_rejects_foreign_tools() {
    let lines = vec![
        json!({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "rm_rf", "input": {}}]}}),
        json!({"type": "result", "subtype": "success", "is_error": false,
            "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ];
    let say: Arc<dyn Fn(String) + Send + Sync> = Arc::new(|_| {});
    assert!(fold_lines(&lines, &["bash".to_string()], &say).is_err());
}

#[test]
fn input_images_become_native_image_blocks() {
    let b = body(vec![
        json!({"role": "user", "content": "look"}),
        json!({"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAA"}]}),
        json!({"type": "function_call", "call_id": "c1", "name": "view", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "c1", "output": "Shown: x.png"}),
        json!({"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,BBB"}]}),
    ]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    // Pasted image rides next to its text.
    let first = &turn.frames[0]["message"]["content"];
    assert_eq!(first[1]["type"], "image");
    assert_eq!(first[1]["source"]["media_type"], "image/png");
    assert_eq!(first[1]["source"]["data"], "AAA");
    // Tool image folds into the preceding tool_result.
    let last = &turn.frames[2]["message"]["content"];
    assert_eq!(last.as_array().unwrap().len(), 1);
    let tr = &last[0];
    assert_eq!(tr["type"], "tool_result");
    assert_eq!(
        tr["content"][0],
        json!({"type": "text", "text": "Shown: x.png"})
    );
    assert_eq!(tr["content"][1]["source"]["data"], "BBB");
}

#[test]
fn carrier_restores_partial_message_natives() {
    // --include-partial-messages saves one native line per content block
    // (same message id). A text turn's carrier is [thinking, text]: the
    // thinking-only partial must not be taken for the text frame, and the
    // next turn's tool carrier must still land on its tool frame.
    let carrier = |msgs: Value| {
        json!({"type": "reasoning", "summary": [], "encrypted_content":
            json!({"type": "claude-subscription-native", "version": 1,
                "messages": msgs}).to_string()})
    };
    let b = body(vec![
        json!({"role": "user", "content": "hi"}),
        carrier(json!([
            {"id": "m1", "role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "s1"}]},
            {"id": "m1", "role": "assistant", "content": [
                {"type": "text", "text": "hello there"}]}])),
        json!({"role": "assistant", "content": "hello there"}),
        json!({"role": "user", "content": "run it"}),
        carrier(json!([
            {"id": "m2", "role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "s2"}]},
            {"id": "m2", "role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "mcp__gray__bash",
                 "input": {"command": "ls"}}]}])),
        json!({"type": "function_call", "call_id": "c1", "name": "bash",
            "arguments": "{\"command\":\"ls\"}"}),
        json!({"type": "function_call_output", "call_id": "c1", "output": "a"}),
    ]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    let kinds = |f: &Value| -> Vec<String> {
        f["message"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap().to_string())
            .collect()
    };
    let roles: Vec<&str> = turn
        .frames
        .iter()
        .map(|f| f["type"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
    assert_eq!(kinds(&turn.frames[1]), ["thinking", "text"]);
    assert_eq!(turn.frames[1]["message"]["content"][0]["signature"], "s1");
    assert_eq!(
        turn.frames[1]["message"]["content"][1]["text"],
        "hello there"
    );
    assert_eq!(kinds(&turn.frames[3]), ["thinking", "tool_use"]);
    assert_eq!(turn.frames[3]["message"]["content"][0]["signature"], "s2");
}

#[test]
fn carrier_with_mismatched_projection_is_not_spliced() {
    // A carrier whose visible text no longer matches (host edited or
    // compacted the message) must leave the re-derived frame alone.
    let b = body(vec![
        json!({"role": "user", "content": "hi"}),
        json!({"type": "reasoning", "summary": [], "encrypted_content":
            json!({"type": "claude-subscription-native", "version": 1,
                "messages": [{"id": "m1", "role": "assistant", "content": [
                    {"type": "text", "text": "original"}]}]}).to_string()}),
        json!({"role": "assistant", "content": "edited"}),
        json!({"role": "user", "content": "next"}),
    ]);
    let turn = prepare_turn(&b, "sonnet").unwrap();
    assert_eq!(turn.frames[1]["message"]["content"][0]["text"], "edited");
}

#[test]
fn resume_key_matches_next_turn_history() {
    // The key recorded after a turn (sent frames + native partial lines)
    // must equal the key of the next turn's history prefix, whether the
    // carrier restored the natives or the frame was re-derived.
    let turn1 = prepare_turn(
        &body(vec![json!({"role": "user", "content": "run it"})]),
        "sonnet",
    )
    .unwrap();
    let mut convo = turn1.frames.clone();
    convo.push(
        json!({"type": "assistant", "message": {"id": "m1", "role": "assistant",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}]}}),
    );
    convo.push(
        json!({"type": "assistant", "message": {"id": "m1", "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "mcp__gray__bash",
            "input": {"command": "ls"}}]}}),
    );
    let recorded = resume_key(&turn1, "sys", &convo);
    let next = |with_carrier: bool| {
        let mut items = vec![json!({"role": "user", "content": "run it"})];
        if with_carrier {
            items.push(
                json!({"type": "reasoning", "summary": [], "encrypted_content":
                json!({"type": "claude-subscription-native", "version": 1,
                    "messages": [convo[1]["message"], convo[2]["message"]]}).to_string()}),
            );
        }
        items.push(
            json!({"type": "function_call", "call_id": "c1", "name": "bash",
            "arguments": "{\"command\":\"ls\"}"}),
        );
        items.push(json!({"type": "function_call_output", "call_id": "c1", "output": "a"}));
        let t = prepare_turn(&body(items), "sonnet").unwrap();
        let split = t
            .frames
            .iter()
            .rposition(|f| f["type"] == "assistant")
            .unwrap()
            + 1;
        resume_key(&t, "sys", &t.frames[..split])
    };
    assert_eq!(next(true), recorded);
    assert_eq!(next(false), recorded);
    // A different system prompt is a different native session snapshot.
    assert_ne!(resume_key(&turn1, "other", &convo), recorded);
}

#[test]
fn api_error_carries_native_status_and_sentence() {
    let d = format!("{API_ERROR}429: You've hit your session limit · resets 2:30am");
    assert_eq!(
        api_error(&d),
        Some((429, "You've hit your session limit · resets 2:30am"))
    );
    assert_eq!(api_error(&format!("{API_ERROR}200: ok")), None);
    assert_eq!(
        api_error("native request failed (nonzero exit without a success result)"),
        None
    );
}

const SID: &str = "6b3513ca-34f8-4571-a982-b90a863b3b02";
const AT: &str = "47d3d750-7c40-421c-b6e3-308ff8ceec2e";

fn answer(model: &str) -> Value {
    json!({"type": "assistant", "uuid": AT, "session_id": SID,
        "message": {"id": "msg_1", "model": model, "role": "assistant",
            "content": [{"type": "text", "text": "hi"}]}})
}

fn result(extra: Value) -> Value {
    let mut r = json!({"type": "result", "subtype": "success", "is_error": false,
        "session_id": SID, "api_error_status": null});
    for (k, v) in extra.as_object().unwrap() {
        r[k] = v.clone();
    }
    r
}

#[test]
fn refused_resume_moves_on_but_rate_limit_does_not() {
    // `--resume` of a pruned session / unknown uuid (claude 2.1.285):
    // exit 1, one error_during_execution result, no assistant, no request.
    let refused = vec![result(json!({"subtype": "error_during_execution",
        "is_error": true, "num_turns": 0}))];
    let e = judge(&refused, false).unwrap_err();
    assert!(replay_after_failed_resume(&e), "{e}");
    assert_eq!(api_error(&e), None);

    // Subscription limit: synthetic assistant + result with the API status.
    let limit_text = "You've hit your session limit · resets 2:30am";
    let limited = vec![
        answer("<synthetic>"),
        result(json!({"is_error": true, "api_error_status": 429, "result": limit_text})),
    ];
    let e = judge(&limited, false).unwrap_err();
    assert!(!replay_after_failed_resume(&e), "{e}");
    assert_eq!(api_error(&e), Some((429, limit_text)));

    // Same rejection without the synthetic line is still an API error,
    // never a "no answer" that would be retried.
    let e = judge(&limited[1..], false).unwrap_err();
    assert!(!replay_after_failed_resume(&e), "{e}");
    assert_eq!(api_error(&e).map(|(c, _)| c), Some(429));

    // Other failures surface unchanged.
    assert!(!replay_after_failed_resume("Claude request timed out"));
    assert!(!replay_after_failed_resume("native stdin closed"));
}

#[test]
fn judge_accepts_answers_and_the_tool_boundary() {
    assert!(judge(&[answer("claude-sonnet-5"), result(json!({}))], true).is_ok());
    // --max-turns 1 ends a tool call with a nonzero exit.
    let boundary = vec![
        answer("claude-sonnet-5"),
        result(json!({"subtype": "error_max_turns", "is_error": true})),
    ];
    assert!(judge(&boundary, false).is_ok());
    let e = judge(&[answer("claude-sonnet-5"), result(json!({}))], false).unwrap_err();
    assert!(e.starts_with("native request failed"), "{e}");
    assert!(judge(&[answer("claude-sonnet-5")], true).is_err());
}

#[test]
fn resume_point_takes_session_and_last_assistant_uuid() {
    let mut first = answer("claude-sonnet-5");
    first["uuid"] = json!("11111111-1111-4111-8111-111111111111");
    let lines = vec![first, answer("claude-sonnet-5"), result(json!({}))];
    assert_eq!(
        resume_point(&lines),
        Some((SID.to_string(), AT.to_string()))
    );
    // Nothing worth resuming: client-side error, upstream error, no result,
    // or ids that would not be safe as argv / paths.
    assert_eq!(
        resume_point(&[answer("<synthetic>"), result(json!({}))]),
        None
    );
    assert_eq!(
        resume_point(&[
            answer("claude-sonnet-5"),
            result(json!({"api_error_status": 529}))
        ]),
        None
    );
    assert_eq!(resume_point(&[answer("claude-sonnet-5")]), None);
    let mut bad = answer("claude-sonnet-5");
    bad["uuid"] = json!("--help");
    assert_eq!(resume_point(&[bad, result(json!({}))]), None);
}

#[test]
fn hash_is_stable_across_builds() {
    // FNV-1a reference vectors: resume files outlive the binary.
    assert_eq!(hash_of(""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(hash_of("a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(hash_of("foobar"), 0x8594_4171_f739_67e8);
}
