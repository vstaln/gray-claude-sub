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
