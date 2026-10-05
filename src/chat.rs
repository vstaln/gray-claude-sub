//! One-shot chat turn over the loopback relay. Mirrors the Hermes DirectSDK
//! `Client.chat.completions.create` flow
//! (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT):
//!
//! history frames replay first (`shouldQuery: false`, zero-turn ack each),
//! the final user/tool-result frame queries. Native answers with exactly one
//! upstream request (the admission relay enforces it); gray owns tools,
//! approvals and compaction — Claude only answers.
//!
//! The relay speaks the OpenAI Responses SSE wire the host already streams,
//! so no host changes are needed: the declared transport points at the
//! per-turn relay URL and the host POSTs its standard body with the
//! per-turn bearer.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::setup;

/// Tool-name prefix native sees (host names restored on the way back).
pub const TOOL_PREFIX: &str = "mcp__gray__";
/// Relay rejection when native retries past the single admitted request.
pub const ADMISSION_CONSUMED: &str = "HERMES_MODEL_ADMISSION_CONSUMED";

/// One translated turn: system text, native history frames, host tool names.
pub struct PreparedTurn {
    pub system: String,
    pub frames: Vec<Value>,
    pub names: Vec<String>,
    pub native_model: String,
}

/// Normalize a tool input schema: strip top-level `oneOf`/`allOf`/`anyOf`
/// (Anthropic hard-400s) and guarantee object schemas carry `properties`.
pub fn normalize_input_schema(schema: &Value) -> Value {
    let mut out = schema.clone();
    if let Some(obj) = out.as_object_mut() {
        for key in ["oneOf", "allOf", "anyOf"] {
            obj.remove(key);
        }
        obj.entry("type".to_string())
            .or_insert(Value::String("object".to_string()));
        if obj.get("type").and_then(Value::as_str) == Some("object")
            && !matches!(obj.get("properties"), Some(Value::Object(_)))
        {
            obj.insert("properties".to_string(), json!({}));
        }
    }
    out
}

fn check_tool_name(name: &str, seen: &HashSet<String>) -> Result<(), String> {
    if name.len() > 50
        || name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        || seen.contains(name)
    {
        return Err(format!(
            "tool names must be unique ASCII identifiers of at most 50 characters: {name:?}"
        ));
    }
    Ok(())
}

fn text_of(blocks: &Value) -> String {
    match blocks {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => b.get("text").and_then(Value::as_str),
                // OpenAI Responses wire parts the host actually sends.
                Some("input_text" | "output_text") => b.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Kind of a Responses input item: the `type` field, or "message" for the
/// EasyInputMessage short form (role present, type absent) the host emits.
fn item_kind(item: &Value) -> &str {
    match item.get("type").and_then(Value::as_str) {
        Some(k) => k,
        None if item.get("role").is_some() => "message",
        None => "",
    }
}

/// Translate an OpenAI Responses body into native history frames.
/// Assistant tool calls go out prefixed; reasoning items carrying our own
/// native carrier restore byte-identical frames when the projection matches.
pub fn prepare_turn(body: &Value, model: &str) -> Result<PreparedTurn, String> {
    let instructions = body
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut system_parts: Vec<String> = Vec::new();
    if !instructions.is_empty() {
        system_parts.push(instructions.to_string());
    }
    let input = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut frames: Vec<Value> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut tool_index: BTreeMap<String, String> = BTreeMap::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for t in tools {
            let name = t.get("name").and_then(Value::as_str).unwrap_or("");
            // Responses tools are functions; anything else is host-owned.
            if t.get("type")
                .and_then(Value::as_str)
                .is_some_and(|k| k != "function")
            {
                continue;
            }
            check_tool_name(name, &seen_names)?;
            seen_names.insert(name.to_string());
            names.push(name.to_string());
            tool_index.insert(name.to_string(), format!("{TOOL_PREFIX}{name}"));
        }
    }
    // First pass: collect tool_call ids so outputs can join them.
    let mut calls: BTreeMap<String, (String, Value)> = BTreeMap::new();
    for item in &input {
        match item_kind(item) {
            "function_call" => {
                let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let args: Value = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(json!({}));
                calls.insert(
                    id.to_string(),
                    (
                        name.to_string(),
                        if args.is_object() { args } else { json!({}) },
                    ),
                );
            }
            "reasoning" => {
                // Foreign/display-only reasoning never reaches native; our own
                // carrier restores below when the projection still matches.
            }
            _ => {}
        }
    }
    for item in &input {
        match item_kind(item) {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("");
                let content = item.get("content").cloned().unwrap_or(Value::Null);
                if role == "system" || role == "developer" {
                    if !frames.is_empty() {
                        return Err("system messages must precede conversation history".into());
                    }
                    system_parts.push(text_of(&content));
                } else if role == "assistant" {
                    let text = text_of(&content);
                    // Signed-carrier replay happens in the second pass
                    // (`restore_carriers`): the carrier arrives as an adjacent
                    // `reasoning` item, not inside this message.
                    if !text.is_empty() {
                        frames.push(json!({"type": "assistant",
                            "message": {"role": "assistant",
                                "content": [{"type": "text", "text": text}]}}));
                    }
                    // Tool calls on this message go out prefixed.
                    for (id, (name, args)) in &calls {
                        let _ = (id, name, args);
                    }
                } else if role == "user" {
                    let text = text_of(&content);
                    let mut blocks: Vec<Value> = Vec::new();
                    if !text.is_empty() {
                        blocks.push(json!({"type": "text", "text": text}));
                    }
                    if blocks.is_empty() {
                        continue;
                    }
                    if let Some(last) = frames.last_mut()
                        && last.get("type").and_then(Value::as_str) == Some("user")
                        && let Some(arr) = last
                            .pointer_mut("/message/content")
                            .and_then(Value::as_array_mut)
                    {
                        arr.extend(blocks);
                        continue;
                    }
                    frames.push(
                        json!({"type": "user", "message": {"role": "user", "content": blocks}}),
                    );
                }
            }
            "function_call" => {
                let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let (rname, args) = calls
                    .get(id)
                    .cloned()
                    .unwrap_or((name.to_string(), json!({})));
                let prefixed = tool_index
                    .get(&rname)
                    .cloned()
                    .unwrap_or_else(|| format!("{TOOL_PREFIX}{rname}"));
                let block = json!({"type": "tool_use", "id": id, "name": prefixed, "input": args});
                if let Some(last) = frames.last_mut()
                    && last.get("type").and_then(Value::as_str) == Some("assistant")
                    && let Some(arr) = last
                        .pointer_mut("/message/content")
                        .and_then(Value::as_array_mut)
                {
                    arr.push(block);
                } else {
                    frames.push(json!({"type": "assistant",
                        "message": {"role": "assistant", "content": [block]}}));
                }
            }
            "function_call_output" => {
                let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let text = item
                    .get("output")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        item.get("output")
                            .map(|v| v.to_string())
                            .unwrap_or_default()
                    });
                let text = if text.is_empty() {
                    "(no output)".to_string()
                } else {
                    text
                };
                let block = json!({"type": "tool_result", "tool_use_id": id, "content": text, "is_error": false});
                if let Some(last) = frames.last_mut()
                    && last.get("type").and_then(Value::as_str) == Some("user")
                    && let Some(arr) = last
                        .pointer_mut("/message/content")
                        .and_then(Value::as_array_mut)
                {
                    arr.push(block);
                    continue;
                }
                frames.push(json!({"type": "user",
                    "message": {"role": "user", "content": [block]}}));
            }
            _ => {}
        }
    }
    // Restore native carriers: `reasoning` items whose encrypted content is
    // our own native messages replace the re-derived assistant frames when
    // the visible projection still matches (host compaction owns history).
    restore_carriers(&input, &mut frames, model);
    let Some(last) = frames.last() else {
        return Err("history must end in a nonempty user/tool-result message".into());
    };
    if last.get("type").and_then(Value::as_str) != Some("user")
        || last
            .pointer("/message/content")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
    {
        return Err(
            "history must end in a nonempty user/tool-result message; assistant prefill is unsupported"
                .into(),
        );
    }
    Ok(PreparedTurn {
        system: system_parts.join("\n\n"),
        frames,
        names,
        native_model: crate::catalog::native_model(model)?,
    })
}

fn restore_carriers(input: &[Value], frames: &mut Vec<Value>, model: &str) {
    // Collect our own carriers in order: each is one turn's saved native
    // assistant messages. A carrier's tool_use ids are the host's call_ids,
    // so re-derived frames and saved messages join on ids the host owns.
    let mut carriers: Vec<Vec<Value>> = Vec::new();
    for item in input {
        if item_kind(item) != "reasoning" {
            continue;
        }
        let blob = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        let Ok(v) = serde_json::from_str::<Value>(blob) else {
            continue;
        };
        let saved: Option<Vec<Value>> = match &v {
            Value::Array(natives) => Some(natives.clone()),
            obj if obj.get("type").and_then(Value::as_str)
                == Some("claude-subscription-native")
                && obj.get("version").and_then(Value::as_u64) == Some(1) =>
            {
                obj.get("messages").and_then(Value::as_array).cloned()
            }
            _ => None,
        };
        // Same-model gate lives in the host (thinking_block stamps it); the
        // relay trusts the host to only send back what it stamped.
        let _ = model;
        if let Some(saved) = saved {
            carriers.push(saved);
        }
    }
    if carriers.is_empty() {
        return;
    }
    let block_ids = |m: &Value| -> Vec<String> {
        m.get("content")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .filter_map(|b| b.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let has_tool_use = |m: &Value| !block_ids(m).is_empty();
    let is_text_only = |f: &Value| {
        f.pointer("/message/content")
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .all(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            })
    };
    // Rebuild: carriers map positionally onto re-derived assistant frames,
    // one turn's frames per carrier. Text-only frames take the carrier's
    // text message; tool_use frames take the native messages covering the
    // same call ids — this restores the signed thinking blocks Anthropic
    // requires alongside a replayed tool_use (without them the model sees
    // a bare call and re-issues it, then confabulates a missing result).
    let mut next_saved: Vec<usize> = vec![0; carriers.len()];
    let mut ci = 0;
    let mut out: Vec<Value> = Vec::with_capacity(frames.len() + carriers.len());
    for f in frames.iter() {
        if f.get("type").and_then(Value::as_str) != Some("assistant") {
            out.push(f.clone());
            continue;
        }
        while ci < carriers.len() && next_saved[ci] >= carriers[ci].len() {
            ci += 1;
        }
        if ci >= carriers.len() {
            out.push(f.clone());
            continue;
        }
        let ids = block_ids(&f["message"]);
        if ids.is_empty() && is_text_only(f) {
            // Text-only frame: take the next saved message that carries no
            // tool_use of its own.
            let saved = &carriers[ci];
            if let Some(pos) = saved[next_saved[ci]..]
                .iter()
                .position(|m| !has_tool_use(m))
            {
                let idx = next_saved[ci] + pos;
                out.push(json!({"type": "assistant", "message": saved[idx].clone()}));
                next_saved[ci] = idx + 1;
                continue;
            }
            out.push(f.clone());
            continue;
        }
        // Tool frame: splice the contiguous run of saved messages whose
        // tool_use ids cover this frame's ids in order.
        let saved = &carriers[ci];
        let mut got: Vec<String> = Vec::new();
        let mut run = 0;
        while next_saved[ci] + run < saved.len() && has_tool_use(&saved[next_saved[ci] + run]) {
            got.extend(block_ids(&saved[next_saved[ci] + run]));
            run += 1;
            if got == ids {
                break;
            }
        }
        if !ids.is_empty() && got == ids {
            for m in &saved[next_saved[ci]..next_saved[ci] + run] {
                out.push(json!({"type": "assistant", "message": m.clone()}));
            }
            next_saved[ci] += run;
        } else {
            out.push(f.clone());
        }
    }
    *frames = out;
}

/// Extra body the relay injects: the inert tool manifest native sees.
pub fn extra_body(names: &[String], tools: &[Value]) -> Value {
    json!({
        "tools": names.iter().map(|n| {
            let t = tools.iter().find(|t| t.get("name").and_then(Value::as_str) == Some(n));
            json!({"name": format!("{TOOL_PREFIX}{n}"),
                "description": t.and_then(|t| t.get("description").and_then(Value::as_str)).unwrap_or(""),
                "input_schema": t.map(|t| normalize_input_schema(t.get("parameters").unwrap_or(&json!({})))).unwrap_or(json!({}))})
        }).collect::<Vec<_>>(),
    })
}

/// Spawn `claude` for one turn: history frames replay first (no-query each),
/// then the final frame queries. Returns native stream-json lines.
pub fn spawn_turn(
    turn: &PreparedTurn,
    extra: &Value,
    system: &str,
    effort: Option<&str>,
    timeout: std::time::Duration,
) -> Result<Vec<Value>, String> {
    if let Some(key) = setup::conflicting_env() {
        return Err(format!(
            "subscription provider refuses conflicting {key}: unset it so native uses your Claude login"
        ));
    }
    let binary = setup::resolve_command().ok_or_else(|| setup::INSTALL_HINT.to_string())?;
    let stage = tempfile_stage()?;
    let settings_path = stage.join("settings.json");
    std::fs::write(
        &settings_path,
        json!({"env": {"CLAUDE_CODE_EXTRA_BODY": extra.to_string()}}).to_string(),
    )
    .map_err(|e| format!("staging settings.json: {e}"))?;
    let mut argv: Vec<String> = vec![
        "-p".into(),
        "--model".into(),
        turn.native_model.clone(),
        "--input-format".into(),
        "stream-json".into(),
        "--output-format".into(),
        "stream-json".into(),
        "--verbose".into(),
        "--include-partial-messages".into(),
        "--tools".into(),
        String::new(),
        "--system-prompt".into(),
        system.to_string(),
        "--settings".into(),
        settings_path.to_string_lossy().into_owned(),
        "--setting-sources".into(),
        String::new(),
        "--strict-mcp-config".into(),
        "--disable-slash-commands".into(),
        // One turn: a tool_use ends the native run; the host executes the
        // call and the result returns as a replayed frame next turn.
        "--max-turns".into(),
        "1".into(),
        "--permission-mode".into(),
        "dontAsk".into(),
        "--no-session-persistence".into(),
        "--mcp-config".into(),
        json!({"mcpServers": {}}).to_string(),
    ];
    if let Some(e) = effort.filter(|e| *e != "off") {
        argv.push("--effort".into());
        argv.push(e.to_string());
    }
    let mut child = std::process::Command::new(&binary)
        .args(&argv)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .envs(setup::child_env())
        .env("ENABLE_TOOL_SEARCH", "false")
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_AUTO_COMPACT", "1")
        .env("DISABLE_COMPACT", "1")
        .env("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off")
        // A turn must never pop a browser out of a stale login.
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
        .map_err(|_| setup::INSTALL_HINT.to_string())?;
    let frames = &turn.frames;
    let debug = std::env::var_os("CLAUDE_SUB_DEBUG").is_some();
    let mut sent: Vec<Value> = Vec::new();
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "native stdin unavailable".to_string())?;
        for (i, frame) in frames.iter().enumerate() {
            let mut f = frame.clone();
            if frame.get("type").and_then(Value::as_str) == Some("user") && i + 1 < frames.len() {
                f["shouldQuery"] = json!(false);
            }
            if debug {
                sent.push(f.clone());
            }
            let line = serde_json::to_string(&f).map_err(|e| format!("frame encode: {e}"))? + "\n";
            stdin
                .write_all(line.as_bytes())
                .map_err(|_| "native stdin closed".to_string())?;
        }
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "native stdout unavailable".to_string())?;
    let lines = read_lines(stdout, timeout)?;
    if debug {
        dump_debug(&sent, &lines);
    }
    let status = child.wait().map_err(|e| format!("native wait: {e}"))?;
    let saw_result = lines
        .iter()
        .any(|v: &Value| v.get("type").and_then(Value::as_str) == Some("result"));
    let saw_assistant = lines
        .iter()
        .any(|v: &Value| v.get("type").and_then(Value::as_str) == Some("assistant"));
    if !saw_result || !saw_assistant {
        return Err("incomplete native response: assistant and one result required".into());
    }
    // error_max_turns is the tool boundary (--max-turns 1): the run can exit
    // nonzero there while still having produced the calls the host needs.
    let tool_boundary = lines.iter().any(|v: &Value| {
        v.get("type").and_then(Value::as_str) == Some("result")
            && v.get("subtype").and_then(Value::as_str) == Some("error_max_turns")
    });
    if !status.success() && !tool_boundary {
        return Err("native request failed (nonzero exit without a success result)".into());
    }
    Ok(lines)
}

/// CLAUDE_SUB_DEBUG=1 dump: the frames sent to native and the stream-json
/// lines received, to /tmp/claude-sub-<pid>-<seq>.ndjson mode 0600.
fn dump_debug(sent: &[Value], lines: &[Value]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("claude-sub-{}-{seq}.ndjson", std::process::id()));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let Ok(mut f) = opts.open(&path) else {
        return;
    };
    use std::io::Write;
    for v in sent {
        let _ = writeln!(f, "sent {}", serde_json::to_string(v).unwrap_or_default());
    }
    for v in lines {
        let _ = writeln!(f, "recv {}", serde_json::to_string(v).unwrap_or_default());
    }
}

fn tempfile_stage() -> Result<std::path::PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("claude-sub-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("staging dir: {e}"))?;
    Ok(dir)
}

fn read_lines(
    stdout: std::process::ChildStdout,
    timeout: std::time::Duration,
) -> Result<Vec<Value>, String> {
    use std::io::BufRead;
    let reader = std::io::BufReader::new(stdout);
    let mut out = Vec::new();
    let start = std::time::Instant::now();
    for line in reader.lines() {
        if start.elapsed() > timeout {
            return Err("Claude request timed out".into());
        }
        let line = line.map_err(|e| format!("native stdout: {e}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|_| {
            format!(
                "invalid native stream-json output: {:?}",
                line.chars().take(300).collect::<String>()
            )
        })?;
        out.push(v);
    }
    Ok(out)
}

/// Fold native stream-json lines into a Responses SSE stream.
/// Returns (sse_bytes, natives, text, calls, usage, stop_reason).
#[allow(clippy::type_complexity)]
pub fn fold_lines(
    lines: &[Value],
    names: &[String],
    say: &Arc<dyn Fn(String) + Send + Sync>,
) -> Result<
    (
        Vec<u8>,
        Vec<Value>,
        String,
        Vec<(String, String, String)>,
        Usage,
        String,
    ),
    String,
> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut natives: Vec<Value> = Vec::new();
    let mut usage = Usage::default();
    let mut stop = "completed".to_string();
    let mut sse = Vec::new();
    let mut sse_text = String::new();
    let emit = |sse: &mut Vec<u8>, payload: &Value| {
        sse.extend_from_slice(b"data: ");
        sse.extend_from_slice(payload.to_string().as_bytes());
        sse.extend_from_slice(b"\n\n");
    };
    // response.created first so the host stream has an id to attach to.
    let resp_id = format!("resp_{}", rand_hex(12));
    emit(
        &mut sse,
        &json!({"type": "response.created", "response": {"id": resp_id, "model": "", "status": "in_progress"}}),
    );
    for line in lines {
        let kind = line.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "assistant" => {
                if line.get("error").is_some_and(|e| !e.is_null())
                    || line.pointer("/message/error").is_some()
                {
                    let detail = line
                        .pointer("/message/content")
                        .and_then(Value::as_array)
                        .map(|blocks| {
                            blocks
                                .iter()
                                .filter_map(|b| b.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    if line.get("error").and_then(Value::as_str) == Some("authentication_failed") {
                        return Err(format!(
                            "Claude Code has no usable login here; run `claude auth login` (native: {detail})"
                        ));
                    }
                    return Err(format!("native API error: {detail}"));
                }
                if let Some(msg) = line.get("message") {
                    natives.push(msg.clone());
                    if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                                        // Progress, not stream: the host replays
                                        // the full text at finalize.
                                        say(format!("…{t}"));
                                        text.push_str(t);
                                    }
                                }
                                Some("thinking") => {
                                    if let Some(t) = b.get("thinking").and_then(Value::as_str) {
                                        thinking.push_str(t);
                                    }
                                }
                                Some("tool_use") => {
                                    let id = b
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    let full = b.get("name").and_then(Value::as_str).unwrap_or("");
                                    let Some(short) = full.strip_prefix(TOOL_PREFIX) else {
                                        return Err(format!(
                                            "native returned a tool outside the current host inventory: {full:?}"
                                        ));
                                    };
                                    if !names.contains(&short.to_string()) {
                                        return Err(format!(
                                            "native returned a tool outside the current host inventory: {full:?}"
                                        ));
                                    }
                                    let input = b.get("input").cloned().unwrap_or(json!({}));
                                    let args = serde_json::to_string(&input)
                                        .unwrap_or_else(|_| "{}".into());
                                    // Partial snapshots can repeat a block;
                                    // the last one carries the full input.
                                    if let Some(existing) =
                                        calls.iter_mut().find(|(eid, _, _)| *eid == id)
                                    {
                                        *existing = (id, short.to_string(), args);
                                    } else {
                                        calls.push((id, short.to_string(), args));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            "result" => {
                if let Some(u) = line.get("usage") {
                    usage = map_usage(u);
                }
                let subtype = line.get("subtype").and_then(Value::as_str).unwrap_or("");
                let is_error = line
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                // error_max_turns with emitted tool calls is the tool
                // boundary (--max-turns 1), not a failure: the host runs the
                // calls and the results return as replayed frames next turn.
                if is_error && subtype != "success" {
                    if !(subtype == "error_max_turns" && !calls.is_empty()) {
                        let detail = line
                            .get("result")
                            .and_then(Value::as_str)
                            .unwrap_or(subtype);
                        return Err(format!("native request failed: {detail}"));
                    }
                }
                for a in &natives {
                    let sr = a.get("stop_reason").and_then(Value::as_str).unwrap_or("");
                    if sr == "max_tokens" || sr == "model_context_window_exceeded" {
                        stop = "incomplete".to_string();
                    }
                }
                let _ = subtype;
            }
            _ => {}
        }
    }
    if !calls.is_empty() && stop == "completed" {
        stop = "tool_use".to_string();
    }
    // Finalize: one function_call + completed function_call_output per call,
    // reasoning carrier for the next turn, then response.completed.
    let mut item_id = 0;
    for (id, name, args) in &calls {
        item_id += 1;
        emit(
            &mut sse,
            &json!({"type": "response.output_item.added",
                "output_index": item_id - 1,
                "item": {"type": "function_call", "id": format!("fc_{item_id}"),
                    "call_id": id, "name": name, "arguments": args}}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.function_call_arguments.done",
                "output_index": item_id - 1,
                "item_id": format!("fc_{item_id}"), "call_id": id,
                "name": name, "arguments": args}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_item.done",
                "output_index": item_id - 1,
                "item": {"type": "function_call", "id": format!("fc_{item_id}"),
                    "call_id": id, "name": name, "arguments": args}}),
        );
    }
    if !text.is_empty() {
        sse_text = text.clone();
        emit(
            &mut sse,
            &json!({"type": "response.output_text.delta", "output_index": 0, "delta": text}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_text.done", "output_index": 0, "text": text}),
        );
    }
    if !natives.is_empty() {
        // The host only attaches the encrypted carrier to history when the
        // turn produced thinking text (thinking_block requires nonempty
        // content). Feed it the native thinking as reasoning deltas so the
        // carrier actually round-trips on replay.
        if !thinking.is_empty() {
            emit(
                &mut sse,
                &json!({"type": "response.reasoning_summary_text.delta",
                    "output_index": item_id, "delta": thinking}),
            );
        }
        let blob = serde_json::to_string(&json!({"type": "claude-subscription-native",
            "version": 1, "messages": natives.clone(),
            "projection": {"content": text.trim(),
                "tool_calls": calls.iter().map(|(id, name, args)| {
                    let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
                    json!({"id": id, "name": format!("{TOOL_PREFIX}{name}"), "input": input})
                }).collect::<Vec<_>>()}}))
        .unwrap_or_else(|_| "[]".into());
        // The host replays `reasoning` items verbatim (same-model gate) and
        // captures `reasoning_item` into the Thinking carrier — this is the
        // native-assistant round-trip, same trick as Anthropic thinking replay.
        item_id += 1;
        emit(
            &mut sse,
            &json!({"type": "response.output_item.added",
                "output_index": item_id - 1,
                "item": {"type": "reasoning", "id": "rs_native",
                    "summary": [], "encrypted_content": blob}}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_item.done",
                "output_index": item_id - 1,
                "item": {"type": "reasoning", "id": "rs_native",
                    "summary": [], "encrypted_content": blob}}),
        );
    }
    let usage_val = json!({"input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_tokens_details": {"cached_tokens": usage.cached_tokens,
            "cache_creation_tokens": usage.cache_write_tokens}});
    emit(
        &mut sse,
        &json!({"type": "response.completed",
            "response": {"id": resp_id, "status": stop.clone(), "usage": usage_val}}),
    );
    sse.extend_from_slice(b"data: [DONE]\n\n");
    Ok((sse, natives, sse_text, calls, usage, stop))
}

/// Map native `result` usage onto inclusive counts.
#[derive(Default, Clone)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// `cache_read_input_tokens` — a subset of `input_tokens`.
    pub cached_tokens: usize,
    /// `cache_creation_input_tokens` — a subset of `input_tokens`.
    pub cache_write_tokens: usize,
}

pub fn map_usage(u: &Value) -> Usage {
    let input = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as usize;
    let output = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as usize;
    let read = u
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let write = u
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    Usage {
        input_tokens: input.saturating_add(read).saturating_add(write),
        output_tokens: output,
        cached_tokens: read,
        cache_write_tokens: write,
    }
}

fn rand_hex(n: usize) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let mut s = format!("{t:08x}{:08x}", std::process::id());
    while s.len() < n {
        s.push('0');
    }
    s[..n].to_string()
}

#[path = "chat_tests.rs"]
#[cfg(test)]
mod tests;
