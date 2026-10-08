//! The Responses→native translation layer, driven by the pooled live
//! sessions in [`crate::live`]. Mirrors the Hermes DirectSDK
//! `Client.chat.completions.create` flow
//! (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT):
//!
//! `prepare_turn` folds an OpenAI Responses request into the stream-json
//! frames a `claude -p --input-format stream-json` session consumes; a
//! cold session resumes a stored native session for its history (stdin
//! `shouldQuery: false` replay is only the last resort) while a warm
//! session takes only the delta; the final user/tool-result frame
//! queries. Native answers with exactly one upstream request per query
//! (the admission relay enforces it); gray owns tools, approvals and
//! compaction — Claude only answers.
//!
//! Tool calls park inside the `gray` MCP server ([`crate::mcp`]) mounted
//! on the live child: `tools/call` blocks on the session socket until a
//! later turn's `function_call_output` supplies the result — the park is
//! the turn boundary `--max-turns 1` used to fake.
//!
//! The relay speaks the OpenAI Responses SSE wire the host already streams,
//! so no host changes are needed: the declared transport points at the
//! per-turn relay URL and the host POSTs its standard body with the
//! per-turn bearer.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::session::{self, ResumePoint};

/// Tool-name prefix native sees (host names restored on the way back).
pub const TOOL_PREFIX: &str = "mcp__gray__";
/// Relay rejection when native retries past the single admitted request.
pub const ADMISSION_CONSUMED: &str = "HERMES_MODEL_ADMISSION_CONSUMED";

/// One translated turn: system text, native history frames, host tool
/// names — and `input`, the request's `input` array verbatim, what
/// [`crate::live`] continuation matching compares absorbed state against.
pub struct PreparedTurn {
    pub system: String,
    pub frames: Vec<Value>,
    pub input: Vec<Value>,
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

/// One visible text block folded into a message's text. Blocks are
/// paragraphs, not run-on continuations — `text_of`, [`fold_lines`]
/// accumulation and [`crate::live`]'s echo matching all share this
/// separator or a replayed answer would never equal its echo.
pub(crate) fn push_text(acc: &mut String, t: &str) {
    if t.is_empty() {
        return;
    }
    if !acc.is_empty() {
        acc.push_str("\n\n");
    }
    acc.push_str(t);
}

pub(crate) fn text_of(blocks: &Value) -> String {
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
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// Anthropic image blocks for the `input_image` data-URL parts in `blocks`.
pub(crate) fn images_of(blocks: &Value) -> Vec<Value> {
    let Value::Array(arr) = blocks else {
        return Vec::new();
    };
    arr.iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("input_image"))
        .filter_map(|b| b.get("image_url").and_then(Value::as_str))
        .filter_map(|u| u.strip_prefix("data:")?.split_once(";base64,"))
        .map(|(mt, data)| {
            json!({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        })
        .collect()
}

/// Kind of a Responses input item: the `type` field, or "message" for the
/// EasyInputMessage short form (role present, type absent) the host emits.
pub(crate) fn item_kind(item: &Value) -> &str {
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
                    blocks.extend(images_of(&content));
                    if blocks.is_empty() {
                        continue;
                    }
                    if let Some(last) = frames.last_mut()
                        && last.get("type").and_then(Value::as_str) == Some("user")
                        && let Some(arr) = last
                            .pointer_mut("/message/content")
                            .and_then(Value::as_array_mut)
                    {
                        // The host sends a tool's image as a separate user
                        // item right after its output: fold it into that
                        // tool_result so the model sees it as the tool's.
                        if text.is_empty()
                            && let Some(tr) = arr.last_mut()
                            && tr.get("type").and_then(Value::as_str) == Some("tool_result")
                        {
                            if let Some(t) = tr["content"].as_str() {
                                tr["content"] = json!([{"type": "text", "text": t}]);
                            }
                            if let Some(parts) = tr["content"].as_array_mut() {
                                parts.extend(blocks);
                                continue;
                            }
                        }
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
        input,
        names,
        native_model: crate::catalog::native_model(model)?,
    })
}

fn restore_carriers(input: &[Value], frames: &mut [Value], model: &str) {
    // Our own carriers, in order: each holds one turn's native assistant
    // messages. Partial-message mode emits one line per content block
    // (thinking, text, tool_use…) under the same message id, so merge them
    // back into whole messages before matching.
    let mut pool: Vec<Value> = Vec::new();
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
        pool.extend(merge_partials(saved.unwrap_or_default()));
    }
    if pool.is_empty() {
        return;
    }
    // Each re-derived assistant frame takes the next saved message whose
    // visible projection (text + tool_use ids) matches it. That restores
    // the signed thinking Anthropic requires alongside a replayed tool_use
    // (without it the model sees a bare call and re-issues it). A frame
    // with no matching carrier stays as re-derived.
    let mut next = 0;
    for f in frames.iter_mut() {
        if f.get("type").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let want = assistant_projection(&f["message"]);
        if let Some(pos) = pool[next..]
            .iter()
            .position(|m| assistant_projection(m) == want)
        {
            *f = json!({"type": "assistant", "message": pool[next + pos].clone()});
            next += pos + 1;
        }
    }
}

/// Merge consecutive partial messages sharing an id into one message.
fn merge_partials(saved: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for m in saved {
        let id = m.get("id").and_then(Value::as_str);
        if let Some(last) = out.last_mut()
            && id.is_some()
            && last.get("id").and_then(Value::as_str) == id
            && let (Some(dst), Some(src)) = (
                last.get_mut("content").and_then(Value::as_array_mut),
                m.get("content").and_then(Value::as_array),
            )
        {
            dst.extend(src.iter().cloned());
            continue;
        }
        out.push(m);
    }
    out
}

/// What the host sees of an assistant message: its text (whitespace
/// ignored) and its tool_use ids. Thinking never projects.
fn assistant_projection(m: &Value) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut ids = Vec::new();
    for b in m
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text.extend(
                b.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .chars()
                    .filter(|c| !c.is_whitespace()),
            ),
            Some("tool_use") => ids.push(
                b.get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            _ => {}
        }
    }
    (text, ids)
}

/// Prefix of a judged-lines error where native produced no answer: a
/// refused resume ends with a result but no assistant line — safe to
/// retry on the next history-loading tier since no upstream request was
/// spent (see [`crate::live::spawn_conversation`]).
const NO_ANSWER: &str = "incomplete native response";

/// Whether a failed `--resume` run may move on to the next way of loading
/// history. Only a run that never answered qualifies: a missing session or
/// unknown `--resume-session-at` uuid ends in an `error_during_execution`
/// result with no assistant line and no request made. An upstream
/// rejection (rate limit, overload) already spent a request — retrying
/// would just hit the limit twice — and a timeout may still be generating,
/// so both surface as-is.
pub(crate) fn replay_after_failed_resume(err: &str) -> bool {
    err.starts_with(NO_ANSWER)
}

/// Where a finished turn's conversation now lives: (native session id,
/// transcript uuid of its last assistant line). `None` for anything not
/// worth resuming: no session, a synthetic (client-side error) answer, an
/// upstream error, or ids that don't look like native uuids.
pub(crate) fn resume_point(lines: &[Value]) -> Option<ResumePoint> {
    let result = lines
        .iter()
        .rev()
        .find(|v| v.get("type").and_then(Value::as_str) == Some("result"))?;
    if result.get("api_error_status").is_some_and(|s| !s.is_null()) {
        return None;
    }
    let last = lines
        .iter()
        .rev()
        .find(|v| v.get("type").and_then(Value::as_str) == Some("assistant"))?;
    if last.pointer("/message/model").and_then(Value::as_str) == Some("<synthetic>") {
        return None;
    }
    let sid = result.get("session_id").and_then(Value::as_str)?;
    let uuid = last.get("uuid").and_then(Value::as_str)?;
    (session::is_uuid(sid) && session::is_uuid(uuid)).then(|| (sid.to_string(), uuid.to_string()))
}

/// Judge a finished native run from its stream-json lines and exit status.
///
/// An upstream rejection (subscription limit, overload, auth) exits nonzero
/// with an is_error result carrying the API status and the native sentence
/// ("You've hit your session limit · resets …"); it is reported as such
/// even without an assistant line, so it is never mistaken for a refused
/// resume (which would be retried). A `result` with
/// `subtype: error_max_turns` (a stray the history carries) also excuses
/// a nonzero exit.
pub(crate) fn judge(lines: &[Value], exit_ok: bool) -> Result<(), String> {
    let is = |v: &Value, t: &str| v.get("type").and_then(Value::as_str) == Some(t);
    let tool_boundary = lines.iter().any(|v| {
        is(v, "result") && v.get("subtype").and_then(Value::as_str) == Some("error_max_turns")
    });
    // An upstream rejection (subscription limit, overload, auth) is a
    // rejection whatever the process exit code: the result carries the API
    // status and the native sentence, and must never degrade to NO_ANSWER —
    // that reads as a refused resume and would replay the burned request.
    if !tool_boundary {
        let api = lines.iter().rev().find_map(|v| {
            is(v, "result")
                .then(|| v.get("api_error_status").and_then(Value::as_u64))
                .flatten()
                .map(|code| (code, v.get("result").and_then(Value::as_str).unwrap_or("")))
        });
        if let Some((code, text)) = api {
            return Err(format!("{API_ERROR}{code}: {text}"));
        }
    }
    if !lines.iter().any(|v| is(v, "result")) || !lines.iter().any(|v| is(v, "assistant")) {
        return Err(format!("{NO_ANSWER}: assistant and one result required"));
    }
    if !exit_ok && !tool_boundary {
        return Err("native request failed (nonzero exit without a success result)".into());
    }
    Ok(())
}

const API_ERROR: &str = "native API error ";

/// Split an upstream rejection (a judged `API_ERROR` line) into the HTTP
/// status the relay should answer with and the native message. `None`
/// for everything else.
pub fn api_error(detail: &str) -> Option<(u16, &str)> {
    let (code, text) = detail.strip_prefix(API_ERROR)?.split_once(": ")?;
    let code: u16 = code.parse().ok().filter(|c| (400..600).contains(c))?;
    Some((code, if text.is_empty() { detail } else { text }))
}

/// FNV-1a: stable across processes and toolchains (resume files written by
/// one build must key the same under the next).
fn hash_of(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Resume key: everything a resumed session fixes — where native files it
/// and what its environment block says (cwd), the model, the system prompt
/// (native snapshots it on the first request) and the conversation as the
/// host sees it. Thinking and partial-message splits don't project, so a
/// carrier that failed to restore still keys the same.
pub(crate) fn resume_key(turn: &PreparedTurn, system: &str, frames: &[Value]) -> u64 {
    let mut runs: Vec<(String, Vec<String>)> = Vec::new();
    for f in frames {
        let role = f
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let msg = f.get("message").cloned().unwrap_or(Value::Null);
        let parts: Vec<String> = if role == "assistant" {
            let (text, ids) = assistant_projection(&msg);
            std::iter::once(text).chain(ids).collect()
        } else {
            msg.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(Value::to_string)
                .collect()
        };
        match runs.last_mut() {
            Some((r, p)) if *r == role && role == "assistant" => {
                // Partial lines of one answer project as one message.
                p[0].push_str(&parts[0]);
                p.extend(parts.into_iter().skip(1));
            }
            Some((r, p)) if *r == role => p.extend(parts),
            _ => runs.push((role, parts)),
        }
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    hash_of(&json!([cwd, turn.native_model, system, runs]).to_string())
}

/// A `rate_limit_event` that means the CLI is waiting out its window, not
/// answering: `{"status": "rejected", "resetsAt", "rateLimitType"}`.
/// `allowed`/`allowed_warning` pass through; an unknown status doesn't
/// qualify (the run's own result line judges it at the end). Shaped like
/// an [`API_ERROR`] so the relay reports the 429, not a generic failure.
pub(crate) fn rate_limit_rejection(v: &Value) -> Option<String> {
    if v.get("type").and_then(Value::as_str) != Some("rate_limit_event") {
        return None;
    }
    let info = v.get("rate_limit_info")?;
    if info.get("status").and_then(Value::as_str) != Some("rejected") {
        return None;
    }
    let kind = info
        .get("rateLimitType")
        .and_then(Value::as_str)
        .unwrap_or("rate limit");
    let reset = info
        .get("resetsAt")
        .and_then(Value::as_u64)
        .and_then(|t| {
            t.checked_sub(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_secs(),
            )
        })
        .map(|secs| format!("resets in ~{}m", (secs / 60).max(1)))
        .unwrap_or_else(|| "reset imminent".to_string());
    Some(format!(
        "{API_ERROR}429: Claude usage limit reached ({kind}) · {reset}"
    ))
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
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut natives: Vec<Value> = Vec::new();
    // Usage reports ride `message.usage` on assistant events and `usage`
    // on `result`; both write the same slot so the last report in stream
    // order wins. The ordering is load-bearing: a settled turn ends on
    // `result`, the only report whose `output_tokens` is final (assistant
    // snapshots are stamped mid-stream — judged on one, a whole turn
    // reads ~2 output tokens and 0 tps), while a drain folding a deferred
    // `result` then fresh assistant events keeps the newer partial
    // instead of misattributing the old query's summary to the new one.
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
                    // Latest call wins: each report carries that call's
                    // full prompt size, which is what the gauge shows.
                    if let Some(u) = msg.get("usage") {
                        usage = map_usage(u);
                    }
                    if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                                        // Progress, not stream: the host replays
                                        // the full text at finalize.
                                        say(format!("…{t}"));
                                        // Distinct blocks (across a thinking
                                        // block or a new message) are
                                        // paragraphs — bare concat glues the
                                        // last word of one onto the next.
                                        push_text(&mut text, t);
                                    }
                                }
                                // Thinking stays inside the native carrier:
                                // Claude surfaces no reasoning traces, so
                                // the host must not either. The signed blocks
                                // still round-trip for replay via `natives`.
                                Some("thinking") => {}
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
                    usage = map_result_usage(u);
                }
                let subtype = line.get("subtype").and_then(Value::as_str).unwrap_or("");
                let is_error = line
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                // error_max_turns with emitted tool calls is the tool
                // boundary (--max-turns 1), not a failure: the host runs the
                // calls and the results return as replayed frames next turn.
                if is_error
                    && subtype != "success"
                    && !(subtype == "error_max_turns" && !calls.is_empty())
                {
                    let detail = line
                        .get("result")
                        .and_then(Value::as_str)
                        .unwrap_or(subtype);
                    return Err(format!("native request failed: {detail}"));
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

/// `result` usage is the settled query's truth where assistant snapshots
/// are mid-stream: its `output_tokens` is final. `iterations` lists every
/// upstream request the query made (a call-parked query resumes on a
/// second one, and the top level sums the legs — input read there would
/// double the live context size), so the last entry's report is the
/// request this drain actually settled on.
fn map_result_usage(u: &Value) -> Usage {
    match u
        .get("iterations")
        .and_then(Value::as_array)
        .and_then(|a| a.last())
    {
        Some(last) => map_usage(last),
        None => map_usage(u),
    }
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
