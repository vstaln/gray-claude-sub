//! A pooled `claude -p` child: one live process per conversation, reused
//! across host turns while each request's history strictly continues what
//! the session last answered. Mirrors the gray-devin-sub pooled-persistent
//! design; the wire is claude's `--input-format stream-json` instead of
//! ACP.
//!
//! Why this replaces the old spawn-per-turn driver ([`crate::chat`] keeps
//! only the translation layer): a live child accepts MANY user messages
//! on one stdin — each produces its own `result` event — so a turn is no
//! longer one process. Host tools mount as a real MCP stdio server named
//! `gray` ([`crate::mcp`]): a model `tool_use` arrives as a `tools/call`
//! request parked on the session socket, and the host's
//! `function_call_output` on a later turn is what unblocks it. The
//! `--max-turns 1` boundary and `CLAUDE_CODE_EXTRA_BODY` manifest staging
//! are gone; the park IS the boundary.
//!
//! Pool rules (devin-sub semantics): at most [`MAX_LIVE`] children; a
//! session leaves the pool for its whole turn and returns on success — or
//! on a failure whose prompt still settled upstream (a `result` with
//! `api_error_status`, a rejected rate-limit event): the upstream state
//! is then known, so the host's identical whole-turn retry continues it.
//! Only a dead wire (stdin write error, stdout EOF, parse garbage,
//! deadline) kills the child — a half-written prompt is never trusted.
//!
//! Matching is by content: same native model, system prompt and effort,
//! then [`continuation`] must split the request into `absorbed` prefix +
//! echo of our last reply + a non-assistant delta. The delta routes by
//! kind: `function_call_output` answers a parked call on the socket (or
//! one already answered — a re-issued historical call after resume),
//! `message` user items merge into one stdin frame. Anything else fails
//! the match.
//!
//! Cold starts keep the old three-tier history load: stored resume point,
//! synthetic transcript, stdin replay — a refused resume still advances
//! the tier BEFORE any upstream request is spent.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::chat::{self, PreparedTurn};
use crate::mcp::{self, Bridge};
use crate::session::{self, ResumePoint};
use crate::setup;

/// Pool bound: at most this many live `claude` children.
const MAX_LIVE: usize = 4;
/// A turn that misses while a matching session is out for a keepalive
/// waits this long for it to come back — a fresh spawn re-bills the whole
/// transcript, which is exactly the miss the refresh exists to prevent.
const KEEPALIVE_WAIT: Duration = Duration::from_secs(60);
/// Poll granularity of the drain: parked-call arrivals share a channel
/// with stdout events only logically, so the wait loop wakes to fold in
/// socket traffic.
const DRAIN_STEP: Duration = Duration::from_millis(100);
/// Grace window before committing to the calls boundary: the assistant
/// event carrying a parked call's `tool_use` can still be inside the
/// stdout pump when the socket request lands.
const CALLS_GRACE: Duration = Duration::from_millis(150);
/// A delta's `function_call_output` answering nothing this session parked
/// or answered means the history belongs to another conversation.
const NOTE_UNDELIVERED: &str = "[Harness note] Your previous reply was interrupted by the user and never delivered; none of its tool calls ran.";
/// Parked calls the host never produced results for (reply undelivered,
/// or a call dropped between turns) get this error so the query unblocks.
const NOTE_SUPERSEDED: &str = "the harness did not return a result for this call (superseded)";

/// Stdout pump output: parsed stream-json events, or a line that wasn't.
enum OutEv {
    Line(Value),
    Bad(String),
}

/// What a drain is waiting on. `waiting` lives in the bridge — these are
/// only the per-drain views of it.
enum Term {
    /// Every owed `result` arrived and no call is parked.
    Done,
    /// The front query parked on tool calls covering every `tool_use`
    /// emitted this drain: the host must answer them next turn.
    Calls,
}

/// How a routed delta resolved. `applied_to` is the absolute index into
/// the request `input` up to which items were delivered upstream — the
/// absorbed prefix for the next turn's continuation match.
enum Route {
    /// The delta doesn't belong to this session; it goes back untouched.
    Diverge,
    /// Delta applied (or safely deferred, counted by `applied_to`).
    Applied {
        applied_to: usize,
        /// Merged user blocks deferred until the parked front's result —
        /// a stdin user frame mid-query interrupts and wedges the child.
        pending_user: Option<Vec<Value>>,
    },
}

/// One live `claude` child plus the replay state needed to recognize and
/// answer its strict-continuation turns.
pub struct LiveSession {
    child: Child,
    /// `Option` so `close` can drop the pipe before the wait.
    stdin: Option<ChildStdin>,
    rx: Receiver<OutEv>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    bridge: Bridge,
    parked_rx: Receiver<String>,
    /// The exact `input` array of the last request this session absorbed.
    absorbed: Vec<Value>,
    /// Call ids emitted in that answer, in emission order.
    reply_call_ids: Vec<String>,
    /// The (trimmed) text emitted in that answer.
    reply_text: String,
    /// toolUseIds this session already answered: a host
    /// `function_call_output` naming one is absorbed silently instead of
    /// diverging (a resume re-issue can land its answer before the delta).
    answered: HashSet<String>,
    /// call_id → MCP result content harvested from loaded history: a
    /// resumed session that re-issues a historical `tool_use` gets
    /// answered from here without involving the host.
    auto: HashMap<String, Value>,
    /// Upstream queries still owed a `result`: live or stdin-queued. A
    /// query parked on tool calls owes nothing until it resumes.
    running: usize,
    native_model: String,
    system: String,
    effort: Option<String>,
    /// Host tool inventory in MCP schema; updated per turn.
    tools: Vec<Value>,
    /// The unprefixed names, for `tool_use` validation expectations.
    names: Vec<String>,
    /// User-activity clock: real turns only (the keepalive never bumps
    /// it) — bounds both the warm window and the reaper.
    last_used: Instant,
    /// Last upstream contact, real turn or refresh: the
    /// `REFRESH_AFTER` clock.
    last_contact: Option<Instant>,
    /// Consecutive keepalive failures; [`crate::keepalive`] stops at its
    /// max.
    pub(crate) failures: u32,
    /// A settled 429 rides the session: keepalives skip it, real turns
    /// still try it.
    rate_limited: bool,
    /// claude's own session id, from the stream (debug/trace only).
    sid: String,
    pid: u32,
    /// Frames written this turn, for the `CLAUDE_SUB_DEBUG` dump.
    sent: Vec<Value>,
}

/// Enough of a checked-out session to guess whether a turn could use it:
/// the real continuation check runs once it's back in the pool. `pid`
/// identifies the child for removal on return.
struct Busy {
    pid: u32,
    native_model: String,
    system: String,
    effort: Option<String>,
}

#[derive(Default)]
struct Pool {
    live: Vec<LiveSession>,
    /// Sessions checked out for a keepalive refresh.
    busy: Vec<Busy>,
}

/// The process-wide session pool.
static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
/// Signalled whenever a checked-out session returns, so a waiting `take`
/// re-checks.
static POOL_CHANGED: Condvar = Condvar::new();
/// Set by [`shutdown_all`]: the sweep stops checking sessions out.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn locked_pool() -> MutexGuard<'static, Pool> {
    POOL.get_or_init(|| Mutex::new(Pool::default()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Pull dead/reapable sessions out of a locked pool; the caller closes
/// them after dropping the lock.
fn reap(pool: &mut Pool) -> Vec<LiveSession> {
    let mut dead = Vec::new();
    let now = Instant::now();
    let mut i = 0;
    while i < pool.live.len() {
        let s = &mut pool.live[i];
        // Dead wire: child exited or the `gray` bridge dropped.
        let dead_wire = !matches!(s.child.try_wait(), Ok(None)) || s.bridge.is_dead();
        if crate::keepalive::reapable(now, s.last_used, s.rate_limited, s.failures, dead_wire) {
            dead.push(pool.live.remove(i));
        } else {
            i += 1;
        }
    }
    dead
}

/// Close `dead` outside the pool lock (teardown can block on the child).
fn close_all(dead: Vec<LiveSession>) {
    for mut s in dead {
        s.close();
    }
}

/// Strict-continuation check, ported from gray-devin-sub: `input` must be
/// `absorbed` plus the echo of the session's last answer plus a
/// non-assistant tail. The echo zone collects call ids and assistant text
/// in order and compares them to what we emitted; reasoning carriers ride
/// along unmatched. Returns `(tail, undelivered)` — `undelivered` when
/// the echo zone is empty but the session's last reply wasn't (the answer
/// never reached the host, so the delta needs the note).
///
/// Miss reasons feed the debug trace: `prefix` (history diverged),
/// `echo` (the replayed answer isn't what this session sent),
/// `empty_delta` (nothing new, or an assistant item sits in the tail —
/// the history mid-edited a turn).
fn continuation<'a>(
    absorbed: &[Value],
    reply_call_ids: &[String],
    reply_text: &str,
    input: &'a [Value],
) -> Result<(&'a [Value], bool), &'static str> {
    if input.len() <= absorbed.len() || !input.starts_with(absorbed) {
        return Err("prefix");
    }
    let rest = &input[absorbed.len()..];
    let mut i = 0;
    let mut call_ids: Vec<String> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    while i < rest.len() && assistant_side(&rest[i]) {
        let item = &rest[i];
        match chat::item_kind(item) {
            "function_call" => call_ids.push(
                item.get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            "message" => texts.push(chat::text_of(item.get("content").unwrap_or(&Value::Null))),
            _ => {}
        }
        i += 1;
    }
    let undelivered = i == 0 && (!reply_call_ids.is_empty() || !reply_text.trim().is_empty());
    if !undelivered
        && (call_ids.as_slice() != reply_call_ids || texts.join("\n\n").trim() != reply_text.trim())
    {
        return Err("echo");
    }
    let tail = &rest[i..];
    if tail.is_empty() || tail.iter().any(assistant_side) {
        return Err("empty_delta");
    }
    Ok((tail, undelivered))
}

/// An input item the assistant side produced: the replayed echo of a
/// session's own answer (assistant message, its calls, reasoning
/// carriers). Anything else is host-side delta material.
fn assistant_side(item: &Value) -> bool {
    match chat::item_kind(item) {
        "function_call" | "reasoning" => true,
        "message" => item.get("role").and_then(Value::as_str) == Some("assistant"),
        _ => false,
    }
}

/// The first pooled session this turn is a strict continuation of, or
/// the deepest reason any candidate reached (`no_session` when empty).
fn find(
    live: &[LiveSession],
    turn: &PreparedTurn,
    effort: Option<&str>,
) -> Result<(usize, usize, bool), &'static str> {
    let mut reason = "no_session";
    let mut depth = 0;
    for (i, s) in live.iter().enumerate() {
        let (miss, d) = if s.native_model != turn.native_model {
            ("model", 1)
        } else if s.system != turn.system {
            ("system", 2)
        } else if s.effort.as_deref() != effort {
            ("effort", 3)
        } else {
            match continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &turn.input) {
                Ok((tail, undelivered)) => {
                    return Ok((i, absorbed_len(turn, tail), undelivered));
                }
                Err(m) => (
                    m,
                    match m {
                        "prefix" => 4,
                        "echo" => 5,
                        _ => 6,
                    },
                ),
            }
        };
        if d > depth {
            depth = d;
            reason = miss;
        }
    }
    Err(reason)
}

/// Index `input` points at where `tail` starts.
fn absorbed_len(turn: &PreparedTurn, tail: &[Value]) -> usize {
    turn.input.len() - tail.len()
}

/// Take the pooled session this turn continues, plus where its delta
/// starts. While a session that could answer is out for a keepalive, wait
/// for it — up to [`KEEPALIVE_WAIT`] — rather than miss and re-bill the
/// whole transcript on a fresh child.
fn take(
    turn: &PreparedTurn,
    effort: Option<&str>,
    deadline: Instant,
) -> (Option<(LiveSession, usize, bool)>, &'static str) {
    let wait_deadline = (Instant::now() + KEEPALIVE_WAIT).min(deadline);
    let mut dead = Vec::new();
    let result = {
        let mut pool = locked_pool();
        loop {
            dead.extend(reap(&mut pool));
            let reason = match find(&pool.live, turn, effort) {
                Ok((i, delta_at, undelivered)) => {
                    let s = pool.live.remove(i);
                    break (Some((s, delta_at, undelivered)), "reuse");
                }
                Err(reason) => reason,
            };
            let warming = pool.busy.iter().any(|b| {
                b.native_model == turn.native_model
                    && b.system == turn.system
                    && b.effort.as_deref() == effort
            });
            let left = wait_deadline.saturating_duration_since(Instant::now());
            if !warming || SHUTDOWN.load(Ordering::Relaxed) {
                break (None, reason);
            }
            if left.is_zero() {
                break (None, "keepalive_busy");
            }
            pool = POOL_CHANGED
                .wait_timeout(pool, left.min(Duration::from_secs(1)))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    };
    close_all(dead);
    result
}

/// Return a session to the pool, evicting least-recently-used past
/// [`MAX_LIVE`], and clear its `busy` marker so waiting takes re-check.
fn give_back(s: LiveSession) {
    let dead = {
        let mut pool = locked_pool();
        pool.busy.retain(|b| b.pid != s.pid);
        let mut dead = reap(&mut pool);
        pool.live.push(s);
        if pool.live.len() > MAX_LIVE {
            let lru = pool
                .live
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            dead.push(pool.live.remove(lru));
        }
        dead
    };
    POOL_CHANGED.notify_all();
    close_all(dead);
}

/// Check a session out of the pool for a keepalive refresh: alive, inside
/// the warm window, upstream contact due, not rate-limited and not parked
/// on tool calls (a parked session is mid-conversation — it must answer,
/// not probe). Listed as `busy` so a matching turn waits for it.
pub(crate) fn take_refresh_candidate() -> Option<LiveSession> {
    if SHUTDOWN.load(Ordering::Relaxed) {
        return None;
    }
    let mut pool = locked_pool();
    let now = Instant::now();
    let i = pool.live.iter_mut().position(|s| {
        crate::keepalive::refresh_due(
            now,
            s.last_used,
            s.last_contact,
            s.rate_limited,
            s.failures,
            matches!(s.child.try_wait(), Ok(None)),
            !s.bridge.unanswered_ids().is_empty(),
        )
    })?;
    let s = pool.live.remove(i);
    pool.busy.push(Busy {
        pid: s.pid,
        native_model: s.native_model.clone(),
        system: s.system.clone(),
        effort: s.effort.clone(),
    });
    Some(s)
}

/// Fold a finished refresh back in: upstream contact happened either way,
/// a dead wire kills the session, anything else counts toward the failure
/// cap the reaper enforces.
pub(crate) fn finish_refresh(mut s: LiveSession, err: Option<&str>) {
    s.last_contact = Some(Instant::now());
    match err {
        None => s.failures = 0,
        Some(e) => {
            if chat::api_error(e).is_some_and(|(c, _)| c == 429) {
                s.rate_limited = true;
            }
            s.failures += 1;
            if !s.settle(e) {
                let mut pool = locked_pool();
                pool.busy.retain(|b| b.pid != s.pid);
                POOL_CHANGED.notify_all();
                return;
            }
        }
    }
    give_back(s);
}

/// Kill sessions the reaper wants (idle past TTL and not warmable,
/// rate-limited past window, dead child, keepalive failures spent).
/// Called by the keepalive sweep; pool accesses reap inline too.
pub(crate) fn reap_idle() {
    let dead = {
        let mut pool = locked_pool();
        reap(&mut pool)
    };
    close_all(dead);
}

/// Kill every pooled session; used on `plugin/shutdown` and stdin EOF.
pub fn shutdown_all() {
    SHUTDOWN.store(true, Ordering::Relaxed);
    let dead = {
        let mut pool = locked_pool();
        std::mem::take(&mut pool.live)
    };
    close_all(dead);
}

/// Errors that leave the session untrusted: the child died, the pipe
/// broke or the wire spoke garbage — the prompt's upstream state is
/// ambiguous. Everything else (an `API_ERROR` result, a refused resume)
/// settled and the session can stay pooled.
fn dead_wire(e: &str) -> bool {
    !e.starts_with("native API error ")
        && !chat::replay_after_failed_resume(e)
        && !e.starts_with("native request failed")
}

/// Build the child argv for one live session: the old per-turn flag set
/// minus `--max-turns` (the MCP park is the tool boundary now) and minus
/// the `--settings`/`CLAUDE_CODE_EXTRA_BODY` staging (tools arrive via the
/// MCP server), plus the `gray` stdio server pointing at our own `mcp`
/// subcommand on this session's socket.
fn argv(
    turn: &PreparedTurn,
    system: &str,
    effort: Option<&str>,
    sock: &std::path::Path,
    resume: Option<&ResumePoint>,
) -> Result<Vec<String>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;
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
        // An empty --tools disables builtins AND masks MCP tools; the
        // server-scoped pattern allows every `gray` tool and nothing
        // else, and allowedTools pre-approves them — under dontAsk an
        // unlisted tool is denied locally before it can park.
        "--tools".into(),
        "mcp__gray".into(),
        "--allowedTools".into(),
        "mcp__gray".into(),
        "--system-prompt".into(),
        system.to_string(),
        "--setting-sources".into(),
        String::new(),
        "--strict-mcp-config".into(),
        "--disable-slash-commands".into(),
        "--permission-mode".into(),
        "dontAsk".into(),
        "--mcp-config".into(),
        json!({"mcpServers": {"gray": {
            "type": "stdio",
            "command": exe.to_string_lossy(),
            "args": ["mcp", sock.to_string_lossy()],
        }}})
        .to_string(),
    ];
    if let Some((sid, uuid)) = resume {
        argv.extend([
            "--resume".into(),
            sid.clone(),
            "--resume-session-at".into(),
            uuid.clone(),
        ]);
    }
    if let Some(e) = effort.filter(|e| *e != "off") {
        argv.push("--effort".into());
        argv.push(e.to_string());
    }
    Ok(argv)
}

/// Host tool inventory in MCP schema — BARE names (claude adds the
/// `mcp__gray__` prefix itself when it shows the tool to the model) and
/// `inputSchema` (what `tools/list` serves).
fn tool_inventory(names: &[String], tools: &[Value]) -> Vec<Value> {
    names
        .iter()
        .map(|n| {
            let t = tools
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some(n));
            json!({"name": n,
                "description": t.and_then(|t| t.get("description").and_then(Value::as_str)).unwrap_or(""),
                "inputSchema": t.map(|t| chat::normalize_input_schema(t.get("parameters").unwrap_or(&json!({})))).unwrap_or(json!({}))})
        })
        .collect()
}

impl LiveSession {
    /// The child's stderr tail, for dead-wire diagnostics.
    fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|t| t.iter().cloned().collect::<Vec<_>>().join(" · "))
            .unwrap_or_default()
    }

    /// One stdin frame. A write error is a dead wire: a half-written
    /// prompt is never left ambiguous — the caller kills the session.
    fn send(&mut self, v: &Value) -> Result<(), String> {
        let line = serde_json::to_string(v).map_err(|e| format!("frame encode: {e}"))? + "\n";
        let Some(stdin) = self.stdin.as_mut() else {
            return Err("native stdin closed".to_string());
        };
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|_| "native stdin closed".to_string())
    }

    /// A querying user frame: merged content blocks (text + Anthropic
    /// image blocks) as one stdin `user` message.
    fn send_user(&mut self, blocks: Vec<Value>) -> Result<(), String> {
        let frame = json!({"type": "user",
            "message": {"role": "user", "content": blocks}});
        self.send(&frame)?;
        if std::env::var_os("CLAUDE_SUB_DEBUG").is_some() {
            self.sent.push(frame);
        }
        self.running += 1;
        Ok(())
    }

    /// Write history frames on a cold session: every user frame but the
    /// last is `shouldQuery: false` (native appends assistant frames at
    /// once but queues user frames into the next querying message), so
    /// exactly the final frame queries.
    fn write_history(&mut self, frames: &[Value]) -> Result<(), String> {
        for (i, frame) in frames.iter().enumerate() {
            let mut f = frame.clone();
            if frame.get("type").and_then(Value::as_str) == Some("user") && i + 1 < frames.len() {
                f["shouldQuery"] = json!(false);
            }
            if std::env::var_os("CLAUDE_SUB_DEBUG").is_some() {
                self.sent.push(f.clone());
            }
            self.send(&f)?;
        }
        // Every `user` frame yields its own `result` event — even a
        // `shouldQuery:false` replay frame (num_turns:0, empty). Assistant
        // frames yield init/echo only. `running` must count user frames,
        // not just the one querying frame: with `1`, the first replayed
        // frame's empty result ended the drain early and the turn judged
        // NO_ANSWER, so any conversation with prior history could never
        // cold-start (found via resume e2e).
        self.running = frames
            .iter()
            .filter(|f| f.get("type").and_then(Value::as_str) == Some("user"))
            .count();
        if self.running == 0 {
            return Err("history contains no user frames".into());
        }
        Ok(())
    }

    /// Harvest `call_id → tool_result` content out of loaded frames: a
    /// resumed session that re-issues a historical `tool_use` is answered
    /// from this table without involving the host.
    fn seed_auto(&mut self, frames: &[Value]) {
        for f in frames {
            let Some(arr) = f.pointer("/message/content").and_then(Value::as_array) else {
                continue;
            };
            for b in arr {
                if b.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let Some(id) = b.get("tool_use_id").and_then(Value::as_str) else {
                    continue;
                };
                let content = match b.get("content") {
                    Some(Value::String(s)) => json!([{"type": "text", "text": s}]),
                    Some(Value::Array(parts)) => Value::Array(parts.clone()),
                    _ => continue,
                };
                self.auto
                    .insert(id.to_string(), json!({"content": content}));
            }
        }
    }

    /// Route a continuation delta: `function_call_output` items answer
    /// their parked calls, user items merge into one querying stdin
    /// frame. Validated BEFORE applying — a `call_id` that matches no
    /// parked or already-answered call means divergence, and the session
    /// goes back untouched.
    ///
    /// A user frame must never land while the front query is parked on a
    /// tool call: mid-query stdin is an interrupt that leaves the parked
    /// call's query result-less and wedges the child. When the front is
    /// parked the merged user frame defers — the caller sends it after
    /// the front's `result` arrives — and `applied_to` then marks only
    /// the delivered prefix so an unsent frame re-arrives next turn.
    fn route_delta(
        &mut self,
        delta: &[Value],
        undelivered: bool,
        delta_at: usize,
    ) -> Result<Route, String> {
        // A parked front query blocks everything behind it; it owes a
        // result again once its calls are answered below.
        let front_parked = !self.bridge.unanswered_ids().is_empty();
        let mut answers: Vec<(String, String)> = Vec::new();
        let mut user_blocks: Vec<Value> = Vec::new();
        let mut user_at: Option<usize> = None;
        for (j, item) in delta.iter().enumerate() {
            match chat::item_kind(item) {
                "function_call_output" => {
                    let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                    if self.answered.contains(id) {
                        // Already answered (a re-issued historical call):
                        // absorb silently — it's not a live parked call.
                        continue;
                    }
                    let parked = self.bridge.unanswered_ids();
                    if !parked.contains(id) {
                        return Ok(Route::Diverge);
                    }
                    let text = item
                        .get("output")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            item.get("output")
                                .map(|v| v.to_string())
                                .unwrap_or_default()
                        });
                    answers.push((id.to_string(), text));
                }
                "message" if item.get("role").and_then(Value::as_str) == Some("user") => {
                    user_at.get_or_insert(j);
                    let content = item.get("content").cloned().unwrap_or(Value::Null);
                    let text = chat::text_of(&content);
                    if !text.is_empty() {
                        user_blocks.push(json!({"type": "text", "text": text}));
                    }
                    user_blocks.extend(chat::images_of(&content));
                }
                // continuation() already rejected assistant-side tail
                // items; anything else is foreign.
                _ => return Ok(Route::Diverge),
            }
        }
        // Apply: parked calls answer first so the front query resumes
        // before the new user frame lands behind it. Any parked call the
        // host produced no result for gets an error answer — leaving it
        // parked would block the front (and every queued frame) forever.
        for (id, text) in answers {
            let text = if text.is_empty() {
                "(no output)".to_string()
            } else {
                text
            };
            if self.bridge.answer_text(&id, &text) {
                if debug_on() {
                    self.sent.push(json!({"mcp_answer": id}));
                }
                self.answered.insert(id.clone());
                self.auto
                    .insert(id, json!({"content": [{"type": "text", "text": text}]}));
            }
        }
        for id in self.bridge.answer_open_error(NOTE_SUPERSEDED) {
            self.answered.insert(id);
        }
        if front_parked {
            // The previously parked query owes a result again.
            self.running += 1;
        }
        if undelivered && !user_blocks.is_empty() {
            user_blocks.insert(0, json!({"type": "text", "text": NOTE_UNDELIVERED}));
        }
        if user_blocks.is_empty() {
            return Ok(Route::Applied {
                applied_to: delta_at + delta.len(),
                pending_user: None,
            });
        }
        if front_parked {
            // Deferred: the frame goes in after the front's result.
            // Absorbed state marks only the contiguous applied prefix —
            // the delta's user items (and anything behind them, which
            // `answered` absorbs on replay) come back next turn.
            return Ok(Route::Applied {
                applied_to: delta_at + user_at.unwrap_or(0),
                pending_user: Some(user_blocks),
            });
        }
        self.send_user(user_blocks)?;
        Ok(Route::Applied {
            applied_to: delta_at + delta.len(),
            pending_user: None,
        })
    }

    /// Sync the served inventory with this turn's tools: the `gray`
    /// server reads it live and emits `tools/list_changed` on drift.
    /// `false` when the bridge is dead (the session can't serve tools).
    fn update_tools(&mut self, names: &[String], tools: Vec<Value>) -> bool {
        if self.bridge.is_dead() {
            return false;
        }
        self.bridge.set_tools(tools.clone());
        self.tools = tools;
        self.names = names.to_vec();
        true
    }

    /// Fold one stdout event into this drain's view. Rate-limit
    /// rejections short-circuit as settled `API_ERROR`s; a repeating
    /// `system/init` is ignored except for the `gray` server status —
    /// a failed MCP mount on a tool-bearing turn is a dead session.
    fn on_line(
        &mut self,
        v: Value,
        lines: &mut Vec<Value>,
        expected: &mut HashSet<String>,
    ) -> Result<(), String> {
        if let Some(msg) = chat::rate_limit_rejection(&v) {
            return Err(msg);
        }
        match v.get("type").and_then(Value::as_str) {
            Some("system") => {
                if v.get("subtype").and_then(Value::as_str) == Some("init")
                    && !self.tools.is_empty()
                {
                    let failed = v
                        .get("mcp_servers")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .any(|s| {
                            s.get("name").and_then(Value::as_str) == Some("gray")
                                && s.get("status").and_then(Value::as_str) != Some("connected")
                        });
                    if failed {
                        return Err(MCP_MOUNT_FAILED.to_string());
                    }
                }
            }
            Some("assistant") => {
                if let Some(blocks) = v.pointer("/message/content").and_then(Value::as_array) {
                    for b in blocks {
                        if b.get("type").and_then(Value::as_str) != Some("tool_use") {
                            continue;
                        }
                        // Expected = the calls fold_lines will accept:
                        // prefixed and inside the current inventory.
                        let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                        let Some(short) = name.strip_prefix(chat::TOOL_PREFIX) else {
                            continue;
                        };
                        if !self.names.iter().any(|n| n == short) {
                            continue;
                        }
                        if let Some(id) = b.get("id").and_then(Value::as_str) {
                            expected.insert(id.to_string());
                        }
                    }
                }
            }
            Some("result") => {
                self.running = self.running.saturating_sub(1);
                if let Some(sid) = v.get("session_id").and_then(Value::as_str) {
                    self.sid = sid.to_string();
                }
            }
            Some("rate_limit_event") => {
                // Subscription-quota telemetry: `unifiedWindows` refreshes
                // every window at once — write-through to the /usage cache
                // so quota rows stay warm between probes.
                crate::usage::note_event(&v);
            }
            _ => {}
        }
        if let Some(sid) = v.get("session_id").and_then(Value::as_str) {
            self.sid = sid.to_string();
        }
        lines.push(v);
        Ok(())
    }

    /// Fold a parked-call arrival: auto-answerable re-issues and denied
    /// calls (keepalive) resolve instantly — the query never blocks. A
    /// real new call parks the front query: the first such arrival flips
    /// `front_parked` and its result stops being owed (parallel calls on
    /// the same front don't decrement `running` again).
    fn on_parked(
        &mut self,
        id: String,
        deny_calls: bool,
        parked: &mut HashSet<String>,
        front_parked: &mut bool,
    ) {
        if deny_calls {
            if self
                .bridge
                .answer_error(&id, "cache-refresh probe cannot run tools")
            {
                self.answered.insert(id);
            }
            return;
        }
        if let Some(result) = self.auto.get(&id).cloned() {
            if self.bridge.answer(&id, result) {
                self.answered.insert(id);
            }
            return;
        }
        parked.insert(id);
        if !*front_parked && self.running > 0 {
            *front_parked = true;
            self.running -= 1;
        }
    }

    /// Fold immediately-available stdout events and parked arrivals.
    /// Returns `Some(Err)` on dead wire or a settled rejection.
    fn collect_pending(
        &mut self,
        lines: &mut Vec<Value>,
        expected: &mut HashSet<String>,
        deny_calls: bool,
        parked: &mut HashSet<String>,
        front_parked: &mut bool,
    ) -> Option<String> {
        loop {
            match self.rx.try_recv() {
                Ok(OutEv::Line(v)) => {
                    if let Err(e) = self.on_line(v, lines, expected) {
                        return Some(e);
                    }
                }
                Ok(OutEv::Bad(s)) => {
                    return Some(format!("invalid native stream-json output: {s}"));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Some("native stdout closed".to_string());
                }
            }
        }
        while let Ok(id) = self.parked_rx.try_recv() {
            self.on_parked(id, deny_calls, parked, front_parked);
        }
        None
    }

    /// Drive the session until the turn boundary: every owed `result`
    /// arrived (Done) or the front query parked on calls covering every
    /// emitted `tool_use` (Calls). `deny_calls` is the keepalive mode:
    /// tool calls get an instant `isError` instead of parking.
    /// Errors: dead wire (kill) or settled upstream rejection (keep).
    fn drain(&mut self, deadline: Instant, deny_calls: bool) -> Result<(Term, Vec<Value>), String> {
        self.drain_inner(deadline, deny_calls).map_err(|e| {
            // A dead wire is where the child's own diagnostics live.
            let tail = self.stderr_tail();
            if dead_wire(&e) && !tail.is_empty() {
                format!("{e} · native stderr: {tail}")
            } else {
                e
            }
        })
    }

    fn drain_inner(
        &mut self,
        deadline: Instant,
        deny_calls: bool,
    ) -> Result<(Term, Vec<Value>), String> {
        let mut lines = Vec::new();
        let mut expected: HashSet<String> = HashSet::new();
        // Calls parked before this drain (defensive — routing flushes
        // them) are already the parked front: they owe no result and
        // don't decrement `running` again.
        let mut parked: HashSet<String> = self.bridge.unanswered_ids();
        let mut front_parked = !parked.is_empty();
        loop {
            if let Some(e) = self.collect_pending(
                &mut lines,
                &mut expected,
                deny_calls,
                &mut parked,
                &mut front_parked,
            ) {
                return Err(e);
            }
            if self.running == 0 && self.bridge.unanswered_ids().is_empty() {
                // Nothing in flight, nothing parked: turn boundary.
                return Ok((Term::Done, lines));
            }
            if !parked.is_empty() && parked.is_superset(&expected) {
                // The front query is waiting on answers only the host has.
                // Give the stdout pump a beat first — the assistant event
                // carrying a parked call's tool_use may still be in flight.
                match self.rx.recv_timeout(CALLS_GRACE) {
                    Ok(OutEv::Line(v)) => {
                        self.on_line(v, &mut lines, &mut expected)?;
                        continue;
                    }
                    Ok(OutEv::Bad(s)) => {
                        return Err(format!("invalid native stream-json output: {s}"));
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if parked.is_superset(&expected) {
                            return Ok((Term::Calls, lines));
                        }
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err("native stdout closed".into());
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err("Claude request timed out".into());
            }
            if self.bridge.is_dead() && !self.tools.is_empty() {
                // No transport, no tool answers: this query can only hang.
                return Err("gray MCP bridge closed".into());
            }
            match self
                .rx
                .recv_timeout(DRAIN_STEP.min(deadline.saturating_duration_since(Instant::now())))
            {
                Ok(OutEv::Line(v)) => self.on_line(v, &mut lines, &mut expected)?,
                Ok(OutEv::Bad(s)) => {
                    return Err(format!("invalid native stream-json output: {s}"));
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("native stdout closed".into());
                }
            }
            while let Ok(id) = self.parked_rx.try_recv() {
                self.on_parked(id, deny_calls, &mut parked, &mut front_parked);
            }
        }
    }

    /// Fold a drain error: a dead wire or a second in-flight query (a
    /// queued frame's upstream fate — its late lines would pollute a
    /// later drain — is ambiguous) kills the session; a lone rejected
    /// query settled upstream, so flush its parked calls and reset the
    /// owed count. Returns whether the session survives.
    fn settle(&mut self, e: &str) -> bool {
        if dead_wire(e) || self.running > 1 {
            self.close();
            return false;
        }
        self.running = 0;
        self.bridge.answer_open_error(NOTE_SUPERSEDED);
        if chat::api_error(e).is_some_and(|(c, _)| c == 429) {
            self.rate_limited = true;
        }
        true
    }

    /// Record what this session just answered so the next request
    /// matches as its strict continuation. `applied_to` bounds the
    /// absorbed prefix: only input items actually delivered upstream
    /// count (a user frame deferred past a re-park is not in the child's
    /// transcript, so it must re-arrive).
    fn absorb(&mut self, turn: &PreparedTurn, lines: &[Value], applied_to: usize) {
        let (ids, text) = echo_of(lines, &self.names);
        self.absorbed = turn.input[..applied_to].to_vec();
        self.reply_call_ids = ids;
        self.reply_text = text.trim().to_string();
        self.last_used = Instant::now();
        self.last_contact = Some(Instant::now());
    }

    /// Record a failed first turn whose prompt still settled upstream:
    /// the input counts as absorbed (its frames are already in the
    /// child's transcript) but no reply is on record, so an identical
    /// retry misses by design and respawns, while a later turn of this
    /// conversation continues the warm session with only its tail.
    /// Ported from devin-sub's `absorb_failed`.
    fn absorb_failed(&mut self, turn: &PreparedTurn) {
        self.system = turn.system.clone();
        self.absorbed = turn.input.clone();
        self.reply_call_ids = Vec::new();
        self.reply_text = String::new();
        self.last_used = Instant::now();
        self.last_contact = Some(Instant::now());
    }

    /// The keepalive probe: one minimal querying user frame drained to
    /// its result. Tool calls get an instant `isError` — the probe
    /// refreshes the cache, it never gets to run tools.
    pub(crate) fn refresh(&mut self, deadline: Instant) -> Result<(), String> {
        self.send_user(vec![json!({"type": "text",
            "text": crate::keepalive::PROMPT})])?;
        let (term, lines) = self.drain(deadline, true)?;
        if matches!(term, Term::Calls) {
            return Err("keepalive probe parked on tool calls".into());
        }
        chat::judge(&lines, true)
    }

    /// Best-effort teardown: the bridge closes (parked calls resolve
    /// `isError`, the socket file goes), stdin drops, then kill.
    fn close(&mut self) {
        self.bridge.close();
        drop(self.stdin.take());
        let wait_deadline = Instant::now() + Duration::from_secs(2);
        while self.child.try_wait().ok().flatten().is_none() && Instant::now() < wait_deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.close();
    }
}

/// The reply a turn produced, for echo matching: `tool_use` ids in
/// emission order (in-inventory only — the same filter drain uses) and
/// the concatenated text.
fn echo_of(lines: &[Value], names: &[String]) -> (Vec<String>, String) {
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    let mut text = String::new();
    for v in lines {
        if v.get("type").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = v.pointer("/message/content").and_then(Value::as_array) else {
            continue;
        };
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                        chat::push_text(&mut text, t);
                    }
                }
                Some("tool_use") => {
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                    let ok = name
                        .strip_prefix(chat::TOOL_PREFIX)
                        .is_some_and(|short| names.iter().any(|n| n == short));
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                    if ok && !id.is_empty() && seen.insert(id.to_string()) {
                        ids.push(id.to_string());
                    }
                }
                _ => {}
            }
        }
    }
    (ids, text)
}

/// `system/init` reporting the `gray` stdio mount failed is a spawn-time
/// flake — the child's `claude-sub mcp` shim never connected or claude's
/// handshake timed out — not a verdict on the session. The failed child
/// closes itself, so one respawn on a fresh socket almost always mounts.
const MCP_MOUNT_FAILED: &str = "gray MCP server failed to connect";

/// Spawn a live child: bind the `gray` socket, write `tail` frames into
/// it, drain to the boundary. The caller owns judge/tier semantics.
fn spawn_live(
    turn: &PreparedTurn,
    tools: Vec<Value>,
    effort: Option<&str>,
    resume: Option<&ResumePoint>,
    tail: &[Value],
    deadline: Instant,
) -> Result<(LiveSession, Term, Vec<Value>), String> {
    let binary = setup::resolve_command().ok_or_else(|| setup::INSTALL_HINT.to_string())?;
    let sock = mcp::sock_path();
    let (parked_tx, parked_rx) = channel();
    let bridge = Bridge::bind(sock.clone(), parked_tx)
        .map_err(|e| format!("mcp socket {}: {e}", sock.display()))?;
    let argv = argv(turn, &turn.system, effort, &sock, resume)?;
    let mut child = Command::new(&binary)
        .args(&argv)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
    let pid = child.id();
    let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
    if let Some(err) = child.stderr.take() {
        let tail_buf = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                if let Ok(mut t) = tail_buf.lock() {
                    if t.len() >= 20 {
                        t.pop_front();
                    }
                    t.push_back(line);
                }
            }
        });
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "native stdout unavailable".to_string())?;
    let (tx, rx) = channel::<OutEv>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ev = match serde_json::from_str::<Value>(line) {
                Ok(v) => OutEv::Line(v),
                Err(_) => OutEv::Bad(line.chars().take(300).collect()),
            };
            if tx.send(ev).is_err() {
                break;
            }
        }
    });
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "native stdin unavailable".to_string())?;
    let stdin = Some(stdin);
    let mut s = LiveSession {
        child,
        stdin,
        rx,
        stderr_tail,
        bridge,
        parked_rx,
        absorbed: Vec::new(),
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        answered: HashSet::new(),
        auto: HashMap::new(),
        running: 0,
        native_model: turn.native_model.clone(),
        system: turn.system.clone(),
        effort: effort.map(str::to_string),
        tools,
        names: turn.names.clone(),
        last_used: Instant::now(),
        last_contact: None,
        failures: 0,
        rate_limited: false,
        sid: String::new(),
        pid,
        sent: Vec::new(),
    };
    // The bridge is bound before the child spawns; claude's
    // initialize/tools/list can arrive as soon as it mounts `gray`.
    s.bridge.set_tools(s.tools.clone());
    s.seed_auto(&turn.frames);
    if let Err(e) = s.write_history(tail) {
        // A dead pipe mid-load is untrusted — but still drain: a refused
        // resume's error result is worth reading for the tier fallback.
        let outcome = s.drain(deadline, false);
        s.close();
        return match outcome {
            Ok((_, lines)) => Err(classify(&lines).unwrap_or(e)),
            Err(_) => Err(e),
        };
    }
    let (term, lines) = match s.drain(deadline, false) {
        Ok(t) => t,
        Err(e) => {
            s.close();
            return Err(e);
        }
    };
    if let Some(sid) = lines
        .iter()
        .find_map(|v| v.get("session_id").and_then(Value::as_str))
    {
        session::adopt(sid);
        s.sid = sid.to_string();
    }
    Ok((s, term, lines))
}

/// Classify a finished drain for the caller: `None` = answered, else the
/// judge verdict (API_ERROR / NO_ANSWER / generic).
fn classify(lines: &[Value]) -> Option<String> {
    chat::judge(lines, true).err()
}

/// `run_turn` for a conversation with no pooled continuation: the
/// three-tier cold start — stored resume point, synthetic transcript,
/// stdin replay — where a refused resume (no request spent) advances the
/// tier and anything else surfaces.
fn spawn_conversation(
    turn: &PreparedTurn,
    tools: Vec<Value>,
    effort: Option<&str>,
    deadline: Instant,
) -> Result<(LiveSession, Term, Vec<Value>), String> {
    let frames = &turn.frames;
    let split = frames
        .iter()
        .rposition(|f| f.get("type").and_then(Value::as_str) == Some("assistant"))
        .map_or(0, |i| i + 1);
    let prev_key = (split > 0).then(|| chat::resume_key(turn, &turn.system, &frames[..split]));
    if let Some(prev) = prev_key {
        let history = &frames[..split];
        let tail = &frames[split..];
        // Lazily: synthesize writes a transcript file only when the stored
        // point is absent or the resume was refused.
        let stored = || session::lookup(prev);
        let synthetic = || session::synthesize(history);
        for at in [
            &stored as &dyn Fn() -> Option<ResumePoint>,
            &synthetic as &dyn Fn() -> Option<ResumePoint>,
        ] {
            let Some(at) = at() else { continue };
            match spawn_live_retried(turn, tools.clone(), effort, Some(&at), tail, deadline) {
                // A refused resume ends Done with an error result and no
                // assistant line — judge reads it as NO_ANSWER and the
                // child is a dead end: kill and advance the tier.
                Ok((mut s, term, lines))
                    if matches!(term, Term::Done)
                        && chat::judge(&lines, true)
                            .err()
                            .is_some_and(|e| chat::replay_after_failed_resume(&e)) =>
                {
                    s.close();
                    continue;
                }
                r => return r,
            }
        }
    }
    spawn_live_retried(turn, tools, effort, None, frames, deadline)
}

/// One respawn on a fresh socket when the child's `gray` MCP mount failed
/// at init — a spawn-time flake, retried once, never looping.
fn spawn_live_retried(
    turn: &PreparedTurn,
    tools: Vec<Value>,
    effort: Option<&str>,
    resume: Option<&ResumePoint>,
    tail: &[Value],
    deadline: Instant,
) -> Result<(LiveSession, Term, Vec<Value>), String> {
    match spawn_live(turn, tools.clone(), effort, resume, tail, deadline) {
        Err(e) if e == MCP_MOUNT_FAILED => spawn_live(turn, tools, effort, resume, tail, deadline),
        r => r,
    }
}

/// Record where this conversation now lives (the session plus the
/// transcript uuid of its last assistant line) so a future cold start
/// resumes instead of replaying.
fn record_point(turn: &PreparedTurn, lines: &[Value]) {
    if let Some((sid, uuid)) = chat::resume_point(lines)
        && session::prune_native_tool_results(&sid)
    {
        let mut convo = turn.frames.clone();
        convo.extend(
            lines
                .iter()
                .filter(|v| v.get("type").and_then(Value::as_str) == Some("assistant"))
                .map(|v| json!({"type": "assistant", "message": v["message"].clone()})),
        );
        session::save(chat::resume_key(turn, &turn.system, &convo), &sid, &uuid);
    }
}

/// CLAUDE_SUB_DEBUG=1 dump: the frames written and the stream-json lines
/// received, to /tmp/claude-sub-<pid>-<seq>.ndjson mode 0600, headed by a
/// `{"live": ...}` row naming the child pid and the reuse mode so session
/// reuse is observable.
fn dump_debug(sent: &[Value], lines: &[Value], meta: &Value) {
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
    let _ = writeln!(
        f,
        "live {}",
        serde_json::to_string(meta).unwrap_or_default()
    );
    for v in sent {
        let _ = writeln!(f, "sent {}", serde_json::to_string(v).unwrap_or_default());
    }
    for v in lines {
        let _ = writeln!(f, "recv {}", serde_json::to_string(v).unwrap_or_default());
    }
}

fn debug_on() -> bool {
    std::env::var_os("CLAUDE_SUB_DEBUG").is_some()
}

/// Run one host turn against a pooled-or-fresh live session. Returns the
/// native stream-json lines [`crate::chat::fold_lines`] folds to SSE.
///
/// A pooled continuation writes only its delta; everything else cold-
/// starts. On a settled upstream rejection the session stays pooled
/// (the host's identical retry continues it); only a dead wire kills it.
pub fn run_turn(
    body: &Value,
    model: &str,
    effort: Option<&str>,
    timeout: Duration,
) -> Result<Vec<Value>, String> {
    if let Some(key) = setup::conflicting_env() {
        return Err(format!(
            "subscription provider refuses conflicting {key}: unset it so native uses your Claude login"
        ));
    }
    let turn = chat::prepare_turn(body, model)?;
    let tools = tool_inventory(
        &turn.names,
        body.get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .as_slice(),
    );
    let deadline = Instant::now() + timeout;
    let (claimed, miss_reason) = take(&turn, effort, deadline);
    let (mut s, mut applied_to, pending_user, mode) = match claimed {
        Some((mut s, delta_at, undelivered)) => {
            let names = turn.names.clone();
            s.sent.clear();
            if !s.update_tools(&names, tools.clone()) {
                // The bridge died: the session can't serve tools anymore.
                s.close();
                let (s2, term, lines) = spawn_conversation(&turn, tools, effort, deadline)?;
                return finish_spawn(s2, term, lines, &turn, "fresh:bridge_dead");
            }
            match s.route_delta(&turn.input[delta_at..], undelivered, delta_at) {
                Ok(Route::Applied {
                    applied_to,
                    pending_user,
                }) => (s, applied_to, pending_user, "reuse".to_string()),
                Ok(Route::Diverge) => {
                    // The delta doesn't continue this session after all —
                    // give it back untouched and cold-start the request.
                    give_back(s);
                    let (s2, term, lines) = spawn_conversation(&turn, tools, effort, deadline)?;
                    return finish_spawn(s2, term, lines, &turn, "fresh:diverged");
                }
                Err(e) => {
                    // Dead wire mid-delta: the request is still
                    // answerable, on a fresh child.
                    s.close();
                    let (s2, term, lines) = spawn_conversation(&turn, tools, effort, deadline)?;
                    return finish_spawn(s2, term, lines, &turn, &format!("fresh:{e}"));
                }
            }
        }
        None => {
            let (s, term, lines) = spawn_conversation(&turn, tools, effort, deadline)?;
            return finish_spawn(s, term, lines, &turn, &format!("fresh:{miss_reason}"));
        }
    };
    // Reused session: the delta routed, now drain to the boundary. A
    // deferred user frame goes in only once the parked front's result
    // landed — mid-query stdin is an interrupt that wedges the child.
    let lines = match s.drain(deadline, false) {
        Ok((term, mut lines)) => {
            match term {
                Term::Done => {
                    if let Some(blocks) = pending_user {
                        if s.send_user(blocks).is_err() {
                            // Dead wire writing the deferred frame: the
                            // turn is still answerable on a fresh child.
                            s.close();
                            let (s2, t2, l2) = spawn_conversation(&turn, tools, effort, deadline)?;
                            return finish_spawn(s2, t2, l2, &turn, "fresh:send_dead");
                        }
                        applied_to = turn.input.len();
                        match s.drain(deadline, false) {
                            Ok((_, lines2)) => lines.extend(lines2),
                            Err(e) => {
                                if s.settle(&e) {
                                    give_back(s);
                                }
                                return Err(e);
                            }
                        }
                    }
                    lines
                }
                // Still parked — the deferred user frame rides next turn.
                Term::Calls => lines,
            }
        }
        Err(e) => {
            if debug_on() {
                dump_debug(
                    &s.sent,
                    &[],
                    &json!({"mode": mode, "pid": s.pid, "sid": s.sid, "error": e}),
                );
            }
            if s.settle(&e) {
                give_back(s);
            }
            return Err(e);
        }
    };
    let term_is_calls = !s.bridge.unanswered_ids().is_empty();
    if debug_on() {
        dump_debug(
            &s.sent,
            &lines,
            &json!({"mode": mode, "pid": s.pid, "sid": s.sid}),
        );
    }
    if term_is_calls {
        s.absorb(&turn, &lines, applied_to);
        give_back(s);
        return Ok(lines);
    }
    if let Err(e) = chat::judge(&lines, true) {
        // Settled upstream state: keep the session with the absorb state
        // that matched this turn, so an identical retry re-prompts the
        // same delta.
        if chat::api_error(&e).is_some_and(|(c, _)| c == 429) {
            s.rate_limited = true;
        }
        give_back(s);
        return Err(e);
    }
    record_point(&turn, &lines);
    s.absorb(&turn, &lines, applied_to);
    give_back(s);
    Ok(lines)
}

/// Fold a cold-started drain: judge the Done lines (a refused resume has
/// already advanced through the tiers inside [`spawn_conversation`]),
/// pool the session and hand the lines back.
fn finish_spawn(
    mut s: LiveSession,
    term: Term,
    lines: Vec<Value>,
    turn: &PreparedTurn,
    mode: &str,
) -> Result<Vec<Value>, String> {
    if debug_on() {
        dump_debug(
            &s.sent,
            &lines,
            &json!({"mode": mode, "pid": s.pid, "sid": s.sid}),
        );
    }
    match term {
        Term::Done => {
            if let Err(e) = chat::judge(&lines, true) {
                if dead_wire(&e) {
                    s.close();
                } else {
                    // Settled upstream failure on a fresh child: the input
                    // counts as absorbed so this conversation owns the
                    // session, but an identical retry respawns by design.
                    if chat::api_error(&e).is_some_and(|(c, _)| c == 429) {
                        s.rate_limited = true;
                    }
                    s.absorb_failed(turn);
                    give_back(s);
                }
                return Err(e);
            }
            record_point(turn, &lines);
            s.absorb(turn, &lines, turn.input.len());
            give_back(s);
            Ok(lines)
        }
        Term::Calls => {
            s.absorb(turn, &lines, turn.input.len());
            give_back(s);
            Ok(lines)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_records_what_fold_sent() {
        // `reply_text` feeds next turn's continuation match, so it must be
        // the same text the host saw — block separators included, or every
        // continuation misses and respawns.
        let lines = vec![
            json!({"type": "assistant", "message": {"role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "tool_use", "id": "c1",
                        "name": "mcp__gray__bash", "input": {"command": "ls"}},
                    {"type": "text", "text": "b"}]}}),
            json!({"type": "result", "subtype": "success", "is_error": false}),
        ];
        let names = vec!["bash".to_string()];
        let say: Arc<dyn Fn(String) + Send + Sync> = Arc::new(|_| {});
        let (_, _, folded, _, _, _) = chat::fold_lines(&lines, &names, &say).unwrap();
        let (ids, reply_text) = echo_of(&lines, &names);
        assert_eq!(ids, vec!["c1".to_string()]);
        assert_eq!(reply_text, folded);
        assert_eq!(reply_text, "a\n\nb");
    }

    #[test]
    fn continuation_accepts_the_replayed_echo() {
        // The host replays the absorbed prefix, the assistant message it
        // saw (one item, whole text), the call + its output, then the new
        // user tail.
        let absorbed = vec![json!({"role": "user", "content": "hi"})];
        let input = vec![
            absorbed[0].clone(),
            json!({"role": "assistant",
                "content": [{"type": "output_text", "text": "a\n\nb"}]}),
            json!({"type": "function_call", "call_id": "c1",
                "name": "bash", "arguments": "{}"}),
            json!({"type": "function_call_output",
                "call_id": "c1", "output": "out"}),
            json!({"role": "user", "content": "next"}),
        ];
        let (tail, undelivered) =
            continuation(&absorbed, &["c1".to_string()], "a\n\nb", &input).unwrap();
        assert!(!undelivered);
        assert_eq!(tail.len(), 2, "{tail:?}");
        // A glued echo (the old text_of would have produced it) misses.
        assert_eq!(
            continuation(&absorbed, &["c1".to_string()], "ab", &input),
            Err("echo")
        );
    }
}
