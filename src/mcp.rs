//! The `gray` MCP server: host tools mounted inside the persistent
//! `claude` child over a real MCP stdio transport.
//!
//! Claude spawns MCP stdio servers itself, so the sidecar can't speak to
//! the child on a socket directly. Instead the child's `--mcp-config`
//! points at `claude-sub mcp <sock>` — a pipe subcommand ([`bridge_main`])
//! that forwards JSON-RPC lines between its own stdio and a per-session
//! unix socket the sidecar listens on ([`Bridge`]). A model `tool_use`
//! therefore reaches us as a `tools/call` request that carries the
//! assistant block's id in `params._meta["claudecode/toolUseId"]`; the
//! response stays parked until the host's `function_call_output` arrives
//! on a later turn ([`crate::live`]) — the in-process park replaces the
//! old `--max-turns 1` + transcript-replay tool boundary.
//!
//! Wire notes (verified on claude 2.1.285):
//! * requests are newline-delimited JSON-RPC; `initialize` answers with
//!   the client's own protocol version, `tools/list` with the current
//!   host inventory, `tools/call` never answers itself — it registers a
//!   parked call and a waiter thread writes the result when the turn
//!   driver supplies it (or `isError` when the session dies);
//! * notifications and response objects are dropped; unknown requests
//!   get `-32601`, never a hang;
//! * `notifications/tools/list_changed` is sent best-effort when the
//!   host inventory changes mid-session.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

/// The bridge stream: a unix socket on unix, a loopback TCP stream on
/// Windows (no unix sockets there). `Tcp` exists everywhere so the
/// Windows transport can be exercised on unix with
/// `CLAUDE_SUB_MCP_TRANSPORT=tcp`.
enum Stream {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
    Tcp(std::net::TcpStream),
}

impl Stream {
    fn try_clone(&self) -> std::io::Result<Stream> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.try_clone().map(Stream::Unix),
            Stream::Tcp(s) => s.try_clone().map(Stream::Tcp),
        }
    }

    /// End the stream both ways so the peer's read loop hits EOF.
    fn shutdown(&self) {
        let _ = match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.shutdown(std::net::Shutdown::Both),
            Stream::Tcp(s) => s.shutdown(std::net::Shutdown::Both),
        };
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.read(buf),
            Stream::Tcp(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.write(buf),
            Stream::Tcp(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.flush(),
            Stream::Tcp(s) => s.flush(),
        }
    }
}

/// The bound side of the transport. Nonblocking so the accept loop can
/// poll `dead` — a listener has no shutdown we could signal instead.
enum Listener {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixListener),
    Tcp {
        inner: std::net::TcpListener,
        /// First line a client must send: a unix socket hides behind
        /// mode-0700 directories, an open port does not — the token keeps
        /// stray loopback connections from parking `tools/call`s.
        token: String,
    },
}

/// True on Windows (no unix sockets) or when the TCP transport is forced
/// for development.
fn use_tcp() -> bool {
    cfg!(windows)
        || std::env::var_os("CLAUDE_SUB_MCP_TRANSPORT").is_some_and(|v| v == "tcp")
}

/// A per-session shared secret for the TCP transport: RandomState's keys
/// are OS-seeded, so hashing time+pid under fresh states is unguessable
/// enough for a loopback handshake.
fn auth_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut a = RandomState::new().build_hasher();
    a.write_u128(nanos);
    a.write_u32(std::process::id());
    let mut b = RandomState::new().build_hasher();
    b.write_u64(a.finish());
    format!("{:016x}{:016x}", a.finish(), b.finish())
}

/// Bind the transport; returns the listener, the endpoint string
/// `claude-sub mcp <endpoint>` connects to, and the file to unlink on
/// close (the socket path on unix, none on TCP).
fn listen() -> std::io::Result<(Listener, String, Option<PathBuf>)> {
    if use_tcp() {
        let inner = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        inner.set_nonblocking(true)?;
        let port = inner.local_addr()?.port();
        let token = auth_token();
        let listener = Listener::Tcp {
            inner,
            token: token.clone(),
        };
        return Ok((listener, format!("tcp:{port}:{token}"), None));
    }
    #[cfg(unix)]
    {
        let path = sock_path();
        if let Some(dir) = path.parent() {
            let mut mkdir = std::fs::DirBuilder::new();
            mkdir.recursive(true);
            use std::os::unix::fs::DirBuilderExt;
            mkdir.mode(0o700);
            mkdir.create(dir)?;
        }
        let listener = std::os::unix::net::UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;
        let endpoint = path.to_string_lossy().into_owned();
        Ok((Listener::Unix(listener), endpoint, Some(path)))
    }
    #[cfg(not(unix))]
    unreachable!("tcp transport is mandatory without unix sockets")
}

/// Accept one connection. On TCP the client's first line must be the
/// listener token; anything else fails `PermissionDenied` and the accept
/// loop keeps going instead of dying on a stray probe.
fn accept(listener: &Listener) -> std::io::Result<Stream> {
    match listener {
        #[cfg(unix)]
        Listener::Unix(l) => Ok(Stream::Unix(l.accept()?.0)),
        Listener::Tcp { inner, token } => {
            let (mut s, _) = inner.accept()?;
            s.set_read_timeout(Some(Duration::from_secs(10)))?;
            let mut want = token.clone();
            want.push('\n');
            let mut got = vec![0u8; want.len()];
            let ok = s.read_exact(&mut got).is_ok() && got == want.as_bytes();
            let _ = s.set_read_timeout(None);
            if !ok {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "bridge client failed the token handshake",
                ));
            }
            Ok(Stream::Tcp(s))
        }
    }
}

/// Client side of the transport (`claude-sub mcp <endpoint>`): a unix
/// socket path, or `tcp:PORT:TOKEN` — connect, then send the token line
/// before the stdio pump starts.
fn connect(endpoint: &str) -> std::io::Result<Stream> {
    if let Some(rest) = endpoint.strip_prefix("tcp:") {
        let bad = || {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad tcp endpoint")
        };
        let (port, token) = rest.split_once(':').ok_or_else(bad)?;
        let port: u16 = port.parse().map_err(|_| bad())?;
        let mut s = std::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))?;
        s.write_all(token.as_bytes())?;
        s.write_all(b"\n")?;
        return Ok(Stream::Tcp(s));
    }
    #[cfg(unix)]
    return std::os::unix::net::UnixStream::connect(endpoint).map(Stream::Unix);
    #[cfg(not(unix))]
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "unknown bridge endpoint",
    ))
}

/// A `tools/call` request parked upstream maps toolUseId → the JSON-RPC
/// request id the bridge child sent; the name/arguments already rode the
/// assistant `tool_use` line to the host, so they aren't stored.

#[derive(Default)]
struct State {
    /// Host tool inventory in MCP schema (`name`/`description`/`inputSchema`).
    tools: Vec<Value>,
    /// Calls the model is waiting on, by `claudecode/toolUseId` → JSON-RPC
    /// request id.
    parked: HashMap<String, Value>,
    /// Results the turn driver supplied, by the same id.
    answers: HashMap<String, Value>,
    /// Set on session close: every parked call resolves `isError`.
    closed: bool,
}

/// Shared bridge state: the condvar wakes parked-call waiters when an
/// answer lands or the session closes.
struct Shared {
    state: Mutex<State>,
    answered: Condvar,
    /// The accepted connection's write half (a `try_clone` of the reader).
    write: Mutex<Option<Stream>>,
    /// The socket transport is gone — a dead bridge means tool calls can
    /// never arrive, so the session itself is untrusted.
    dead: AtomicBool,
}

/// One session's `gray` MCP endpoint: the listener, the (single) accepted
/// connection and the parked-call table.
pub struct Bridge {
    endpoint: String,
    cleanup: Option<PathBuf>,
    shared: Arc<Shared>,
    accept: Option<std::thread::JoinHandle<()>>,
}

impl Bridge {
    /// Bind the session transport and start the accept/dispatch thread.
    /// `arrivals` receives the toolUseId of every registered `tools/call`
    /// so the turn driver can multiplex them against stdout events.
    pub fn bind(arrivals: Sender<String>) -> std::io::Result<Bridge> {
        let (listener, endpoint, cleanup) = listen()?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            answered: Condvar::new(),
            write: Mutex::new(None),
            dead: AtomicBool::new(false),
        });
        let accept_shared = shared.clone();
        let accept = std::thread::spawn(move || accept_loop(listener, accept_shared, arrivals));
        Ok(Bridge {
            endpoint,
            cleanup,
            shared,
            accept: Some(accept),
        })
    }

    /// The endpoint `claude-sub mcp <endpoint>` connects to: a unix
    /// socket path, or `tcp:PORT:TOKEN` on the TCP transport.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Replace the served inventory; notify the client when it changed and
    /// a connection is live.
    pub fn set_tools(&self, tools: Vec<Value>) {
        let changed = {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if st.tools == tools {
                false
            } else {
                st.tools = tools;
                true
            }
        };
        if changed {
            self.notify(json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}));
        }
    }

    /// toolUseIds of calls still waiting on a host answer.
    pub fn unanswered_ids(&self) -> HashSet<String> {
        let st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.parked
            .keys()
            .filter(|id| !st.answers.contains_key(*id))
            .cloned()
            .collect()
    }

    /// Supply the host's result for a parked call: the waiter thread then
    /// writes the `tools/call` response. `false` when the id isn't parked.
    pub fn answer(&self, tool_use_id: &str, result: Value) -> bool {
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if !st.parked.contains_key(tool_use_id) {
            return false;
        }
        st.answers.insert(tool_use_id.to_string(), result);
        self.shared.answered.notify_all();
        true
    }

    /// `answer` with a text result (the normal `function_call_output`).
    pub fn answer_text(&self, tool_use_id: &str, text: &str) -> bool {
        self.answer(
            tool_use_id,
            json!({"content": [{"type": "text", "text": text}]}),
        )
    }

    /// `answer` with an `isError` text result (superseded / closed calls).
    pub fn answer_error(&self, tool_use_id: &str, text: &str) -> bool {
        self.answer(
            tool_use_id,
            json!({"content": [{"type": "text", "text": text}], "isError": true}),
        )
    }

    /// Resolve every still-parked call as an error (the host moved on
    /// without producing a result for it). Returns the affected ids.
    pub fn answer_open_error(&self, text: &str) -> Vec<String> {
        let ids: Vec<String> = self.unanswered_ids().into_iter().collect();
        for id in &ids {
            self.answer_error(id, text);
        }
        ids
    }

    /// The socket transport died (peer closed or a write failed).
    pub fn is_dead(&self) -> bool {
        self.shared.dead.load(Ordering::Relaxed)
    }

    /// Write a protocol message on the live connection. Best-effort: a
    /// dead socket only flags the bridge.
    fn write_msg(&self, v: &Value) {
        let mut guard = self.shared.write.lock().unwrap_or_else(|e| e.into_inner());
        let Some(conn) = guard.as_mut() else { return };
        if write_line(conn, v).is_err() {
            drop(guard);
            self.flag_dead();
        }
    }

    fn notify(&self, v: Value) {
        self.write_msg(&v);
    }

    /// Mark the transport dead and wake every waiter so parked calls
    /// resolve instead of hanging their threads.
    fn flag_dead(&self) {
        self.shared.dead.store(true, Ordering::Relaxed);
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.closed = true;
        self.shared.answered.notify_all();
    }

    /// Close the bridge: parked calls resolve `isError`, the socket file is
    /// removed and the accept thread stopped.
    pub fn close(&mut self) {
        self.flag_dead();
        // Shut the connection down so the read loop's `lines()` hits EOF.
        if let Some(conn) = self
            .shared
            .write
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            conn.shutdown();
        }
        if let Some(t) = self.accept.take() {
            // The accept loop either polls on `closed` (≤60ms) or sits in
            // the connection read loop, which the shutdown just ended.
            let _ = t.join();
        }
        if let Some(path) = self.cleanup.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.close();
    }
}

/// Serialize one JSON-RPC line on a stream.
fn write_line(w: &mut impl Write, v: &Value) -> std::io::Result<()> {
    let line = serde_json::to_string(v).map_err(std::io::Error::other)? + "\n";
    w.write_all(line.as_bytes())?;
    w.flush()
}

/// Accept the (single) bridge-child connection and dispatch its requests
/// until EOF or close.
fn accept_loop(listener: Listener, shared: Arc<Shared>, arrivals: Sender<String>) {
    loop {
        if shared.dead.load(Ordering::Relaxed)
            || shared.state.lock().map(|st| st.closed).unwrap_or(true)
        {
            return;
        }
        match accept(&listener) {
            Ok(conn) => serve_conn(conn, &shared, &arrivals),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(60));
            }
            // A stray loopback connection failed the token handshake.
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {}
            Err(_) => return,
        }
    }
}

/// Read JSON-RPC lines off the accepted connection. Requests are answered
/// inline (initialize/tools/list/unknown) or parked (`tools/call`);
/// notifications and stray responses are dropped.
fn serve_conn(conn: Stream, shared: &Arc<Shared>, arrivals: &Sender<String>) {
    let Ok(writer) = conn.try_clone() else {
        shared.dead.store(true, Ordering::Relaxed);
        return;
    };
    *shared.write.lock().unwrap_or_else(|e| e.into_inner()) = Some(writer);
    let reader = BufReader::new(conn);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        trace("mcp<", &v);
        let is_request = v.get("method").is_some() && v.get("id").is_some();
        if !is_request {
            // Notifications and stray responses carry nothing we need.
            continue;
        }
        let id = v.get("id").cloned().unwrap_or(Value::Null);
        match v.get("method").and_then(Value::as_str) {
            Some("initialize") => {
                // Echo the client's protocol version verbatim.
                let pv = v
                    .pointer("/params/protocolVersion")
                    .cloned()
                    .unwrap_or(json!("2024-11-05"));
                write_or_die(
                    shared,
                    // `listChanged` is what makes the client honor the
                    // tools/list_changed notification — without it the
                    // first (often empty) inventory is cached for good.
                    &json!({"jsonrpc": "2.0", "id": id, "result": {
                        "protocolVersion": pv,
                        "capabilities": {"tools": {"listChanged": true}},
                        "serverInfo": {"name": "gray", "version": "1"},
                    }}),
                );
            }
            Some("ping") => {
                write_or_die(shared, &json!({"jsonrpc": "2.0", "id": id, "result": {}}));
            }
            Some("tools/list") => {
                let tools = shared
                    .state
                    .lock()
                    .map(|st| st.tools.clone())
                    .unwrap_or_default();
                write_or_die(
                    shared,
                    &json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tools}}),
                );
            }
            Some("tools/call") => {
                // The toolUseId ties this request to the assistant event's
                // tool_use block 1:1 — it's the key the host answers by.
                let tool_use_id = v
                    .pointer("/params/_meta/claudecode~1toolUseId")
                    .or_else(|| v.pointer("/params/_meta/claudecode/toolUseId"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if tool_use_id.is_empty() {
                    write_or_die(
                        shared,
                        &json!({"jsonrpc": "2.0", "id": id, "error":
                            {"code": -32602, "message": "tools/call without claudecode/toolUseId"}}),
                    );
                    continue;
                }
                {
                    let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    st.parked.insert(tool_use_id.clone(), id);
                }
                let _ = arrivals.send(tool_use_id.clone());
                // The read loop must never block on a parked call: each
                // call's response rides its own waiter thread.
                let shared2 = shared.clone();
                std::thread::spawn(move || answer_waiter(shared2, tool_use_id));
            }
            _ => {
                write_or_die(
                    shared,
                    &json!({"jsonrpc": "2.0", "id": id, "error":
                        {"code": -32601, "message": "method not found"}}),
                );
            }
        }
        if shared.state.lock().map(|st| st.closed).unwrap_or(true) {
            break;
        }
    }
    // The bridge child went away: no more calls can arrive.
    shared.dead.store(true, Ordering::Relaxed);
    if let Ok(mut st) = shared.state.lock() {
        st.closed = true;
    }
    shared.answered.notify_all();
}

/// CLAUDE_SUB_DEBUG wire trace for the bridge side (stderr, no secrets —
/// tool args are already host-visible).
fn trace(tag: &str, v: &Value) {
    if std::env::var_os("CLAUDE_SUB_DEBUG").is_some() {
        eprintln!("{tag} {}", serde_json::to_string(v).unwrap_or_default());
    }
}

/// Write an inline response; a dead socket flips the bridge dead so the
/// session stops trusting it.
fn write_or_die(shared: &Arc<Shared>, v: &Value) {
    trace("mcp>", v);
    let mut guard = shared.write.lock().unwrap_or_else(|e| e.into_inner());
    let Some(conn) = guard.as_mut() else { return };
    if write_line(conn, v).is_err() {
        drop(guard);
        shared.dead.store(true, Ordering::Relaxed);
        if let Ok(mut st) = shared.state.lock() {
            st.closed = true;
        }
        shared.answered.notify_all();
    }
}

/// The per-call waiter: sleeps until the turn driver answers this
/// toolUseId or the bridge closes, then writes the `tools/call` result.
fn answer_waiter(shared: Arc<Shared>, tool_use_id: String) {
    let result = {
        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(answer) = st.answers.get(&tool_use_id) {
                break Some(answer.clone());
            }
            if st.closed {
                break Some(
                    json!({"content": [{"type": "text", "text": "session closed"}],
                        "isError": true}),
                );
            }
            st = shared
                .answered
                .wait_timeout(st, Duration::from_millis(250))
                .unwrap_or_else(|e| e.into_inner())
                .0;
            if shared.dead.load(Ordering::Relaxed) {
                break Some(
                    json!({"content": [{"type": "text", "text": "session closed"}],
                        "isError": true}),
                );
            }
        }
    };
    let req_id = {
        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.answers.remove(&tool_use_id);
        st.parked.remove(&tool_use_id).unwrap_or(Value::Null)
    };
    if let Some(result) = result {
        write_or_die(
            &shared,
            &json!({"jsonrpc": "2.0", "id": req_id, "result": result}),
        );
    }
}

/// A fresh unix socket path under the sidecar cache root.
#[cfg(unix)]
pub fn sock_path() -> PathBuf {
    let tag = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    crate::session::cache_root().join(format!("mcp-{}-{tag:x}.sock", std::process::id()))
}

/// The `claude-sub mcp <endpoint>` subcommand: a two-thread pipe between
/// the inherited stdio (what claude speaks MCP on) and the session
/// transport. Either side closing ends the bridge.
pub fn bridge_main(endpoint: &str) -> i32 {
    let conn = match connect(endpoint) {
        Ok(c) => c,
        Err(_) => return 1,
    };
    let Ok(mut to_sock) = conn.try_clone() else {
        return 1;
    };
    let stdin = std::io::stdin();
    let pump = std::thread::spawn(move || {
        let reader = BufReader::new(stdin);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if to_sock
                .write_all(line.as_bytes())
                .and_then(|()| to_sock.write_all(b"\n"))
                .and_then(|()| to_sock.flush())
                .is_err()
            {
                break;
            }
        }
        // stdin gone means claude is done with us.
        to_sock.shutdown();
    });
    {
        let reader = BufReader::new(conn);
        let mut stdout = std::io::stdout();
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if stdout
                .write_all(line.as_bytes())
                .and_then(|()| stdout.write_all(b"\n"))
                .and_then(|()| stdout.flush())
                .is_err()
            {
                break;
            }
        }
    }
    // Either pump ending is enough: don't linger on the other half.
    drop(pump);
    0
}
