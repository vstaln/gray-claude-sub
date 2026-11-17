//! The `gray` mount has to survive the client reconnecting. Claude Code
//! drops and reopens connections on a healthy session: a probe sibling is
//! closed as the real transport lands, and a mount that hits Claude's own
//! 30s timeout respawns `claude-sub mcp` against this same endpoint.
//! Ending the bridge the instant the session conn closed turned that churn
//! into a permanent failure the host could only report as a bare "failed".

use super::*;

use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc::channel;
use std::time::Duration;

fn bridge() -> Bridge {
    let (tx, _rx) = channel();
    Bridge::bind(tx).expect("bind the bridge")
}

/// Connect exactly as `claude-sub mcp <endpoint>` does — TCP plus the token
/// line on Windows, a unix socket elsewhere — and return both halves.
fn client(endpoint: &str) -> (Stream, BufReader<Stream>) {
    let write = connect(endpoint).expect("connect to the bridge");
    arm_read_timeout(&write);
    let read = write.try_clone().expect("clone");
    (write, BufReader::new(read))
}

/// A hung test is worse than a failed one: cap every read.
fn arm_read_timeout(s: &Stream) {
    let d = Some(Duration::from_secs(10));
    match s {
        #[cfg(unix)]
        Stream::Unix(s) => {
            let _ = s.set_read_timeout(d);
        }
        Stream::Tcp(s) => {
            let _ = s.set_read_timeout(d);
        }
    }
}

/// Wait for a condition the bridge reaches asynchronously, instead of
/// sleeping a fixed time and hoping the scheduler cooperated.
fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

/// One request/response round trip.
fn call(write: &mut Stream, read: &mut BufReader<Stream>, msg: Value) -> Value {
    write
        .write_all(format!("{msg}\n").as_bytes())
        .expect("write request");
    write.flush().expect("flush request");
    let mut line = String::new();
    read.read_line(&mut line).expect("read response");
    serde_json::from_str(&line).expect("json response")
}

/// `initialize` — what makes a conn the session transport.
fn initialize(write: &mut Stream, read: &mut BufReader<Stream>, id: Value) -> Value {
    call(
        write,
        read,
        json!({"jsonrpc": "2.0", "id": id, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "claude-code", "version": "2.1.309"}}}),
    )
}

#[test]
fn a_reconnected_transport_is_served() {
    let b = bridge();

    // First transport: it speaks, so it claims the write half — then the
    // client drops it, the way a mount timeout or a respawn does.
    let (mut w1, mut r1) = client(b.endpoint());
    let first = initialize(&mut w1, &mut r1, json!(1));
    assert_eq!(first["id"], json!(1), "the first transport is served");
    drop(r1);
    drop(w1);

    // Let the accept loop and the read loop see the close.
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        !b.is_dead(),
        "a closed conn is a reconnect window, not the end of the mount"
    );

    // The respawned shim connects to the same endpoint and has to be
    // served — this is the connection a real mount failure never got.
    let (mut w2, mut r2) = client(b.endpoint());
    let second = initialize(&mut w2, &mut r2, json!(2));
    assert_eq!(second["id"], json!(2), "the reopened transport is served");
}

#[test]
fn a_probe_conn_that_never_speaks_does_not_arm_the_window() {
    let b = bridge();
    let (mut w, mut r) = client(b.endpoint());
    let probe = call(
        &mut w,
        &mut r,
        json!({"jsonrpc": "2.0", "id": "p1", "method": "server/discover"}),
    );
    assert_eq!(probe["id"], json!("p1"), "discover is answered");
    drop(r);
    drop(w);
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        !b.is_dead(),
        "probe conns are disposable and never end the mount"
    );
}

#[test]
fn the_window_is_what_ends_a_mount_that_never_comes_back() {
    let b = bridge();
    let (mut w, mut r) = client(b.endpoint());
    initialize(&mut w, &mut r, json!(1));
    drop(r);
    drop(w);
    wait_for("the session conn to be reported gone", || {
        b.shared
            .state
            .lock()
            .map(|st| st.session_gone_since.is_some())
            .unwrap_or(false)
    });

    let st = b.shared.state.lock().unwrap();
    assert!(
        !st.lost_within(RECONNECT_GRACE),
        "inside the grace the mount is still reconnectable"
    );
    assert!(
        st.lost_within(Duration::ZERO),
        "past the grace a silent mount is a lost mount"
    );
}

#[test]
fn closing_the_bridge_ends_it_at_once() {
    let mut b = bridge();
    let (mut w, mut r) = client(b.endpoint());
    initialize(&mut w, &mut r, json!(1));
    b.close();
    assert!(b.is_dead(), "an explicit close is not a reconnect window");
}

#[test]
fn a_conn_keeps_serving_after_its_first_request() {
    // The accept loop's listener is nonblocking; on Windows the accepted
    // socket inherits that, and the first pump read after the buffered
    // request fails WouldBlock — the conn dies right after answering
    // `server/discover`, so claude never gets past its probe and the mount
    // never lands. A second request on the same conn is the tripwire.
    let b = bridge();
    let (mut w, mut r) = client(b.endpoint());
    let probe = call(
        &mut w,
        &mut r,
        json!({"jsonrpc": "2.0", "id": "p1", "method": "server/discover"}),
    );
    assert_eq!(probe["id"], json!("p1"), "discover is answered");
    let second = call(
        &mut w,
        &mut r,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    );
    assert_eq!(
        second["id"],
        json!(2),
        "the same conn keeps reading after its first request"
    );
}

#[test]
fn a_conn_superseded_by_a_newer_transport_cannot_arm_the_window() {
    // Claude's respawn can claim the transport before the old conn's EOF is
    // processed. That conn's death then says nothing about the mount: the
    // live transport is the new one.
    let b = bridge();
    let (mut wa, mut ra) = client(b.endpoint());
    initialize(&mut wa, &mut ra, json!(1));

    let (mut wb, mut rb) = client(b.endpoint());
    initialize(&mut wb, &mut rb, json!(2));

    // The live transport still serves.
    let served = call(
        &mut wb,
        &mut rb,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    );
    assert_eq!(
        served["id"],
        json!(3),
        "the newer transport owns the session"
    );

    // Now the superseded conn's read loop ends and reports the loss, exactly
    // as `serve_conn` does on EOF. Driven directly so the assertion cannot
    // race the scheduler.
    drop(ra);
    drop(wa);
    b.shared.conn_lost(Some(1));

    let st = b.shared.state.lock().unwrap();
    assert!(
        !st.lost_within(Duration::ZERO),
        "the conn that died was not the transport any more"
    );
}
