//! Setup-time probes of the user's Claude CLI: binary presence and login
//! state. Port of the Hermes DirectSDK `directsdk_setup.py` probes
//! (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT).
//!
//! Both probes are offline with respect to Anthropic: `auth status` reads
//! the local credential store. Anything unexpected degrades to the pinned
//! catalog rather than failing setup.

use std::path::PathBuf;

/// `claude` is missing (or not on PATH): install hint, never a spawn panic.
pub const INSTALL_HINT: &str = "`claude` not found on PATH. Install it with \
    `npm install -g @anthropic-ai/claude-code`, then `claude auth login`. \
    Override the binary with CLAUDE_SUB_COMMAND=/path/to/claude.";
/// `claude` is present but has no usable login here.
pub const LOGIN_HINT: &str =
    "Claude Code is installed but not logged in. Run `claude auth login` and retry.";

/// Login state. `Unknown` degrades to the pinned catalog, never to a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginState {
    LoggedIn,
    LoggedOut,
    Unknown,
}

/// Probe login state with the CLI's own offline `auth status`.
///
/// `auth status` reads the local credential store (no network, no browser,
/// no window) and reports `{"loggedIn": …}`: exit 0 parses the verdict,
/// anything unexpected is `Unknown`. `CLAUDE_SUB_PROBE_TIMEOUT_SECS`
/// overrides the default 15s (tests use 1s…5s).
pub fn probe_login() -> LoginState {
    let binary = match resolve_command() {
        Some(b) => b,
        None => return LoginState::Unknown,
    };
    if conflicting_env().is_some() {
        return LoginState::Unknown;
    }
    let timeout_secs: u64 = std::env::var("CLAUDE_SUB_PROBE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    let mut child = match std::process::Command::new(&binary)
        .args(["auth", "status"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .envs(child_env())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return LoginState::Unknown,
    };
    let status = wait_timeout(&mut child, std::time::Duration::from_secs(timeout_secs));
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
        return LoginState::Unknown;
    }
    match status {
        Some(Ok(s)) if s.success() => {
            let mut out = String::new();
            if let Some(mut stdout) = child.stdout.take()
                && std::io::Read::read_to_string(&mut stdout, &mut out).is_ok()
                && let Ok(value) = serde_json::from_str::<serde_json::Value>(&out)
                && value.get("loggedIn").and_then(serde_json::Value::as_bool) == Some(true)
            {
                return LoginState::LoggedIn;
            }
            LoginState::LoggedOut
        }
        Some(Ok(_)) => LoginState::LoggedOut,
        _ => LoginState::Unknown,
    }
}

fn wait_timeout(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Option<std::io::Result<std::process::ExitStatus>> {
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(Ok(status)),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => return Some(Err(e)),
        }
    }
}

fn env_override() -> Option<String> {
    std::env::var("CLAUDE_SUB_COMMAND")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| {
            std::env::var("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND")
                .ok()
                .filter(|v| !v.is_empty())
        })
}

/// Resolve the `claude` binary: explicit override, then PATH.
pub fn resolve_command() -> Option<String> {
    if let Some(v) = env_override() {
        return Some(v);
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["claude", "claude.exe", "claude.cmd"] {
            let p: PathBuf = dir.join(name);
            if p.is_file() {
                return Some(p.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// Fail-closed auth env: a set value would forward the subscription bearer
/// somewhere else, or confuse native backend routing.
pub fn conflicting_env() -> Option<String> {
    for key in [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_FOUNDRY_API_KEY",
    ] {
        if std::env::var_os(key).is_some() {
            return Some(key.to_string());
        }
    }
    for key in [
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ] {
        if let Some(v) = std::env::var_os(key)
            && !matches!(
                v.to_string_lossy().to_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        {
            return Some(key.to_string());
        }
    }
    None
}

/// Child env for every spawn: never inherit a conflicting value, always
/// quiet the CLI's nonessential traffic.
pub fn child_env() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            ![
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_FOUNDRY_API_KEY",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY",
            ]
            .contains(&k.as_str())
        })
        .collect();
    for (k, v) in [
        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
        ("DISABLE_TELEMETRY", "1"),
        ("DISABLE_ERROR_REPORTING", "1"),
    ] {
        if !out.iter().any(|(k2, _)| k2 == k) {
            out.push((k.to_string(), v.to_string()));
        }
    }
    out
}

#[path = "setup_tests.rs"]
#[cfg(test)]
mod tests;
