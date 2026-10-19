//! Setup-time probes of the user's Claude CLI: binary presence and login
//! state. Port of the Hermes DirectSDK `directsdk_setup.py` probes
//! (NousResearch/hermes-plugin-claude-subscription-directsdk, MIT).
//!
//! Both probes are offline with respect to Anthropic: `auth status` reads
//! the local credential store. Anything unexpected degrades to `Unknown`
//! rather than failing setup.

use std::path::PathBuf;

/// `claude` is missing (or not on PATH): install hint, never a spawn panic.
pub const INSTALL_HINT: &str = "`claude` not found on PATH. Install it with \
    `npm install -g @anthropic-ai/claude-code`, then `claude auth login`. \
    Override the binary with CLAUDE_SUB_COMMAND=/path/to/claude.";
/// `claude` is present but has no usable login here.
pub const LOGIN_HINT: &str =
    "Claude Code is installed but not logged in. Run `claude auth login` and retry.";

/// Login state. `Unknown` never blocks a turn.
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
///
/// On Windows the PATH entries are shim files, not spawn targets: the bare
/// `claude` is a POSIX script (`CreateProcess` → os error 193), and
/// `claude.cmd` routes through `cmd.exe`, which cannot carry arguments
/// containing newlines (Rust refuses them, `os error 1231`-class
/// InvalidInput). The npm `.cmd` shim ends by exec'ing the real PE
/// (`node_modules\@anthropic-ai\claude-code\bin\claude.exe`), so we resolve
/// through the shim to the `.exe` it wraps. A `.cmd` we cannot unwrap is
/// still returned as a last resort — simple probes like `auth status`
/// carry no newlines and survive cmd.exe.
pub fn resolve_command() -> Option<String> {
    if let Some(v) = env_override() {
        return Some(v);
    }
    let path = std::env::var_os("PATH")?;
    let names: &[&str] = if cfg!(windows) {
        &["claude.exe", "claude.cmd", "claude"]
    } else {
        &["claude", "claude.exe", "claude.cmd"]
    };
    let mut cmd_fallback: Option<PathBuf> = None;
    for dir in std::env::split_paths(&path) {
        for name in names {
            let p: PathBuf = dir.join(name);
            if !p.is_file() {
                continue;
            }
            #[cfg(windows)]
            {
                let lower = name.to_ascii_lowercase();
                if lower.ends_with(".cmd") || lower.ends_with(".bat") {
                    if let Some(real) = unwrap_cmd_shim(&p) {
                        return Some(real.to_string_lossy().into_owned());
                    }
                    cmd_fallback.get_or_insert(p);
                    continue;
                }
                if !lower.ends_with(".exe") && !is_pe_binary(&p) {
                    continue; // POSIX shim — unspawnable on Windows
                }
            }
            return Some(p.to_string_lossy().into_owned());
        }
    }
    cmd_fallback.map(|p| p.to_string_lossy().into_owned())
}

/// First bytes are `MZ`: the file is a PE executable and `CreateProcess`
/// will run it regardless of its (missing) extension.
#[cfg(windows)]
fn is_pe_binary(p: &PathBuf) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 2];
    std::fs::File::open(p)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok_and(|_| magic == *b"MZ")
}

/// Resolve the executable an npm-style `.cmd`/`.bat` shim execs. Such
/// shims end in `"<path>"  %*` where `<path>` may embed `%~dp0`/`%dp0%`
/// (the shim's own directory) or `%ENV%`. Returns the last quoted token
/// that resolves to a real file.
#[cfg(windows)]
fn unwrap_cmd_shim(shim: &PathBuf) -> Option<PathBuf> {
    let text = std::fs::read_to_string(shim).ok()?;
    let dir = shim.parent()?.to_path_buf();
    let mut found = None;
    let mut rest = text.as_str();
    while let Some(start) = rest.find('"') {
        let tail = &rest[start + 1..];
        let Some(end) = tail.find('"') else { break };
        let token = &tail[..end];
        rest = &tail[end + 1..];
        if !(token.contains('\\') || token.contains('/')) {
            continue;
        }
        // %~dp0/%dp0% expand to the shim's own directory (trailing slash
        // included) — substitute before generic %VAR% expansion so `dp0`
        // is never looked up in env.
        let dir_str = format!("{}{}", dir.display(), std::path::MAIN_SEPARATOR);
        let mut expanded = token
            .replace("%~dp0", &dir_str)
            .replace("%dp0%", &dir_str)
            .replace("%~dps0", &dir_str);
        let mut out = String::with_capacity(expanded.len());
        while let Some(off) = expanded.find('%') {
            out.push_str(&expanded[..off]);
            let tail2 = &expanded[off + 1..];
            match tail2.find('%') {
                Some(e) => {
                    let var = &tail2[..e];
                    if let Ok(v) = std::env::var(var) {
                        out.push_str(&v);
                    } else {
                        out.push('%');
                        out.push_str(var);
                        out.push('%');
                    }
                    expanded = tail2[e + 1..].to_string();
                }
                None => {
                    out.push_str(&expanded);
                    expanded.clear();
                }
            }
        }
        out.push_str(&expanded);
        let candidate = PathBuf::from(&out);
        let candidate = if candidate.is_absolute() || out.starts_with("\\\\") {
            candidate
        } else {
            // Root-relative remainder like `\node_modules\…` (a prefix-less
            // `\` path would survive join() only as drive-rooted) — strip
            // one leading separator so it lands under the shim directory.
            dir.join(PathBuf::from(out.trim_start_matches(['\\', '/'])))
        };
        if candidate.is_file() {
            found = Some(candidate);
        }
    }
    found
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
