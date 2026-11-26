//! The exact protocol-1.2 Claude subscription provider declaration.

use gray_plugin::{
    AuthMethodDecl, PROVIDER_CREDENTIALS, ProviderDecl, ProviderHeaderDecl,
    ProviderRequestPolicyDecl, ProviderTransportDecl,
};

pub const PLUGIN_NAME: &str = "claude-sub";
pub const PLUGIN_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PROVIDER_ID: &str = "claude-subscription";
pub const AUTH_METHOD_ID: &str = "claude-login";
/// The operator command: `/claude tools …` owns the tool allowlist.
pub const CLAUDE_COMMAND: &str = "/claude";

/// A protocol-1.2 manifest value. Credentials stay with the user's own
/// `claude auth login`: the `external-login` method performs no OAuth, it
/// only names the login the chat path probes before spawning.
const DEFAULT_CLAUDE_COMMANDS: &[(&str, &str)] = &[
    ("compact", "Compact conversation history in Claude Code"),
    ("review", "Review changes or a pull request"),
    ("plan", "Plan carefully before making changes"),
    ("init", "Initialize project CLAUDE.md guide"),
    (
        "security-review",
        "Run a security review of pending changes",
    ),
    ("pr-comments", "Inspect pull request review comments"),
];

const OPERATOR_COMPLETIONS: &[(&str, &str)] = &[
    ("commands", "list Claude Code slash commands"),
    ("tools", "show or set upstream harness tool allowlist"),
    ("tools all", "allow every harness tool the host sends"),
    ("tools default", "bash, file and subagent tools"),
    ("tools bash", "allow bash only"),
    ("account", "manage Claude logins"),
    ("account list", "list Claude accounts and health"),
    ("account status", "show which Claude account serves now"),
    ("account usage", "show Claude account quota usage"),
    ("account use", "pin a Claude account"),
    ("account remove", "remove a Claude account"),
    ("account strategy", "set account selection strategy"),
    ("account reset", "clear cooldowns on Claude accounts"),
];

fn commands_file() -> Option<std::path::PathBuf> {
    gray_core::paths::gray_home().map(|h| h.join("claude-sub").join("commands.json"))
}

pub fn cached_commands() -> Vec<(String, String)> {
    if let Some(path) = commands_file()
        && let Ok(text) = std::fs::read_to_string(path)
        && let Ok(names) = serde_json::from_str::<Vec<String>>(&text)
        && !names.is_empty()
    {
        return names
            .into_iter()
            .filter_map(|raw| {
                let name = raw.trim().trim_start_matches('/').to_string();
                if name.is_empty() {
                    return None;
                }
                let desc = DEFAULT_CLAUDE_COMMANDS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, d)| (*d).to_string())
                    .unwrap_or_else(|| "Claude Code slash command".to_string());
                Some((name, desc))
            })
            .collect();
    }
    DEFAULT_CLAUDE_COMMANDS
        .iter()
        .map(|(n, d)| ((*n).to_string(), (*d).to_string()))
        .collect()
}

pub fn save_commands(names: &[String]) {
    let cleaned: Vec<String> = names
        .iter()
        .map(|s| s.trim().trim_start_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if cleaned.is_empty() {
        return;
    }
    if let Some(path) = commands_file() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec(&cleaned) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }
    sync_completion_file(&cached_commands());
}

pub fn sync_completion_file(commands: &[(String, String)]) {
    let Some(home) = gray_core::paths::gray_home() else {
        return;
    };
    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for (arg, desc) in OPERATOR_COMPLETIONS {
        seen.push((*arg).to_string());
        entries.push(serde_json::json!({"arg": arg, "description": desc}));
    }
    for (name, desc) in commands {
        let arg = name.trim_start_matches('/').trim();
        if arg.is_empty() || seen.iter().any(|s| s == arg) {
            continue;
        }
        seen.push(arg.to_string());
        entries.push(serde_json::json!({"arg": arg, "description": desc}));
    }
    let Ok(bytes) = serde_json::to_vec(&serde_json::json!({"completion": entries})) else {
        return;
    };
    let plugins_dir = home.join("plugins");
    let _ = std::fs::create_dir_all(&plugins_dir);
    for stem in ["claude", PLUGIN_NAME] {
        let path = plugins_dir.join(format!("{stem}-completion.json"));
        let tmp = plugins_dir.join(format!("{stem}-completion.json.tmp"));
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

fn strip_claude_prefix(raw: &str) -> Option<&str> {
    let trimmed = raw.trim_start();
    let rest = trimmed
        .strip_prefix(CLAUDE_COMMAND)
        .or_else(|| trimmed.strip_prefix("claude"))?;
    let next = rest.chars().next()?;
    if !next.is_whitespace() {
        return None;
    }
    let body = rest.trim_start();
    (!body.is_empty()).then_some(body)
}

fn format_commands(commands: &[(String, String)]) -> String {
    if commands.is_empty() {
        return "no Claude Code slash commands discovered yet".to_string();
    }
    let mut lines =
        vec!["Claude Code slash commands (run as `/claude <command> [args…]`):".to_string()];
    for (name, desc) in commands {
        if desc.is_empty() {
            lines.push(format!("  /claude {name}"));
        } else {
            lines.push(format!("  /claude {name:<20} {desc}"));
        }
    }
    lines.join("\n")
}

pub fn manifest() -> gray_plugin::Manifest {
    sync_completion_file(&cached_commands());
    gray_plugin::Manifest {
        name: PLUGIN_NAME.to_string(),
        version: PLUGIN_VERSION.to_string(),
        tools: Vec::new(),
        commands: vec![CLAUDE_COMMAND.to_string()],
        hooks: Vec::new(),
        protocol: Some("1.2".to_string()),
        subcommands: Vec::new(),
        capabilities: vec![
            PROVIDER_CREDENTIALS.to_string(),
            gray_plugin::capabilities::HOST_SAY.to_string(),
        ],
        providers: vec![provider()],
        provider_errors: Vec::new(),
    }
}

pub fn run_command(name: &str, argv: &[String]) -> Option<serde_json::Value> {
    run_command_with_context(name, argv, None, None)
}

pub fn run_command_with_context(
    name: &str,
    argv: &[String],
    raw: Option<&str>,
    provider_id: Option<&str>,
) -> Option<serde_json::Value> {
    if name != CLAUDE_COMMAND {
        return None;
    }
    if argv.is_empty() {
        return Some(serde_json::json!({}));
    }
    let commands = cached_commands();
    match argv[0].as_str() {
        "account" | "tools" => Some(serde_json::json!({
            "text": crate::settings::command(argv),
        })),
        "commands" => Some(serde_json::json!({
            "text": format_commands(&commands),
        })),
        other => {
            let cmd = other.trim_start_matches('/');
            if commands.iter().any(|(c, _)| c == cmd) {
                if provider_id.is_some_and(|p| {
                    p != PROVIDER_ID && p.rsplit_once(':').map(|(_, id)| id) != Some(PROVIDER_ID)
                }) {
                    return Some(serde_json::json!({
                        "text": format!(
                            "switch to a Claude model first (`/claude`) to run /{cmd}"
                        ),
                    }));
                }
                let prompt = match raw.and_then(strip_claude_prefix) {
                    Some(body) if body.starts_with('/') => body.to_string(),
                    Some(body) => format!("/{body}"),
                    None if argv.len() > 1 => format!("/{cmd} {}", argv[1..].join(" ")),
                    None => format!("/{cmd}"),
                };
                return Some(serde_json::json!({
                    "prompt": prompt,
                }));
            }
            Some(serde_json::json!({
                "text": crate::settings::command(argv),
            }))
        }
    }
}

/// Claude subscription provider. Requests go to the loopback relay the
/// sidecar opens per chat turn (see `chat`); the host adds bearer, policy,
/// and every declared header from this declaration. The bearer is a
/// per-turn relay token minted by the sidecar, never the user's OAuth token.
pub fn provider() -> ProviderDecl {
    ProviderDecl {
        id: PROVIDER_ID.to_string(),
        name: "Claude subscription".to_string(),
        transport: ProviderTransportDecl {
            kind: "openai-responses".to_string(),
            base_url: "https://127.0.0.1:1/"
                .parse()
                .expect("loopback placeholder"),
            authorization: gray_plugin::ProviderAuthorizationDecl {
                kind: "bearer".to_string(),
                secret_name: "relay_token".to_string(),
            },
            request: ProviderRequestPolicyDecl {
                prompt_cache_key: false,
                // Relay sidecar spawning per-turn `claude` children: the
                // host must never verbatim-replay its requests.
                warm_replay: false,
                store: false,
                include_reasoning_encrypted: true,
                previous_response_id: false,
                tool_choice: Some("auto".to_string()),
                parallel_tool_calls: Some(true),
                text_verbosity: Some("low".to_string()),
                // Claude's native prompt cache lives ~1h, not the host's
                // 5-minute default — the warmth timer and cold-cache
                // notices must know, or they flag healthy idles as misses.
                cache_ttl_secs: Some(3600),
            },
            headers: vec![ProviderHeaderDecl {
                name: "session-id".to_string(),
                value: None,
                source: Some(gray_plugin::ProviderHeaderSourceDecl::SessionId),
                required: true,
            }],
        },
        auth_methods: vec![AuthMethodDecl {
            id: AUTH_METHOD_ID.to_string(),
            name: "Claude Code login".to_string(),
            kind: "api_key".to_string(),
            operations: vec!["models".to_string(), "chat".to_string()],
        }],
    }
}

#[path = "manifest_tests.rs"]
#[cfg(test)]
mod tests;
