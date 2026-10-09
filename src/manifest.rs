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
pub fn manifest() -> gray_plugin::Manifest {
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

/// `command/run` result for a claimed command, `None` for names this
/// sidecar doesn't answer. A bare `/claude` answers nothing (`{}`) so the
/// host falls back to the provider-login shortcut — connect → model
/// picker on Claude's rows, i.e. "switch to Claude" — while
/// `/claude tools …` owns the tool allowlist (see [`crate::settings`]),
/// answered as `{"text": …}`.
pub fn run_command(name: &str, argv: &[String]) -> Option<serde_json::Value> {
    if name == CLAUDE_COMMAND {
        if argv.is_empty() {
            return Some(serde_json::json!({}));
        }
        return Some(serde_json::json!({
            "text": crate::settings::command(argv),
        }));
    }
    None
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
