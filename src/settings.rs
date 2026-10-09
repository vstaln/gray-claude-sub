//! The host-tool allowlist the `gray` MCP server advertises upstream.
//!
//! Default is bash-only: a harness may carry far more (search, discord,
//! device tools, …) but a Claude turn sees — and may call — only `bash`.
//! `/claude tools …` in the REPL (the sidecar's `command/run`) and
//! `claude-sub tools …` in a shell are the same operation on the same
//! file, `<gray home>/claude-sub/tools.json`. `chat::prepare_turn`
//! re-reads it on every request, so a mid-session toggle applies to the
//! next turn with no respawn, and the file is safe to hand-edit —
//! missing or corrupt reads fail closed to bash.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::{Value, json};

/// `*` in the allowlist stands for every function tool the host sends.
const ALL: &str = "*";
/// The default (and reset) surface.
const BASH: &str = "bash";

/// Which host tools upstream may use. `prepare_turn` advertises only
/// `allows`-passing tools to the child's `gray` MCP server, and the fold
/// rejects a `tool_use` for anything else — so the host's inventory
/// error, not a silent execution, answers a call the policy doesn't
/// cover.
pub struct ToolPolicy {
    allow: BTreeSet<String>,
}

impl Default for ToolPolicy {
    /// Missing/corrupt file lands here: bash only.
    fn default() -> Self {
        Self {
            allow: BTreeSet::from([BASH.to_string()]),
        }
    }
}

impl ToolPolicy {
    /// `name` may run upstream: it's listed, or the policy is `all`.
    pub fn allows(&self, name: &str) -> bool {
        self.allow.contains(ALL) || self.allow.contains(name)
    }

    /// The persisted policy, or the bash-only default. Never errors: a
    /// turn must never fail on a settings read.
    pub fn load() -> Self {
        let Some(path) = file_path() else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            return Self::default();
        };
        let allow: BTreeSet<String> = value
            .get("allow")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .filter(|name| valid_name(name) || name == ALL)
                    .collect()
            })
            .unwrap_or_default();
        // An empty stored list is not a policy — keep the default so a
        // truncated write can't silently disarm the inventory.
        if allow.is_empty() {
            Self::default()
        } else {
            Self { allow }
        }
    }

    fn save(&self) -> Result<String, String> {
        let path = file_path().ok_or("cannot resolve GRAY_HOME")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
        }
        let allow: Vec<&String> = self.allow.iter().collect();
        std::fs::write(&path, json!({"allow": allow}).to_string()).map_err(|e| format!("{e}"))?;
        Ok(path.to_string_lossy().into_owned())
    }
}

/// `<gray home>/claude-sub/tools.json` — the same GRAY_HOME rule the host
/// uses (`gray_core::paths`), so a repo-local GRAY_HOME scopes the policy.
fn file_path() -> Option<PathBuf> {
    gray_core::paths::gray_home().map(|h| h.join("claude-sub").join("tools.json"))
}

/// Tool names are ASCII identifiers the host's schema already constrains;
/// the policy file honors the same bound so junk can never name a tool.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 50
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// The `/claude` command body: `tools` runs the policy command, anything
/// else prints usage plus the current policy. A bare `/claude` never
/// reaches this — `manifest::run_command` leaves it unanswered so the
/// host's provider-login shortcut (connect → model picker) takes it.
pub fn command(argv: &[String]) -> String {
    match argv.first().map(String::as_str) {
        Some("tools") => tools_command(&argv[1..]),
        _ => format!("{USAGE}\n\n{}", status_line(&ToolPolicy::load())),
    }
}

const USAGE: &str = "\
/claude — switch the session to Claude (connect → model picker).
/claude tools — which harness tools the upstream Claude model may call.
  /claude tools          show the current policy
  /claude tools all      allow every harness tool the host sends
  /claude tools bash     bash only (the default)
  /claude tools +name    also allow one tool, e.g. /claude tools +discord_send
  /claude tools -name    remove one
Same from a shell: `gray claude-sub tools …`.";

fn status_line(policy: &ToolPolicy) -> String {
    if policy.allow.contains(ALL) {
        "upstream tools: all harness tools".to_string()
    } else {
        format!(
            "upstream tools: {}",
            policy.allow.iter().cloned().collect::<Vec<_>>().join(", ")
        )
    }
}

/// `tools` with no args prints the policy; each arg is `all` (allow
/// everything), `bash` (reset to default), `+name`/`-name` (edit the
/// list) or a bare name (same as `+name`). Processed left to right; the
/// first invalid arg stops with its message, nothing written.
pub fn tools_command(args: &[String]) -> String {
    if args.is_empty() {
        return format!("{}\n\n{USAGE}", status_line(&ToolPolicy::load()));
    }
    let mut policy = ToolPolicy::load();
    for arg in args {
        match arg.as_str() {
            "all" => policy.allow = BTreeSet::from([ALL.to_string()]),
            "bash" => policy.allow = BTreeSet::from([BASH.to_string()]),
            _ => {
                let (add, name) = match arg.strip_prefix('-') {
                    Some(rest) => (false, rest),
                    None => (true, arg.strip_prefix('+').unwrap_or(arg)),
                };
                if policy.allow.contains(ALL) {
                    return "policy is `all`; reset with `/claude tools bash` before \
                            editing the list"
                        .to_string();
                }
                if !valid_name(name) {
                    return format!("bad tool name: {arg}");
                }
                if add {
                    policy.allow.insert(name.to_string());
                } else {
                    policy.allow.remove(name);
                }
            }
        }
    }
    if policy.allow.is_empty() {
        return "refusing to write an empty tool list — `/claude tools bash` resets".to_string();
    }
    match policy.save() {
        Ok(path) => format!("{}\n(saved to {path})", status_line(&policy)),
        Err(e) => format!("could not save policy: {e}"),
    }
}
