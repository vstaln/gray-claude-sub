# gray-claude-sub

> **Claude Pro/Max subscription model-provider sidecar plugin for the [gray](https://github.com/vstaln/gray) agent harness.**

`gray-claude-sub` drives the official `claude` CLI fully inert (`--tools ''`, no MCP servers, no slash commands, `dontAsk`, no session persistence) as a request-scoped model provider. Gray owns the agent loop, tools, approvals, and compaction — Claude only answers.

---

## Highlights

* **Zero Token / Credential Leaks**: Uses your existing official Claude Code login (`claude auth login`). Never handles, stores, or logs API keys or tokens.
* **Single-Request Admission Relay**: Drives an internal request-scoped loopback admission relay that enforces exactly one upstream request per turn and absorbs redundant recovery attempts.
* **Native Assistant Replay Carrier**: Assistant messages round-trip in signed reasoning replay carriers (`NATIVE_ITEM_ID`) so future turns restore byte-identical upstream frames without prompt divergence.
* **Pinned Model Catalog**: Exact context windows without guessing (`opus`, `sonnet`, `fable`: 1,000,000 tokens; `haiku`: 200,000 tokens).

---

## Requirements

| Requirement | Details |
|---|---|
| **Claude Code CLI** | Official `@anthropic-ai/claude-code` CLI installed and on `$PATH`. |
| **Authentication** | Logged in via `claude auth login` on your Claude Pro or Max subscription. |
| **Gray Harness** | [Gray](https://github.com/vstaln/gray) agent framework (Protocol v1.1 or v1.2 sidecar support). |

The plugin validates dependencies at every seam:
* If `claude` is not found, it reports an informative installation hint rather than a process crash.
* If custom `ANTHROPIC_BASE_URL` or conflicting cloud backend variables are set, it fails closed to prevent leaking subscription authentication.

---

## Installation & Build

### Building from Source

```sh
git clone https://github.com/vstaln/gray-claude-sub.git
cd gray-claude-sub
cargo build --release
```

The resulting binary is located at `./target/release/claude-sub`.

### Installing in Gray

Point Gray to the binary or install it directly into `$GRAY_HOME/plugins` (default `~/.gray/plugins`):

```sh
# Copy binary to your Gray plugin directory
mkdir -p ~/.gray/plugins/claude-sub
cp target/release/claude-sub ~/.gray/plugins/claude-sub/claude-sub

# Or install via gray plugin management
gray install plugin claude-sub
```

Authenticate with Claude Code if you haven't already:

```sh
claude auth login
```

---

## Model Selection

Once registered, Claude subscription models appear under the `claude-sub/` prefix:

```sh
# Select model in interactive mode
/model claude-sub/sonnet
/model claude-sub/opus
/model claude-sub/haiku
/model claude-sub/fable

# Or launch directly with Gray CLI
gray -m claude-sub/sonnet -p "Review this codebase"
```

### Pinned Model Catalog

| Model ID | Target Model | Context Window |
|---|---|---|
| `claude-sub/sonnet` | Claude 3.7 Sonnet | 1,000,000 tokens |
| `claude-sub/opus` | Claude 3.5 Opus | 1,000,000 tokens |
| `claude-sub/haiku` | Claude 3.5 Haiku | 200,000 tokens |
| `claude-sub/fable` | Claude 3.7 Sonnet (Fable) | 1,000,000 tokens |

---

## Wire Protocol & Architecture

```
┌──────────────────┐               ┌──────────────────┐               ┌───────────────────┐
│                  │  stdio NDJSON │                  │  stdin/stdout │                   │
│   Gray Harness   │ ────────────> │ gray-claude-sub  │ ────────────> │  claude CLI       │
│   (owns tools &  │ <──────────── │ (admission relay │ <──────────── │  (inert runner)   │
│   approvals)     │ Protocol 1.2  │ & frame encoder) │  stream-json  │                   │
└──────────────────┘               └──────────────────┘               └───────────────────┘
```

The sidecar communicates over standard I/O using newline-delimited JSON (NDJSON):
* `plugin/manifest`: Declares provider capabilities and supported model IDs.
* `provider/models`: Returns the pinned catalog.
* `provider/chat`: Parks the request intent and initiates a relayed turn.
* `provider/auth/*`: Reports external CLI authentication state.

---

## Repository Structure

```text
├── src/
│   ├── main.rs              # Protocol-1.2 sidecar entry point
│   ├── chat.rs              # Request translation, streaming parser, and CLI runner
│   ├── relay.rs             # Single-admission loopback relay proxy
│   ├── catalog.rs           # Pinned model catalog definitions
│   ├── models.rs            # Provider model metadata
│   ├── manifest.rs          # Plugin protocol manifest
│   ├── setup.rs             # CLI dependency discovery and verification
└── reference/               # Complete upstream Hermes DirectSDK test & eval suite
```

---

## License

MIT License — Copyright (c) 2026 Vstalin Grady
