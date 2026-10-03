# gray-claude-sub

Claude Pro/Max subscription provider for the [gray](https://github.com/vstaln/gray) agent harness.

Drives the official `claude` CLI fully inert (`--tools ''`, no MCP servers, no slash commands, no session persistence) as a request-scoped model provider. Gray owns tools, approvals, and compaction — Claude only answers.

## Components Included

1. **Protocol 1.2 Sidecar Plugin** (`claude-sub` binary):
   - Runs out-of-process as a Gray sidecar plugin communicating via NDJSON over stdio.
   - Per-turn loopback admission relay enforcing exactly one upstream request per turn.
   - Pinned context windows (`opus`/`sonnet`/`fable`: 1M tokens, `haiku`: 200K tokens).
2. **Direct In-Process Provider** (`claude_sub::direct_provider`):
   - Native Rust implementation of `gray_core::agent::Provider` for embedded usage.
   - Automatic tool name translation (`mcp__gray__*` -> host names).
   - Signed native reasoning item carrier for state reconstruction across turns.
3. **Reference Implementation & Evals** (`reference/`):
   - Upstream Hermes DirectSDK Python implementation (`hermes-plugin-claude-subscription-directsdk`).
   - Admission breakpoint tests, streaming cache verification, and replay test fixtures.

## Requirements

- [Claude Code CLI](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (`npm install -g @anthropic-ai/claude-code`)
- Authenticated via `claude auth login`
- [Gray](https://github.com/vstaln/gray) harness

## License

MIT
