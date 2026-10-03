# gray-claude-sub

Claude Pro/Max subscription provider sidecar plugin for the [gray](https://github.com/vstaln/gray) agent harness.

Drives the official `claude` CLI fully inert (`--tools ''`, no MCP servers, no slash commands, no session persistence) as a request-scoped model provider. Gray owns tools, approvals, and compaction — Claude only answers.

## Architecture

- **Wire protocol**: Gray Plugin Protocol v1.2 (NDJSON over stdio).
- **Single-request admission relay**: Forwards only the first upstream Messages request and enforces inert tools.
- **Credential security**: Uses your existing `claude auth login` via the official Claude Code CLI. No tokens or API keys are handled or stored by Gray.

## Requirements

- [Claude Code CLI](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (`npm install -g @anthropic-ai/claude-code`)
- Logged in via `claude auth login`
- [Gray](https://github.com/vstaln/gray) harness

## License

MIT
