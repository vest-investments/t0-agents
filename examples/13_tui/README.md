# TUI Agent — Typer + Rich CLI (Milestone M6)

Interactive streaming REPL for an MCP-backed agent, powered by the new
`[cli]` extra. Demonstrates:

- **Token-level streaming** — `Agent.stream()` deltas rendered inside a
  Rich `Live` panel that refreshes 20×/second.
- **Tool-call spinners** — `🔧 add(a=2,b=3)` turns into `✓ add(a=2,b=3)`
  (or `❌` on error) as the tool completes.
- **Rich-rendered elicitation** — `set_reminder` calls `ctx.elicit()`
  with a structured schema; the Phase F `rich_elicitation_handler`
  renders a per-field prompt (boolean `Confirm`, `enum` choices,
  optional description).
- **Episode continuity** — the REPL threads the last `episode_id` into
  each new turn, so memory / TensorZero's episode state stays coherent
  across the whole session.

## Prerequisites

```bash
docker compose up -d           # boots gateway + clickhouse from repo root
export GROQ_API_KEY=gsk_...    # or whichever provider the function uses
uv sync --extra cli            # installs typer + rich + uvicorn + fastapi
```

## Run the TUI

```bash
uv run t0-agents run examples.13_tui.agent:assistant
```

You'll see a cyan panel labelled `tui-assistant`; type at the `You:`
prompt. Sample session:

```
You: What is 2 + 3?
╭─ tui-assistant ─────────────────────────────╮
│ 🔧 add({"a":2,"b":3})                       │
│ ✓ add({"a":2,"b":3})                        │
│                                             │
│ 2 + 3 is 5.                                 │
╰─────────────────────────────────────────────╯

You: Remind me tomorrow at 9 to water the plants.
╭─ tui-assistant ─────────────────────────────╮
│ 🔧 set_reminder({"when":"tomorrow 9","...}) │
╰─────────────────────────────────────────────╯
Reminder at tomorrow 9: water the plants
confirm (y/n):                      ← Rich prompt
channel [email/sms/push] (email):
note:
╭─ tui-assistant ─────────────────────────────╮
│ ✓ set_reminder({"when":"tomorrow 9","...}) │
│                                             │
│ Scheduled reminder for 'tomorrow 9' via…    │
╰─────────────────────────────────────────────╯
```

Quit with `Ctrl-D` or `Ctrl-C` at the `You:` prompt.

### `--verbose` — show reasoning text

Enable `-v` (or `--verbose`) to render `ThoughtEvent` / `ThoughtDeltaEvent`
inline. Useful with reasoning models; ignore for chat-only models.

```bash
uv run t0-agents run examples.13_tui.agent:assistant --verbose
```

## Run as a dev HTTP server

```bash
uv run t0-agents serve examples.13_tui.agent:assistant \
    --port 8080 --deployment tui-demo
```

This wraps the same factory in
[`t0_agents.serve.create_app(share_client=True)`](../../docs/serving.md)
and runs uvicorn against the Phase D non-streaming routes (`POST /chat`,
`/feedback`, `/health`, `/info`). Streaming is off by default because
this factory wires `rich_elicitation_handler` into a shared `Client` —
that handler talks to the local terminal, so it does not make sense over
HTTP, and the WS streaming route additionally needs a per-connection
handler (FastMCP Q3) which shared-Client mode cannot give it.

For full streaming + browser elicitation, see
[`examples/12_streaming_http/`](../12_streaming_http/), which wires
`client_factory` + `elicitation_client_factory` into `create_app`
directly.

## Inspect the config

```bash
uv run t0-agents info examples.13_tui.agent:assistant
```

Prints a table of every `AgentConfig` field the factory wires up. The
factory context is entered just long enough to read `agent.config`, so
your gateway connection opens and closes in under a second.

## Layout

```
examples/13_tui/
├── README.md          ← this file
├── agent.py           ← @asynccontextmanager factory consumed by the CLI
└── server.py          ← FastMCP server: add, set_reminder
```

## LLM caveat

Smaller models (`gpt-oss-20b`, Llama-3-8B, etc.) occasionally ask "are
you sure?" in chat before invoking `set_reminder`. The elicitation modal
still appears, just one turn later. Larger models consistently call the
tool directly. See
[`examples/12_streaming_http/README.md`](../12_streaming_http/README.md)
for the long-form discussion.
