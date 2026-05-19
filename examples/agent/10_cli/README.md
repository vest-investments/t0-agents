# 10 — CLI (Typer + Rich TUI agent)

Interactive streaming REPL for an MCP-backed agent, powered by the new
`[cli]` extra. Demonstrates:

- **Token-level streaming** — `Run.events()` deltas rendered inside a
  Rich `Live` panel that refreshes 20×/second.
- **Tool-call spinners** — `🔧 add(a=2,b=3)` turns into `✓ add(a=2,b=3)`
  (or `❌` on error) as the tool completes.
- **Rich-rendered elicitation** — `set_reminder` calls `ctx.elicit()`
  with a structured schema; the `rich_elicitation_handler` renders a
  per-field prompt (boolean `Confirm`, `enum` choices, optional
  description).
- **Episode continuity** — the REPL threads the last `episode_id` into
  each new turn, so memory / TensorZero's episode state stays coherent
  across the whole session.

## Prerequisites

```bash
docker compose --env-file .env -f examples/agent/01_first_agent/docker-compose.yml up -d
export GROQ_API_KEY=gsk_...    # or whichever provider the function uses
uv sync --extra cli            # installs typer + rich + uvicorn + fastapi
```

This example shares the `agent` chat function with
[`examples/agent/01_first_agent/`](../01_first_agent/) — boot that
gateway.

## Run the TUI

```bash
uv run t0-agents run examples.agent.10_cli.agent:assistant
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
uv run t0-agents run examples.agent.10_cli.agent:assistant --verbose
```

## Run as a dev HTTP server

```bash
uv run t0-agents serve examples.agent.10_cli.agent:assistant \
    --port 8080 --deployment tui-demo
```

This wraps the same factory in
[`t0_agents.serve.create_app(share_client=True)`](../../../docs/serving.md)
and runs uvicorn against the non-streaming HTTP routes (`POST /chat`,
`/feedback`, `/health`, `/info`). Streaming is off by default because
this factory wires `rich_elicitation_handler` into a shared `Client` —
that handler talks to the local terminal, so it does not make sense over
HTTP, and the WS streaming route additionally needs a per-connection
elicitation handler (one handler per `Client`) which shared-Client mode
cannot give it.

For full streaming + browser elicitation, see
[`examples/agent/07_streaming/`](../07_streaming/), which wires
`client_factory` + `elicitation_client_factory` into `create_app`
directly.

## Inspect the config

```bash
uv run t0-agents info examples.agent.10_cli.agent:assistant
```

Prints a table of every `AgentConfig` field the factory wires up. The
factory context is entered just long enough to read `agent.config`, so
your gateway connection opens and closes in under a second.

## Advanced — audit-logging elicitation handler

`advanced.py` ships a second factory, `assistant_with_audit`, that
wires the same assistant with a **custom elicitation handler layered
on top of `make_rich_elicitation_handler`**. Every `ctx.elicit` call
and its outcome (`accept` / `decline` / `cancel`) gets appended to
`tui-audit.jsonl` in the current working directory.

Two patterns from the Advanced tiers of [`docs/cli.md`](../../../docs/cli.md)
are exercised:

- Composing on `make_rich_elicitation_handler()` so the Rich `Live`
  panel still pauses and resumes around every prompt — the wrapper
  only *adds* behaviour, it doesn't replace the rendering.
- Calling `get_active_renderer()` inside the handler so the audit log
  can tag each record with the agent name the live panel is driving,
  without having to plumb the renderer through FastMCP's fixed
  `ElicitationHandler` signature.

```bash
uv run t0-agents run examples.agent.10_cli.advanced:assistant_with_audit
```

Issue a request that triggers elicitation (e.g. `Remind me tomorrow at
9 to water the plants.`), answer the Rich prompts, then inspect the
log:

```bash
cat tui-audit.jsonl
```

Expected shape (one JSON object per line):

```
{"kind":"elicit_request","mode":"form","message":"Reminder at tomorrow 9: water the plants","agent":"audited-assistant","ts":1712345678.1}
{"kind":"elicit_response","action":"accept","has_content":true,"ts":1712345682.4}
```

Delete the file between runs for a clean log, or `tail -f
tui-audit.jsonl` in another terminal to watch entries as they land.

## Layout

```
examples/agent/10_cli/
├── README.md          ← this file
├── agent.py           ← @asynccontextmanager factory consumed by the CLI
├── advanced.py        ← audit-logging elicitation handler + factory
└── server.py          ← FastMCP server: add, set_reminder
```

## LLM caveat

Smaller models (`gpt-oss-20b`, Llama-3-8B, etc.) occasionally ask "are
you sure?" in chat before invoking `set_reminder`. The elicitation modal
still appears, just one turn later. Larger models consistently call the
tool directly. See
[`examples/agent/07_streaming/README.md`](../07_streaming/README.md)
for the long-form discussion.
