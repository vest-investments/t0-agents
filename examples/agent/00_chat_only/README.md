# 00 — Toolless chat

The shortest possible agent: one user message, one inference, one
text response.  No MCP server.  No MCP client.  No `list_tools()`
roundtrip.

```
You:    Explain monads in two sentences.
Agent:  A monad is a design pattern that wraps a value in a context
        (like "maybe-failure" or "asynchrony") and chains operations
        on it without leaking the context out.  Each step receives
        the inner value, returns a new wrapped value, and the monad
        threads the context through automatically.
```

**Aha! moment:** `Agent(config, t0=t0)` is the entire stack.  v1.0
made `tool_provider` optional, so chat-only agents don't pay any
MCP allocation cost — the loop runs one inference and returns the
text response.  Compare to [`../01_first_agent/`](../01_first_agent/)
to see what tools add on top of this baseline.

## What it demonstrates

1. The optional `tool_provider` contract — chat-only `Agent`
   construction with **no** `Client` parameter.
2. Zero MCP plumbing visible in `main.py` (no `fastmcp` import,
   no FastMCP server allocated at runtime).
3. The same `agent.invoke(message)` → `Run[T]` → `await`
   protocol the rest of the examples build on.
4. The same `Response`/`Turns`/`Episode ID`/`Inference IDs`
   diagnostic shape — only here `Turns: 1` confirms a single
   inference, no tool calls.

## When to use this template

- You want a pure conversational LLM with TensorZero's episode
  tracking and ClickHouse-backed observability, **but no tools**.
- You want the smallest possible reference for "is my gateway
  reachable?" — boot the compose, run the script, observe a
  non-empty `Response:`.

For agent loops that discover and execute MCP tools, head to
[`../01_first_agent/`](../01_first_agent/) instead.

## Prerequisites

- Python 3.11+
- `uv sync`
- `GROQ_API_KEY=...` in `.env` at the project root.

## Start the gateway

```bash
docker compose -f examples/agent/00_chat_only/docker-compose.yml \
    --env-file .env up -d
```

## Run

```bash
uv run python examples/agent/00_chat_only/main.py
```

## Stop the gateway

```bash
docker compose -f examples/agent/00_chat_only/docker-compose.yml down
```

## What to observe

- The `Response:` line is one natural-language answer — the LLM
  did the whole job in a single inference.
- `Turns: 1` confirms the loop fired once and saw no tool calls.
- `Episode ID` is a TensorZero-assigned UUID.  Pass it to
  `agent.invoke(..., episode_id=...)` next time you want
  conversational continuity, and use it to filter the ClickHouse
  dashboard.

## Related

- [Repository `README.md`](../../README.md) — the same 4-line
  toolless-chat snippet inline.
- [`docs/glow/getting-started.md`](../../docs/glow/getting-started.md)
  — the toolless-chat unlock described from the glow angle.
- [`docs/api-reference.md`](../../docs/api-reference.md) —
  signature reference for `run_agent` /
  `StreamingAgentLoop.stream` with the optional `tool_provider`
  parameter.
