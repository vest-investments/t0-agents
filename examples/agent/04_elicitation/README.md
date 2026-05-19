# 04 — Elicitation (human-in-the-loop)

Human-in-the-loop approval and structured input via MCP Elicitation.

## What it demonstrates

1. A FastMCP server with elicitation-enabled tools (`delete_task` for approval, `create_task` for structured input) alongside a normal tool (`list_tasks`).
2. A CLI-based `elicitation_handler` that prompts the user in the terminal.
3. The handler distinguishes approval from structured input by checking `response_type is None`.
4. `AgentConfig(tool_timeout=None)` to disable the per-tool timeout while waiting for human input.
5. Decline handling — rejecting an approval returns a message the LLM can react to.

## How elicitation works

When a tool calls `ctx.elicit()`, it pauses mid-execution and sends a request back to the client. The client's `elicitation_handler` runs (prompting the human), and the response is sent back to the server. The tool then resumes with the result.

From the agent loop's perspective, a tool with elicitation looks like a slow tool call. No changes to the agent, TensorZero configuration, or tool schemas are needed.

```
Agent loop                   Client                         Server
    |                            |                              |
    |  result = call_tool(name)  |                              |
    | --------------------------> |  tools/call ---------------->|
    |                            |                              |  tool starts
    |                            |                              |  ctx.elicit("Approve?")
    |                            |  <--- elicitation/create --- |  tool PAUSED
    |                            |                              |
    |                            |  handler() runs in terminal  |
    |                            |                              |
    |                            |  response ------------------>|  tool RESUMES
    |  <------------------------ |  <-- tools/call response --- |
```

## The handler

The handler receives four parameters. The key one is `response_type`:

- **`None`** — Binary approval. The handler asks y/n and returns `ElicitResult(action="accept", content={})` or `ElicitResult(action="decline")`.
- **A type** — Structured input. The handler inspects `params.requestedSchema` to discover field names and types, prompts for each, and returns `ElicitResult(action="accept", content={...})`.

All `input()` calls use `asyncio.to_thread()` to avoid blocking the event loop.

## Prerequisites

- Python 3.11+
- `uv sync` (install dependencies)
- `GROQ_API_KEY` in your environment

## Run

Boot the bundled gateway from this folder, then run the example. Because `-f` changes Docker Compose's project directory, `--env-file` is required so it picks up the repo-root `.env`:

```bash
docker compose --env-file ../../../.env -f examples/agent/04_elicitation/docker-compose.yml up -d
uv run python examples/agent/04_elicitation/main.py
docker compose -f examples/agent/04_elicitation/docker-compose.yml down
```

## What to observe

1. The agent calls `list_tasks` first — no elicitation, returns immediately with a list of project tasks.
2. The agent calls `delete_task("T-103")` — you see an approval prompt. Type `y` to approve or `n` to decline.
3. If you approve, the tool returns "Deleted task: T-103 (Update README)". If you decline, the tool returns "Deletion cancelled" and the LLM adapts.
4. The agent calls `create_task` — you see prompts for title, priority, and assignee. Type values for each field.
5. The LLM receives all tool results and produces a final summary.

## Learn more

See [docs/elicitation.md](../../../docs/elicitation.md) for the full guide covering handler patterns, content format, timeout guidance, and constraints.
