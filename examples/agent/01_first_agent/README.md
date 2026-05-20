# 01 — Your first agent

A coffee shop concierge with two tools.  You ask one question; the LLM
picks the right tool order on its own; you get one synthesised answer.

```
You:    What should I order, and how long will I wait?
Agent:  Today's specialty is an Ethiopian Yirgacheffe pour-over, and
        the queue is about a 7-minute wait — good time to grab it.
```

**Aha! moment:** the LLM decides the order of the two tool calls,
combines their returns, and never has to be told what to do next.  The
whole "agent loop" — discover tools → infer → execute → feed back → repeat —
runs unchanged whether you have two tools or twenty.

## What it demonstrates

1. Connecting to a TensorZero gateway with `AsyncTensorZeroGateway`.
2. Hosting an in-process FastMCP server with two tools.
3. Building an `Agent` declaratively with `AgentConfig`.
4. `agent.invoke(message)` → `Run[T]` → `await` for the final
   `AgentResult` — the canonical v1.0 surface.

## Prerequisites

- Python 3.11+
- `uv sync`
- `GROQ_API_KEY=...` in `.env` at the project root.

## Start the gateway

```bash
docker compose -f examples/agent/01_first_agent/docker-compose.yml \
    --env-file .env up -d
```

## Run

```bash
uv run python examples/agent/01_first_agent/main.py
```

## Stop the gateway

```bash
docker compose -f examples/agent/01_first_agent/docker-compose.yml down
```

## What to observe

- The `Response` line is one natural-language answer that references
  both `get_specialty_today` and `estimate_wait_minutes`.
- `Turns: 2` confirms the loop fired one tool-call turn and one
  final-answer turn.
- `Episode ID` is the TensorZero-assigned UUID.  Pass it to
  `agent.invoke(..., episode_id=...)` next time you want continuity
  (memory, follow-ups), and use it to grep the ClickHouse dashboard.

If you only need conversational LLM access without tools, see
[`../00_chat_only/`](../00_chat_only/) for the same `Agent` surface
with no MCP plumbing.

Once this works, head to [`02_agent_class`](../02_agent_class/) for
configuration shapes, then [`03_output_modes`](../03_output_modes/)
for typed JSON / Pydantic extraction.
