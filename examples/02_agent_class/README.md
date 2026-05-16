# 02 — Agent class

The declarative `Agent` class.  Configure once with `AgentConfig`, then
drive every invocation through `agent.invoke(message)` — no more passing
every knob to `run_agent` on every call.  Three runnable files, one per
tier.

| File | Tier | What it adds |
|---|---|---|
| `basic.py` | Basic | The minimal `Agent(AgentConfig(...))` builder — same demo as `01_first_agent` but via the declarative wrapper. |
| `intermediate.py` | Intermediate | A system prompt baked into the `AgentConfig` — the agent's personality stays consistent across `invoke()` calls. |
| `advanced.py` | Advanced | The `system="..."` per-call escape hatch — different personalities in three back-to-back `invoke()` calls without touching the config. |

See [`docs/agent.md`](../../docs/agent.md) for the full reference.

## Quick start

```bash
# 1. Set your API key
export GROQ_API_KEY=...

# 2. Start the gateway
docker compose -f examples/02_agent_class/docker-compose.yml --env-file .env up -d

# 3. Pick a tier and run it
uv run python examples/02_agent_class/basic.py
uv run python examples/02_agent_class/intermediate.py
uv run python examples/02_agent_class/advanced.py
```

## When to use `Agent` vs `run_agent`

| Use `Agent` when | Use `run_agent` when |
|---|---|
| You want a clean one-liner (`agent.invoke(msg)`) | You need full control over the inference call |
| Configuration is static across calls | Configuration varies per call |
| You want to add memory, guardrails, or streaming on the same object | You only need the low-level MCP bridge |

Both paths produce the same `AgentResult`. `Agent.invoke()` delegates to
`run_agent` internally, so there is no performance difference.
