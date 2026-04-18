# Example 08: Agent Class

Runnable one-liner demo of the `Agent` class: configure once with
`AgentConfig`, then call `agent.run(message)` instead of passing every
knob to `run_agent` on every call. See
[`docs/agent.md`](../../docs/agent.md) for the full reference.

## Quick start

```bash
# 1. Set your API key
export GROQ_API_KEY=...

# 2. Start the gateway
docker compose -f examples/08_agent_class/docker-compose.yml --env-file .env up -d

# 3. Run the example
uv run python examples/08_agent_class/main.py
```

## When to use `Agent` vs `run_agent`

| Use `Agent` when | Use `run_agent` when |
|---|---|
| You want a clean one-liner (`agent.run(msg)`) | You need full control over the inference call |
| Configuration is static across calls | Configuration varies per call |
| You want to add memory, guardrails, or streaming on the same object | You only need the low-level MCP bridge |

Both paths produce the same `AgentResult`. `Agent.run()` delegates to
`run_agent` internally, so there is no performance difference.
