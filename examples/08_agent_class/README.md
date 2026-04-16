# Example 08: Agent Class

Demonstrates the `Agent` wrapper that ships with Milestone M1. Instead of
passing configuration to `run_agent` on every call, you configure once with
`AgentConfig` and invoke `agent.run(message)`.

## Quick start

```bash
# 1. Set your API key
export GROQ_API_KEY=...

# 2. Start the gateway
docker compose -f examples/08_agent_class/docker-compose.yml up -d

# 3. Run the example
uv run python examples/08_agent_class/main.py
```

## When to use Agent vs run_agent

| Use `Agent` when | Use `run_agent` when |
|---|---|
| You want a clean one-liner (`agent.run(msg)`) | You need full control over the inference call |
| Configuration is static across calls | Configuration varies per call |
| You plan to add memory, guardrails, or streaming later (Phases B–E) | You only need the low-level MCP bridge |

Both paths produce the same `AgentResult`. `Agent.run()` delegates to
`run_agent` internally, so there is no performance difference.
