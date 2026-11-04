# Examples

Two parallel tours of t0-agents.

- [`examples/glow/`](glow/) — the Pythonic decorator surface.
  8 examples, one shared gateway. Start here if you want the
  fastest path to a working agent.
- [`examples/agent/`](agent/) — the explicit `Agent` class
  surface. 15 examples plus two advanced tiers, every collaborator
  visible. For library authors and the moments when the decorator
  is in your way.

Both tracks are progressive: each step adds one concept. Pick a
track; don't mix.

## Side-by-side

| Concept                  | glow                         | agent                                |
|--------------------------|------------------------------|--------------------------------------|
| Hello agent              | `glow/01_hello.py`           | `agent/01_first_agent/`              |
| Structured output        | `glow/02_structured.py`      | `agent/03_output_modes/`             |
| Streaming events         | `glow/03_streaming.py`       | `agent/07_streaming/`                |
| Sequential pipeline      | `glow/04_sequential.py`      | `agent/12_concurrency/01_*.py`       |
| Parallel fan-out         | `glow/05_parallel.py`        | `agent/12_concurrency/02_*.py`       |
| Orchestrator             | `glow/06_orchestrator.py`    | `agent/08_multi_agent/`              |
| DI / typed tools         | `glow/07_di_tools.py`        | (agent track: explicit construction) |
| Typed MCP servers        | `glow/08_mcp_typing.py`      | `agent/14_multi_server/`             |

The agent track also covers ground glow doesn't:
elicitation (`04_elicitation`), memory (`05_memory`), guardrails
(`06_guardrails`), serving / FastAPI (`09_serving`), testing
(`11_testing`), observability / feedback (`13_observability`), and
the MCP-submodule internals (`15_mcp_internals`).

## How to run

Every example carries its own `docker-compose.yml`,
`config/tensorzero.toml`, and `.env.example` (where applicable).
Boot the gateway from inside the example folder; tear down with
`docker compose down` when finished.

A typical first run:

```bash
cp examples/agent/.env.example examples/agent/.env
$EDITOR examples/agent/.env                          # add GROQ_API_KEY
docker compose --env-file examples/agent/.env \
    -f examples/agent/01_first_agent/docker-compose.yml up -d
uv run python examples/agent/01_first_agent/main.py
docker compose -f examples/agent/01_first_agent/docker-compose.yml down
```

The same shape applies to every Docker-backed example in either track.
