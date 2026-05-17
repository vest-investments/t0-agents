# `glow` examples (experimental)

Runnable examples for the `t0_agents.glow` sub-package — a Pythonic
high-level surface over t0-agents v1.0.

The layer is experimental. Public API is unstable until further
notice.

## Layout

A self-contained TensorZero gateway boots from this directory:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
```

Three TZ functions are pre-configured under `config/`:

- `agent` — chat (used by `01_hello.py`; also reused by the
  glow-integration test's researcher / writer sub-agents).
- `plan_trip` — JSON-structured (used by `02_structured.py`); schema
  matches the `TripPlan(BaseModel)` defined in the example.
- `orchestrator` — chat (used by the M3 integration test
  `test_orchestrator_with_glow_sub_agents` and forthcoming
  orchestrator-style examples).

Once the gateway is up, run any example as
`uv run python examples/glow/<name>.py`. Each prints a `SUCCESS:`
banner on completion.

## Examples

| #  | File                  | Milestone | Demonstrates                                            |
|----|-----------------------|-----------|---------------------------------------------------------|
| 01 | `01_hello.py`         | M2        | 4-line hello-world: `@app.agent` + docstring system prompt + `-> str` |
| 02 | `02_structured.py`    | M2        | Return type → `StructuredOutput`: `await plan(q)` is a `TripPlan` instance |
| 03 | `03_streaming.py`     | M2+       | `FluentRun.events()` with selectors (coming in M3 work) |
| 04 | `04_composition.py`   | M4        | `compose([...])`, `compose({...})`, `compose({a, b})`   |
| 05 | `05_di_tools.py`      | M3        | `@app.tool` + `Depends` injection                       |
| 06 | `06_mcp_typing.py`    | M4        | `MCPServer` class + `Annotated` MCP injection           |

When you're done:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env down
```
