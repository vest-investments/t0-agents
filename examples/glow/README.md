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

- `agent` — chat (used by `01_hello.py`, `04_sequential.py`,
  `05_parallel.py`, and the glow-integration test's researcher /
  writer sub-agents).
- `plan_trip` — JSON-structured (used by `02_structured.py`); schema
  matches the `TripPlan(BaseModel)` defined in the example.
- `orchestrator` — chat (used by `06_orchestrator.py` and the glow
  integration test `test_orchestrator_with_glow_sub_agents`).

Once the gateway is up, run any example as
`uv run python examples/glow/<name>.py`. Each prints a `SUCCESS:`
banner on completion.

## Examples

| #  | File                   | Demonstrates                                                                                       |
|----|------------------------|----------------------------------------------------------------------------------------------------|
| 01 | `01_hello.py`          | 4-line hello-world: `@app.agent` + docstring system prompt + `-> str`                              |
| 02 | `02_structured.py`     | Return type → `StructuredOutput`: `await plan(q)` is a `TripPlan` instance                         |
| 03 | `03_streaming.py`      | `FluentRun.events()` with `Thought` + `ToolCall.map()` + `Text` selectors                          |
| 04 | `04_sequential.py`     | Sequential pipeline: chain two `@app.agent` callables with an ad-hoc step in between               |
| 05 | `05_parallel.py`       | Parallel fan-out: `asyncio.gather` over `@app.agent` wrappers with a plain Python reducer          |
| 06 | `06_orchestrator.py`   | Orchestrator: `Depends(<wrapper>)` parameters install sub-agents as `<name>_ask` MCP tools         |
| 07 | `07_di_tools.py`       | `@app.tool` + `Depends` injection                                                                  |
| 08 | `08_mcp_typing.py`     | `MCPServer` class + `Annotated` MCP injection                                                      |

When you're done:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env down
```
