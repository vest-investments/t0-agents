# glow examples (experimental)

The 4-line "hello agent" lives here.

```python
from t0_agents.glow import App
app = App(t0=t0)

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""

await tutor("How do I say hello in formal Japanese?")  # -> str
```

One `docker-compose.yml` + one `config/tensorzero.toml` power all
eight examples. Boot the gateway once from this directory:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
```

Then run any example with `uv run python examples/glow/<name>.py`.
Each prints a `SUCCESS:` banner on completion.

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

The same three patterns (sequential, parallel, orchestrator) are
spelled out without glow under
[`../agent/12_concurrency/`](../agent/12_concurrency/) — useful
side-by-side reading for the "what does the decorator actually do?"
question.

When you're done:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env down
```

> **Experimental.** The `t0_agents.glow` public surface is unstable
> until further notice; expect breaking changes between minor
> releases. The explicit `Agent` track in
> [`../agent/`](../agent/) is the canonical, supported surface.
