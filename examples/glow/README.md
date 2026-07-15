# glow examples (experimental)

The 8-line "hello agent" lives here — module-level `App()`, no gateway
ceremony:

```python
from t0_agents.glow import App, llm

app = App()  # resolves TENSORZERO_URL / TENSORZERO_CONFIG at first use

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""
    return llm()

async with app:
    print(await tutor("How do I say hello in formal Japanese?"))  # -> str
```

Two ways to give the examples a gateway (GROQ_API_KEY in `.env` either
way):

```bash
# HTTP gateway (docker) — powers examples 02–09 as written:
docker compose -f examples/glow/docker-compose.yml --env-file .env up -d

# ...or docker-free, embedded in-process (01_hello):
export TENSORZERO_CONFIG=examples/glow/config/tensorzero.toml
```

Then run any example with `uv run python examples/glow/<name>.py`
(each prints a `SUCCESS:` banner), or drive one straight from the CLI:

```bash
t0-agents run examples.glow.01_hello:app --agent tutor "hi"
```

## Examples

| #  | File                   | Demonstrates                                                                                       |
|----|------------------------|----------------------------------------------------------------------------------------------------|
| 01 | `01_hello.py`          | 8-line hello-world: module-level `App()`, docstring system prompt, `-> str`, docker-free embedded  |
| 02 | `02_structured.py`     | Return type → `StructuredOutput`: `await plan(q)` is a `TripPlan` instance                         |
| 03 | `03_streaming.py`      | `FluentRun.events()` with `Thought` + `ToolCall.map()` + `Text` selectors                          |
| 04 | `04_sequential.py`     | Sequential pipeline: chain two `@app.agent` callables with an ad-hoc step in between               |
| 05 | `05_parallel.py`       | Parallel fan-out: `asyncio.gather` over `@app.agent` wrappers with a plain Python reducer          |
| 06 | `06_orchestrator.py`   | Orchestrator: `Depends(<wrapper>)` parameters install sub-agents as `<name>_ask` MCP tools         |
| 07 | `07_di_tools.py`       | `@app.tool` + `Depends` injection                                                                  |
| 08 | `08_mcp_typing.py`     | `MCPServer` class + `Annotated` MCP injection                                                      |
| 09 | `09_agent_tools.py`    | Per-agent `tools=[...]` filtering: callables, `MCPServer` namespaces, glob strings                 |

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
