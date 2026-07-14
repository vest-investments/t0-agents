# `Depends` — tools and sub-agents

> Part of the glow ladder.  Covers the two roles of
> [FastAPI](https://fastapi.tiangolo.com/)'s `Depends(...)` sentinel
> inside glow: value-style injection for `@app.tool` parameters and
> sub-agent injection for `@app.agent` parameters.

`Depends(provider)` is FastAPI's injection sentinel.  Glow uses it in
two places with different semantics — they look alike but the runtime
treats them differently.

## Inside `@app.tool` — value-style injection

Strip the dependency parameter from the tool's auto-generated JSON
Schema (so the LLM never sees it) and re-inject it at call time:

```python
from t0_agents.glow import Depends

async def get_db() -> DB: ...

@app.tool
async def lookup_order(order_id: str, db: DB = Depends(get_db)) -> Order:
    return await db.orders.get(order_id)
```

The schema advertised to the LLM has a single `order_id` parameter;
the `db` argument is resolved by glow at call time and never appears
in the model's tool-call payload.

Providers may themselves take `Depends` (recursive resolution); sync,
async, and sync-returning-awaitable shapes are all supported.

### `dependency_overrides` for tests

Swap providers at test time without re-instantiating the decorator:

```python
app.dependency_overrides[get_db] = lambda: InMemoryDB(...)
```

The override dict is consulted on every call, not snapshotted at
registration — drop in a mock, run the assertion, clear the override.
The runnable test pattern lives at
[`examples/glow/07_di_tools.py`](../../examples/glow/07_di_tools.py).

## Inside `@app.agent` — sub-agent injection

When the parameter is typed as a glow agent wrapper and the provider
is itself a glow-registered agent, glow installs it as a sub-agent on
the parent `AgentConfig.sub_agents` declaratively:

```python
@app.agent("orchestrator")
async def orchestrator(
    query: str,
    research: Agent = Depends(researcher),
    write: Agent    = Depends(writer),
) -> str:
    """Calls research → write."""
    return llm()
```

The sub-agents become `{name}_ask` tools on the orchestrator's MCP
server (the v1.0 [multi-agent composition](../multi-agent.md) path).

Value-style `Depends(get_db)` on `@app.agent` raises
`ConfigurationError` — the runtime only has two defined behaviours for
`@app.agent` parameters (the user message and sub-agents), so glow
rejects ambiguous shapes loudly.  The fail-fast surfaces at
`app.build()` time, not at the first inference.

## Restricting tools per agent — `tools=`

By default every agent on an `App` sees every `@app.tool`, every mounted
`MCPServer`, and every `<sub_agent>_ask` tool.  Pass `tools=` to
`@app.agent` to narrow that to an explicit allow-list:

```python
@app.agent("analyst", tools=[get_weather, Fred, "customers_get_*"])
async def analyst(q: str) -> str:
    """Sees only get_weather, every Fred-namespaced tool, and customers_get_*."""
    return llm()

@app.agent("chitchat", tools=[])          # explicitly toolless
async def chitchat(q: str) -> str:
    return llm()
```

Entries may be `@app.tool` callables (matched by name), `MCPServer`
classes or instances (matched by their `namespace` prefix — a namespace is
**required**, else `build()` raises), and glob strings.  Omitting `tools=`
keeps the default (the agent sees everything — back-compatible); `tools=[]`
is explicitly toolless.  A pattern matching no live tool warns at
`app.build()`, naming the available tools — so a mistyped prefix surfaces
as a build-time diagnostic rather than a silently missing tool.

Sub-agent `<name>_ask` tools are never filtered: the allow-list applies to
the real composed tools only, so delegation stays available.

This is how a consumer keeps a dangerous tool off an agent even while its
MCP server is mounted for others — vest-ai, for example, mounts its
order-placement server but omits those tools from the assistant agent's
`tools=`, so they can never be called.

## When to use which

| Decorator     | Parameter shape                                       | What `Depends` does                              |
|---------------|-------------------------------------------------------|--------------------------------------------------|
| `@app.tool`   | `db: DB = Depends(get_db)`                            | hides `db` from the LLM schema; injects at call  |
| `@app.tool`   | `db: DB = Depends(other_tool_provider)`               | recursive provider resolution                    |
| `@app.agent`  | `research: Agent = Depends(researcher)`               | installs `researcher` as a sub-agent             |
| `@app.agent`  | `db: DB = Depends(get_db)` (value injection)          | **error** — raises `ConfigurationError`          |
| `@app.agent`  | `tools=[fn, Server, "glob_*"]`                        | restricts the agent to an allow-list of tools    |

## Runnable demos

- [`examples/glow/07_di_tools.py`](../../examples/glow/07_di_tools.py) —
  `@app.tool` + `Depends` schema-stripping + `app.dependency_overrides`
  exercised through a direct MCP `call_tool` round-trip.
- [`examples/glow/09_agent_tools.py`](../../examples/glow/09_agent_tools.py)
  — per-agent `tools=` filtering: two agents on one App with disjoint tool
  subsets, verified via each agent's tool provider.
- [`examples/glow/06_orchestrator.py`](../../examples/glow/06_orchestrator.py)
  — sub-agent injection via `Depends(<wrapper>)`.

---

For the canonical v1.0 sub-agent surface and the wiring details (how
`{name}_ask` tools are mounted on the orchestrator's MCP server), see
[`docs/multi-agent.md`](../multi-agent.md).
