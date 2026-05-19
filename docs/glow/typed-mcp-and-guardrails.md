# Typed MCP and guardrails

> Part of the glow ladder.  Covers how to declare MCP servers via
> Python typing and how to attach guardrails as `Annotated[...]`
> metadata.

Both patterns share the same idea: lift configuration from imperative
construction (`AgentConfig(...)` kwargs) into the type system, where
the decorator harvests it at registration time.

## Typed MCP — class-based and `Annotated`

Two patterns coexist:

```python
from fastmcp import FastMCP
from typing import Annotated, Any
from t0_agents.glow import App, MCPServer

# 1. Class-based — preferred for shared servers.
class MathMCP(MCPServer, server=_math_mcp_instance, namespace="math"): ...

app = App(t0=t0, servers=[MathMCP])

# 2. Annotated — per-parameter override on a single agent.
@app.agent("calc")
async def calc(query: str, math: Annotated[Any, MCPServer(url="http://...")]) -> str:
    """Math assistant with a remote MCP server."""
```

Class-based declarations register the spec on `__init_subclass__`;
annotation-based declarations are harvested by `@app.agent`'s
parameter walk.  `App.build()` mounts every spec on the App-owned
composed FastMCP via `FastMCP.mount()` for in-memory servers and
`fastmcp.server.create_proxy(url)` for remote HTTP/SSE/STDIO endpoints.
Remote auth headers and transport options pass through
`create_proxy(url, ...)`.

Every glow agent on a given `App` sees every mounted server — per-agent
MCP subsets are a [known limitation](limitations.md#no-per-agent-mcp-subsets);
spin up multiple `App` instances for strict isolation.

Runnable demo:
[`examples/glow/08_mcp_typing.py`](../../examples/glow/08_mcp_typing.py).

## Guardrails as type metadata

The v1.0 [guardrail factories](../guardrails.md) (`require_approval`,
`max_cost`, `max_tool_calls`, `block_tools`, `require_parsed`,
`regex_guardrail`) can be attached as `Annotated` metadata on
`@app.agent` parameter or return annotations.  `regex_guardrail` lives
in `t0_agents.guardrails.regex` and is also re-exported from
`t0_agents.glow` so the typed-metadata pattern composes cleanly:

```python
from typing import Annotated
from t0_agents.glow import regex_guardrail
from t0_agents.guardrails import max_cost

no_pii = regex_guardrail(r"\b\d{3}-\d{2}-\d{4}\b", "no SSN")
under_budget = max_cost(0.50)

@app.agent("support")
async def support(query: str) -> Annotated[str, no_pii, under_budget]:
    """Customer support."""
```

`@app.agent`'s decoration-time walk harvests every Annotated metadata
entry satisfying the `Guardrail` Protocol and folds it into
`AgentConfig.guardrails`.  Runtime path is unchanged — the canonical
[`GuardrailsMiddleware`](../guardrails.md) does the work.

### Phase placement

`phase="before_run"` guardrails (e.g.
`regex_guardrail(pattern, msg, phase="before_run")`) also work attached
to a parameter annotation; the metadata harvest is symmetric across
parameter and return positions.

```python
@app.agent("filter")
async def filter(
    query: Annotated[str, regex_guardrail(r"\bsecret\b", "no secrets", phase="before_run")],
) -> str:
    """Reject queries containing 'secret'."""
```

## When to drop down to `AgentConfig.guardrails=[...]`

The metadata path covers the common case (declarative, type-checked
guardrails per agent).  When you need:

- a guardrail shared across multiple agents declared in different
  modules,
- a runtime-configured guardrail (the rule changes per request),
- a guardrail whose phase or scope can't be expressed at decoration
  time,

— drop down to `wrapper._agent.config.guardrails` directly (see
[Escape hatch](escape-hatch.md)) or build the `Agent` with explicit
`AgentConfig(guardrails=[...])`.

---

For the canonical v1.0 narratives, see
[`docs/guardrails.md`](../guardrails.md) (the full Guardrail Protocol,
the built-in factories, the `GuardrailsMiddleware`) and
[`docs/multi-server.md`](../multi-server.md) (mounting multiple FastMCP
servers behind one agent).
