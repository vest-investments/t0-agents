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
from t0_agents.glow import App, MCPServer, llm

# 1. Class-based — preferred for shared servers.
class MathMCP(MCPServer, server=_math_mcp_instance, namespace="math"): ...

app = App(t0=t0, servers=[MathMCP])

# 2. Annotated — per-parameter override on a single agent.
@app.agent("calc")
async def calc(query: str, math: Annotated[Any, MCPServer(url="http://...")]) -> str:
    """Math assistant with a remote MCP server."""
    return llm()
```

Class-based declarations register the spec on `__init_subclass__`;
annotation-based declarations are harvested by `@app.agent`'s
parameter walk.  `App.build()` mounts every spec on the App-owned
composed FastMCP via `FastMCP.mount()` for in-memory `server=` specs and
`fastmcp.server.create_proxy(...)` for remote `url=` endpoints (HTTP,
SSE, STDIO — the transport is inferred from the URL unless `headers=` /
`auth=` are set; see below).

Every glow agent on a given `App` sees every mounted server — per-agent
MCP subsets are a [known limitation](limitations.md#no-per-agent-mcp-subsets);
spin up multiple `App` instances for strict isolation.

Runnable demo:
[`examples/glow/08_mcp_typing.py`](../../examples/glow/08_mcp_typing.py).

### Instances in `servers=`

`servers=` accepts `MCPServer` **instances**, not only subclasses — so a
per-mount configuration needs no module-level subclass:

```python
app = App(t0=t0, servers=[
    MCPServer(server=_math_mcp_instance, namespace="math"),
    MCPServer(url="https://api.example.com/mcp", namespace="ex"),
])
```

A subclass is still right for a server shared across modules (it also
types as a parameter annotation); an instance is convenient for a one-off
mount assembled at startup.

### Authenticated remote mounts — `headers=` / `auth=`

A `url=` mount carries per-request identity directly:

```python
import httpx

class ServiceAuth(httpx.Auth):
    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request):
        request.headers["X-Service-Auth-Token"] = self._token
        yield request

customers = MCPServer(
    url="https://mcp-customers/mcp",
    namespace="customers",
    headers={                          # static, sent on every request
        "X-Vest-Customer-Id": customer_id,
        "X-Vest-Customer-Token": customer_token,
    },
    auth=ServiceAuth(service_token),   # an httpx.Auth, applied per request
)
app = App(t0=t0, servers=[customers])
```

`headers` is a static dict; `auth` is any
[`httpx.Auth`](https://www.python-httpx.org/advanced/authentication/) —
its `auth_flow` runs per request, the place for a rotating token or a
signature.  When either is set the mount rides an explicit transport
chosen by `transport=` (`"http"` default, or `"sse"`) so the headers and
auth reach the server; without them the bare URL is passed to
`create_proxy`, which infers the transport as before.  Setting
`headers=`/`auth=` on an in-memory `server=` mount is a configuration
error — they have no meaning off an HTTP transport.

This is the first-class replacement for hand-building a
`create_proxy(MCPConfig({...: RemoteMCPServer(headers=..., auth=...)}))`
and wrapping it in a locally-defined `MCPServer` subclass.

### Unprefixed mounts — `prefix=False`

By default a mount's tool names are prefixed with its `namespace`
(`customers_retrieve_user_profile`), keeping multi-server setups
collision-free.  A single-server mount that must present names verbatim
(e.g. matching a legacy tool inventory) opts out:

```python
# tool names stay verbatim: retrieve_user_profile
#                           (not customers_retrieve_user_profile)
MCPServer(server=customers, namespace="customers", prefix=False)
```

`namespace` is then a logical label only.  An unprefixed mount has no
prefix to filter on, so select its tools by explicit name or glob in
`tools=` (`tools=["retrieve_user_profile", "retrieve_user_portfolio"]`)
rather than by passing the server — the latter is a loud error.  Keep the
default `prefix=True` for multi-server apps so their tool names stay
collision-free.

## Human-in-the-loop — `elicitation_handler=`

When a mounted MCP tool calls `ctx.elicit(...)` (for example a trade
confirmation), glow's App-owned `Client` needs a resolver — otherwise the
call surfaces as "Elicitation not supported".  Pass one to the App:

```python
from t0_agents.core.elicitation_forwarder import ElicitationForwarder

forwarder = ElicitationForwarder(on_request=send_to_user)  # your return channel
app = App(t0=t0, servers=[...], elicitation_handler=forwarder.handler)

# ... later, when the human answers on that channel:
await forwarder.resolve(elicitation_id, {"action": "accept", "content": {...}})
```

`elicitation_handler` is any FastMCP
[`ElicitationHandler`](https://gofastmcp.com/clients/elicitation).
`ElicitationForwarder` is the batteries-included one: it bridges an
in-process `ctx.elicit` to an out-of-process channel (a WebSocket, or an
SSE + POST pair) and blocks until the human answers.  A `fastmcp.Client`
binds exactly one handler and the App owns one Client, so this is one
resolver per App — construct one App per session/tenant for isolated HITL
(the "one App per drive" shape the rest of glow assumes).  With no handler
the default is unchanged: elicitation is unsupported.

## Guardrails as type metadata

The v1.0 [guardrail factories](../guardrails.md) (`require_approval`,
`max_cost`, `max_tool_calls`, `block_tools`, `require_parsed`,
`regex_guardrail`) can be attached as `Annotated` metadata on
`@app.agent` parameter or return annotations.  `regex_guardrail` lives
in `t0_agents.guardrails.regex` and is also re-exported from
`t0_agents.glow` so the typed-metadata pattern composes cleanly:

```python
from typing import Annotated
from t0_agents.glow import llm, regex_guardrail
from t0_agents.guardrails import max_cost

no_pii = regex_guardrail(r"\b\d{3}-\d{2}-\d{4}\b", "no SSN")
under_budget = max_cost(0.50)

@app.agent("support")
async def support(query: str) -> Annotated[str, no_pii, under_budget]:
    """Customer support."""
    return llm()
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
    return llm()
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
