# Elicitation in your FastMCP server

> **Audience.** You maintain a **FastMCP server** that exposes tools
> to a downstream `t0-agents`-based agent. That agent runs in another
> process (an API, a CLI, a worker). You want one or more of your
> tools to pause mid-execution and ask the human a question —
> "approve this delete?", "what title should I use?", "pick a
> region" — and then continue based on the answer.
>
> This guide is self-contained for the **server side**. It does not
> assume you read any other guide. It does not assume `t0-agents` is
> imported anywhere in this codebase — your server is plain
> `fastmcp`. The companion guide
> [`elicitation-in-api.md`](elicitation-in-api.md) covers what the
> agent's process needs to do to consume your tools.

---

## 1. What this guide covers

Everything a FastMCP server needs to do to host elicitation-capable
tools and run them in production:

- Transport requirement (only some transports support elicitation).
- The four mainstream `ctx.elicit()` shapes — binary, scalar,
  constrained choice, structured input — and when to use each.
- Two dynamic-survey topologies: API supplies the questions vs the
  server generates them with its own LLM.
- Schema constraints imposed by the MCP spec.
- Returning a usable result on decline / cancel.
- Graceful degradation when the client has no elicitation handler.
- Server-side timeouts.
- Tests in isolation (no `t0-agents`, no TensorZero, no real human).
- Deployment shape (Dockerfile, transport flag, health check,
  observability).

Out of scope: how the agent process (`t0-agents` consumer) wires up
the handler, the SSE return-channel, the request fan-out across
multiple MCP servers. All of that is in
[`elicitation-in-api.md`](elicitation-in-api.md).

---

## 2. Where your server sits in the stack

```
┌─────────────────────────┐   MCP over    ┌─────────────────────────┐
│ Your FastMCP server     │  HTTP / SSE   │ Downstream agent (API,  │
│ (this guide)            │ ◄──────────► │  worker, CLI built on   │
│                         │   bi-direc-   │  t0-agents)             │
│  @mcp.tool              │   tional      │                         │
│  async def my_tool():   │   session     │  Client(                │
│    ctx.elicit(...)      │               │    {"mcpServers": ...}, │
│                         │               │    elicitation_handler= │
│                         │               │  )                      │
└─────────────────────────┘               └─────────────────────────┘
                                                   ▲ ▼
                                              human-facing
                                              return channel
                                              (UI, terminal, ...)
```

The MCP transport between the two processes must allow the server
to send a message **back** to the client mid-tool. That is what
elicitation requires, and it constrains your transport choice.

---

## 3. Transport requirement

Elicitation needs a stateful, bidirectional MCP session. Pick the
right transport when you call `mcp.run(...)` or you will hit either
silent hangs or `McpError("Elicitation not supported")` at runtime.

| Transport | `mcp.run(...)` argument | Elicitation? | When to use |
|---|---|---|---|
| Streamable HTTP (modern, default) | `transport="http"` | ✓ | **Production HTTP deployments** — recommended. |
| SSE + HTTP (legacy) | `transport="sse"` | ✓ | Older clients pinned below the streamable-http upgrade. |
| stdio | `transport="stdio"` | ✓ | Local development, embedded subprocess, single-machine integration. |
| In-memory | (the agent imports your `FastMCP` instance directly into its process) | ✓ | Tests; single-process toy deployments. |
| Stateless HTTP | `transport="http", stateless_http=True` or equivalent | ✗ — **broken silently** | Do not enable for elicitation-capable servers. Each request gets its own session — there is no back-channel. |

For the setup described in `elicitation-in-api.md` (separate Python
API connecting over HTTP), **streamable HTTP** is the right choice
and the default for `transport="http"` in modern FastMCP — you do
not need any extra flag. Just do not enable the stateless mode.

### 3.1 Minimal entrypoint

```python
# src/your_server/entrypoint.py
from fastmcp import FastMCP

from your_server.tools import register_all

mcp = FastMCP("YourServer")
register_all(mcp)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

The downstream agent connects to `http://your-host:8000/mcp`.

---

## 4. Tool patterns

All five patterns share the same shape: import `Context`, accept it
as a parameter, call `await ctx.elicit(...)`, branch on
`result.action`. The differences are the `response_type` argument
and how `result.data` is consumed.

### 4.1 Binary confirmation (approve / reject)

The simplest pattern: pause before doing something irreversible.

```python
# src/your_server/tools/approval.py
from fastmcp import Context, FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool
    async def delete_resource(resource_id: str, ctx: Context) -> str:
        """Delete a resource (requires human confirmation)."""
        result = await ctx.elicit(
            f"Delete resource {resource_id!r}? This cannot be undone.",
            response_type=None,  # None ⇒ binary approval, no data
        )
        if result.action == "accept":
            # ... actually delete here ...
            return f"Deleted: {resource_id}"
        return f"Deletion cancelled for {resource_id}"
```

Three things to internalise:

1. **Return a string** in both branches. Decline is a normal outcome,
   not an error. Raising an exception triggers retry / error logic on
   the agent side; returning a readable string lets the LLM adapt.
2. **`response_type=None`** signals "no data, just yes/no". The
   handler's response payload is an empty dict `{}` on accept.
3. `result.action` is one of `"accept"`, `"decline"`, `"cancel"`.
   Most code treats `decline` and `cancel` the same (the user did
   not approve), as in this example.

### 4.2 Structured input — flat dataclass

When you need multiple fields at once. Each field must be a
primitive (string, int, float, bool, enum) — see §5.

```python
from dataclasses import dataclass

from fastmcp import Context, FastMCP


@dataclass
class TaskIntake:
    title: str
    priority: str       # "low" | "medium" | "high" — constrain via prompt
    assignee: str


def register(mcp: FastMCP) -> None:
    @mcp.tool
    async def create_task(ctx: Context) -> str:
        """Collect task details from the user and create the task."""
        result = await ctx.elicit(
            "Please provide the new task details.",
            response_type=TaskIntake,
        )
        if result.action != "accept":
            return "Task creation cancelled"
        data = result.data   # typed as TaskIntake
        # ... persist the task ...
        return (
            f"Created task: {data.title!r} "
            f"(priority={data.priority}, assignee={data.assignee})"
        )
```

`result.data` arrives as a `TaskIntake` instance validated against
the dataclass. If the user submits invalid data (missing required
fields), the client surfaces the validation error before reaching
your tool — `result.action` will not be `"accept"`.

### 4.3 Constrained choice (single-select)

```python
@mcp.tool
async def pick_region(ctx: Context) -> str:
    """Single-select region from a fixed list."""
    result = await ctx.elicit(
        "Which region?",
        response_type=["us-east-1", "us-west-2", "eu-west-1"],
    )
    if result.action != "accept":
        return "No region selected"
    return f"Region set to {result.data}"
```

The list-of-strings form is FastMCP's encoding for a single-select
constrained choice. `result.data` is the chosen string.

Multi-select (since FastMCP 2.14.0) uses a nested list:
`response_type=[["a", "b", "c"]]`. `result.data` is the list of
selected items.

### 4.4 Multi-turn within one tool

Calling `ctx.elicit()` more than once in the same tool. Each call
is an independent round-trip to the client handler.

```python
@mcp.tool
async def plan_meeting(ctx: Context) -> str:
    """Sequential elicitations: title → duration → urgency."""
    r1 = await ctx.elicit("Meeting title?", response_type=str)
    if r1.action != "accept":
        return "Cancelled at title"

    r2 = await ctx.elicit("Duration in minutes?", response_type=int)
    if r2.action != "accept":
        return f"Cancelled after title={r1.data!r}"

    r3 = await ctx.elicit("Urgent?", response_type=bool)
    if r3.action != "accept":
        return f"Cancelled after duration={r2.data}"

    return (
        f"Planned: {r1.data!r} for {r2.data} min "
        f"(urgent={r3.data})"
    )
```

For scalar types (`str`, `int`, `bool`), FastMCP wraps the response
in `{"value": <scalar>}` on the wire; `result.data` is unwrapped on
the server side so you receive the primitive directly.

### 4.5 Ask a passed-in list of questions (API-driven survey)

When the **agent process** decides what to ask — for example, it
called its own TensorZero function to generate adaptive questions
and passes them in — your tool only needs to iterate.

```python
import json
from dataclasses import dataclass
from typing import Any

from fastmcp import Context, FastMCP


@dataclass
class QuestionSpec:
    prompt: str
    type: str                  # "str" | "int" | "bool" | "choice"
    options: list[str] | None = None   # required when type == "choice"


_TYPE_MAP: dict[str, type] = {"str": str, "int": int, "bool": bool}


def _resolve_response_type(q: QuestionSpec) -> Any:
    if q.type == "choice":
        opts = q.options or []
        if len(opts) >= 2 and all(isinstance(o, str) for o in opts):
            return opts
        return None  # malformed
    return _TYPE_MAP.get(q.type)


def register(mcp: FastMCP) -> None:
    @mcp.tool
    async def ask_questions(questions: list[QuestionSpec], ctx: Context) -> str:
        """Iterate elicitations over a caller-supplied question list.

        Returns a JSON string with the collected answers.  If the user
        declines a question, returns the partial set with
        ``complete: false``.
        """
        answers: list[dict[str, Any]] = []
        for q in questions:
            rt = _resolve_response_type(q)
            if rt is None:
                answers.append({"prompt": q.prompt, "answer": None, "skipped": True})
                continue
            r = await ctx.elicit(q.prompt, response_type=rt)
            if r.action != "accept":
                return json.dumps({
                    "answers": answers,
                    "complete": False,
                    "stopped_at": q.prompt,
                })
            answers.append({"prompt": q.prompt, "answer": r.data})

        return json.dumps({"answers": answers, "complete": True})
```

This is the right shape for the architecture in
`elicitation-in-api.md`: the API generates the questions using its
own TensorZero gateway and feeds them to `ask_questions(...)`. Your
server stays stateless and has no LLM dependency.

### 4.6 Server-generated questions (self-contained survey)

When **your server** owns the question generation — typically
because it has access to data the agent does not, or because the
generation is a server-side concern (privacy, latency, custom
prompts) — you can give the server its own TensorZero gateway via a
FastMCP lifespan and call it from inside the tool.

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, cast

from fastmcp import Context, FastMCP
from tensorzero import AsyncTensorZeroGateway, Text


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[None]:
    """Open the TensorZero gateway for the server's lifetime."""
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://tensorzero:3000"),
    )
    async with await gateway as t0:
        mcp.state["t0"] = t0     # stash on the FastMCP instance
        yield


mcp = FastMCP("YourServer", lifespan=lifespan)


@mcp.tool
async def run_survey(topic: str, ctx: Context) -> str:
    """Generate adaptive questions about `topic`, then ask the human."""
    t0: AsyncTensorZeroGateway = ctx.fastmcp.state["t0"]

    # 1. Generate questions via your own TensorZero function
    resp = await t0.inference(
        function_name="question_generator",
        input={"messages": [{"role": "user", "content": topic}]},
    )
    text = next(
        (b.text for b in resp.content if isinstance(b, Text) and b.text),
        None,
    )
    if text is None:
        return '{"complete": false, "error": "no questions generated"}'

    import json
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return '{"complete": false, "error": "malformed question generator output"}'

    # 2. Iterate elicitations (same logic as §4.5 inside)
    # ... reuse _resolve_response_type / response_type mapping ...
    # ... return the same JSON shape as ask_questions ...
```

Notes:

- The `lifespan` opens the gateway once on server startup. The tool
  reads it from `ctx.fastmcp.state` (or from a module-level captured
  via closure). This avoids opening a new gateway per call.
- The TensorZero function `question_generator` must be configured
  separately in **your** `tensorzero.toml`. The agent process and
  this server can share one TensorZero gateway URL or use distinct
  gateways — your call.
- This pattern adds a dependency (`tensorzero` on the server). The
  §4.5 pattern keeps the server LLM-free; prefer §4.5 unless you
  have a reason to centralise question generation here.

---

## 5. Schema constraints

MCP elicitation allows only **flat objects with primitive
properties**. This applies to every shape your `response_type` can
take:

| Pattern | Supported? | Notes |
|---|---|---|
| `None` (binary approval) | ✓ | No data exchanged |
| `str` / `int` / `float` / `bool` | ✓ | Wrapped as `{"value": <scalar>}` on the wire |
| `Literal["a", "b", "c"]` or `Enum` | ✓ | Maps to MCP `enum` |
| `["a", "b", "c"]` (list of strings) | ✓ | Single-select constrained choice |
| `[["a", "b", "c"]]` (nested list, ≥ FastMCP 2.14.0) | ✓ | Multi-select |
| flat `@dataclass` / `BaseModel` / `TypedDict` with primitive fields | ✓ | Each field is its own primitive |
| Nested objects (dataclass with another dataclass as a field) | ✗ | MCP spec rejects |
| Arrays of objects (`list[SomeModel]`) | ✗ | MCP spec rejects |
| `Optional[...]` | ⚠ partial | Handle as `None`-able primitives or split flows |
| Union of primitives | ⚠ partial | Test against the specific handler |

When you need richer input, decompose: one elicitation collects the
primitive answers, a follow-up elicitation in the same tool handles
the next layer. Or split across multiple tools.

---

## 6. Decline / cancel semantics

The agent loop treats your tool's **return value** as a tool result.
A decline is not an error — it is a valid outcome.

| Tool behaviour | What the LLM sees | When to choose |
|---|---|---|
| Return `"Deletion cancelled"` | A normal `ToolResult` with that string | Default. Lets the LLM acknowledge the decline and propose an alternative. |
| Return `"ERROR: user declined deletion"` | A normal `ToolResult` whose content the agent loop treats opaquely | When you want the LLM to surface the decline as a hard stop, not retry. |
| Raise `fastmcp.exceptions.ToolError("...")` | `ToolResult` whose `result` starts with `"ERROR: tool execution failed: ..."` | When decline = unrecoverable, e.g. policy says "if not approved, abort entire task." |

Whichever you pick, document it in your tool's docstring so the
prompt author can guide the LLM accordingly.

---

## 7. Handling clients without an elicitation handler

A client that does not register an `elicitation_handler` will
reject the `elicitation/create` request. From the tool's view this
raises `McpError`:

```python
from fastmcp import Context, FastMCP
from mcp import McpError


@mcp.tool
async def delete_resource(resource_id: str, ctx: Context) -> str:
    try:
        result = await ctx.elicit(
            f"Delete {resource_id!r}?", response_type=None
        )
    except McpError as exc:
        # Client did not register a handler — degrade explicitly
        if "not supported" in str(exc).lower():
            return (
                f"ERROR: this tool requires an elicitation-capable client; "
                f"refusing to delete {resource_id!r} without explicit approval"
            )
        raise   # propagate other MCP errors

    if result.action != "accept":
        return f"Deletion cancelled for {resource_id}"
    # ... actually delete ...
    return f"Deleted: {resource_id}"
```

**Never proceed silently** when the handler is missing. The whole
point of the elicitation is human approval. Either degrade visibly
(return an `ERROR:` string), or raise so the loop captures the
failure.

---

## 8. Server-side timeouts

The decorator `@mcp.tool(timeout=...)` covers the **entire** tool
runtime, including `ctx.elicit()` waits. Default is too tight for
humans.

| Tool kind | Recommended `timeout` |
|---|---|
| No elicitation | The FastMCP default (≈30s) is fine |
| One short approval | `300.0` (5 min) |
| Long surveys / structured input | `600.0`–`1800.0` (10–30 min) |

The tighter of (server-side timeout) and (client-side
`tool_timeout`) wins. Coordinate with the agent author so the
client side is not the unintentional bottleneck — by convention
their `AgentConfig.tool_timeout=None` disables the client timeout
and yields to the server's value.

---

## 9. Testing in isolation

Test your server without `t0-agents`, without TensorZero, and
without a real human. FastMCP's in-memory `Client` plus a scripted
`elicitation_handler` is enough:

```python
# tests/test_my_server.py
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult

from your_server.entrypoint import mcp


@pytest.mark.asyncio
async def test_delete_resource_approved() -> None:
    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("delete_resource", {"resource_id": "x"})
        assert result.data == "Deleted: x"


@pytest.mark.asyncio
async def test_delete_resource_declined() -> None:
    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("delete_resource", {"resource_id": "x"})
        assert result.data == "Deletion cancelled for x"


@pytest.mark.asyncio
async def test_delete_resource_no_handler() -> None:
    """When the client has no handler, the tool degrades per §7."""
    async with Client(mcp) as client:
        result = await client.call_tool("delete_resource", {"resource_id": "x"})
        assert "ERROR" in result.data
```

For **structured input** tests, the handler returns
`ElicitResult(action="accept", content={"field": value, ...})`. For
**scalar input** (str / int / bool), the handler returns
`content={"value": <scalar>}` — FastMCP wraps scalars in a `value`
field on the wire.

---

## 10. Deployment

### 10.1 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

ENV PORT=8000
EXPOSE 8000

CMD ["uv", "run", "python", "-m", "your_server.entrypoint"]
```

### 10.2 Environment

Keep credentials out of the FastMCP config. Read them from env vars
at startup; pass them into tool factories or stash on `mcp.state`
via lifespan.

### 10.3 Health and readiness

FastMCP HTTP mode exposes a default health endpoint at the
transport level. For orchestrators that demand a custom endpoint
(e.g. Kubernetes `/healthz`), add an in-process route alongside
the MCP one.

### 10.4 Observability — log every elicitation

```python
import logging

logger = logging.getLogger(__name__)

result = await ctx.elicit(prompt, response_type=None)
logger.info(
    "elicitation tool=%s action=%s prompt=%r",
    "delete_resource", result.action, prompt,
)
```

This is the audit trail for HITL decisions. Include the
`elicitation_id` if FastMCP exposes it on `params` in your version.
Mask sensitive fields when logging `result.data` for structured
inputs — elicitation often carries PII.

---

## 11. Pitfalls

| # | Pitfall | Fix |
|---|---|---|
| 1 | Server deployed with **stateless HTTP** — elicitation hangs or errors immediately | Use streamable HTTP. It is the default for `mcp.run(transport="http")` in modern FastMCP; do not enable any stateless flag. See §3. |
| 2 | Default tool timeout (~30s) too short for humans | `@mcp.tool(timeout=300.0)` minimum on tools that elicit. See §8. |
| 3 | Tool raises on decline instead of returning a string | The agent loop treats raised exceptions as tool errors and may retry. Decline is a normal outcome — return a readable string. See §6. |
| 4 | `response_type` is a nested dataclass / list-of-objects | MCP rejects nested schemas. Flatten or decompose into multiple elicitations. See §5. |
| 5 | Tool catches `McpError("not supported")` and proceeds with the destructive action | Either degrade (return `"ERROR: ..."`) or let the error propagate. Never proceed silently. See §7. |
| 6 | Tool calls `ctx.elicit()` from a task spawned with `asyncio.create_task` that outlives the tool body | The `Context` is bound to the running tool call. Background tasks lose their elicitation channel. Keep elicits in the foreground coroutine. |
| 7 | Server logs the full `result.data` for structured-input tools including sensitive fields | Mask fields before logging — elicitation often carries PII. |
| 8 | Schema uses `default` values to mean "optional" | MCP's elicitation schema does not honour defaults the way OpenAPI does. Treat every field as required; split into multiple elicitations for optional sub-flows. |
| 9 | Survey tool with `t0.inference()` opens a new gateway per call | Open the gateway once in a FastMCP `lifespan`; stash on `mcp.state`; read from `ctx.fastmcp.state` inside the tool. See §4.6. |
| 10 | Multi-tenant deployment where the same server process serves many concurrent agents | Fine — every `ctx.elicit()` call is isolated to the calling tool invocation. The MCP session demultiplexes responses automatically. No shared state to coordinate. |

---

## 12. Cross-references

- The MCP protocol spec for elicitation:
  <https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/elicitation>
- FastMCP elicitation docs:
  <https://gofastmcp.com/servers/elicitation>
- Companion guide for the agent process:
  [`elicitation-in-api.md`](elicitation-in-api.md) — covers the
  handler, the SSE return-channel, per-session lifecycle, and the
  wire format between the API and the human-facing UI.
