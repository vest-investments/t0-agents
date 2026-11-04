# Elicitation

Human-in-the-loop approval and structured input via MCP Elicitation.

---

## What is elicitation

Elicitation is an MCP protocol feature that lets tools **pause mid-execution**
to ask a human a question. The tool resumes after the human responds. Two
patterns:

| Pattern | Server code | Use case |
|---------|------------|----------|
| Binary approval | `ctx.elicit("Delete 3 files?", response_type=None)` | Confirm before destructive actions |
| Structured input | `ctx.elicit("Details?", response_type=MyModel)` | Collect data the tool needs to proceed |

**Key point for t0-agents consumers:** elicitation requires **no changes** to
`run_agent`, the adapter layer, or any production code. The existing agent loop
handles it transparently.

---

## How it works

### Protocol flow

```
run_agent                     Client                         Server
    |                            |                              |
    |  result = call_tool(name)  |                              |
    | -------------------------► |  tools/call ----------------►|
    |                            |                              |  tool starts
    |                            |                              |  ctx.elicit("Approve?")
    |                            |  ◄--- elicitation/create --- |  tool PAUSED
    |                            |                              |
    |                            |  elicitation_handler() runs  |
    |                            |  (human interacts)           |
    |                            |                              |
    |                            |  response ------------------►|  tool RESUMES
    |                            |                              |  tool returns result
    |  ◄------------------------ |  ◄-- tools/call response --- |
    |  result = CallToolResult   |                              |
```

From `run_agent`'s perspective, a tool with elicitation looks like a slow tool
call. The `await client.call_tool(...)` simply takes longer because the tool is
paused waiting for human input.

### What changes for the caller

One thing: pass `elicitation_handler` when creating the `Client`. Everything
else -- `run_agent` signature, TensorZero configuration, tool schemas -- stays
the same.

```python
from fastmcp import Client
from t0_agents import run_agent

client = Client(mcp_server, elicitation_handler=my_handler)
async with client:
    result = await run_agent(t0, client, "Deploy to prod", "agent")
```

TensorZero is a stateless inference gateway. It does not know about elicitation
and needs zero changes.

---

## The handler interface

### Signature

```python
from fastmcp.client.elicitation import ElicitResult

async def elicitation_handler(
    message: str,
    response_type: type | None,
    params: Any,
    context: Any,
) -> ElicitResult:
    ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable prompt from the tool (e.g., `"Approve: deploy?"`) |
| `response_type` | `type \| None` | `None` for approval (no data expected). Otherwise, a type generated from the server's JSON schema. |
| `params` | `ElicitRequestParams` | Raw MCP parameters including `requestedSchema`. Rarely needed -- `message` and `response_type` cover most cases. |
| `context` | `RequestContext` | Request metadata. Rarely needed in typical handlers. |

### Response actions

| Action | Meaning | When to use |
|--------|---------|-------------|
| `"accept"` | Approved / data provided | User confirmed or submitted data |
| `"decline"` | Explicitly refused | User chose not to proceed |
| `"cancel"` | Dismissed without choosing | User closed the prompt |

Return an `ElicitResult`:

```python
# Accept (approval or with data)
return ElicitResult(action="accept", content={})
return ElicitResult(action="accept", content={"name": "Alice", "age": 30})

# Decline
return ElicitResult(action="decline")

# Cancel
return ElicitResult(action="cancel")
```

---

## Approval vs structured input

### Detecting the pattern

The handler distinguishes approval from structured input by checking
`response_type`:

```python
async def elicitation_handler(message, response_type, params, context):
    if response_type is None:
        # Binary approval -- no data expected
        ...
    else:
        # Structured input -- return data matching response_type
        ...
```

### Approval (`response_type is None`)

Return `ElicitResult(action="accept", content={})` to approve, or
`ElicitResult(action="decline")` to reject. The `content` must be `{}` (empty
dict), not `None`.

### Content format by response type

| Server `response_type` | Handler `content` format | Example |
|------------------------|-------------------------|---------|
| `None` (approval) | `{}` | `ElicitResult(action="accept", content={})` |
| `dataclass` / `BaseModel` | `{field: value, ...}` | `ElicitResult(action="accept", content={"name": "Alice", "age": 30})` |
| `str` | `{"value": <str>}` | `ElicitResult(action="accept", content={"value": "Alice"})` |
| `int` | `{"value": <int>}` | `ElicitResult(action="accept", content={"value": 42})` |
| `bool` | `{"value": <bool>}` | `ElicitResult(action="accept", content={"value": True})` |
| `["a", "b", "c"]` (constrained choice) | `{"value": <choice>}` | `ElicitResult(action="accept", content={"value": "medium"})` |

**Important:** Scalar types (`str`, `int`, `bool`) and constrained choices
(`list[str]`) require `content={"value": <scalar>}`. FastMCP wraps these in a
`ScalarElicitationType[T]` internally. This wrapping is transparent on the
server side (`result.data` returns the unwrapped value), but **handlers must use
the `{"value": ...}` format**.

---

## Timeout

`tool_timeout` (default `30.0` seconds) wraps the **entire** `call_tool()`
duration, including elicitation wait time. If the human takes longer than the
timeout to respond, the call fails with `McpError`. `None` disables the
per-call timeout on every path — `run_agent`, `AgentConfig`, and
`@app.agent(tool_timeout=None)` alike.

When using elicitation-enabled tools, set `tool_timeout` to `None` (no timeout)
or a generous value:

```python
result = await run_agent(
    t0, client, "Deploy to production", "agent",
    tool_timeout=None,  # Disable timeout for elicitation
)
```

| Scenario | Recommended `tool_timeout` |
|----------|--------------------------|
| No elicitation-capable tools | `30.0` (default) |
| Mixed tools (some with elicitation) | `300.0` (5 minutes) |
| All tools may elicit | `None` (disable timeout) |

**Trade-off:** `tool_timeout=None` removes timeout protection for **all** tools,
not just elicitation-capable ones. If you have a mix of fast tools and
elicitation tools, use a generous value (e.g., `300.0`) rather than `None`.

**Remote mounts have a second fuse.** `tool_timeout` is the wall-clock
deadline, but a glow `url` mount also crosses an httpx **read window** on the
wire (default 300 s of silence). A human deliberating for six minutes over a
silent connection kills the call at ~300 s even with `tool_timeout=None`.
Three levers, combined as needed: raise the window per server with
[`MCPServer(timeout=...)`](glow/typed-mcp-and-guardrails.md#per-server-read-timeout--timeout),
have the server emit a heartbeat (`ctx.info(...)` every few seconds — any SSE
frame resets the window), and keep the client-side window **above the
server's elicitation wait, with margin**. The full layer-by-layer picture —
including why a dead stream plus `tool_timeout=None` hangs instead of
erroring — lives in
[Error handling § The timeout stack](error-handling.md#the-timeout-stack).

---

## Server-side patterns

These patterns show how to use `ctx.elicit()` inside MCP tool functions. They
run on the server (the FastMCP side), not in the handler.

### Binary approval

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("MyServer")

@mcp.tool
async def approve_action(action: str, ctx: Context) -> str:
    """Request approval before performing an action."""
    result = await ctx.elicit(f"Approve: {action}?", response_type=None)
    if result.action == "accept":
        return f"Approved: {action}"
    return f"Rejected: {action}"
```

### Reusable approval helper

```python
async def require_approval(ctx: Context, summary: str) -> bool:
    """Ask the human to approve an action. Returns True if approved."""
    result = await ctx.elicit(f"Approve: {summary}?", response_type=None)
    return result.action == "accept"

@mcp.tool
async def delete_files(files: list[str], ctx: Context) -> str:
    if not await require_approval(ctx, f"Delete {len(files)} files"):
        return "Rejected by user"
    # proceed with deletion...
    return f"Deleted {len(files)} files"
```

### Structured input (dataclass)

```python
from dataclasses import dataclass

@dataclass
class UserInfo:
    name: str
    age: int

@mcp.tool
async def collect_info(ctx: Context) -> str:
    """Collect user information via elicitation."""
    result = await ctx.elicit(
        "Please provide your information",
        response_type=UserInfo,
    )
    if result.action == "accept":
        return f"Hello {result.data.name}, age {result.data.age}"
    return "Info not provided"
```

### Constrained choice

```python
@mcp.tool
async def pick_priority(ctx: Context) -> str:
    """Constrained choice from low/medium/high."""
    result = await ctx.elicit("Priority?", response_type=["low", "medium", "high"])
    if result.action == "accept":
        return f"Priority: {result.data}"
    return "No priority set"
```

### Multi-turn elicitation

A tool can call `ctx.elicit()` multiple times. Each call triggers a separate
handler invocation. The handler must be stateful (or inspect the `message`
parameter) to respond differently to each prompt.

```python
@mcp.tool
async def multi_step(ctx: Context) -> str:
    """Two sequential elicitations: name then confirmation."""
    r1 = await ctx.elicit("Step 1: name?", response_type=str)
    if r1.action != "accept":
        return "Cancelled at step 1"
    r2 = await ctx.elicit("Step 2: confirm?", response_type=None)
    if r2.action != "accept":
        return "Rejected at step 2"
    return f"Completed: {r1.data}"
```

---

## Constraints

| # | Constraint | Impact |
|---|-----------|--------|
| 1 | **Flat schemas only** -- Elicitation schemas must be flat objects with primitive properties (string, number, integer, boolean, enum). No nesting, no arrays of objects. | `dataclass` / `BaseModel` fields must be primitive types. |
| 2 | **Bidirectional session required** -- Elicitation needs a stateful MCP session. | In-memory, stdio, and SSE transports work. Stateless HTTP mode breaks elicitation. See [Production](production.md). |
| 3 | **Timeout covers elicitation** -- Both server-side (`@mcp.tool(timeout=X)`) and client-side (`tool_timeout`) timeouts include human wait time, and remote mounts add an httpx read window (default 300 s of silence). | Set `tool_timeout=None` or a generous value; for remote mounts also see [Timeout](#timeout) (heartbeats / `MCPServer(timeout=)`). |
| 4 | **Scalar wrapping** -- Scalar types (str, int, bool) are wrapped in `{"value": <type>}` for MCP protocol compatibility. | Handlers must use `content={"value": ...}` for scalar responses. Transparent on the server side. |
| 5 | **`tool_choice` is evaluated on every turn** -- Passing `tool_choice={"specific": <tool_name>}` to `run_agent` forces the LLM to emit a tool call on *each* turn, including the terminating turn where the model needs to emit final text. The gateway propagates the provider's validation failure. | For elicitation-driven flows (approval, structured input), omit `tool_choice` and rely on prompt guidance to route the first turn. |

---

## Reference handlers

The library ships two handlers suitable for tests and prototypes:

```python
from t0_agents.mcp import auto_approve_handler, auto_decline_handler
```

| Handler | Behaviour | Use case |
|---------|-----------|----------|
| `auto_approve_handler` | Approves binary approval requests (`response_type is None`); declines structured-input requests. | Smoke tests for approval flows. |
| `auto_decline_handler` | Declines every request. | Verifying that tools degrade gracefully when the user refuses. |

Both are for testing only. Production code should implement a real handler --
either a CLI prompt (Intermediate example below), a stateful multi-turn handler
(Advanced example below), or a remote channel using the forwarder described in
[Agent](agent.md). The [Serving](serving.md) docs show how the forwarder is
wired into a remote HTTP session.

---

## Examples

### Easy -- auto-approve handler

Approve all elicitation requests unconditionally. Useful for testing.

```python
import asyncio
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from tensorzero import AsyncTensorZeroGateway
from t0_agents import run_agent

async def auto_approve(message, response_type, params, context):
    return ElicitResult(action="accept", content={})

async def main():
    gateway = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    )
    client = Client(mcp_server, elicitation_handler=auto_approve)

    async with await gateway as t0, client:
        result = await run_agent(
            t0, client, "Deploy the service", "agent",
            tool_timeout=None,
        )
        print(result.response)

asyncio.run(main())
```

### Intermediate -- response-type dispatch

Route approval and structured input to different logic based on
`response_type`.

```python
from fastmcp.client.elicitation import ElicitResult

async def interactive_handler(message, response_type, params, context):
    if response_type is None:
        # Binary approval -- ask via CLI
        answer = input(f"{message} [y/n]: ")
        if answer.lower() == "y":
            return ElicitResult(action="accept", content={})
        return ElicitResult(action="decline")
    else:
        # Structured input -- collect fields from the schema
        schema = params.requestedSchema
        content = {}
        for field_name, field_def in schema.get("properties", {}).items():
            value = input(f"{field_name} ({field_def.get('type', '?')}): ")
            # Coerce to the expected type
            if field_def.get("type") == "integer":
                content[field_name] = int(value)
            else:
                content[field_name] = value
        return ElicitResult(action="accept", content=content)
```

### Advanced -- stateful multi-turn handler

Track elicitation count to respond differently to sequential prompts from the
same tool.

```python
from fastmcp.client.elicitation import ElicitResult

def make_multi_turn_handler(responses: list[ElicitResult]) -> ...:
    """Create a handler that returns responses in sequence."""
    call_count = 0

    async def handler(message, response_type, params, context):
        nonlocal call_count
        result = responses[call_count]
        call_count += 1
        return result

    return handler

# Use with a multi-turn tool:
handler = make_multi_turn_handler([
    ElicitResult(action="accept", content={"value": "Alice"}),  # name
    ElicitResult(action="accept", content={}),                   # confirm
])
client = Client(mcp_server, elicitation_handler=handler)
```

---

Elicitation lets the loop pause for interactive input during a single run. Continuity *across* runs — remembering what was said yesterday, or in a different process, or after a redeploy — is a separate concern, handled by a `MemoryProvider` wired into `AgentConfig`. That is the subject of the next chapter: [Memory](memory.md).
