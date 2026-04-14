# Integration Contract: Elicitation in t0-agents (Final)

> **Version:** 1.0 (final)
> **Date:** 2026-04-14
> **Supersedes:** `integration-contract-elicitation-v02.md`
> **Scope:** Human-in-the-loop approval AND structured user input via MCP Elicitation in a headless library where TensorZero is a stateless inference gateway.

---

## 1. Architectural Decision

**Use Elicitation for everything.** One MCP-native mechanism covers both binary approval and structured input.

| Need | Pattern | `response_type` |
|------|---------|-----------------|
| Binary approval | `ctx.elicit("Approve X?", response_type=None)` | `None` |
| Choice from options | `ctx.elicit("Pick one", response_type=["a","b","c"])` | `list[str]` |
| Scalar input | `ctx.elicit("How many?", response_type=int)` | `int`, `str`, `bool` |
| Structured input | `ctx.elicit("Details?", response_type=MyModel)` | `dataclass` / `BaseModel` |

### Discarded Alternatives

| Alternative | Reason for rejection |
|-------------|---------------------|
| **FastMCP Approval provider** | Requires `fastmcp[apps]`, Prefab UI renderer, and a host supporting `structuredContent` in sandboxed iframes. Incompatible with headless. *(source: apps.md > "Approval")* |
| **Middleware `on_call_tool`** | Docs do not show calling `ctx.elicit()` from middleware. Middleware can `raise ToolError` (programmatic deny) but has no documented path to pause-and-ask-a-human. *(source: servers-middleware.md > "Operation Hooks")* |
| **Custom callbacks outside FastMCP** | Loses MCP context, re-invents what elicitation already does, and adds a second abstraction for consumers to implement. |

---

## 2. How Elicitation Flows Through the System

### 2.1 Protocol Flow

MCP sessions are bidirectional. The server can send requests (like `elicitation/create`) to the client while a `tools/call` response is still pending.

```
Agent loop                    Client                         Server
    |                            |                              |
    |  result = call_tool(name)  |                              |
    | -------------------------► |  tools/call ----------------►|
    |                            |                              |  tool starts executing
    |                            |                              |  ctx.elicit("Approve?")
    |                            |  ◄--- elicitation/create --- |  tool PAUSED (awaiting)
    |                            |                              |
    |                            |  elicitation_handler() runs  |
    |                            |  (human interacts)           |
    |                            |                              |
    |                            |  response ------------------►|  ctx.elicit() returns
    |                            |                              |  tool RESUMES
    |                            |                              |  tool returns final result
    |  ◄----------------------- |  ◄-- tools/call response --- |
    |  result = CallToolResult   |                              |
```

**Key conclusion:** From `call_tool()`'s perspective, elicitation is **transparent**. It looks like a slow tool call. The `await client.call_tool(...)` simply takes longer because the tool is paused waiting for human input.

### 2.2 What `run_agent` Needs to Change

**Nothing structural.** The agent loop already awaits `call_tool()`. Whether that call takes 50ms (no elicitation) or 60s (human thinking) is irrelevant to the loop.

The only concern is **timeout** (see section 3).

### 2.3 Client Lifecycle

The `Client` is created and owned by the **caller**, not by `run_agent`:

```python
# Current API (unchanged)
async def run_agent(
    t0: AsyncTensorZeroGateway,
    mcp_client: Client[Any],    # <-- caller passes pre-connected Client
    user_message: str,
    function_name: str,
    ...
) -> AgentResult
```

The caller configures `elicitation_handler` when creating the Client:

```python
# Caller's code
client = Client("server.py", elicitation_handler=my_handler)
async with client:
    result = await run_agent(t0, client, "Do the thing", "agent")
```

**No new parameter on `run_agent`.** This preserves the existing contract where the caller owns the Client's configuration and lifecycle.

### 2.4 TensorZero's Role

TensorZero is a **stateless inference gateway**. It receives messages + tool definitions, returns a response. It does not know about elicitation and needs **zero changes**.

```
+------------------------------------------------------+
|  run_agent loop                                      |
|                                                      |
|  while not done:                                     |
|      response = t0.inference(messages, tools)         |  <-- T0: stateless, no changes
|      if has_tool_calls(response):                    |
|          for call in tool_calls:                     |
|              result = await client.call_tool(...)     |  <-- may trigger elicitation
|              messages.append(result)                  |     (transparent to loop)
|      else:                                           |
|          done = True                                 |
+------------------------------------------------------+
```

### 2.5 Adapter Layer

`serialize_tool_result` in `adapter.py` needs **zero changes**. The `CallToolResult` contains only the tool's final return value. Elicitation metadata is consumed inside the tool during execution — it is not present in the result.

**Verified in FastMCP source:** `CallToolResult` exposes `.data`, `.content`, `.structured_content`, `.is_error` — no elicitation fields.

---

## 3. Timeout Interaction (Verified)

### 3.1 The Problem

`_execute_single_tool_call` in `loop.py:110-111`:

```python
mcp_result = await mcp_client.call_tool(
    block.name, block.arguments, timeout=timeout
)
```

Default `tool_timeout=30.0` seconds. Elicitation adds human wait time to the call duration.

### 3.2 Verified Behavior

**Source:** `mcp/shared/session.py:284-303` (MCP SDK)

The timeout wraps `anyio.fail_after(timeout)` around the response `receive()` call. The entire `tools/call` request-response cycle — including any intermediate elicitation roundtrip — is under this single timeout.

```python
# mcp/shared/session.py:291
with anyio.fail_after(timeout):
    response_or_error = await response_stream_reader.receive()
```

**Conclusion:** If `tool_timeout=30` and the human takes 45 seconds to respond to an elicitation prompt, the call times out with `McpError` code `-32000`.

### 3.3 Resolution

**Timeout is the caller's responsibility.** When using elicitation-enabled tools, the caller should set `tool_timeout` to `None` (no timeout) or a generous value:

```python
# With elicitation: disable or increase timeout
result = await run_agent(
    t0, client, "Do the thing", "agent",
    tool_timeout=None,  # or tool_timeout=300.0
)
```

This matches the existing design: `tool_timeout` is already a caller-configurable parameter. No new parameters needed.

**Trade-off:** Setting `tool_timeout=None` removes timeout protection for all tools, not just elicitation-capable ones. If the caller has a mix of fast tools and elicitation tools, they should use a generous timeout (e.g., 300s) rather than `None`. A per-tool timeout mechanism is a future enhancement, not a blocker for this integration.

---

## 4. Error Handling (Verified)

### 4.1 No Handler Present

**Source:** `mcp/client/session.py:75-82`

When a tool calls `ctx.elicit()` but the Client has no `elicitation_handler`, the default MCP callback returns:

```python
ErrorData(code=INVALID_REQUEST, message="Elicitation not supported")
```

Code: `-32600` (INVALID_REQUEST)

The server receives this as a `JSONRPCError` and raises `McpError` inside the tool. This propagates to `call_tool()`, which raises `McpError` to the caller.

### 4.2 Current Error Contract Handles It

`_execute_single_tool_call` in `loop.py:123-131` already catches `McpError`:

```python
except McpError as e:
    return ToolResult(
        name=name,
        result=f"ERROR: MCP protocol error (possibly timeout): {e}",
        id=block.id,
    )
```

**No code change needed.** The LLM receives the error as a `ToolResult` string and can decide how to proceed (retry, skip, or explain the failure to the user).

### 4.3 Client Capability Advertisement

**Source:** `mcp/client/session.py:148-161`

During session initialization, the Client only advertises elicitation capability if the callback is NOT the default:

```python
elicitation = (
    ElicitationCapability(form=FormElicitationCapability(), ...)
    if self._elicitation_callback is not _default_elicitation_callback
    else None
)
```

This means a properly configured server could check client capabilities before calling `ctx.elicit()`. However, this is server-side logic — t0-agents doesn't need to act on it.

---

## 5. In-Memory Testing (Verified)

### 5.1 Bidirectional Session Support

**Source:** `fastmcp/client/transports/memory.py:32-82`

The in-memory transport creates true bidirectional memory streams via `create_client_server_memory_streams()`. The server runs in the same `anyio` task group. This supports all session features including elicitation.

### 5.2 Test Pattern

```python
from fastmcp import FastMCP, Client, Context
from fastmcp.client.elicitation import ElicitResult

mcp = FastMCP("TestServer")

@mcp.tool
async def needs_approval(ctx: Context) -> str:
    result = await ctx.elicit("Approve?", response_type=None)
    if result.action == "accept":
        return "approved"
    return "rejected"

async def test_elicitation_approve():
    async def handler(message, response_type, params, context):
        return ElicitResult(action="accept", content={})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("needs_approval", {})
        assert result.data == "approved"
```

The existing `Client(mcp)` pattern used throughout the test suite works for elicitation. No transport changes needed.

---

## 6. The Abstraction Point: `elicitation_handler`

### 6.1 Handler Signature

```python
from fastmcp.client.elicitation import ElicitResult, ElicitRequestParams, RequestContext

async def elicitation_handler(
    message: str,
    response_type: type | None,
    params: ElicitRequestParams,
    context: RequestContext,
) -> ElicitResult | object
```

*(source: clients.md > "User Elicitation")*

### 6.2 Parameter Semantics

| Parameter | Type | Meaning |
|-----------|------|---------|
| `message` | `str` | Human-readable prompt to display |
| `response_type` | `type \| None` | `None` = approval (no data). Otherwise, a type auto-generated from the server's JSON schema |
| `params` | `ElicitRequestParams` | Raw MCP params including `requestedSchema` |
| `context` | `RequestContext` | Request metadata |

### 6.3 Response Contract

```python
# Accept with data (or implicit accept by returning the object directly)
return ElicitResult(action="accept", content=response_type(value=user_input))
return response_type(value=user_input)  # implicit accept

# Accept approval (response_type=None) — send empty object
return ElicitResult(action="accept", content={})

# Decline (user chose not to provide info)
return ElicitResult(action="decline")

# Cancel (user cancelled the operation)
return ElicitResult(action="cancel")
```

### 6.4 Distinguishing Approval from Input

```python
async def elicitation_handler(message, response_type, params, context):
    if response_type is None:
        # APPROVAL — binary accept/decline
        approved = await present_approval_prompt(message)
        if approved:
            return ElicitResult(action="accept", content={})
        return ElicitResult(action="decline")
    else:
        # STRUCTURED INPUT — collect data matching the schema
        data = await present_input_form(message, params.requestedSchema)
        if data is None:
            return ElicitResult(action="decline")
        return response_type(**data)
```

---

## 7. Server-Side Patterns (Reference)

### 7.1 Binary Approval

```python
@mcp.tool
async def delete_files(files: list[str], ctx: Context) -> str:
    result = await ctx.elicit(
        f"Delete {len(files)} files?", response_type=None
    )
    if result.action != "accept":
        return "Rejected by user"
    # proceed with deletion...
```

### 7.2 Reusable Approval Helper

```python
async def require_approval(ctx: Context, summary: str) -> bool:
    result = await ctx.elicit(f"Approve: {summary}?", response_type=None)
    return result.action == "accept"
```

### 7.3 Structured Input

```python
from dataclasses import dataclass

@dataclass
class TaskDetails:
    title: str
    priority: str  # flat fields only — MCP restriction

@mcp.tool
async def create_task(ctx: Context) -> str:
    result = await ctx.elicit("Task details?", response_type=TaskDetails)
    match result:
        case AcceptedElicitation(data=task):
            return f"Created: {task.title}"
        case DeclinedElicitation():
            return "Cancelled"
        case CancelledElicitation():
            return "Cancelled"
```

### 7.4 Multi-Turn Elicitation

```python
@mcp.tool
async def plan_meeting(ctx: Context) -> str:
    r1 = await ctx.elicit("Meeting title?", response_type=str)
    if r1.action != "accept":
        return "Cancelled"
    r2 = await ctx.elicit("Duration in minutes?", response_type=int)
    if r2.action != "accept":
        return "Cancelled"
    return f"Meeting '{r1.data}' for {r2.data} minutes"
```

---

## 8. Constraints and Limitations

| # | Constraint | Impact | Source |
|---|-----------|--------|--------|
| 1 | **Flat JSON schemas only** — Elicitation schemas must be flat objects with primitive properties (string, number, integer, boolean, enum). No nesting, no arrays of objects. | `dataclass` / `BaseModel` fields must be primitives. | `python-sdk-fastmcp-server-elicitation.md` |
| 2 | **Client must support elicitation** — If no handler, `McpError(-32600, "Elicitation not supported")` is raised. | Already handled by `_execute_single_tool_call`'s `McpError` catch. | `mcp/client/session.py:75-82` |
| 3 | **Stateful session required** — Elicitation requires bidirectional MCP session. Stateless HTTP mode breaks it. | Not an issue for current usage (in-memory and stdio transports are stateful). | `deployment.md > "Enabling Stateless Mode"` |
| 4 | **Timeout covers elicitation** — Both server-side (`@mcp.tool(timeout=X)`) and client-side (`call_tool(timeout=X)`) timeouts include human wait time. | Caller must set `tool_timeout=None` or generous value when using elicitation. | `mcp/shared/session.py:284-303` |
| 5 | **Scalar wrapping** — Scalar types (str, int, bool) are wrapped in `{"value": <type>}` for MCP compatibility. FastMCP unwraps automatically on the server side. | Transparent to both tool author and handler implementor. | `servers-elicitation.md > "Scalar Types"` |
| 6 | **Background task elicitation** — Requires Redis/Valkey backend and `fastmcp[tasks]`. | Out of scope for initial implementation. | `python-sdk-fastmcp-server-tasks-elicitation.md` |

---

## 9. Summary of Changes to t0-agents

| Component | Change | Reason |
|-----------|--------|--------|
| `run_agent` signature | **None** | Caller configures handler on Client |
| `_execute_single_tool_call` | **None** | `McpError` already caught |
| `serialize_tool_result` | **None** | `CallToolResult` shape unchanged |
| `build_tool_result` | **None** | Same reason |
| `convert_tools` | **None** | Tool schemas unchanged |
| `AgentResult` | **None** | Return type unchanged |
| Test infrastructure | **Add** elicitation tools + tests | Validate the integration |
| Documentation | **Add** elicitation guide + examples | Enable consumers |
