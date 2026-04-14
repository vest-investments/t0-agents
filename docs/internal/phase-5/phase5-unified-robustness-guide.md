# Phase 5: Robustness and Error Handling — Unified Reference Guide

> **Scope:** All documentation needed to implement F5.1–F5.8 for a TensorZero + FastMCP integration.
> Every claim is traced to a specific documentation file with source and original URL.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Exception Taxonomy](#2-exception-taxonomy)
3. [TensorZero Response Format for Tool Calls](#3-tensorzero-response-format-for-tool-calls)
4. [Sending Tool Results Back to TensorZero](#4-sending-tool-results-back-to-tensorzero)
5. [F5.1 — ToolError from FastMCP](#5-f51--toolerror-from-fastmcp)
6. [F5.2 — McpError (timeout, protocol)](#6-f52--mcperror-timeout-protocol)
7. [F5.3 — arguments is None (invalid JSON from LLM)](#7-f53--arguments-is-none-invalid-json-from-llm)
8. [F5.4 — Timeout in call_tool](#8-f54--timeout-in-call_tool)
9. [F5.5 — LLM Can Retry After Error](#9-f55--llm-can-retry-after-error)
10. [F5.6 — ValidationError from FastMCP](#10-f56--validationerror-from-fastmcp)
11. [F5.7 — Maximum Turn Limit](#11-f57--maximum-turn-limit)
12. [F5.8 — Tools with Empty or None Description](#12-f58--tools-with-empty-or-none-description)
13. [MCP Client Choice: mcp.ClientSession vs FastMCP Client](#13-mcp-client-choice-mcpclientsession-vs-fastmcp-client)
14. [Complete Reference Implementation](#14-complete-reference-implementation)
15. [Conversion: MCP Tools → TensorZero Format](#15-conversion-mcp-tools--tensorzero-format)
16. [Quick Decision Matrix](#16-quick-decision-matrix)
17. [Import Reference](#17-import-reference)
18. [Sources](#18-sources)

---

## 1. Architecture Overview

The agent loop has three participants:

```
TensorZero Gateway ←→ Your Agent Loop ←→ FastMCP Server
(LLM inference)        (orchestrator)     (tool execution)
```

The flow per turn is:

1. Your code calls `t0.inference(...)` → TensorZero returns content blocks.
2. For each `ToolCall` block, your code calls `session.call_tool(...)` or `client.call_tool(...)` → FastMCP executes the tool.
3. Your code collects results as `ToolResult` blocks and sends them back to TensorZero in the next inference call.
4. The loop continues until the LLM produces only text (no tool calls).

**Every error in step 2 must be caught and converted to a `ToolResult` string** — TensorZero has no dedicated error field; errors are communicated to the LLM as the `result` string.

---

## 2. Exception Taxonomy

### FastMCP exceptions (`fastmcp.exceptions`)

| Exception          | Purpose                                        | Relevance       |
|--------------------|------------------------------------------------|-----------------|
| `FastMCPError`     | Base error for all FastMCP errors              | Parent class    |
| `ToolError`        | Error in tool operations                       | **F5.1**        |
| `ValidationError`  | Error in validating parameters or return values| **F5.6**        |
| `ResourceError`    | Error in resource operations                   | Not applicable  |
| `PromptError`      | Error in prompt operations                     | Not applicable  |
| `ClientError`      | Error in client operations                     | Edge cases      |
| `InvalidSignature` | Invalid function signature for FastMCP         | Setup-time only |

> **Source:** `python-sdk-fastmcp-exceptions.md` — Original URL: https://gofastmcp.com/python-sdk/fastmcp-exceptions.md

### MCP protocol exception (`mcp` package)

| Exception  | Purpose                                      | Relevance       |
|------------|----------------------------------------------|-----------------|
| `McpError` | Wraps `TimeoutError` and `JSONRPCError`      | **F5.2, F5.4**  |

> **Source:** `python-sdk-fastmcp-client-mixins-tools.md` — Original URL: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md

### Key distinction

| Error Type       | Origin                  | When it fires                                  |
|------------------|-------------------------|------------------------------------------------|
| `ToolError`      | Tool function raised it | Business logic errors during tool execution    |
| `McpError`       | MCP protocol layer      | Timeout, JSON-RPC error, transport failure     |
| `ValidationError`| Pydantic validation     | Tool arguments don't match expected schema     |

---

## 3. TensorZero Response Format for Tool Calls

When TensorZero returns a `tool_call` content block, it provides **both validated and raw fields**:

```
type: "tool_call"
├── arguments  (object | null)   — validated; NULL if LLM produced invalid JSON
├── id         (string)          — content block ID (pass back in tool_result)
├── name       (string | null)   — validated tool name; NULL if invalid
├── raw_arguments (string)       — raw string from the model (may be malformed)
├── raw_name      (string)       — raw tool name string from the model
```

### Key implications

- `arguments == None` → the LLM produced invalid JSON → **F5.3 detection signal**.
- `name == None` → the LLM hallucinated a tool name that doesn't exist.
- `raw_arguments` is always a string — log it for debugging even when `arguments` is `None`.
- `raw_name` is always a string — log it even when `name` is `None`.

> **Source:** `gateway-api-reference.md` > "API Reference: Inference" > Response > `content` field
> **Original URL:** https://www.tensorzero.com/docs/gateway/api-reference/inference.md

---

## 4. Sending Tool Results Back to TensorZero

Tool results are sent as content blocks inside a **user** message:

```json
{
  "type": "tool_result",
  "id": "123456789",
  "name": "get_temperature",
  "result": "25"
}
```

| Field    | Type   | Required | Notes                                      |
|----------|--------|----------|--------------------------------------------|
| `id`     | string | yes      | Must match the `id` from the `tool_call`   |
| `name`   | string | yes      | Must match the tool name                   |
| `result` | string | yes      | **Must be a string** — not an object/number|

### Critical: There is no `is_error` field

TensorZero's `tool_result` has **no dedicated error field**. You communicate errors to the LLM by putting the error description in the `result` string. The LLM receives it as part of the conversation and can decide to retry or adjust.

```python
t0_tool_result = ToolResult(
    name=content_block.name or content_block.raw_name,
    result=f"Error: {str(error_message)}",  # free-form string
    id=content_block.id,
)
```

> **Source:** `gateway-api-reference.md` > "API Reference: Inference" > Request > `input.messages` > content blocks
> **Original URL:** https://www.tensorzero.com/docs/gateway/api-reference/inference.md

---

## 5. F5.1 — ToolError from FastMCP

> `ToolError` from FastMCP is caught and sent as a ToolResult with an error message to the LLM.

### FastMCP side: What raises ToolError

When a tool encounters an error, the developer can raise `ToolError` explicitly:

```python
from fastmcp.exceptions import ToolError

@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ToolError("Division by zero is not allowed.")
    return a / b
```

FastMCP also converts **any unhandled exception** (e.g., `ValueError`, `TypeError`) into an MCP error response. The behavior depends on `mask_error_details`:

- **`mask_error_details=False` (default):** All exception details are included in the error response.
- **`mask_error_details=True`:** Only `ToolError` messages pass through; other exceptions get a generic message like `"Error executing tool 'divide'"`.

> "By default, all exceptions (including their details) are logged and converted into an MCP error response to be sent back to the client LLM. This helps the LLM understand failures and react appropriately."

> "Error messages from ToolError are always sent to clients, regardless of mask_error_details setting."

> **Source:** `servers-tools.md` > "Error Handling" — Original URL: https://gofastmcp.com/servers/tools

### Client-side behavior

How you catch `ToolError` depends on which MCP client you use (see [Section 13](#13-mcp-client-choice-mcpclientsession-vs-fastmcp-client)):

**If using `mcp.ClientSession` (as in the T0 reference example):**
The low-level `session.call_tool()` returns `CallToolResult` with an `isError` flag. It does **not** raise `ToolError` — you check the flag manually:

```python
mcp_result = await session.call_tool(name, arguments)
if mcp_result.isError:
    error_text = mcp_result.content[0].text
```

**If using FastMCP's `Client`:**
`client.call_tool()` raises `ToolError` by default (controllable with `raise_on_error`):

```python
# Option A: Catch exception (default behavior)
try:
    result = await client.call_tool("my_tool", arguments)
except ToolError as e:
    error_text = str(e)

# Option B: Disable auto-raise, check flag
result = await client.call_tool("my_tool", arguments, raise_on_error=False)
if result.is_error:
    error_text = result.content[0].text
```

> **Source:** `clients.md` > "Error Handling" — Original URL: https://gofastmcp.com/clients/tools

### TensorZero side: Catch and send as ToolResult

```python
from tensorzero import ToolResult

# After catching the error (however you detect it):
t0_tool_result = ToolResult(
    name=content_block.name or content_block.raw_name,
    result=f"ToolError: {error_text}",
    id=content_block.id,
)
tool_results.append(t0_tool_result)
```

> **Source:** `gateway-api-reference.md` > tool_result content block

---

## 6. F5.2 — McpError (timeout, protocol)

> `McpError` from mcp (timeout, protocol) is caught and sent as a ToolResult with error to the LLM.

### What raises McpError

`McpError` is thrown by the MCP protocol layer — not by tool code. It wraps `TimeoutError` and `JSONRPCError`. From the FastMCP docs:

> **Raises:**
> - `ToolError`: If the tool call results in an error.
> - `McpError`: If the tool call request results in a TimeoutError | JSONRPCError
> - `RuntimeError`: If called while the client is not connected.

This applies regardless of whether you use `mcp.ClientSession` or FastMCP `Client` — both can raise `McpError` at the protocol level.

> **Source:** `python-sdk-fastmcp-client-mixins-tools.md` — Original URL: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md

### When McpError occurs

| Trigger              | Cause                                              |
|----------------------|----------------------------------------------------|
| `TimeoutError`       | Tool execution exceeded timeout                    |
| `JSONRPCError`       | Malformed protocol message, transport failure      |
| Connection failure   | Server crashed, pipe broken, network issue         |

### Implementation

```python
from mcp.shared.exceptions import McpError  # verify exact path for your mcp version

try:
    mcp_result = await session.call_tool(name, arguments)
except McpError as e:
    t0_tool_result = ToolResult(
        name=content_block.name or content_block.raw_name,
        result=f"McpError (protocol/timeout): {e}",
        id=content_block.id,
    )
    tool_results.append(t0_tool_result)
```

---

## 7. F5.3 — arguments is None (invalid JSON from LLM)

> `block.arguments is None` (invalid JSON from the LLM) is detected and reported as an error to the LLM.

### Detection (TensorZero side)

TensorZero parses the LLM's tool call JSON. If parsing fails:

```
content_block.arguments = None    (null)
content_block.raw_arguments = "<the raw malformed string>"
```

Similarly, if the tool name is invalid:

```
content_block.name = None         (null)
content_block.raw_name = "<the raw string>"
```

> **Source:** `gateway-api-reference.md` > Response > `content` field (tool_call)

### Why check before calling FastMCP

FastMCP's `call_tool()` accepts `arguments: dict[str, Any] | None = None`, so passing `None` won't crash immediately. However, the server will likely reject it if the tool requires parameters — and you'd be making an unnecessary round-trip.

> **Source:** `python-sdk-fastmcp-client-mixins-tools.md` — call_tool signature

### Implementation

Check **before** calling the MCP server:

```python
for content_block in response.content:
    if isinstance(content_block, ToolCall):
        if content_block.arguments is None or content_block.name is None:
            error_msg = (
                f"Invalid tool call: name={content_block.raw_name!r}, "
                f"arguments={content_block.raw_arguments!r}. "
                "Please provide valid JSON arguments for a valid tool."
            )
            t0_tool_result = ToolResult(
                name=content_block.raw_name or "unknown",
                result=error_msg,
                id=content_block.id,
            )
            tool_results.append(t0_tool_result)
            continue  # skip the call_tool entirely

        # Proceed with valid tool call...
```

**Important:** You must still send a `ToolResult` with the matching `id` — the LLM expects a result for every tool call it made, or the conversation state breaks.

> **Source:** `gateway-api-reference.md` > `tool_result` content block

---

## 8. F5.4 — Timeout in call_tool

> Timeout in `call_tool(..., timeout=N)` generates `McpError` that is caught correctly.

### Client-side timeout parameter

Both MCP client options accept a timeout:

**FastMCP Client:**

```python
result = await client.call_tool(
    "long_running_task",
    {"param": "value"},
    timeout=30.0  # seconds (accepts float, int, or timedelta)
)
```

> **Source:** `clients.md` > "Execution Options" — Original URL: https://gofastmcp.com/clients/tools

**mcp.ClientSession:**

```python
mcp_result = await session.call_tool(
    name,
    arguments,
    # timeout depends on the low-level mcp SDK version
)
```

### What happens on timeout

When the timeout is exceeded, `McpError` is raised (wrapping `TimeoutError`). On the server side, if the tool has a server-side timeout configured, the error includes code `-32000`:

> "When a tool exceeds its timeout, FastMCP returns an MCP error with code `-32000` and a message indicating which tool timed out and how long it ran."

Server-side timeout (configured by the tool author):

```python
@mcp.tool(timeout=30.0)
async def fetch_data(url: str) -> dict:
    """Fetch data with a 30-second timeout."""
    ...
```

> "Tools must explicitly opt-in to timeouts. There is no server-level default timeout setting."

> **Source:** `servers-tools.md` > "Timeouts" — Original URL: https://gofastmcp.com/servers/tools

### Implementation

The timeout exception is a `McpError` — handled identically to F5.2:

```python
try:
    mcp_result = await session.call_tool(name, arguments)
except McpError as e:
    # Covers both timeout and other protocol errors
    t0_tool_result = ToolResult(
        name=content_block.name or content_block.raw_name,
        result=f"Tool call failed (timeout/protocol): {e}",
        id=content_block.id,
    )
    tool_results.append(t0_tool_result)
```

---

## 9. F5.5 — LLM Can Retry After Error

> The LLM can retry a tool after receiving an error.

### Mechanism

This works automatically if errors are sent as `ToolResult` strings. From FastMCP:

> "Tool errors are caught and sent back to the LLM, allowing it to recover gracefully."

> **Source:** `servers-sampling.md` > "Defining Tools" — Original URL: https://gofastmcp.com/servers/sampling

From TensorZero's perspective, the mechanism is:

1. You send the error as a `ToolResult` with `result="Error: ..."`.
2. TensorZero sends this to the LLM in the next inference call as part of the conversation history.
3. The LLM sees the error and can choose to retry (issue another `tool_call`) or respond with text.
4. Your loop continues processing `tool_call` blocks as normal.

No special configuration is needed on either side. The loop structure naturally supports retries — the `while` loop keeps running as long as there are `tool_call` blocks in the response.

> **Source:** `examples/mcp-model-context-protocol/main.py` (T0 project knowledge)

### What makes retry possible

The key design decision: **errors are tool results, not terminal failures.** Both TensorZero and FastMCP treat tool errors as data that flows back into the conversation, not as exceptions that break the loop.

---

## 10. F5.6 — ValidationError from FastMCP

> `ValidationError` from FastMCP (invalid schema parameters) is handled without a crash.

### When it fires

`ValidationError` is raised when tool arguments fail Pydantic validation against the tool's schema. FastMCP has two validation modes:

**Flexible (default, `strict_input_validation=False`):** Coerces compatible types (e.g., `"10"` → `10` for `int`).

**Strict (`strict_input_validation=True`):** Rejects any type mismatch.

| Input Type                         | Flexible (default) | Strict          |
|------------------------------------|--------------------|--------------------|
| `"10"` for `int`                   | ✅ Coerced          | ❌ ValidationError |
| `"3.14"` for `float`              | ✅ Coerced          | ❌ ValidationError |
| `"true"` for `bool`               | ✅ Coerced          | ❌ ValidationError |
| `"abc"` for `int`                  | ❌ ValidationError  | ❌ ValidationError |

> **Source:** `servers-tools.md` > "Validation Modes" — Original URL: https://gofastmcp.com/servers/tools

### How it surfaces to the client

On the server, a `ValidationError` is converted into an MCP error response — the same way `ToolError` is. From the client's perspective:

- **`mcp.ClientSession`:** `result.isError == True` with the validation message in `result.content`.
- **FastMCP `Client`:** Raises `ToolError` (not `ValidationError`) because the server already converted it.

> "By default, all exceptions (including their details) are logged and converted into an MCP error response to be sent back to the client LLM."

> **Source:** `servers-tools.md` > "Error Handling"

### Implementation

Your existing `ToolError` / `isError` catch (F5.1) already covers this. But as a defense-in-depth, include `ValidationError` in your catch chain if using FastMCP Client directly, or rely on `isError` if using `mcp.ClientSession`:

```python
# If using FastMCP Client:
from fastmcp.exceptions import ToolError, ValidationError

try:
    result = await client.call_tool(name, arguments)
except (ToolError, ValidationError) as e:
    error_text = f"Validation/Tool error: {e}"

# If using mcp.ClientSession:
mcp_result = await session.call_tool(name, arguments)
if mcp_result.isError:
    error_text = mcp_result.content[0].text  # includes validation error message
```

### Environment variables

```
FASTMCP_MASK_ERROR_DETAILS=false       # default; set true to hide internals
FASTMCP_STRICT_INPUT_VALIDATION=false  # default; set true for strict mode
```

> **Source:** `more.md` > "Error Handling" — Original URL: https://gofastmcp.com/more

---

## 11. F5.7 — Maximum Turn Limit

> Maximum turn limit for the loop (prevents infinite loop if the LLM always requests tools).

### Neither TensorZero nor your custom loop has a built-in limit

TensorZero does **not** have a built-in turn limit for the agent loop. The loop is entirely in your application code.

FastMCP's `sample()` has an internal safety limit, but that only applies to FastMCP's own sampling loop — **not to your TensorZero integration loop**:

> "An internal safety limit prevents infinite loops."

Since you're building TensorZero → call_tool → TensorZero externally, you must implement your own guard.

> **Source:** `servers-sampling.md` > "Defining Tools" — Original URL: https://gofastmcp.com/servers/sampling
> **Source:** `examples/mcp-model-context-protocol/main.py` (T0 reference: `while True` with no limit)

### Implementation

```python
MAX_TOOL_TURNS = 10  # adjust as needed
tool_turn_count = 0

while True:
    response = await t0.inference(...)

    tool_results = []
    for content_block in response.content:
        if isinstance(content_block, ToolCall):
            tool_turn_count += 1
            if tool_turn_count > MAX_TOOL_TURNS:
                tool_results.append(ToolResult(
                    name=content_block.name or content_block.raw_name or "unknown",
                    result="Error: Maximum tool call limit reached. Provide a final answer now.",
                    id=content_block.id,
                ))
                continue  # still collect results for ALL tool_calls in this turn
            # ... normal tool execution ...

    if tool_results:
        messages.append(Message(role="user", content=tool_results))
        if tool_turn_count > MAX_TOOL_TURNS:
            # One final inference so the LLM can produce a text response
            final = await t0.inference(...)
            messages.append(Message(role="assistant", content=final.content))
            break
    else:
        break  # No tool calls → final text response
```

### Counting strategy

The counter increments **per individual tool call**, not per turn. A single LLM response can contain multiple `ToolCall` blocks — all count toward the limit.

---

## 12. F5.8 — Tools with Empty or None Description

> Tools with empty or `None` `description` do not cause an error in T0.

### FastMCP side: How descriptions are derived

FastMCP uses the function's docstring as the description. If no docstring and no explicit `description` argument, the tool's description will be `None`:

> "Each `@mcp.tool` function is self-contained: its name becomes the tool name, its docstring becomes the description, its type annotations become the JSON Schema"

The `description` parameter on `@mcp.tool` is typed `str | None` — it is optional.

> **Source:** `servers-tools.md` > "Decorator Arguments" — Original URL: https://gofastmcp.com/servers/tools

### TensorZero side: description requirements differ by API

| API                            | `description` requirement |
|--------------------------------|---------------------------|
| Native API (`POST /inference`) — `additional_tools` | **Required** (string)    |
| OpenAI-compatible API (`POST /openai/v1/chat/completions`) — `tools` | Optional |
| Config file (`tensorzero.toml`) — `[tools.tool_name]`              | **Required** (string)    |

> **Source (Native API):** `gateway-api-reference.md` > "additional_tools"
> **Source (OpenAI API):** `gateway-api-reference.md` > "API Reference: Inference (OpenAI-compatible)" > `tools`
> **Source (Config):** `gateway-configuration-reference.md` > `[tools.tool_name]`

### Implementation: Provide fallback on conversion

```python
t0_tools = [
    {
        "name": tool["name"],
        # F5.8: guard against None/empty description
        "description": tool.get("description") or "No description available.",
        "parameters": tool.get("inputSchema") or {
            "type": "object",
            "properties": {},
        },
    }
    for tool in mcp_tools
]
```

---

## 13. MCP Client Choice: mcp.ClientSession vs FastMCP Client

The T0 reference example uses the **low-level** `mcp.ClientSession`:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
```

FastMCP offers a **higher-level** `Client`:

```python
from fastmcp import Client
```

### Error behavior comparison

| Behavior                   | `mcp.ClientSession`                    | FastMCP `Client`                              |
|----------------------------|----------------------------------------|-----------------------------------------------|
| Tool error                 | Returns `CallToolResult` with `isError=True` | Raises `ToolError` (default) or returns `is_error=True` with `raise_on_error=False` |
| Protocol/timeout error     | Raises `McpError`                      | Raises `McpError`                             |
| Raw protocol access        | `session.call_tool()` always returns raw | `client.call_tool_mcp()` for raw protocol     |

### Which to use

If you want **minimal changes** from the T0 reference example: use `mcp.ClientSession`. Check `result.isError` manually.

If you want **higher-level features** (structured results, auto-deserialization, progress handlers): use FastMCP `Client`. Use `raise_on_error=False` or `call_tool_mcp()` for the agent loop pattern.

> **Source (mcp.ClientSession):** `examples/mcp-model-context-protocol/main.py` (T0 reference)
> **Source (FastMCP Client):** `clients.md` > "Error Handling" — Original URL: https://gofastmcp.com/clients/tools
> **Source (call_tool_mcp):** `python-sdk-fastmcp-client-mixins-tools.md`

---

## 14. Complete Reference Implementation

Below is the unified skeleton that addresses all 8 checklist items. It uses `mcp.ClientSession` (matching the T0 reference example) with specific exception handling.

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult

MAX_TOOL_TURNS = 10


async def load_mcp_tools_for_t0(session: ClientSession) -> list[dict]:
    """Load MCP tools and convert to TensorZero additional_tools format.
    Handles empty/None descriptions safely (F5.8)."""
    mcp_tools = await session.list_tools()
    mcp_tools_list = [tool.model_dump() for tool in mcp_tools.tools]

    t0_tools = []
    for tool in mcp_tools_list:
        t0_tools.append({
            "name": tool["name"],
            # F5.8: description is REQUIRED in T0 native API
            "description": tool.get("description") or "No description available.",
            "parameters": tool.get("inputSchema") or {
                "type": "object",
                "properties": {},
            },
        })

    return t0_tools


async def agent_loop(
    t0: AsyncTensorZeroGateway,
    session: ClientSession,
    t0_tools: list[dict],
    function_name: str,
    user_input: str,
    tool_call_timeout: float = 30.0,
):
    """Agent loop with full F5.1–F5.8 error handling."""

    messages = [Message(role="user", content=user_input)]
    episode_id = None
    tool_turn_count = 0

    while True:
        # ── TensorZero inference ──────────────────────────────────
        response = await t0.inference(
            function_name=function_name,
            episode_id=episode_id,
            input={"messages": messages},
            additional_tools=t0_tools,
        )
        episode_id = response.episode_id

        assistant_message = Message(role="assistant", content=response.content)
        messages.append(assistant_message)

        # ── Process content blocks ────────────────────────────────
        tool_results: list[ToolResult] = []

        for content_block in response.content:
            if isinstance(content_block, Text):
                # Final text output — handle as needed
                pass

            elif isinstance(content_block, ToolCall):

                # ── F5.3: Invalid JSON or invalid tool name ───────
                if content_block.arguments is None or content_block.name is None:
                    tool_results.append(ToolResult(
                        name=content_block.raw_name or "unknown",
                        result=(
                            f"Invalid tool call: name={content_block.raw_name!r}, "
                            f"arguments={content_block.raw_arguments!r}. "
                            "Please provide valid JSON arguments for a valid tool."
                        ),
                        id=content_block.id,
                    ))
                    continue

                # ── F5.7: Maximum turn limit ──────────────────────
                tool_turn_count += 1
                if tool_turn_count > MAX_TOOL_TURNS:
                    tool_results.append(ToolResult(
                        name=content_block.name,
                        result=(
                            "Error: Maximum tool call limit reached. "
                            "Please provide a final answer now."
                        ),
                        id=content_block.id,
                    ))
                    continue

                # ── F5.1, F5.2, F5.4, F5.6: Execute tool with error handling
                try:
                    mcp_result = await session.call_tool(
                        content_block.name,
                        content_block.arguments,
                    )

                    # Check isError flag (covers ToolError + ValidationError
                    # on the server side)  — F5.1, F5.6
                    if mcp_result.isError:
                        result_text = (
                            f"Error from tool '{content_block.raw_name}': "
                            + (mcp_result.content[0].text
                               if mcp_result.content else "Unknown error")
                        )
                    else:
                        # Success: join all content blocks
                        result_text = "\n".join(
                            block.text
                            for block in mcp_result.content
                            if hasattr(block, "text")
                        )

                except McpError as e:
                    # F5.2 / F5.4: Protocol error or timeout
                    result_text = (
                        f"MCP protocol error calling '{content_block.raw_name}': "
                        f"{type(e).__name__}: {e}"
                    )

                except Exception as e:
                    # Catch-all for unexpected errors (defense in depth)
                    result_text = (
                        f"Unexpected error calling '{content_block.raw_name}': "
                        f"{type(e).__name__}: {e}"
                    )

                tool_results.append(ToolResult(
                    name=content_block.name,
                    result=result_text,
                    id=content_block.id,
                ))

        # ── F5.5: LLM retries automatically via the loop ─────────
        if tool_results:
            messages.append(Message(role="user", content=tool_results))

            # If we hit the turn limit, do one final inference
            if tool_turn_count > MAX_TOOL_TURNS:
                final = await t0.inference(
                    function_name=function_name,
                    episode_id=episode_id,
                    input={"messages": messages},
                    additional_tools=t0_tools,
                )
                messages.append(
                    Message(role="assistant", content=final.content)
                )
                break
        else:
            # No tool calls → LLM produced a final text response
            break

    return messages, episode_id


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "my_mcp_server"],
        env={},
    )

    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as t0:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                t0_tools = await load_mcp_tools_for_t0(session)

                messages, episode_id = await agent_loop(
                    t0=t0,
                    session=session,
                    t0_tools=t0_tools,
                    function_name="my_function",
                    user_input="What is the weather in Mexico City?",
                )


if __name__ == "__main__":
    asyncio.run(main())
```

### Checklist coverage in the implementation

| Line / Block | Covers |
|---|---|
| `if content_block.arguments is None or content_block.name is None` | **F5.3** |
| `if tool_turn_count > MAX_TOOL_TURNS` | **F5.7** |
| `if mcp_result.isError` | **F5.1** (ToolError) + **F5.6** (ValidationError) |
| `except McpError` | **F5.2** (protocol) + **F5.4** (timeout) |
| Loop continues after errors → `messages.append(...)` | **F5.5** (retry) |
| `tool.get("description") or "No description available."` | **F5.8** |
| `except Exception` (catch-all) | Defense-in-depth for unforeseen errors |

---

## 15. Conversion: MCP Tools → TensorZero Format

```python
async def load_mcp_tools_for_t0(session):
    """Load MCP tools and convert to TensorZero additional_tools format."""
    mcp_tools = await session.list_tools()
    mcp_tools_list = [tool.model_dump() for tool in mcp_tools.tools]

    t0_tools = []
    for tool in mcp_tools_list:
        t0_tools.append({
            "name": tool["name"],
            "description": tool.get("description") or "No description available.",
            "parameters": tool.get("inputSchema") or {
                "type": "object",
                "properties": {},
            },
        })

    return t0_tools
```

### Field mapping

| MCP `Tool` field  | TensorZero `additional_tools` field | Notes                    |
|--------------------|--------------------------------------|--------------------------|
| `name`             | `name`                               | Required on both sides   |
| `description`      | `description`                        | Optional in MCP, **required** in T0 native API — provide fallback |
| `inputSchema`      | `parameters`                         | JSON Schema object       |

> **Source (MCP):** `getting-started.md` > "Tools" — `inputSchema` is generated from type hints
> **Source (T0):** `gateway-api-reference.md` > `additional_tools`

---

## 16. Quick Decision Matrix

| Item | What to Check | Exception / Condition | Action |
|---|---|---|---|
| **F5.1** | Tool business error | `mcp_result.isError == True` | Extract `.content[0].text` → send as `ToolResult.result` |
| **F5.2** | Protocol error | `except McpError` | `str(e)` → send as `ToolResult.result` |
| **F5.3** | Invalid JSON | `content_block.arguments is None` | Skip `call_tool`, send parse error as `ToolResult.result` |
| **F5.4** | Timeout | `except McpError` (wraps TimeoutError) | Same as F5.2 |
| **F5.5** | Retry | Loop structure | Errors are tool results → loop continues → LLM decides |
| **F5.6** | Schema validation | `mcp_result.isError == True` | Server converts ValidationError → same as F5.1 |
| **F5.7** | Infinite loop | `tool_turn_count > MAX_TOOL_TURNS` | Send limit message → one final inference → break |
| **F5.8** | Empty description | `tool.get("description") is None` | `or "No description available."` on conversion |

---

## 17. Import Reference

```python
# ── TensorZero ────────────────────────────────────────────────
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult

# ── MCP low-level client (as used in T0 reference example) ────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

# ── FastMCP exceptions (if using FastMCP Client instead) ──────
from fastmcp.exceptions import ToolError
from fastmcp.exceptions import ValidationError
from fastmcp.exceptions import FastMCPError
from fastmcp.exceptions import ClientError

# ── FastMCP Client (alternative to mcp.ClientSession) ────────
from fastmcp import Client

# ── FastMCP middleware (if needed on server side) ─────────────
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.error_handling import RetryMiddleware
```

---

## 18. Sources

### FastMCP Documentation

| File | Section | Original URL |
|------|---------|-------------|
| `python-sdk-fastmcp-exceptions.md` | Exception hierarchy | https://gofastmcp.com/python-sdk/fastmcp-exceptions.md |
| `servers-tools.md` | Error Handling, Timeouts, Validation Modes, Decorator Arguments | https://gofastmcp.com/servers/tools |
| `clients.md` | Error Handling, Execution Options, Raw Protocol Access | https://gofastmcp.com/clients/tools |
| `python-sdk-fastmcp-client-mixins-tools.md` | call_tool, call_tool_mcp signatures and raises | https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md |
| `servers-sampling.md` | Tool error recovery, safety limit, mask_error_details | https://gofastmcp.com/servers/sampling |
| `servers-middleware.md` | ErrorHandlingMiddleware, error types by operation | https://gofastmcp.com/servers/middleware |
| `more.md` | Environment variables for error handling | https://gofastmcp.com/more |
| `python-sdk-fastmcp-utilities-timeout.md` | Timeout normalization | https://gofastmcp.com/python-sdk/fastmcp-utilities-timeout.md |
| `python-sdk-fastmcp-server-sampling-run.md` | execute_tools, sample_impl | https://gofastmcp.com/python-sdk/fastmcp-server-sampling-run.md |
| `python-sdk-fastmcp-server-sampling-sampling_tool.md` | SamplingTool.from_function | https://gofastmcp.com/python-sdk/fastmcp-server-sampling-sampling_tool.md |
| `getting-started.md` | Tool definition, inputSchema derivation | https://gofastmcp.com/getting-started |

### TensorZero Documentation

| File | Section | Original URL |
|------|---------|-------------|
| `gateway-api-reference.md` | Inference API: additional_tools, tool_call response, tool_result request | https://www.tensorzero.com/docs/gateway/api-reference/inference.md |
| `gateway-api-reference.md` | OpenAI-compatible API: tools field | https://www.tensorzero.com/docs/gateway/api-reference/inference.md |
| `gateway-guides.md` | Tool Use (Function Calling): MCP integration, multi-turn | https://www.tensorzero.com/docs/gateway/guides/tool-use.md |
| `gateway-configuration-reference.md` | `[tools.tool_name]`: description, parameters | https://www.tensorzero.com/docs/gateway/configuration-reference/ |
| `examples/mcp-model-context-protocol/main.py` | Reference agent loop | https://github.com/tensorzero/tensorzero/tree/main/examples/mcp-model-context-protocol |
