# Phase 2: Results Adapter (FastMCP → TensorZero) — Complete Implementation Guide

> This guide consolidates **all** the relevant FastMCP documentation needed to
> complete items F2.1 through F2.8. The TensorZero parts you already have in your
> attached document; here they are complemented with the FastMCP-specific details.

---

## Complete flow you are building

```
T0 inference response (ToolCall content blocks)
  → Your adapter extracts name, arguments, id
    → FastMCP client.call_tool(name, arguments, raise_on_error=False)
      → CallToolResult { .data, .content, .is_error, .structured_content }
        → serialize_tool_result() converts to string
          → T0 receives ToolResult(name, result, id) in the next turn
            → LLM produces a coherent response
```

---

## 1. Execute the tool via FastMCP — F2.1

### 1.1 Signature of `call_tool()`

```python
call_tool(
    self: Client,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    version: ... = None,
    timeout: timedelta | float | int | None = None,
    progress_handler: ProgressHandler | None = None,
    raise_on_error: bool = True,      # ← IMPORTANT for F2.6
    meta: dict[str, Any] | None = None,
    task: bool = False,
) -> CallToolResult | ToolTask
```

> **Source:** `clients.md` > "Calling Tools" (https://gofastmcp.com/clients/tools)
> and `python-sdk-fastmcp-client-mixins-tools.md`

### 1.2 Connection between T0 ToolCall and FastMCP call_tool

The `arguments` field of T0's `ToolCall` is already a `dict` (parsed JSON). FastMCP's
`call_tool()` expects exactly that: `dict[str, Any]`. **No additional parsing is needed.**

```python
from tensorzero import ToolCall
from fastmcp import Client

# T0 gives you:
for content_block in t0_response.content:
    if isinstance(content_block, ToolCall):
        tool_name = content_block.name        # str — validated name
        tool_args = content_block.arguments   # dict — validated arguments
        tool_id   = content_block.id          # str — unique ID

        # FastMCP receives directly:
        result = await client.call_tool(
            name=tool_name,
            arguments=tool_args,
            raise_on_error=False,  # To handle errors manually (F2.6)
        )
```

### 1.3 Prior validations (from T0)

Before calling `call_tool()`, check T0's validated fields:

| T0 field | Problematic value | What to do |
|-----------|-------------------|-----------|
| `content_block.name` | `None` | The model hallucinated a tool name. **DO NOT execute.** |
| `content_block.arguments` | `None` | The model produced invalid JSON. **DO NOT execute.** |

> **Source (T0):** Your attached document, Section 2 "Understanding TensorZero's ToolCall Response"

### 1.4 Client context

FastMCP requires the client to be inside an asynchronous context manager:

```python
from fastmcp import Client

client = Client("http://localhost:8000/mcp")  # or the URL of your server

async with client:
    # Here you can make multiple calls
    tools = await client.list_tools()
    result = await client.call_tool("my_tool", {"arg": "value"})
```

> **Source:** `getting-started.md` > "Call Your Server" and `clients.md`

---

## 2. Anatomy of `CallToolResult` — What you receive from FastMCP

### 2.1 Properties of the object

| Property | Type | Description |
|-----------|------|-------------|
| `.data` | `Any` | Fully hydrated Python objects with support for complex types (datetimes, UUIDs, custom classes). **Exclusive to FastMCP.** |
| `.content` | `list[mcp.types.ContentBlock]` | Standard MCP content blocks (`TextContent`, `ImageContent`, `AudioContent`, etc.) |
| `.structured_content` | `dict[str, Any] \| None` | Standard MCP structured JSON data sent by the server |
| `.is_error` | `bool` | Indicates whether the tool execution failed |

> **Source:** `clients.md` > "Structured Results" > "CallToolResult Properties"

### 2.2 Behavior of `.data`

- For tools **with output schema and complex types**: `.data` contains reconstituted
  Python objects (datetime, UUID, dataclasses, etc.)
- For tools **without output schema or when deserialization fails**: `.data` will be `None`
- **Primitives** (`int`, `str`, `bool`): FastMCP servers automatically wrap them in
  `{"result": value}`. FastMCP clients unwrap them, so `.data` gives you the original value.

```python
result = await client.call_tool("add", {"a": 5, "b": 3})
print(result.data)           # 8 (int, automatically unwrapped)
print(result.content[0].text) # "8" (always a string in content blocks)
```

> **Source:** `clients.md` > "Structured Results" and `servers-tools.md` > "Primitive Type Wrapping"

### 2.3 Behavior of `.content`

Always available. Content blocks **always** contain the string representation:

```python
# Dictionaries → JSON string in content
result = await client.call_tool("get_user", {"id": "123"})
# result.content[0].text → '{\n  "name": "Alice",\n  "age": 30\n}'
```

For tools that return complex data, there may be multiple content blocks
(`TextContent`, `ImageContent`, etc.).

> **Source:** `servers-tools.md` > "Dictionaries and Objects" (MCP Result example)

---

## 3. Serializing the result — F2.2 to F2.5

T0 **always** expects the result as a **string** in the `result` field of `ToolResult`.
Your serializer must handle the various types that FastMCP can return.

### 3.1 Complete decision table

| Case | `.data` | `.content` | `.is_error` | Strategy | Checklist |
|------|---------|------------|-------------|------------|-----------|
| String | `"70"` | `[TextContent("70")]` | `False` | `str(result.data)` or passthrough | **F2.2** |
| Int/Float/Bool | `8` / `3.14` / `True` | `[TextContent("8")]` | `False` | `str(result.data)` | **F2.2** |
| Dict | `{"temp": 70}` | `[TextContent('{"temp": 70}')]` | `False` | `json.dumps(result.data)` | **F2.3** |
| None (no schema) | `None` | `[TextContent("..."), ...]` | `False` | Fallback: concatenate `.content[*].text` | **F2.4** |
| Complex types | `datetime(...)`, `UUID(...)` | `[TextContent("...")]` | `False` | `json.dumps(result.data, default=str)` | **F2.5** |
| Error | — | `[TextContent("error msg")]` | `True` | Extract error text | **F2.6** |

### 3.2 Serializer implementation

```python
import json
from datetime import datetime
from uuid import UUID


def serialize_tool_result(result) -> str:
    """
    Converts a FastMCP CallToolResult into a string for T0's `result` field.

    Args:
        result: CallToolResult from client.call_tool(raise_on_error=False)

    Returns:
        str: Serialized result ready for T0
    """

    # ─── F2.6: Handle errors ─────────────────────────────────────
    if result.is_error:
        # When is_error=True, the error text is in content blocks
        error_texts = []
        for block in result.content:
            if hasattr(block, 'text'):
                error_texts.append(block.text)
        error_msg = "\n".join(error_texts) if error_texts else "Unknown tool error"
        return f"ERROR: {error_msg}"

    # ─── F2.2 / F2.3 / F2.5: Try with .data first ──────────────
    data = result.data

    if data is not None:
        # F2.2 — str passthrough
        if isinstance(data, str):
            return data

        # F2.2 — simple primitives
        if isinstance(data, (int, float, bool)):
            return str(data)

        # F2.3 — dict
        # F2.5 — complex types (datetime, UUID) via default=str
        if isinstance(data, dict):
            return json.dumps(data, default=str, ensure_ascii=False)

        # Other serializable objects (dataclass, Pydantic model, etc.)
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            pass  # Fallback to content blocks

    # ─── F2.4: Fallback to content blocks ────────────────────────
    text_parts = []
    for block in result.content:
        if hasattr(block, 'text'):
            text_parts.append(block.text)
    return "\n".join(text_parts) if text_parts else ""
```

### 3.3 Detail: Why can `.data` be `None`?

FastMCP's documentation explicitly indicates:

> "For tools without output schemas or when deserialization fails, `.data` will be
> `None`. Fall back to content blocks in that case."

This occurs when:
- The server's tool has no return type annotation
- The tool returns types that cannot be mapped to a JSON schema
- Deserialization fails for some reason

Pattern recommended by the documentation:

```python
if result.data is not None:
    print(f"Structured: {result.data}")
else:
    for content in result.content:
        if hasattr(content, 'text'):
            print(f"Text result: {content.text}")
```

> **Source:** `clients.md` > "Structured Results" — code block below "CallToolResult Properties"

### 3.4 Detail: Server-side structured content

The FastMCP server automatically generates structured content based on the return type:

- **Objects** (`dict`, Pydantic, dataclass) → `structuredContent` with the data + `content` with JSON string
- **Primitives with annotation** (`-> int`) → `structuredContent: {"result": 8}` + `content: [TextContent("8")]`
- **Primitives without annotation** → Only `content: [TextContent("8")]`, no `structuredContent`

> **Source:** `servers-tools.md` > "Structured Output" > "Automatic Structured Content Rules"

---

## 4. Error handling — F2.6

### 4.1 Two modes in FastMCP

FastMCP offers two modes for handling tool errors:

**Mode 1: Exception (default)**

```python
from fastmcp.exceptions import ToolError

try:
    result = await client.call_tool("failing_tool", {"param": "value"})
except ToolError as e:
    print(f"Tool failed: {e}")
```

**Mode 2: Manual with `raise_on_error=False`** ← **This is the one you need for integration with T0**

```python
result = await client.call_tool(
    "failing_tool",
    {"param": "value"},
    raise_on_error=False,
)

if result.is_error:
    print(f"Tool failed: {result.content[0].text}")
else:
    print(f"Tool succeeded: {result.data}")
```

> **Source:** `clients.md` > "Error Handling"

### 4.2 Why use `raise_on_error=False`?

For the integration with T0, you need to **always** return a result to the LLM, even
when the tool fails. The LLM can use the error information to adjust its response.
If you let the exception propagate, you break the inference flow.

### 4.3 Server-side error types

On the server side, errors are produced in two ways:

```python
# 1. ToolError — message always visible to the client
from fastmcp.exceptions import ToolError
raise ToolError("Division by zero is not allowed.")

# 2. Standard exceptions — visible by default, maskable
raise ValueError("Both arguments must be numbers.")
```

If the server uses `mask_error_details=True`, only `ToolError` messages will arrive
with detail; other exceptions are converted into a generic message.

> **Source:** `servers-tools.md` > "Error Handling"

### 4.4 Relevant FastMCP exceptions

| Exception | Module | When |
|-----------|--------|--------|
| `ToolError` | `fastmcp.exceptions` | Error in tool execution |
| `McpError` | MCP protocol | Timeout or JSON-RPC error |
| `RuntimeError` | Python stdlib | Client not connected |

> **Source:** `python-sdk-fastmcp-exceptions.md` and `python-sdk-fastmcp-client-mixins-tools.md`

---

## 5. Build the ToolResult for T0 — F2.7

### 5.1 Format T0 expects

**T0 native API:**

```python
from tensorzero import ToolResult, Message

t0_tool_result = ToolResult(
    name=content_block.name,    # str — same name as the ToolCall
    result=serialized_string,   # str — ALWAYS string
    id=content_block.id,        # str — same ID as the ToolCall
)
```

**Tool results go inside a message with `role="user"`:**

```python
messages.append(Message(role="user", content=[t0_tool_result]))
```

> **Source (T0):** Your attached document, Section 5

### 5.2 Complete execution loop

```python
from tensorzero import ToolCall, ToolResult, Message

tool_results = []

for content_block in t0_response.content:
    if isinstance(content_block, ToolCall):
        # Validate that T0 does not report an invalid tool call
        if content_block.name is None or content_block.arguments is None:
            tool_results.append(ToolResult(
                name=content_block.raw_name or "unknown",
                result="ERROR: Invalid tool call from model",
                id=content_block.id,
            ))
            continue

        # Execute via FastMCP
        mcp_result = await client.call_tool(
            name=content_block.name,
            arguments=content_block.arguments,
            raise_on_error=False,
        )

        # Serialize
        serialized = serialize_tool_result(mcp_result)

        # Build ToolResult for T0
        tool_results.append(ToolResult(
            name=content_block.name,
            result=serialized,
            id=content_block.id,
        ))

# Send all results in ONE single user message
if tool_results:
    messages.append(Message(role="user", content=tool_results))
```

### 5.3 Call the next inference

```python
# IMPORTANT: reuse the same episode_id
t0_response_2 = await t0_client.inference(
    function_name="your_function",
    episode_id=episode_id,          # ← the same one from the first inference
    input={"messages": messages},
    additional_tools=t0_tools,
)
```

> **Source (T0):** Your attached document, Section 6 "Episode Management"

---

## 6. Parallel tool calls

If T0 returns multiple `ToolCall` blocks in a single response (when
`parallel_tool_calls = true`), you must execute **all** of them and send **all**
the results in the same user message.

The loop in section 5.2 already handles this correctly: it iterates over all
content blocks and accumulates the results.

> **Source (T0):** Your attached document, Section 8 "Multiple Tool Calls in One Response"

---

## 7. Tool discovery for T0's `additional_tools`

To dynamically register FastMCP tools in T0, you need to list them
and convert their schema to T0's format:

### 7.1 List tools

```python
async with client:
    tools = await client.list_tools()
    # tools → list[mcp.types.Tool]
```

Each `mcp.types.Tool` has:
- `.name` — name of the tool
- `.description` — description
- `.inputSchema` — JSON Schema of the parameters (dict)

> **Source:** `python-sdk-fastmcp-client-mixins-tools.md` > `list_tools`

### 7.2 Convert to T0 format

```python
t0_tools = [
    {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,   # JSON Schema object, NOT file path
    }
    for tool in mcp_tools
]
```

> **Source (T0):** Your attached document, Section 7 "Dynamic (at inference time via additional_tools)"

---

## 8. Mixed content blocks (Text + Tool Calls)

T0's assistant response may contain **both** `text` and `tool_call` blocks:

```json
{
  "content": [
    { "type": "text", "text": "Sure, I can help you with that." },
    { "type": "tool_call", "id": "0", "name": "get_temperature", "arguments": "{...}" }
  ]
}
```

When you resend this message as an assistant message, include **all** content blocks.

> **Source (T0):** Your attached document, Section 8 "Mixed Content Blocks"

---

## 9. Complete end-to-end example

```python
import asyncio
import json
from datetime import datetime
from uuid import UUID

from fastmcp import Client
from fastmcp.exceptions import ToolError
from tensorzero import AsyncTensorZeroGateway, ToolCall, ToolResult, Text, Message


# ─── Serializer ──────────────────────────────────────────────────
def serialize_tool_result(result) -> str:
    """Converts FastMCP CallToolResult → string for T0."""

    # F2.6: errors
    if result.is_error:
        error_texts = [b.text for b in result.content if hasattr(b, 'text')]
        return f"ERROR: {chr(10).join(error_texts) or 'Unknown error'}"

    # F2.2/F2.3/F2.5: try with .data
    data = result.data
    if data is not None:
        if isinstance(data, str):
            return data
        if isinstance(data, (int, float, bool)):
            return str(data)
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            pass

    # F2.4: fallback to content blocks
    return "\n".join(b.text for b in result.content if hasattr(b, 'text'))


# ─── Main flow ──────────────────────────────────────────────────
async def run_tool_loop(
    t0: AsyncTensorZeroGateway,
    mcp_client: Client,
    function_name: str,
    user_query: str,
):
    async with mcp_client:
        # 1. Discover FastMCP tools
        mcp_tools = await mcp_client.list_tools()
        t0_tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

        # 2. First inference
        messages = [Message(role="user", content=[Text(text=user_query)])]
        response = await t0.inference(
            function_name=function_name,
            input={"messages": messages},
            additional_tools=t0_tools,
        )
        episode_id = response.episode_id

        # 3. Tool use loop
        while any(isinstance(b, ToolCall) for b in response.content):
            # Add assistant response to messages
            messages.append(Message(role="assistant", content=response.content))

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if not isinstance(block, ToolCall):
                    continue

                if block.name is None or block.arguments is None:
                    tool_results.append(ToolResult(
                        name=block.raw_name or "unknown",
                        result="ERROR: Invalid tool call",
                        id=block.id,
                    ))
                    continue

                # F2.1: Execute via FastMCP
                mcp_result = await mcp_client.call_tool(
                    name=block.name,
                    arguments=block.arguments,
                    raise_on_error=False,
                )

                # F2.2-F2.6: Serialize
                serialized = serialize_tool_result(mcp_result)

                # F2.7: Build ToolResult for T0
                tool_results.append(ToolResult(
                    name=block.name,
                    result=serialized,
                    id=block.id,
                ))

            # Send results to T0
            messages.append(Message(role="user", content=tool_results))
            response = await t0.inference(
                function_name=function_name,
                episode_id=episode_id,
                input={"messages": messages},
                additional_tools=t0_tools,
            )

        # F2.8: Final response from the LLM
        final_text = "\n".join(
            b.text for b in response.content
            if isinstance(b, Text)
        )
        return final_text
```

---

## 10. Verification checklist

| Item | What to verify | FastMCP key | T0 key |
|------|---------------|---------------|----------|
| **F2.1** | `call_tool(name, arguments)` receives correct values | `arguments` is `dict[str, Any]`, same as T0's `ToolCall.arguments` | `ToolCall.arguments` is a validated dict |
| **F2.2** | `str` passes through unmodified | `.data` returns `str` directly; `.content[0].text` is also a string | `result` field is always a string |
| **F2.3** | `dict` is converted with `json.dumps()` | `.data` returns `dict`; `.structured_content` has the raw dict | `json.dumps(data, default=str)` |
| **F2.4** | `None` falls back to `.content[*].text` | `.data` is `None` when there is no output schema | Concatenate texts with `"\n".join()` |
| **F2.5** | `datetime`, `UUID` are handled | `.data` has hydrated objects; `default=str` serializes them | `json.dumps(data, default=str)` |
| **F2.6** | `.is_error=True` is handled | `raise_on_error=False` → `.is_error` bool; text in `.content` | Serialize as `"ERROR: ..."` |
| **F2.7** | `ToolResult` accepted by T0 | N/A | `ToolResult(name=, result=, id=)` in Message `role="user"` |
| **F2.8** | LLM produces coherent response | N/A | Complete end-to-end test |

---

## 11. Pitfalls and important notes

### FastMCP

1. **Always use `raise_on_error=False`** in the context of integration with T0.
   If you leave the default (`True`), a `ToolError` will break your inference loop.

2. **The client must be in a context manager** (`async with client:`).
   Without this, `call_tool()` raises `RuntimeError`.

3. **`.data` is NOT always available.** It depends on whether the server's tool
   has a return type annotation. Always implement the fallback to `.content`.

4. **Primitives are automatically wrapped.** If the tool returns `int` with
   annotation `-> int`, the server sends `{"result": 8}` as structured content.
   The FastMCP client unwraps it, so `.data` gives you `8` directly.

5. **Multi-server clients prefix names.** If you use a multi-server client,
   the tool names are prefixed with the server name
   (e.g., `weather_get_forecast`). Make sure the names match what
   T0 expects.

### TensorZero

6. **`result` is ALWAYS a string.** No matter what type the tool returns,
   the `result` field of `ToolResult` must be `str`.

7. **Reuse `episode_id`.** The first inference creates an episode;
   all subsequent ones must use the same one.

8. **`additional_tools.parameters` is a JSON Schema object**, not a file path.
   This matches FastMCP's `tool.inputSchema`.

9. **Do not execute invalid tool calls.** If `name` or `arguments` is `None`
   in T0's `ToolCall`, the model hallucinated. Return an error to the LLM.

10. **Mixed content blocks.** The assistant response may have `text` AND
    `tool_call` blocks. Include all when rebuilding the message history.

---

## 12. Sources

| Topic | File | Section / URL |
|------|---------|---------------|
| `call_tool()` signature and options | `clients.md` | "Calling Tools" — https://gofastmcp.com/clients/tools |
| `CallToolResult` properties | `clients.md` | "Structured Results" > "CallToolResult Properties" |
| `.data` fallback to content blocks | `clients.md` | Code block post-Properties |
| Primitive wrapping/unwrapping | `clients.md` (Tip) + `servers-tools.md` | "Primitive Type Wrapping" |
| Error handling (client) | `clients.md` | "Error Handling" |
| Error handling (server) | `servers-tools.md` | "Error Handling" |
| `ToolError` exception | `python-sdk-fastmcp-exceptions.md` | `fastmcp.exceptions.ToolError` |
| `call_tool()` overloads and raises | `python-sdk-fastmcp-client-mixins-tools.md` | `ClientToolsMixin` |
| `list_tools()` | `python-sdk-fastmcp-client-mixins-tools.md` | `list_tools` |
| Structured output rules | `servers-tools.md` | "Structured Output" > "Automatic Structured Content Rules" |
| Client context / connection | `getting-started.md` | "Call Your Server" |
| T0 ToolCall response format | Your attached document | Section 2 |
| T0 ToolResult spec | Your attached document | Section 4 |
| T0 Episode management | Your attached document | Section 6 |
| T0 Dynamic tools | Your attached document | Section 7 |
| T0 Parallel tool calls | Your attached document | Section 8 |
