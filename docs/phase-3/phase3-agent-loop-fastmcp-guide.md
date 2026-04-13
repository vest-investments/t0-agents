# Phase 3: Agent Loop — FastMCP Implementation Guide

> All information in this guide is sourced exclusively from the FastMCP project knowledge documentation (gofastmcp.com).
> This is the **companion** to the TensorZero implementation guide. Together they cover the complete agent loop.

---

## Table of Contents

1. [FastMCP Client — Core Concepts](#1-fastmcp-client--core-concepts)
2. [Connecting to an MCP Server](#2-connecting-to-an-mcp-server)
3. [Listing Tools (Discovery)](#3-listing-tools-discovery)
4. [Converting MCP Tools to TensorZero Format](#4-converting-mcp-tools-to-tensorzero-format)
5. [Calling Tools (Execution)](#5-calling-tools-execution)
6. [Extracting Text from Tool Results](#6-extracting-text-from-tool-results)
7. [Error Handling](#7-error-handling)
8. [Parallel Tool Execution with asyncio.gather](#8-parallel-tool-execution-with-asynciogather)
9. [Client Lifecycle & Session Management](#9-client-lifecycle--session-management)
10. [Complete Integration Code](#10-complete-integration-code)
11. [Gotchas & Easy-to-Miss Details (FastMCP Side)](#11-gotchas--easy-to-miss-details-fastmcp-side)
12. [Checklist Mapping (FastMCP Responsibilities)](#12-checklist-mapping-fastmcp-responsibilities)
13. [Source Citations](#13-source-citations)

---

## 1. FastMCP Client — Core Concepts

The `Client` class is the main entry point for connecting to MCP servers. It handles:

- MCP protocol logic (handshake, tool listing, tool calling)
- Transport negotiation (STDIO, HTTP, SSE, in-memory)
- Session lifecycle via async context managers
- Reentrant context managers with reference counting for concurrent usage

```python
from fastmcp import Client
```

The `Client` class supports multiple transport types inferred automatically from what you pass:

| Input | Transport |
|-------|-----------|
| `Client("http://localhost:8000/mcp")` | Streamable HTTP (recommended for production) |
| `Client("http://localhost:8000/sse")` | SSE (legacy, backward compat only) |
| `Client("server.py")` | STDIO via `PythonStdioTransport` |
| `Client(fastmcp_server_instance)` | In-memory (ideal for testing) |
| `Client({"mcpServers": {...}})` | Multi-server MCPConfig |

> (source: `clients.md` > "Client Transports", URL: https://gofastmcp.com/clients/transports.md;
> `python-sdk-fastmcp-client-transports-inference.md` > `infer_transport`)

---

## 2. Connecting to an MCP Server

### 2.1 Basic pattern: `async with client:`

All client operations **must** happen inside an async context manager:

```python
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async with client:
    tools = await client.list_tools()
    result = await client.call_tool("my_tool", {"arg": "value"})
```

> **Critical:** You cannot call `list_tools()` or `call_tool()` outside `async with client:`. Doing so raises `RuntimeError: Client is not connected`.

> (source: `clients.md` > "Basic Execution"; `getting-started.md` > "Call Your Server")

### 2.2 STDIO transport (subprocess)

When connecting to a local MCP server file:

```python
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

transport = StdioTransport(
    command="python",
    args=["my_server.py"],
    env={"API_KEY": "secret"},  # STDIO does NOT inherit your env
    cwd="/path/to/server"
)
client = Client(transport)
```

**Important:** STDIO servers run in isolated environments. They do **not** inherit your shell's environment variables. You must explicitly pass any configuration the server needs.

For convenience, you can pass a file path directly, but this limits configuration:

```python
client = Client("my_server.py")  # Limited - no env/cwd options
```

> (source: `clients.md` > "STDIO Transport")

### 2.3 HTTP transport

```python
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer your-token-here"}
)
client = Client(transport)
```

Or using auth helpers:

```python
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

client = Client(
    "https://api.example.com/mcp",
    auth=BearerAuth("your-token-here")
)
```

> (source: `clients.md` > "HTTP Transport", URL: https://gofastmcp.com/clients/transports.md)

### 2.4 In-memory transport (for testing)

Connect directly to a `FastMCP` server instance in the same process:

```python
from fastmcp import FastMCP, Client

mcp = FastMCP("TestServer")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

client = Client(mcp)

async with client:
    result = await client.call_tool("greet", {"name": "World"})
```

> (source: `clients.md` > "In-Memory Transport")

### 2.5 Session persistence (STDIO)

STDIO transports maintain sessions across multiple `async with` blocks by default (`keep_alive=True`):

```python
transport = StdioTransport(command="python", args=["server.py"])
client = Client(transport)

async with client:
    await client.ping()

async with client:  # Reuses the same subprocess
    await client.call_tool("process_data", {"file": "data.csv"})
```

For complete isolation between connections:

```python
transport = StdioTransport(command="python", args=["server.py"], keep_alive=False)
```

> (source: `clients.md` > "Session Persistence")

---

## 3. Listing Tools (Discovery)

### 3.1 `list_tools()` — high-level, auto-paginated

```python
async with client:
    tools = await client.list_tools()
    # Returns: list[mcp.types.Tool]
```

This method automatically fetches **all pages** if the server paginates results. Each `Tool` object has:

- `tool.name` — the tool name (string)
- `tool.description` — the tool description (string or None)
- `tool.inputSchema` — the JSON Schema for the tool's parameters (dict)

> (source: `python-sdk-fastmcp-client-mixins-tools.md` > `list_tools`)

### 3.2 `list_tools_mcp()` — low-level, raw protocol

```python
async with client:
    result = await client.list_tools_mcp()
    # Returns: mcp.types.ListToolsResult
    # Access tools via: result.tools
```

Use this for manual pagination control with the `cursor` parameter.

> (source: `python-sdk-fastmcp-client-mixins-tools.md` > `list_tools_mcp`)

---

## 4. Converting MCP Tools to TensorZero Format

This is the bridge between FastMCP and TensorZero. The `mcp.types.Tool` objects from `list_tools()` must be converted to TensorZero's `additional_tools` format.

```python
async with client:
    mcp_tools = await client.list_tools()

    # Convert to TensorZero format
    t0_tools = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools
    ]
```

**Key mapping:**

| FastMCP `mcp.types.Tool` attribute | TensorZero `additional_tools` field |
|-------------------------------------|--------------------------------------|
| `tool.name` | `"name"` |
| `tool.description` | `"description"` |
| `tool.inputSchema` | `"parameters"` |

**Important notes:**
- `tool.inputSchema` is already a JSON Schema `dict` — pass it directly as `parameters`
- TensorZero's `parameters` in `additional_tools` expects the schema **object** itself (not a file path). This is different from the TOML config where `parameters` is a file path.
- `tool.description` may be `None` if the server tool has no docstring — use `or ""` to avoid passing `None`

> (source: `python-sdk-fastmcp-client-mixins-tools.md` > `list_tools`;
> `servers-tools.md` > tool decorator arguments;
> TensorZero guide Section 2.3 for the target format)

---

## 5. Calling Tools (Execution)

### 5.1 `call_tool()` — high-level, raises on error

```python
async with client:
    result = await client.call_tool("tool_name", {"arg1": "value1"})
```

**Signature:**
```python
call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    version: str | None = None,      # Specific tool version
    timeout: float | None = None,     # Timeout in seconds
    progress_handler: ProgressHandler | None = None,
    raise_on_error: bool = True,      # Raise ToolError on failure
    meta: dict[str, Any] | None = None,
) -> CallToolResult
```

**Returns a `CallToolResult`** with the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `.data` | `Any` | Fully hydrated Python objects (FastMCP exclusive). `None` if no output schema. |
| `.content` | `list[ContentBlock]` | Standard MCP content blocks (`TextContent`, `ImageContent`, etc.) |
| `.structured_content` | `dict \| None` | Raw structured JSON data |
| `.is_error` | `bool` | Whether the tool execution failed |

**By default, `call_tool()` raises `ToolError` on failure.** This is different from `call_tool_mcp()` which does not raise.

> (source: `clients.md` > "Calling Tools", URL: https://gofastmcp.com/clients/tools.md;
> `python-sdk-fastmcp-client-mixins-tools.md` > `call_tool`)

### 5.2 `call_tool_mcp()` — low-level, no auto-raise

```python
async with client:
    result = await client.call_tool_mcp("tool_name", {"arg": "value"})
    # Returns: mcp.types.CallToolResult (raw protocol object)
    
    if result.isError:
        print(f"Tool failed: {result.content}")
    else:
        print(f"Success: {result.content}")
```

Use this when you need raw protocol access or want to handle errors manually.

> (source: `clients.md` > "Raw Protocol Access";
> `python-sdk-fastmcp-client-mixins-tools.md` > `call_tool_mcp`)

---

## 6. Extracting Text from Tool Results

This is the critical step for bridging FastMCP tool results back to TensorZero's `tool_result.result` field, which **must be a string**.

### 6.1 Using `.content` blocks (recommended for TensorZero integration)

```python
result = await client.call_tool("tool_name", arguments)

# Extract text from content blocks
result_text = "\n".join(
    block.text
    for block in result.content
    if hasattr(block, 'text')
)
```

This pattern handles:
- `TextContent` blocks (have `.text`)
- Skips `ImageContent`, `AudioContent`, etc. that don't have `.text`

### 6.2 Using `.data` (FastMCP structured results)

```python
result = await client.call_tool("tool_name", arguments)

if result.data is not None:
    # Structured data — serialize to string for TensorZero
    result_text = str(result.data)
else:
    # Fall back to content blocks
    result_text = "\n".join(
        block.text
        for block in result.content
        if hasattr(block, 'text')
    )
```

**Note:** FastMCP servers automatically wrap primitive results (like `int`, `str`, `bool`) in a `{"result": value}` structure. FastMCP clients automatically unwrap this, so `.data` gives you the original value.

> (source: `clients.md` > "Structured Results" and "For tools without output schemas")

### 6.3 Complete `execute_tool` function for the agent loop

```python
async def execute_tool(
    client: Client,
    tool_name: str,
    arguments: dict,
) -> str:
    """Execute an MCP tool and return the result as a string for TensorZero."""
    result = await client.call_tool(tool_name, arguments)
    
    # Extract text from content blocks
    text_parts = [
        block.text
        for block in result.content
        if hasattr(block, 'text')
    ]
    return "\n".join(text_parts) if text_parts else ""
```

---

## 7. Error Handling

### 7.1 Default behavior: `ToolError` on failure

```python
from fastmcp.exceptions import ToolError

async with client:
    try:
        result = await client.call_tool("tool_name", {"arg": "value"})
    except ToolError as e:
        print(f"Tool failed: {e}")
```

> (source: `clients.md` > "Error Handling";
> `python-sdk-fastmcp-exceptions.md` > `ToolError`)

### 7.2 Manual error handling with `raise_on_error=False`

```python
async with client:
    result = await client.call_tool(
        "tool_name",
        {"arg": "value"},
        raise_on_error=False
    )
    
    if result.is_error:
        error_text = result.content[0].text if result.content else "Unknown error"
        print(f"Tool failed: {error_text}")
    else:
        print(f"Success: {result.data}")
```

> (source: `clients.md` > "Error Handling")

### 7.3 Error handling in the agent loop

When an MCP tool fails, you still need to send a `tool_result` back to TensorZero so the LLM knows the tool failed and can react:

```python
async def execute_tool_safe(
    client: Client,
    tool_name: str,
    arguments: dict,
) -> str:
    """Execute a tool, returning error text on failure instead of raising."""
    try:
        result = await client.call_tool(tool_name, arguments)
        text_parts = [
            block.text for block in result.content
            if hasattr(block, 'text')
        ]
        return "\n".join(text_parts) if text_parts else ""
    except ToolError as e:
        return f"Error executing tool '{tool_name}': {e}"
```

This way, even if a tool fails, the loop continues — the error message goes back to the LLM as a `tool_result`, and the LLM can decide how to proceed.

### 7.4 Exception hierarchy

From the FastMCP exceptions module:

| Exception | Description |
|-----------|-------------|
| `FastMCPError` | Base error for all FastMCP errors |
| `ToolError` | Error in tool operations (raised by `call_tool()` on failure) |
| `ClientError` | Error in client operations |
| `ValidationError` | Error in parameter/return value validation |
| `ResourceError` | Error in resource operations |

```python
from fastmcp.exceptions import ToolError, ClientError, FastMCPError
```

> (source: `python-sdk-fastmcp-exceptions.md`)

---

## 8. Parallel Tool Execution with asyncio.gather

When TensorZero returns multiple `tool_call` blocks in one response (F3.5, F3.8), you can execute them in parallel using `asyncio.gather`.

### 8.1 Basic parallel execution

```python
import asyncio

# Assuming tool_call_blocks is a list of ToolCall objects from TensorZero's response
results = await asyncio.gather(*[
    client.call_tool(block.name, block.arguments)
    for block in tool_call_blocks
])

# Convert results to TensorZero ToolResult objects
tool_results = []
for block, result in zip(tool_call_blocks, results):
    text_parts = [
        content.text for content in result.content
        if hasattr(content, 'text')
    ]
    tool_results.append(ToolResult(
        name=block.name,
        result="\n".join(text_parts) if text_parts else "",
        id=block.id,
    ))
```

### 8.2 Parallel execution with error handling

```python
import asyncio

async def execute_one_tool(client, block):
    """Execute a single tool call, catching errors."""
    try:
        result = await client.call_tool(block.name, block.arguments)
        text_parts = [c.text for c in result.content if hasattr(c, 'text')]
        return "\n".join(text_parts) if text_parts else ""
    except ToolError as e:
        return f"Error: {e}"

# Execute all tool calls concurrently
result_texts = await asyncio.gather(*[
    execute_one_tool(client, block)
    for block in tool_call_blocks
])

# Build TensorZero ToolResult list
tool_results = [
    ToolResult(name=block.name, result=text, id=block.id)
    for block, text in zip(tool_call_blocks, result_texts)
]

# ALL results go in ONE user message (TensorZero requirement)
messages.append(Message(role="user", content=tool_results))
```

### 8.3 Will `asyncio.gather` break FastMCP?

The `Client` class supports reentrant context managers with reference counting. Multiple concurrent `call_tool()` calls within the same `async with client:` block are safe. The client uses a background task to manage the session lifecycle and events to coordinate between tasks.

> (source: `python-sdk-fastmcp-client-client.md` > `Client` class docstring — describes reference counting and concurrent usage support)

---

## 9. Client Lifecycle & Session Management

### 9.1 The client context manager

The `Client` uses `async with` for lifecycle management:

```python
client = Client("http://localhost:8000/mcp")

async with client:
    # Session is active here
    tools = await client.list_tools()
    result = await client.call_tool("my_tool", {"arg": "value"})
# Session is closed here
```

### 9.2 Keeping the client alive for the entire agent loop

For the agent loop, you want the client session to span the **entire conversation**, not just one tool call. The correct pattern:

```python
async def agent_loop():
    client = Client("http://localhost:8000/mcp")
    
    async with client:
        # Discover tools ONCE
        mcp_tools = await client.list_tools()
        t0_tools = [
            {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}
            for t in mcp_tools
        ]
        
        messages = []
        episode_id = None
        
        while True:
            # ... TensorZero inference loop ...
            # ... tool execution via client.call_tool() ...
            pass
```

**Do NOT create a new `async with client:` block for each tool call.** That would tear down and recreate the session each time, wasting resources and potentially losing server-side session state.

### 9.3 Client constructor parameters

```python
client = Client(
    transport,                          # See section 2 for options
    timeout=30.0,                       # Request timeout (seconds or timedelta)
    init_timeout=10.0,                  # Connection timeout (0 to disable)
)
```

> (source: `python-sdk-fastmcp-client-client.md` > `Client` constructor args;
> `clients.md` > "Client Transports")

---

## 10. Complete Integration Code

This is the full agent loop combining TensorZero and FastMCP:

```python
import asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult


async def execute_tool(client: Client, name: str, arguments: dict) -> str:
    """Execute an MCP tool and return result as string."""
    try:
        result = await client.call_tool(name, arguments)
        text_parts = [
            block.text for block in result.content
            if hasattr(block, 'text')
        ]
        return "\n".join(text_parts) if text_parts else ""
    except ToolError as e:
        return f"Error executing tool '{name}': {e}"


async def main():
    # --- Initialize TensorZero gateway ---
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as t0:

        # --- Initialize FastMCP client ---
        mcp_client = Client("http://localhost:8000/mcp")

        async with mcp_client:

            # --- Discover tools (ONCE) ---
            mcp_tools = await mcp_client.list_tools()
            t0_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                }
                for tool in mcp_tools
            ]

            print(f"Discovered {len(t0_tools)} tools: "
                  f"{[t['name'] for t in t0_tools]}")

            # --- Agent loop state ---
            messages = []
            episode_id = None

            while True:
                # Prompt user if no pending tool results
                if not messages or messages[-1]["role"] == "assistant":
                    user_input = input("\n[User]\n")
                    if user_input.lower() in ("quit", "exit"):
                        break
                    messages.append(Message(role="user", content=user_input))

                # --- Call TensorZero ---
                response = await t0.inference(
                    function_name="my_copilot",
                    episode_id=episode_id,
                    input={"messages": messages},
                    additional_tools=t0_tools,
                )

                # Capture episode_id from first response
                episode_id = response.episode_id

                # Append assistant response to history
                messages.append(
                    Message(role="assistant", content=response.content)
                )

                # --- Process content blocks ---
                tool_results = []
                tool_call_blocks = []

                for block in response.content:
                    if isinstance(block, Text):
                        print(f"\n[Assistant]\n{block.text}")
                    elif isinstance(block, ToolCall):
                        print(f"\n[Tool Call: {block.raw_name}]")
                        print(block.raw_arguments)
                        tool_call_blocks.append(block)

                # --- Execute tools ---
                if tool_call_blocks:
                    # Option A: Sequential execution
                    for block in tool_call_blocks:
                        result_text = await execute_tool(
                            mcp_client, block.name, block.arguments
                        )
                        print(f"[Tool Result: {block.name}] {result_text[:200]}")
                        tool_results.append(ToolResult(
                            name=block.name,
                            result=result_text,
                            id=block.id,
                        ))

                    # Option B: Parallel execution (uncomment to use)
                    # result_texts = await asyncio.gather(*[
                    #     execute_tool(mcp_client, b.name, b.arguments)
                    #     for b in tool_call_blocks
                    # ])
                    # tool_results = [
                    #     ToolResult(name=b.name, result=t, id=b.id)
                    #     for b, t in zip(tool_call_blocks, result_texts)
                    # ]

                # --- Append tool results as SINGLE user message ---
                if tool_results:
                    messages.append(
                        Message(role="user", content=tool_results)
                    )
                # If no tool_results → loop continues → user prompted


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 11. Gotchas & Easy-to-Miss Details (FastMCP Side)

### 1. `call_tool()` raises `ToolError` by default
Unlike `call_tool_mcp()`, the high-level `call_tool()` raises `ToolError` on failure. In the agent loop, **always catch this** or use `raise_on_error=False`, otherwise a single tool failure crashes the entire loop.

> (source: `python-sdk-fastmcp-client-mixins-tools.md` > `call_tool` — "Unlike call_tool_mcp, this method raises a ToolError if the tool call results in an error.")

### 2. Arguments must be a `dict`, not a JSON string
FastMCP's `call_tool()` expects `arguments: dict[str, Any]`. TensorZero's response provides `block.arguments` as a **validated dict** (in the Python SDK). So you can pass it directly:
```python
await client.call_tool(block.name, block.arguments)
```
No need to serialize/deserialize.

> (source: `python-sdk-fastmcp-client-mixins-tools.md` > `call_tool` signature)

### 3. `.content` blocks may include non-text types
Always check for `.text` attribute existence when extracting text:
```python
# SAFE
text_parts = [b.text for b in result.content if hasattr(b, 'text')]

# UNSAFE — will crash on ImageContent, AudioContent, etc.
text_parts = [b.text for b in result.content]
```

> (source: `clients.md` > "Structured Results" — `.content` is `list[mcp.types.ContentBlock]` which includes `TextContent`, `ImageContent`, `AudioContent`, etc.)

### 4. `tool.description` may be `None`
If a tool function has no docstring and no explicit `description=` parameter, the `Tool.description` will be `None`. Always use `tool.description or ""` when building `t0_tools`.

> (source: `servers-tools.md` > `@tool` decorator arguments — description is `str | None`)

### 5. STDIO servers don't inherit environment variables
If your MCP server needs API keys or config, you must pass them explicitly:
```python
transport = StdioTransport(
    command="python",
    args=["server.py"],
    env={"API_KEY": os.environ["API_KEY"]}
)
```

> (source: `clients.md` > "STDIO Transport" — "STDIO servers run in isolated environments by default. They do not inherit your shell's environment variables.")

### 6. Client context must span the full loop
Do **not** open/close the client context per tool call. The `async with client:` block should wrap the entire agent loop:
```python
# ✅ CORRECT
async with client:
    while True:
        result = await client.call_tool(...)

# ❌ WRONG — tears down session each iteration
while True:
    async with client:
        result = await client.call_tool(...)
```

### 7. The MCP `Tool.inputSchema` is the full JSON Schema object
It's not a path to a file. Pass it directly to TensorZero's `additional_tools[].parameters`:
```python
{"parameters": tool.inputSchema}  # ✅ Dict object
# NOT
{"parameters": "tools/my_tool.json"}  # ❌ File path (only for TOML config)
```

### 8. FastMCP primitive result wrapping
FastMCP servers automatically wrap primitive return values (`int`, `str`, `bool`) in `{"result": value}` for structured output. The FastMCP **client** automatically unwraps this in `.data`. But the `.content[0].text` will be the raw string representation (e.g., `"8"` for an int return). For TensorZero, you're extracting `.content` text, so this is transparent.

> (source: `clients.md` — "FastMCP servers automatically wrap primitive results (like int, str, bool) in a {"result": value} structure. FastMCP clients automatically unwrap this, so you get the original value in .data.")

### 9. Timeout support
You can set per-call timeouts to prevent the agent loop from hanging on slow tools:
```python
result = await client.call_tool("slow_tool", args, timeout=30.0)
```

> (source: `clients.md` > "Execution Options")

---

## 12. Checklist Mapping (FastMCP Responsibilities)

| Item | FastMCP Responsibility | Key Detail |
|------|----------------------|------------|
| **F3.1** Single-turn | `client.call_tool()` executes the tool, extract `.content` text as `ToolResult.result` | Section 5, 6 |
| **F3.2** Multi-turn | Same `client` session stays open for the whole loop; tool discovery once, execution many times | Section 9.2 |
| **F3.3** `episode_id` | N/A — TensorZero concern, but client session must stay alive | Section 9.2 |
| **F3.4** Termination | N/A — TensorZero concern; FastMCP just executes when called | — |
| **F3.5** Parallel results | `asyncio.gather()` with `client.call_tool()` is safe (reentrant client) | Section 8 |
| **F3.6** Message history | Tool result text extraction: `"\n".join(b.text for b in result.content if hasattr(b, 'text'))` | Section 6 |
| **F3.7** Continuous conversation | Client session open for entire conversation duration | Section 9.2 |
| **F3.8** `asyncio.gather` | Client supports concurrent `call_tool()` within same context | Section 8.3 |

---

## 13. Source Citations

| Section | File | Original URL |
|---------|------|-------------|
| Client class | `python-sdk-fastmcp-client-client.md` | https://gofastmcp.com/python-sdk/fastmcp-client-client.md |
| Tool methods (list, call) | `python-sdk-fastmcp-client-mixins-tools.md` | https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md |
| Client tools guide | `clients.md` > "Calling Tools" | https://gofastmcp.com/clients/tools.md |
| Client transports guide | `clients.md` > "Client Transports" | https://gofastmcp.com/clients/transports.md |
| Quickstart & call example | `getting-started.md` > "Call Your Server" | https://gofastmcp.com/getting-started/quickstart.md |
| Transport inference | `python-sdk-fastmcp-client-transports-inference.md` | https://gofastmcp.com/python-sdk/fastmcp-client-transports-inference.md |
| Server tool definitions | `servers-tools.md` | https://gofastmcp.com/servers/tools.md |
| Error handling | `servers-tools.md` > "Error Handling" | https://gofastmcp.com/servers/tools.md |
| Exceptions | `python-sdk-fastmcp-exceptions.md` | https://gofastmcp.com/python-sdk/fastmcp-exceptions.md |
| Deployment & transports | `deployment.md` | https://gofastmcp.com/deployment/ |
