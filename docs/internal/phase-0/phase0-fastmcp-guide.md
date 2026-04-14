# Phase 0 Guide: Infrastructure — FastMCP Side (P0.3, P0.4, P0.5)

## Summary

This document covers items P0.3, P0.4, and P0.5 of Phase 0, corresponding to FastMCP. It complements the TensorZero guide (P0.1, P0.2) that you already have.

---

## P0.3 — Test FastMCP server instantiated with at least 2 tools using `@mcp.tool`

### Server instantiation

```python
from fastmcp import FastMCP

mcp = FastMCP(name="TestServer")
```

(source: `servers-tools.md` > "The @tool Decorator", URL: https://gofastmcp.com/servers/tools.md)

### Registering tools with `@mcp.tool`

Decorating a Python function with `@mcp.tool` automatically registers it as an MCP tool. FastMCP:

- Uses the **function name** as the tool name.
- Uses the **docstring** as the tool description.
- Generates an **input schema** based on the signature and type annotations.
- Handles parameter validation and error reporting.

```python
from fastmcp import FastMCP

mcp = FastMCP(name="TestServer")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@mcp.tool
def get_temperature(location: str, units: str = "celsius") -> dict:
    """Get the current temperature in a given location."""
    # Simulation for testing
    temps = {"Tokyo": 22, "London": 15, "NYC": 25}
    return {"location": location, "temperature": temps.get(location, 20), "units": units}
```

(source: `servers-tools.md` > "The @tool Decorator", URL: https://gofastmcp.com/servers/tools.md)

### Important restrictions

- **`*args` and `**kwargs` are not supported:** FastMCP needs to generate a complete parameter schema for the MCP protocol, which is not possible with variable argument lists.
- **Return type annotations:** If your tool returns a primitive type (`int`, `str`, `bool`) and you want structured output, **you need the return annotation** (e.g. `-> int`). Without it, the tool only produces `content` blocks, not `structuredContent`.
- **Sync and async functions:** Both are supported. Sync functions are automatically run in a threadpool to avoid blocking the event loop.

(source: `servers-tools.md` > "Async Support" and "Structured Output > Primitives and Collections", URL: https://gofastmcp.com/servers/tools.md)

### Decorator customization

You can override name, description, and add metadata:

```python
@mcp.tool(
    name="find_products",
    description="Search the product catalog with optional category filtering.",
    tags={"catalog", "search"},
    meta={"version": "1.2", "author": "product-team"}
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    """Internal function description (ignored if description is provided above)."""
    return [{"id": 2, "name": "Another Product"}]
```

Parameters of the `@mcp.tool` decorator:

| Parameter | Type | Description |
|---|---|---|
| `name` | `str \| None` | Explicit tool name. If not provided, uses the function name |
| `description` | `str \| None` | Description. If provided, the docstring is ignored for this purpose |
| `tags` | `set[str] \| None` | Tags for categorizing the tool |
| `timeout` | `float \| None` | Execution timeout in seconds (v3.0.0+) |
| `annotations` | `ToolAnnotations \| dict \| None` | Additional metadata (readOnlyHint, destructiveHint, etc.) |
| `output_schema` | `dict[str, Any] \| None` | Custom JSON schema for the tool output (v2.10.0+) |
| `meta` | `dict[str, Any] \| None` | Meta info passed to the MCP client (v2.11.0+) |

(source: `servers-tools.md` > "Decorator Arguments", URL: https://gofastmcp.com/servers/tools.md)

### Behavior with duplicate tools

```python
mcp = FastMCP(
    name="StrictServer",
    on_duplicate_tools="error"  # "warn" (default), "error", "replace", "ignore"
)
```

(source: `servers-tools.md` > "Duplicate Tools", URL: https://gofastmcp.com/servers/tools.md)

### Error handling in tools

You can use standard exceptions or FastMCP's `ToolError`:

```python
from fastmcp.exceptions import ToolError

@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        # ToolError always sends the message to the client, regardless of mask_error_details
        raise ToolError("Division by zero is not allowed.")
    return a / b
```

To hide internal error details:

```python
mcp = FastMCP(name="SecureServer", mask_error_details=True)
```

(source: `servers-tools.md` > "Error Handling", URL: https://gofastmcp.com/servers/tools.md)

### ToolResult for full control over the response

If you need explicit control over content + structured_content + meta:

```python
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

@mcp.tool
def advanced_tool() -> ToolResult:
    """Tool with full control over output."""
    return ToolResult(
        content=[TextContent(type="text", text="Human-readable summary")],
        structured_content={"data": "value", "count": 42},
        meta={"execution_time_ms": 145}
    )
```

**`ToolResult` fields:**

- **`content`** — Traditional MCP content blocks. Can be a string (auto-converted to `TextContent`), a list of content blocks, or a serializable value.
- **`structured_content`** — Dict with structured data. If only `structured_content` is provided, it is also used as `content` (converted to a JSON string).
- **`meta`** — Runtime metadata (metrics, debugging info). Separate from the `meta` of the `@mcp.tool()` decorator.

(source: `servers-tools.md` > "ToolResult and Metadata", URL: https://gofastmcp.com/servers/tools.md)

### Automatic structured output

FastMCP automatically generates structured output alongside traditional content blocks:

- **Object-like results** (`dict`, Pydantic models, dataclasses) → Always converted to structured content.
- **Non-object results** (`int`, `str`, `list`) → Only converted to structured content if there is an output schema (i.e., a type annotation).
- Primitives are automatically wrapped under `{"result": value}`.

(source: `servers-tools.md` > "Structured Output", URL: https://gofastmcp.com/servers/tools.md)

### P0.3 Verification

```python
# You can verify directly without a client:
tools = mcp.list_tools()
assert len(tools) >= 2
for tool in tools:
    print(f"Tool: {tool.name} — {tool.description}")
```

(source: `development.md` > "Writing Tests > Single Behavior Per Test", URL: https://gofastmcp.com/development/tests.md)

---

## P0.4 — FastMCP `Client(mcp_instance)` in-memory connects and `list_tools()` returns the tools

### Creating the in-memory client

FastMCP's `Client` automatically infers the appropriate transport. When you pass a `FastMCP` instance, it uses **in-memory transport** (no network or subprocess):

```python
from fastmcp import FastMCP, Client

server = FastMCP("TestServer")

@server.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@server.tool
def get_temperature(location: str) -> dict:
    """Get the current temperature."""
    return {"location": location, "temperature": 22}

client = Client(server)  # In-memory, no network or subprocess
```

(source: `clients.md` > "Creating a Client" and "In-Memory Transport", URL: https://gofastmcp.com/clients/overview.md)

### Characteristics of in-memory transport

- Connects directly to the FastMCP instance **within the same Python process**.
- Eliminates subprocess and network complexity.
- The server shares the **memory space and environment variables** of the process.
- Ideal for **testing and development**.

(source: `clients.md` > "Choosing a Transport > In-memory transport", URL: https://gofastmcp.com/clients/overview.md)

### Using the client with `async with`

**All client operations require the `async with` context manager:**

```python
import asyncio
from fastmcp import Client, FastMCP

server = FastMCP("TestServer")

@server.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@server.tool
def get_temperature(location: str) -> dict:
    """Get the current temperature."""
    return {"location": location, "temperature": 22}

client = Client(server)

async def main():
    async with client:
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

asyncio.run(main())
```

(source: `clients.md` > "Creating a Client", URL: https://gofastmcp.com/clients/overview.md)

### `list_tools()` — details

```python
list_tools(self: Client, max_pages: int = 250) -> list[mcp.types.Tool]
```

- Returns `list[mcp.types.Tool]` — a list of all available Tool objects.
- If the server paginates results, `list_tools()` **automatically fetches all pages** and returns the complete list.
- `max_pages` (default 250): Maximum number of pages to fetch before raising an error.
- Raises `RuntimeError` if the page limit is reached before pagination finishes.
- Raises `McpError` if the request results in a `TimeoutError | JSONRPCError`.

For manual pagination, use `list_tools_mcp(cursor=...)` which returns the raw MCP protocol object with `nextCursor`.

(source: `python-sdk-fastmcp-client-mixins-tools.md` > "list_tools", URL: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md)

### P0.4 Verification

```python
async def test_client_connects_and_lists_tools():
    server = FastMCP("TestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def get_temperature(location: str) -> dict:
        """Get temperature."""
        return {"location": location, "temperature": 22}

    async with Client(server) as client:
        tools = await client.list_tools()
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "add" in tool_names
        assert "get_temperature" in tool_names
```

(source: `servers-testing.md` > "Testing with Pytest Fixtures", URL: https://gofastmcp.com/servers/testing.md)

---

## P0.5 — `call_tool()` in-memory executes a tool and returns `CallToolResult` with `.data`

### Basic execution

```python
async with client:
    result = await client.call_tool("add", {"a": 5, "b": 3})
    # result -> CallToolResult with structured and unstructured data

    # Access structured data (automatically deserialized)
    print(result.data)  # 8

    # Access traditional content blocks
    print(result.content[0].text)  # "8"
```

Arguments are passed as a dictionary.

(source: `clients.md` > "Calling Tools > Basic Execution", URL: https://gofastmcp.com/clients/tools.md)

### `call_tool()` — full signature

```python
call_tool(
    self: Client,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    version: str | int | None = None,
    timeout: timedelta | float | int | None = None,
    progress_handler: ProgressHandler | None = None,
    raise_on_error: bool = True,
    meta: dict[str, Any] | None = None,
    task: bool = False,
    task_id: str | None = None,
    ttl: int | None = None,
) -> CallToolResult | ToolTask
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `name` | (required) | Name of the tool to call |
| `arguments` | `None` | Dict with tool arguments |
| `timeout` | `None` | Timeout (aborts if execution takes longer) |
| `raise_on_error` | `True` | If `True`, raises `ToolError` on error |
| `meta` | `None` | Additional metadata (trace_id, user_id, etc.) |

**Returns:** `CallToolResult` (or `ToolTask` if `task=True`).

**Raises:**
- `ToolError` if the tool call results in an error.
- `McpError` if it results in a `TimeoutError | JSONRPCError`.
- `RuntimeError` if called without being connected.

**Difference from `call_tool_mcp()`:** `call_tool()` raises `ToolError` on error. `call_tool_mcp()` returns the raw `CallToolResult` with an `isError` flag without raising an exception.

(source: `python-sdk-fastmcp-client-mixins-tools.md` > "call_tool", URL: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md)

### `CallToolResult` properties

| Property | Type | Description |
|---|---|---|
| `.data` | `Any` | Fully hydrated Python objects with support for complex types (datetime, UUID, custom classes). **Exclusive to FastMCP.** |
| `.content` | `list[mcp.types.ContentBlock]` | Standard MCP content blocks (`TextContent`, `ImageContent`, `AudioContent`, etc.) |
| `.structured_content` | `dict[str, Any] \| None` | Standard MCP structured JSON data as sent by the server |
| `.is_error` | `bool` | Boolean indicating whether execution failed |

**Important note about `.data`:**
- For tools without an output schema or when deserialization fails, `.data` will be `None`. In that case, use `.content` blocks as a fallback.
- FastMCP servers automatically wrap primitive results (such as `int`, `str`, `bool`) in `{"result": value}`. FastMCP clients automatically unwrap this, so you receive the original value in `.data`.

(source: `clients.md` > "Calling Tools > Structured Results > CallToolResult Properties", URL: https://gofastmcp.com/clients/tools.md)

### Fallback when `.data` is `None`

```python
async with client:
    result = await client.call_tool("legacy_tool", {"param": "value"})

    if result.data is not None:
        print(f"Structured: {result.data}")
    else:
        for content in result.content:
            if hasattr(content, 'text'):
                print(f"Text result: {content.text}")
```

(source: `clients.md` > "Calling Tools > Structured Results", URL: https://gofastmcp.com/clients/tools.md)

### Error handling on the client

**Default (raise_on_error=True):**

```python
from fastmcp.exceptions import ToolError

async with client:
    try:
        result = await client.call_tool("potentially_failing_tool", {"param": "value"})
        print("Tool succeeded:", result.data)
    except ToolError as e:
        print(f"Tool failed: {e}")
```

**Manual (raise_on_error=False):**

```python
async with client:
    result = await client.call_tool(
        "potentially_failing_tool",
        {"param": "value"},
        raise_on_error=False
    )

    if result.is_error:
        print(f"Tool failed: {result.content[0].text}")
    else:
        print(f"Tool succeeded: {result.data}")
```

(source: `clients.md` > "Calling Tools > Error Handling", URL: https://gofastmcp.com/clients/tools.md)

### Execution options

```python
async with client:
    # With timeout (aborts if it takes more than 2 seconds)
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        timeout=2.0
    )

    # With progress handler
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )

    # With metadata
    result = await client.call_tool(
        name="send_email",
        arguments={"to": "user@example.com", "subject": "Hello"},
        meta={"trace_id": "abc-123", "request_source": "mobile_app"}
    )
```

(source: `clients.md` > "Calling Tools > Execution Options" and "Sending Metadata", URL: https://gofastmcp.com/clients/tools.md)

### P0.5 Verification

```python
async def test_call_tool_returns_result_with_data():
    server = FastMCP("TestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with Client(server) as client:
        result = await client.call_tool("add", {"a": 5, "b": 3})
        assert result.data == 8
        assert result.content[0].text == "8"
        assert result.is_error is False
```

(source: `development.md` > "In-Memory Testing" and `clients.md` > "Basic Execution", URLs: https://gofastmcp.com/development/tests.md, https://gofastmcp.com/clients/tools.md)

---

## P0.6 — pytest configured with `asyncio_mode = "auto"` in `pyproject.toml`

### Prerequisite

```bash
pip install pytest-asyncio
```

### Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

This eliminates the need to decorate each async test with `@pytest.mark.asyncio`.

(source: `servers-testing.md` > "Prerequisites", URL: https://gofastmcp.com/servers/testing.md)

### Recommended testing pattern with fixtures

The FastMCP documentation recommends this pattern:

```python
import pytest
from fastmcp import FastMCP, Client
from fastmcp.client.transports import FastMCPTransport

from my_project.main import mcp  # your MCP server

@pytest.fixture
async def main_mcp_client():
    async with Client(transport=mcp) as mcp_client:
        yield mcp_client

async def test_list_tools(main_mcp_client: Client[FastMCPTransport]):
    list_tools = await main_mcp_client.list_tools()
    assert len(list_tools) == 5
```

(source: `servers-testing.md` > "Testing with Pytest Fixtures", URL: https://gofastmcp.com/servers/testing.md)

**Important detail — do NOT open clients in fixtures for simple tests:**

> Note that you should **not** open FastMCP clients in your fixtures as it can create hard-to-diagnose issues with event loops.

The recommended alternative pattern is to create the server in the fixture and open the client **inside the test**:

```python
@pytest.fixture
def weather_server():
    server = FastMCP("WeatherServer")

    @server.tool
    def get_temperature(city: str) -> dict:
        temps = {"NYC": 72, "LA": 85, "Chicago": 68}
        return {"city": city, "temp": temps.get(city, 70)}

    return server

async def test_temperature_tool(weather_server):
    async with Client(weather_server) as client:
        result = await client.call_tool("get_temperature", {"city": "LA"})
        assert result.data == {"city": "LA", "temp": 85}
```

(source: `development.md` > "Using Fixtures", URL: https://gofastmcp.com/development/tests.md)

### Parametrized tests

```python
@pytest.mark.parametrize(
    "first_number, second_number, expected",
    [
        (1, 2, 3),
        (2, 3, 5),
        (3, 4, 7),
    ],
)
async def test_add(
    first_number: int,
    second_number: int,
    expected: int,
    main_mcp_client: Client[FastMCPTransport],
):
    result = await main_mcp_client.call_tool(
        name="add", arguments={"x": first_number, "y": second_number}
    )
    assert result.data is not None
    assert isinstance(result.data, int)
    assert result.data == expected
```

(source: `servers-testing.md` > "Using the pytest parametrize decorator", URL: https://gofastmcp.com/servers/testing.md)

---

## Complete example: test suite for Phase 0 (P0.3 + P0.4 + P0.5 + P0.6)

```python
# tests/test_phase0_fastmcp.py

import pytest
from fastmcp import FastMCP, Client


@pytest.fixture
def test_server():
    """P0.3 — FastMCP server with at least 2 tools."""
    server = FastMCP("Phase0TestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Adds two integer numbers together."""
        return a + b

    @server.tool
    def get_temperature(location: str, units: str = "celsius") -> dict:
        """Get the current temperature in a given location."""
        temps = {"Tokyo": 22, "London": 15, "NYC": 25}
        return {
            "location": location,
            "temperature": temps.get(location, 20),
            "units": units,
        }

    return server


# --- P0.3 ---
async def test_server_has_at_least_two_tools(test_server):
    """P0.3: The server has at least 2 registered tools."""
    tools = test_server.list_tools()
    assert len(tools) >= 2


# --- P0.4 ---
async def test_client_connects_and_list_tools(test_server):
    """P0.4: In-memory client connects and list_tools returns the tools."""
    async with Client(test_server) as client:
        tools = await client.list_tools()
        assert len(tools) >= 2
        tool_names = {t.name for t in tools}
        assert "add" in tool_names
        assert "get_temperature" in tool_names


# --- P0.5 ---
async def test_call_tool_add_returns_result_with_data(test_server):
    """P0.5: call_tool executes 'add' and returns CallToolResult with .data."""
    async with Client(test_server) as client:
        result = await client.call_tool("add", {"a": 5, "b": 3})
        assert result.data == 8
        assert result.is_error is False


async def test_call_tool_get_temperature_returns_result_with_data(test_server):
    """P0.5: call_tool executes 'get_temperature' and returns CallToolResult with .data."""
    async with Client(test_server) as client:
        result = await client.call_tool(
            "get_temperature", {"location": "Tokyo", "units": "celsius"}
        )
        assert result.data is not None
        assert result.data["location"] == "Tokyo"
        assert result.data["temperature"] == 22
        assert result.data["units"] == "celsius"
        assert result.is_error is False
```

```toml
# pyproject.toml (P0.6)
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```bash
# Run
pytest tests/test_phase0_fastmcp.py -v
```

---

## Gotchas and details not to overlook

1. **`*args` / `**kwargs` not supported in tools.** FastMCP needs to generate a complete schema. If your function uses them, it will fail at registration.

2. **Return type annotations matter for `.data`.** Without a `-> int` annotation, a tool that returns `8` will produce `result.data == None` and you will only have `result.content[0].text == "8"`. With `-> int`, you will get `result.data == 8` because FastMCP generates structured output that it can deserialize.

3. **The client MUST be used with `async with`.** All operations require the context manager for lifecycle management. Calling `call_tool()` outside the context manager raises `RuntimeError`.

4. **In-memory transport shares memory and environment variables** with the client process. Unlike STDIO transport, there is no process isolation.

5. **`call_tool()` raises `ToolError` by default.** If you prefer to handle errors manually, use `raise_on_error=False` and check `result.is_error`.

6. **Do not open clients in fixtures (per docs).** It can create issues with event loops. The recommended pattern is to return the server from the fixture and open the client inside each test with `async with Client(server)`.

7. **Automatic primitive wrapping/unwrapping.** FastMCP servers wrap `int` in `{"result": 8}` and FastMCP clients automatically unwrap it. This is transparent but important to know when debugging raw responses.

---

## Sources

| File | Section | Original URL |
|---|---|---|
| `servers-tools.md` | Tools (decorator, args, structured output, error handling) | https://gofastmcp.com/servers/tools.md |
| `clients.md` | Creating a Client, In-Memory Transport, Calling Tools | https://gofastmcp.com/clients/overview.md, https://gofastmcp.com/clients/tools.md |
| `python-sdk-fastmcp-client-mixins-tools.md` | list_tools(), call_tool() API reference | https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md |
| `python-sdk-fastmcp-client-transports-memory.md` | FastMCPTransport (in-memory) | https://gofastmcp.com/python-sdk/fastmcp-client-transports-memory.md |
| `servers-testing.md` | Testing your FastMCP Server (pytest fixtures) | https://gofastmcp.com/servers/testing.md |
| `development.md` | In-Memory Testing, Using Fixtures | https://gofastmcp.com/development/tests.md |
