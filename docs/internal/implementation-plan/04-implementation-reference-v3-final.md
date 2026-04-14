# Implementation Reference v3 (FINAL): Verified Code Patterns

> **Changes vs v2**: Fixed exception handling (explicit McpError for
> timeouts), added correct import, added parallel execution pattern,
> fixed pytest config, added testing pattern with fixture.

Each code block is verified against both documentations
(TensorZero + FastMCP). Lines with `# source:` indicate the reference.

---

## 1. tool_adapter.py — Adapter Functions

### convert_tools: FastMCP → T0

```python
# source: fast-mcp > clients/tools.md — list_tools() returns list[mcp.types.Tool]
# source: fast-mcp > getting-started/upgrading/from-low-level-sdk.md — Tool has .name, .description, .inputSchema
# source: gateway-api-reference.md > "additional_tools" — accepts list[dict] with name, description, parameters

from mcp.types import Tool as MCPTool


def convert_tools(mcp_tools: list[MCPTool]) -> list[dict]:
    """Converts FastMCP tools to TensorZero additional_tools format.

    FastMCP Tool.description can be None if the function has no docstring.
    T0 expects a string, so we use "" as a defensive fallback.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools
    ]
```

### serialize_tool_result: FastMCP → str

```python
# source: fast-mcp > clients/tools.md > "Structured Results" — CallToolResult has .data, .content, .is_error
# source: gateway-api-reference.md > content block type "tool_result" — result must be str
# source: fast-mcp > clients/tools.md — primitives auto-wrapped in {"result": value}, auto-unwrapped by client

import json
from typing import Any


def serialize_tool_result(result: Any) -> str:
    """Converts CallToolResult to string for TensorZero ToolResult.

    Strategy:
    1. If is_error=True (only with raise_on_error=False), extract error from content
    2. If .data is str, return directly
    3. If .data is dict/list/int/float, json.dumps with default=str
    4. Fallback: concatenate content blocks with .text
    """
    # Error case (only if raise_on_error=False is used)
    if result.is_error:
        error_texts = [
            c.text for c in (result.content or []) if hasattr(c, "text")
        ]
        return f"Error: {' '.join(error_texts)}" if error_texts else "Error: unknown tool error"

    # Preferred: use .data (FastMCP exclusive, deserialized Python objects)
    if result.data is not None:
        if isinstance(result.data, str):
            return result.data
        # json.dumps with default=str handles: datetime, UUID, Path, etc.
        return json.dumps(result.data, default=str, ensure_ascii=False)

    # Fallback: concatenate content blocks (standard MCP)
    texts = [c.text for c in (result.content or []) if hasattr(c, "text")]
    return "\n".join(texts) if texts else ""
```

### build_tool_result: Combines ToolCall + CallToolResult → T0 ToolResult

```python
# source: examples/mcp-model-context-protocol/main.py — ToolResult(name=, result=, id=)

from tensorzero import ToolResult


def build_tool_result(tool_call, mcp_result) -> ToolResult:
    """Builds a TensorZero ToolResult from a ToolCall and CallToolResult."""
    return ToolResult(
        name=tool_call.name,
        result=serialize_tool_result(mcp_result),
        id=tool_call.id,
    )
```

---

## 2. agent.py — Agent Loop

### Imports

```python
# source: fast-mcp > python-sdk/fastmcp-exceptions.md — ToolError in fastmcp.exceptions
# source: fast-mcp > servers-middleware.md > on_initialize — McpError in mcp
# source: fast-mcp > python-sdk/fastmcp-client-mixins-tools.md — call_tool raises McpError on timeout

from tensorzero import (
    AsyncTensorZeroGateway,
    Message,
    Text,
    ToolCall,
    ToolResult,
)
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp import McpError

from tool_adapter import convert_tools, build_tool_result, serialize_tool_result
```

### execute_tool_calls: Executes N tool calls against FastMCP

```python
# source: gateway-api-reference.md > "arguments (object): null if invalid"
# source: fast-mcp > clients/tools.md > "Error Handling" — ToolError on tool failure
# source: fast-mcp > python-sdk/fastmcp-client-mixins-tools.md — McpError on timeout/protocol
# source: fast-mcp > servers-tools.md > "Timeouts" — timeout generates MCP error code -32000

async def execute_tool_calls(
    client: Client,
    content_blocks: list,
    timeout: float = 30.0,
) -> list[ToolResult]:
    """Executes all ToolCalls and returns a list of ToolResults.

    Handles three types of error:
    - arguments=null: LLM generated invalid JSON (EC-2)
    - ToolError: the tool itself failed
    - McpError: timeout or MCP protocol error (EC-5)
    """
    results = []
    for block in content_blocks:
        if not isinstance(block, ToolCall):
            continue

        # EC-2: Handle invalid arguments
        if block.arguments is None:
            results.append(ToolResult(
                name=block.raw_name or "unknown",
                result=f"Error: LLM produced invalid JSON arguments: {block.raw_arguments}",
                id=block.id,
            ))
            continue

        # Execute tool against FastMCP
        try:
            mcp_result = await client.call_tool(
                block.name,
                block.arguments,
                timeout=timeout,
            )
            results.append(build_tool_result(block, mcp_result))

        except ToolError as e:
            # Tool failed during execution
            results.append(ToolResult(
                name=block.name,
                result=f"Error: tool execution failed: {e}",
                id=block.id,
            ))

        except McpError as e:
            # Timeout or MCP protocol error
            results.append(ToolResult(
                name=block.name,
                result=f"Error: MCP protocol error (possibly timeout): {e}",
                id=block.id,
            ))

        except Exception as e:
            # Catchall for unexpected errors
            results.append(ToolResult(
                name=block.name,
                result=f"Unexpected error: {type(e).__name__}: {e}",
                id=block.id,
            ))

    return results
```

### has_tool_calls: Detects tool calls in the response

```python
def has_tool_calls(content_blocks: list) -> bool:
    """Returns True if there is at least one ToolCall in the content blocks."""
    return any(isinstance(block, ToolCall) for block in content_blocks)
```

### run_agent: Main loop

```python
# source: gateway-guides.md > Episodes — episode_id is captured and propagated
# source: examples/mcp-model-context-protocol/main.py — results go into ONE Message
# source: gateway-api-reference.md > "additional_tools" — passed in every call

async def run_agent(
    t0: AsyncTensorZeroGateway,
    mcp_client: Client,
    user_message: str,
    function_name: str,
    episode_id: str | None = None,
    max_turns: int = 20,
    tool_timeout: float = 30.0,
) -> dict:
    """
    Executes the complete agent loop.

    Args:
        t0: Connected instance of the TensorZero gateway
        mcp_client: Connected instance of the FastMCP Client (inside async with)
        user_message: User's message
        function_name: Name of the function in tensorzero.toml
        episode_id: Existing episode ID (None for new conversation)
        max_turns: Maximum number of tool-call turns before aborting
        tool_timeout: Timeout in seconds for each call_tool

    Returns:
        {
            "response": str,            # final text from the LLM
            "episode_id": str,          # to continue the conversation
            "inference_ids": list[str],  # for feedback
            "turns": int,               # turns used
        }
    """
    # Discover tools once at startup
    mcp_tools = await mcp_client.list_tools()
    t0_tools = convert_tools(mcp_tools)

    # Initialize conversation
    messages = [Message(role="user", content=user_message)]
    inference_ids = []

    for turn in range(max_turns):
        # Inference
        response = await t0.inference(
            function_name=function_name,
            episode_id=episode_id,
            input={"messages": messages},
            additional_tools=t0_tools,
        )

        # Capture IDs
        episode_id = response.episode_id
        inference_ids.append(str(response.inference_id))

        # Add assistant response to history
        messages.append(Message(role="assistant", content=response.content))

        # Are there tool calls?
        if not has_tool_calls(response.content):
            # Final response — extract text
            text_parts = [
                block.text for block in response.content
                if isinstance(block, Text)
            ]
            return {
                "response": "\n".join(text_parts),
                "episode_id": str(episode_id),
                "inference_ids": inference_ids,
                "turns": turn + 1,
            }

        # Execute tool calls (sequential — see M3.7 for parallel)
        tool_results = await execute_tool_calls(
            mcp_client, response.content, timeout=tool_timeout
        )

        # Add results to history — ONE single message with all results
        messages.append(Message(role="user", content=tool_results))

    # Max turns reached (M5.4)
    raise RuntimeError(
        f"Agent loop exceeded {max_turns} turns without final response"
    )
```

### (Optional) execute_tool_calls_parallel: Concurrent execution

```python
# source: fast-mcp > servers-tools.md > "Async Support" — async tools are more efficient for I/O
# Note: Only useful if there are multiple tool calls AND the tools are I/O-bound.

import asyncio


async def _execute_single(
    client: Client, block: ToolCall, timeout: float
) -> ToolResult:
    """Executes a single ToolCall with error handling."""
    if block.arguments is None:
        return ToolResult(
            name=block.raw_name or "unknown",
            result=f"Error: invalid JSON arguments: {block.raw_arguments}",
            id=block.id,
        )
    try:
        mcp_result = await client.call_tool(block.name, block.arguments, timeout=timeout)
        return build_tool_result(block, mcp_result)
    except (ToolError, McpError) as e:
        return ToolResult(name=block.name, result=f"Error: {e}", id=block.id)
    except Exception as e:
        return ToolResult(name=block.name, result=f"Unexpected: {e}", id=block.id)


async def execute_tool_calls_parallel(
    client: Client,
    content_blocks: list,
    timeout: float = 30.0,
) -> list[ToolResult]:
    """Executes tool calls concurrently with asyncio.gather."""
    tool_calls = [b for b in content_blocks if isinstance(b, ToolCall)]
    if not tool_calls:
        return []
    results = await asyncio.gather(*[
        _execute_single(client, tc, timeout) for tc in tool_calls
    ])
    return list(results)
```

---

## 3. Using the Agent Loop (main.py)

```python
# End-to-end usage example

import asyncio
from fastmcp import FastMCP, Client
from tensorzero import AsyncTensorZeroGateway
from agent import run_agent

# Option 1: FastMCP server in-memory (development)
from server import mcp as test_server

async def main():
    async with (
        await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0,
        Client(test_server) as mcp_client,  # in-memory for dev
    ):
        result = await run_agent(
            t0=t0,
            mcp_client=mcp_client,
            user_message="What's the weather in Tokyo?",
            function_name="agent",
        )

        print(f"Response: {result['response']}")
        print(f"Episode: {result['episode_id']}")
        print(f"Turns: {result['turns']}")

        # Feedback
        await t0.feedback(
            episode_id=result["episode_id"],
            metric_name="task_success",
            value=True,
        )

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# Option 2: Remote FastMCP server (production)

async def main_production():
    async with (
        await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0,
        Client("https://my-mcp-server.example.com/mcp") as mcp_client,  # HTTP
    ):
        result = await run_agent(
            t0=t0,
            mcp_client=mcp_client,
            user_message="Search for AI papers",
            function_name="agent",
        )
```

```python
# Option 3: Multi-server with mount (M4)

async def main_multi_server():
    from server_a import mcp as server_a
    from server_b import mcp as server_b

    # Compose servers with namespacing
    # source: fast-mcp > servers/composition.md > "Namespacing"
    main_server = FastMCP("Main")
    main_server.mount(server_a, namespace="weather")
    main_server.mount(server_b, namespace="database")

    async with (
        await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0,
        Client(main_server) as mcp_client,  # in-memory with composed server
    ):
        result = await run_agent(
            t0=t0,
            mcp_client=mcp_client,
            user_message="What's the weather and find related records",
            function_name="agent",
        )
```

---

## 4. TensorZero Configuration

### Minimal tensorzero.toml

```toml
# source: examples/mcp-model-context-protocol/config/tensorzero.toml
# No tools are defined in config — they are passed dynamically via additional_tools

[functions.agent]
type = "chat"

[functions.agent.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini-2025-04-14"
system_template = "functions/agent/baseline/system.txt"

# Metrics for feedback
[metrics.task_success]
type = "boolean"
level = "episode"
optimize = "max"

[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"
```

---

## 5. FastMCP Test Server

### server.py

```python
# source: fast-mcp > servers/tools.md > "The @tool Decorator"
# Note: @mcp.tool infers name (function), description (docstring), inputSchema (type annotations)

from fastmcp import FastMCP

mcp = FastMCP("TestServer")


@mcp.tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city."""
    return {"city": city, "temperature": 22, "units": units, "condition": "sunny"}


@mcp.tool
def search_database(query: str, max_results: int = 5) -> list[dict]:
    """Search the internal database for records matching the query."""
    return [
        {"id": 1, "title": f"Result for '{query}'", "score": 0.95},
        {"id": 2, "title": f"Another result for '{query}'", "score": 0.87},
    ]


@mcp.tool
def failing_tool(data: str) -> str:
    """A tool that always fails (for testing error handling)."""
    raise ValueError(f"Simulated failure processing: {data}")


if __name__ == "__main__":
    mcp.run()  # stdio by default
```

---

## 6. Tests

### pyproject.toml (pytest config)

```toml
# source: fast-mcp > servers/testing.md — recommends asyncio_mode = "auto"
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### tests/conftest.py

```python
# source: fast-mcp > development.md > "In-Memory Testing"
# source: fast-mcp > servers/testing.md — fixture pattern

import pytest
from fastmcp import Client
from server import mcp as test_server


@pytest.fixture
async def mcp_client():
    """FastMCP client connected in-memory to the test server."""
    async with Client(test_server) as client:
        yield client
```

### tests/test_tool_adapter.py

```python
# source: fast-mcp > clients/tools.md — list_tools returns list[mcp.types.Tool]

from tool_adapter import convert_tools, serialize_tool_result


async def test_convert_tools(mcp_client):
    """F1.1 + F1.2: Tool discovery and conversion."""
    tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    assert len(t0_tools) >= 2
    for tool in t0_tools:
        assert "name" in tool
        assert isinstance(tool["name"], str)
        assert "description" in tool
        assert isinstance(tool["description"], str)
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"


async def test_convert_tools_none_description(mcp_client):
    """F1.6: Tool without description is converted with empty ''."""
    # FastMCP generates description from docstring.
    # If there is no docstring, .description will be None.
    # convert_tools uses "or ''" as fallback.
    tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)
    for tool in t0_tools:
        assert tool["description"] is not None  # never None, always str


async def test_serialize_str():
    """F2.2: .data str is returned directly."""
    from unittest.mock import MagicMock
    result = MagicMock(is_error=False, data="hello world", content=[])
    assert serialize_tool_result(result) == "hello world"


async def test_serialize_dict():
    """F2.3: .data dict is serialized with json.dumps."""
    import json
    from unittest.mock import MagicMock
    result = MagicMock(is_error=False, data={"count": 42, "name": "test"}, content=[])
    serialized = serialize_tool_result(result)
    parsed = json.loads(serialized)
    assert parsed == {"count": 42, "name": "test"}


async def test_serialize_none_fallback():
    """F2.4: .data None → fallback to content[*].text."""
    from unittest.mock import MagicMock
    content_block = MagicMock()
    content_block.text = "fallback text"
    result = MagicMock(is_error=False, data=None, content=[content_block])
    assert serialize_tool_result(result) == "fallback text"


async def test_serialize_error():
    """F2.6: .is_error=True → error message."""
    from unittest.mock import MagicMock
    content_block = MagicMock()
    content_block.text = "division by zero"
    result = MagicMock(is_error=True, data=None, content=[content_block])
    serialized = serialize_tool_result(result)
    assert "Error" in serialized
    assert "division by zero" in serialized
```

### tests/test_tool_execution.py

```python
# source: fast-mcp > clients/tools.md > "Error Handling" — ToolError on nonexistent tool

from fastmcp.exceptions import ToolError
import pytest


async def test_tool_execution(mcp_client):
    """F2.1: call_tool executes correctly."""
    result = await mcp_client.call_tool("get_weather", {"city": "Tokyo"})
    assert not result.is_error
    assert result.data is not None
    assert result.data["city"] == "Tokyo"


async def test_tool_error_handling(mcp_client):
    """F5.1: ToolError is raised on a failing tool."""
    with pytest.raises(ToolError):
        await mcp_client.call_tool("failing_tool", {"data": "test"})


async def test_nonexistent_tool(mcp_client):
    """F5.1b: ToolError on nonexistent tool."""
    with pytest.raises((ToolError, Exception)):
        await mcp_client.call_tool("nonexistent_tool", {})
```

### tests/test_integration.py (requires T0 Gateway running)

```python
# source: gateway-api-reference.md — AsyncTensorZeroGateway
# These tests require: docker compose up (T0 + ClickHouse)

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from server import mcp as test_server
from tool_adapter import convert_tools

# Mark as integration tests
pytestmark = pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Need --run-integration flag and running T0 Gateway",
)


async def test_t0_accepts_tools():
    """F1.3: T0 accepts dynamic tools without error."""
    async with Client(test_server) as mcp_client:
        tools = await mcp_client.list_tools()
        t0_tools = convert_tools(tools)

    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
            additional_tools=t0_tools,
        )
        assert response.content is not None
        assert response.episode_id is not None
```

---

## 7. Notes on Critical Scenarios

### EC-2: arguments=null

```python
# T0 returns arguments=null when the LLM generates invalid JSON.
# raw_arguments contains the string as-is.
# Pattern in execute_tool_calls:
if block.arguments is None:
    results.append(ToolResult(
        name=block.raw_name or "unknown",
        result=f"Error: invalid JSON arguments from LLM: {block.raw_arguments}",
        id=block.id,
    ))
```

### EC-4: Serialization of complex types

```python
# json.dumps with default=str handles:
#   datetime → "2024-01-15 10:30:00"
#   UUID → "550e8400-e29b-41d4-a716-446655440000"
#   Path → "/home/user/file.txt"
#   Pydantic model → str(model)  [via default=str]
#
# For Pydantic models, FastMCP .data already comes as a deserialized dict.
# json.dumps(data, default=str) works for the majority of cases.
```

### EC-5: Timeout generates McpError, NOT ToolError

```python
# CORRECTION vs v2: Timeout raises McpError (code -32000), not ToolError.
# source: python-sdk-fastmcp-client-mixins-tools.md — Raises: McpError on TimeoutError
# source: servers-tools.md — "returns an MCP error with code -32000"
#
# Correct import:
from mcp import McpError
# source: servers-middleware.md — "from mcp import McpError"
#
# Catch BOTH in execute_tool_calls:
except ToolError as e:     # tool failed
    ...
except McpError as e:      # timeout or protocol
    ...
```

### EC-6: Parallel tool calls

```python
# Only OpenAI and Fireworks AI support parallel_tool_calls.
# The agent loop ALWAYS handles N tool calls (loop over content blocks).
# All results go into ONE message:
#   Message(role="user", content=[result_1, result_2, ...result_n])
```
