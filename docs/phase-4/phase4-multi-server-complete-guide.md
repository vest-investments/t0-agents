# Phase 4: Multi-Server Integration Guide — TensorZero + FastMCP

> **Complete Guide** — All TensorZero and FastMCP documentation consolidated.
> No `[FastMCP — fill manually]` placeholders remain.

---

## 1. Architecture Overview

The multi-server pattern works as follows:

1. **FastMCP** composes N MCP servers (via `mount()` or `Client({"mcpServers": ...})`), applying namespace prefixes to tool names.
2. Your **application code** discovers tools from the composed FastMCP server and converts them to TensorZero's dynamic tool format.
3. Tools are passed to TensorZero via `additional_tools` at inference time.
4. TensorZero sends all tools to the LLM; the LLM invokes tools by their **namespaced name**.
5. Your application routes `ToolCall` responses back to FastMCP, which dispatches to the correct underlying server.

```
┌──────────────┐
│  Server A    │──┐
│  (tools: x)  │  │   mount / compose        ┌──────────────────┐
└──────────────┘  ├──────────────────────────▶│  FastMCP (main)  │
┌──────────────┐  │   namespace prefixes      │  tools: a_x,     │
│  Server B    │──┘   a_x, b_y               │         b_y      │
│  (tools: y)  │                              └────────┬─────────┘
└──────────────┘                                       │
                                                       │ list_tools()
                                                       ▼
                                              ┌──────────────────┐
                                              │  Your App Code   │
                                              │  (convert tools  │
                                              │   to T0 format)  │
                                              └────────┬─────────┘
                                                       │ additional_tools
                                                       ▼
                                              ┌──────────────────┐
                                              │  TensorZero GW   │
                                              │  POST /inference  │
                                              └──────────────────┘
```

---

## 2. TensorZero Configuration (tensorzero.toml)

The TensorZero function **does not need to declare any static tools** because all tools come dynamically from FastMCP at inference time via `additional_tools`.

### Minimal function config

```toml
# tensorzero.toml

[functions.multi_server_agent]
type = "chat"
# No `tools = [...]` needed — tools are injected dynamically

[functions.multi_server_agent.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini-2025-04-14"
system_template = "functions/multi_server_agent/system.txt"
```

**Key points from the docs:**

- The function must be `type = "chat"` — only chat functions can call tools.
  *(source: gateway-guides.md > "Tool Use (Function Calling)")*

- When no static `tools = [...]` is declared and `additional_tools` are provided at inference time, TensorZero automatically sets `tool_choice` to `"auto"`.
  *(source: gateway-api-reference.md > OpenAI-compatible endpoint)*

### Optional: tool_choice and parallel_tool_calls

```toml
[functions.multi_server_agent]
type = "chat"
tool_choice = "auto"           # "auto" | "required" | "none" | { specific = "tool_name" }
parallel_tool_calls = true     # Allow multiple tool calls per turn (OpenAI/Fireworks only)
```

*(source: gateway-configuration-reference.md > `[functions.function_name]` > `tool_choice`, `parallel_tool_calls`)*

---

## 3. FastMCP Composition — Option A: `mount()` with Namespaces

### 3.1 Basic Mount Pattern

`mount()` creates a **live, dynamic connection** between servers. When a client interacts with a mounted server's objects through the parent server, requests are forwarded to the mounted server in real-time. Changes to the mounted server are immediately reflected.

```python
from fastmcp import FastMCP

server_a = FastMCP("ServerA")
server_b = FastMCP("ServerB")

@server_a.tool
def list_tables() -> str:
    """List available database tables."""
    return "users, orders, products"

@server_b.tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

main = FastMCP("Main")
main.mount(server_a, namespace="a")
main.mount(server_b, namespace="b")

# Tools are now:
# - a_list_tables
# - b_search_docs
```

*(source: servers-composition.md > "Composing Servers", Original URL: https://gofastmcp.com/servers/composition.md)*

### 3.2 Namespacing Rules

When mounting with a namespace, the `Namespace` transform is applied automatically:

| Component Type | Without Namespace | With `namespace="api"` |
| -------------- | ----------------- | ---------------------- |
| Tool           | `my_tool`         | `api_my_tool`          |
| Prompt         | `my_prompt`       | `api_my_prompt`        |
| Resource       | `data://info`     | `data://api/info`      |
| Template       | `data://{id}`     | `data://api/{id}`      |

**Tool name format:** `{namespace}_{original_tool_name}` (underscore separator).

*(source: servers-composition.md > "How Namespacing Works")*

### 3.3 Mount Without Namespace

When a server is mounted without a namespace (`namespace=None`), its tools, resources, templates, and prompts are accessible with their original names. Multiple servers can be mounted without namespaces, and they will be tried in order until a match is found.

```python
main = FastMCP("Main")
main.mount(server_a)  # Tools keep original names
main.mount(server_b)  # Tried after server_a
```

> ⚠️ **For Phase 4**, always use namespaces to avoid collisions.

*(source: python-sdk-fastmcp-server-server.md > `mount()`)*

### 3.4 mount() Signature

```python
mount(
    self,
    server: FastMCP,
    namespace: str | None = None,
    as_proxy: bool | None = None,       # Deprecated
    tool_names: dict[str, str] | None = None,
    prefix: str | None = None           # Deprecated — use namespace
) -> None
```

**Parameters:**

- `server`: The FastMCP server to mount.
- `namespace`: Optional namespace prefix. If `None`, objects keep original names.
- `tool_names`: Optional mapping of original tool names to custom names. Use this to override namespaced names. Keys are the **original** tool names from the mounted server.
- `prefix`: **Deprecated** — use `namespace` instead.

*(source: python-sdk-fastmcp-server-server.md > `mount()`)*

### 3.5 Mounting External Servers (Remote / Subprocess)

Mount remote HTTP servers or subprocess-based MCP servers using `create_proxy()`:

```python
from fastmcp import FastMCP
from fastmcp.server import create_proxy

mcp = FastMCP("Orchestrator")

# Mount a remote HTTP server (URLs work directly)
mcp.mount(create_proxy("http://api.example.com/mcp"), namespace="api")

# Mount local Python scripts (file paths work directly)
mcp.mount(create_proxy("./my_server.py"), namespace="local")
```

#### Mounting npm / uvx Packages

```python
from fastmcp import FastMCP
from fastmcp.server import create_proxy

mcp = FastMCP("Orchestrator")

# Mount npm package via config
github_config = {
    "mcpServers": {
        "default": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        }
    }
}
mcp.mount(create_proxy(github_config), namespace="github")

# Mount Python tool via config
sqlite_config = {
    "mcpServers": {
        "default": {
            "command": "uvx",
            "args": ["mcp-server-sqlite", "--db", "data.db"]
        }
    }
}
mcp.mount(create_proxy(sqlite_config), namespace="db")
```

Or use explicit transport classes:

```python
from fastmcp import FastMCP
from fastmcp.server import create_proxy
from fastmcp.client.transports import NpxStdioTransport, UvxStdioTransport

mcp = FastMCP("Orchestrator")

mcp.mount(
    create_proxy(NpxStdioTransport(package="@modelcontextprotocol/server-github")),
    namespace="github"
)
mcp.mount(
    create_proxy(UvxStdioTransport(tool_name="mcp-server-sqlite", tool_args=["--db", "data.db"])),
    namespace="db"
)
```

*(source: servers-composition.md > "Mounting External Servers")*

### 3.6 Dynamic Composition

Because `mount()` creates a live link, you can add components to a child server **after** mounting and they'll be immediately available through the parent:

```python
main = FastMCP("Main")
main.mount(dynamic_server, namespace="dynamic")

# Add a tool AFTER mounting - it's accessible through main
@dynamic_server.tool
def added_later() -> str:
    return "Added after mounting!"
```

*(source: servers-composition.md > "Dynamic Composition")*

### 3.7 Conflict Resolution

When mounting multiple servers with the same namespace (or no namespace), the **most recently mounted** server takes precedence for conflicting component names:

```python
server_a = FastMCP("A")
server_b = FastMCP("B")

@server_a.tool
def shared_tool() -> str:
    return "From A"

@server_b.tool
def shared_tool() -> str:
    return "From B"

main = FastMCP("Main")
main.mount(server_a)
main.mount(server_b)

# shared_tool returns "From B" (most recently mounted)
```

> ⚠️ Use namespaces to **prevent** this ambiguity entirely.

*(source: servers-composition.md > "Conflict Resolution")*

### 3.8 Performance Considerations

Operations like `list_tools()` on the parent are affected by the performance of all mounted servers:

| Scenario                    | Latency        |
| --------------------------- | -------------- |
| Local tools                 | 1-2ms          |
| HTTP-based mounted servers  | 300-400ms      |

Deep mounting hierarchies and servers with slow initialization also add overhead. If low latency is critical, consider implementing caching strategies or limiting mounting depth.

*(source: servers-composition.md > "Performance Considerations")*

#### ProxyProvider Caching (v3.2.0+)

`ProxyProvider` caches the backend's component lists so that individual lookups don't require a separate backend connection. The cache refreshes whenever an explicit `list_*` call is made, and entries expire after a configurable TTL (default 300 seconds):

```python
from fastmcp.server.providers.proxy import ProxyProvider, ProxyClient

# Default 300s TTL
provider = ProxyProvider(lambda: ProxyClient("http://backend/mcp"))

# Custom TTL
provider = ProxyProvider(lambda: ProxyClient("http://backend/mcp"), cache_ttl=60)

# Disable caching for dynamic backends
provider = ProxyProvider(lambda: ProxyClient("http://backend/mcp"), cache_ttl=0)
```

*(source: servers-providers.md > "Component List Caching")*

---

## 4. FastMCP Composition — Option B: `Client({"mcpServers": ...})`

### 4.1 Client-Side Multi-Server Configuration

`Client` accepts a config dictionary with multiple servers. When the config contains **multiple servers**, it creates a composite client by mounting all servers on a single FastMCP instance, with each server's name used as its namespace prefix.

> **Important:** If the config contains exactly **one** server, a direct connection is established **without** prefixing.

```python
from fastmcp import Client

config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "transport": "http"
        },
        "assistant": {
            "command": "python",
            "args": ["./assistant.py"],
            "env": {"LOG_LEVEL": "INFO"}
        }
    }
}

client = Client(config)

async with client:
    # Tools are namespaced by server name
    weather = await client.call_tool("weather_get_forecast", {"city": "NYC"})
    answer = await client.call_tool("assistant_ask", {"question": "What?"})

    # Resources use prefixed URIs
    icons = await client.read_resource("weather://weather/icons/sunny")
```

*(source: clients.md > "Multi-Server Configuration", Original URL: https://gofastmcp.com/clients/overview.md)*

### 4.2 Naming Pattern

In the multiserver case, tools are accessible with the pattern `{server_name}_{tool_name}` and resources with `protocol://{server_name}/path/to/resource`.

*(source: python-sdk-fastmcp-client-transports-config.md > `MCPConfigTransport`)*

### 4.3 Tool Transformations in Config

FastMCP supports tool transformations within the configuration dictionary. You can change names, descriptions, tags, and arguments:

```python
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "transport": "http",
            "tools": {
                "weather_get_forecast": {
                    "name": "miami_weather",
                    "description": "Get the weather for Miami",
                    "arguments": {
                        "city": {
                            "default": "Miami",
                            "hide": True,
                        }
                    }
                }
            }
        }
    }
}
```

To filter tools by tag, use `include_tags` or `exclude_tags` at the server level:

```python
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "include_tags": ["forecast"]  # Only tools with this tag
        }
    }
}
```

*(source: clients.md > "Tool Transformations")*

### 4.4 create_proxy() with Multi-Server Config

You can also use `create_proxy()` with a multi-server config dict to create a unified proxy with automatic namespacing:

```python
from fastmcp.server import create_proxy

config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "http"
        },
        "calendar": {
            "url": "https://calendar-api.example.com/mcp",
            "transport": "http"
        }
    }
}

# Creates unified proxy with prefixed components:
# - weather_get_forecast
# - calendar_add_event
composite = create_proxy(config, name="Composite")
```

*(source: servers-providers.md > "Multi-Server Proxies")*

---

## 5. Tool Format Conversion: MCP → TensorZero

This is the critical bridge between FastMCP and TensorZero.

### MCP tool schema

An MCP tool (from `session.list_tools()` or `client.list_tools()`) has this shape:

```python
{
    "name": "some_tool_name",        # string (already namespaced after mount)
    "description": "What it does",   # string
    "inputSchema": { ... }           # JSON Schema object
}
```

### TensorZero `additional_tools` format

Each dynamic tool passed to `additional_tools` must have:

| Field          | Type   | Required | Description                             |
|----------------|--------|----------|-----------------------------------------|
| `name`         | string | yes      | The name of the tool                    |
| `description`  | string | yes      | A description of what the tool does     |
| `parameters`   | object | yes      | A JSON Schema defining the tool's parameters |
| `strict`       | bool   | no       | Enforce strict schema validation (default: `false`) |

*(source: gateway-api-reference.md > "API Reference: Inference" > `additional_tools` > "Function Tools")*

### Conversion code (verbatim from official TensorZero example)

```python
# Load the tools from the MCP server
mcp_tools = await session.list_tools()
mcp_tools = [tool.model_dump() for tool in mcp_tools.tools]

# Convert MCP tool format to TensorZero tool format
t0_tools = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"],
    }
    for tool in mcp_tools
]
```

*(source: examples/mcp-model-context-protocol/main.py)*

### Multi-server adaptation

When using FastMCP with mounted/namespaced servers, the `tool["name"]` returned by FastMCP's `list_tools()` **already includes the namespace prefix** (e.g., `a_list_tables`, `b_search_docs`). The conversion code above remains identical — the namespaced names flow through transparently because TensorZero treats them as opaque strings.

### Using FastMCP Client instead of raw session

If using Option B (`Client`), the tools come via `client.list_tools()`:

```python
from fastmcp import Client

async with Client(config) as client:
    mcp_tools = await client.list_tools()
    t0_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools
    ]
```

*(source: python-sdk-fastmcp-client-mixins-tools.md > `list_tools`)*

---

## 6. Making Inference Requests

### Python SDK (TensorZero native client)

```python
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:

    response = await t0.inference(
        function_name="multi_server_agent",
        episode_id=episode_id,          # None for first turn
        input={
            "messages": messages,
        },
        additional_tools=t0_tools,       # All combined tools from all servers
        # Optional overrides:
        # tool_choice="auto",
        # parallel_tool_calls=True,
        # allowed_tools=["a_list_tables", "b_search_docs"],  # restrict subset
    )
```

*(source: examples/mcp-model-context-protocol/main.py)*

### Restricting tools per request

```python
response = await t0.inference(
    function_name="multi_server_agent",
    input={"messages": messages},
    additional_tools=t0_tools,
    allowed_tools=["a_list_tables", "b_search_docs"],  # Only these two
)
```

> The names should be the configuration keys. For dynamic tools, the `name` field is the key.

*(source: gateway-guides.md > "Restricting allowed tools at inference time")*

---

## 7. Processing ToolCall Responses

### ToolCall content block fields

| Field            | Type   | Description                                                      |
|------------------|--------|------------------------------------------------------------------|
| `name`           | string | The **validated** name of the tool (`null` if invalid)           |
| `raw_name`       | string | The name generated by the model (which might be invalid)         |
| `arguments`      | object | The **validated** arguments (`null` if invalid)                  |
| `raw_arguments`  | string | The raw arguments string from the model (might be invalid JSON)  |
| `id`             | string | The ID of the content block (needed for tool_result)             |

*(source: gateway-api-reference.md > Response > Chat Function > `tool_call` content block)*

### Routing tool calls back to FastMCP

#### Option A: Using a composed server with in-memory client

```python
from fastmcp import Client

# 'main' is your composed FastMCP server with mounted sub-servers
async with Client(main) as client:
    for block in response.content:
        if isinstance(block, ToolCall):
            # FastMCP routes to the correct server based on namespace
            mcp_result = await client.call_tool(
                block.name,       # e.g., "a_list_tables"
                block.arguments   # parsed dict
            )
            # mcp_result is a CallToolResult
            result_text = "\n".join(
                [b.text for b in mcp_result.content]
            )
```

#### Option B: Using raw MCP session (as in TensorZero example)

```python
# 'session' is an MCP ClientSession connected to the composed server
mcp_tool_result = await session.call_tool(
    content_block.name,       # The namespaced tool name
    content_block.arguments   # The parsed arguments dict
)

t0_tool_result_text = "\n".join(
    [block.text for block in mcp_tool_result.content]
)
```

*(source: examples/mcp-model-context-protocol/main.py)*

### FastMCP Client.call_tool() details

```python
call_tool(
    self,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    version: VersionSpec | None = None,
    timeout: timedelta | float | int | None = None,
    progress_handler: ProgressHandler | None = None,
    raise_on_error: bool = True,
    meta: dict[str, Any] | None = None,
) -> CallToolResult
```

Unlike `call_tool_mcp`, this method **raises a ToolError** if the tool call results in an error (when `raise_on_error=True`, which is the default).

*(source: python-sdk-fastmcp-client-mixins-tools.md > `call_tool`)*

### Appending tool results to the TensorZero conversation

```python
if tool_results:
    user_message = Message(
        role="user",
        content=tool_results,  # list of ToolResult
    )
    messages.append(user_message)
```

Tool results must include:

| Field    | Type   | Required | Description                              |
|----------|--------|----------|------------------------------------------|
| `id`     | string | yes      | Must match the `id` from the `tool_call` |
| `name`   | string | yes      | The tool name                            |
| `result` | string | yes      | The result text                          |

*(source: gateway-api-reference.md > `input.messages` > content block type `tool_result`)*

---

## 8. Complete Agent Loop Pattern

### Using Option A (mount-based composition)

```python
import asyncio
from fastmcp import FastMCP, Client
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult


async def run_agent():
    # --- Step 1: Compose FastMCP servers ---
    server_a = FastMCP("DatabaseTools")
    server_b = FastMCP("DocTools")

    @server_a.tool
    def list_tables() -> str:
        """List available database tables."""
        return "users, orders, products"

    @server_a.tool
    def query_table(table: str, query: str) -> str:
        """Query a database table."""
        return f"Results from {table}: [mock data for '{query}']"

    @server_b.tool
    def search_docs(query: str) -> str:
        """Search documentation."""
        return f"Doc results for: {query}"

    main = FastMCP("Main")
    main.mount(server_a, namespace="db")
    main.mount(server_b, namespace="docs")

    # --- Step 2: Discover tools via FastMCP Client ---
    async with Client(main) as mcp_client:
        mcp_tools = await mcp_client.list_tools()

        # Tools will be: db_list_tables, db_query_table, docs_search_docs
        t0_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

        # --- Step 3: Run agent loop with TensorZero ---
        async with await AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000"
        ) as t0:
            messages = []
            episode_id = None

            while True:
                # Prompt user if no pending tool results
                if not messages or not isinstance(messages[-1].content, list):
                    user_input = input("\n[User]\n")
                    if user_input.lower() == "quit":
                        break
                    messages.append(Message(role="user", content=user_input))

                # Call TensorZero with all combined tools
                response = await t0.inference(
                    function_name="multi_server_agent",
                    episode_id=episode_id,
                    input={"messages": messages},
                    additional_tools=t0_tools,
                )
                episode_id = response.episode_id

                # Append assistant response
                messages.append(
                    Message(role="assistant", content=response.content)
                )

                # Process content blocks
                tool_results = []
                for block in response.content:
                    if isinstance(block, Text):
                        print(f"\n[Assistant]\n{block.text}")

                    elif isinstance(block, ToolCall):
                        print(f"\n[Tool Call: {block.raw_name}]")
                        print(block.raw_arguments)

                        if block.name is None:
                            print(f"[ERROR] Invalid tool name: {block.raw_name}")
                            tool_results.append(ToolResult(
                                name=block.raw_name,
                                result="Error: tool not found",
                                id=block.id,
                            ))
                            continue

                        # FastMCP routes to the correct server
                        mcp_result = await mcp_client.call_tool(
                            block.name, block.arguments
                        )
                        result_text = "\n".join(
                            [b.text for b in mcp_result.content]
                        )

                        print(f"\n[Tool Result]\n{result_text}")

                        tool_results.append(ToolResult(
                            name=block.name,
                            result=result_text,
                            id=block.id,
                        ))

                if tool_results:
                    messages.append(
                        Message(role="user", content=tool_results)
                    )
                # Loop continues — if tool results exist, inference runs again


asyncio.run(run_agent())
```

### Using Option B (Client config dict)

```python
import asyncio
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult


async def run_agent():
    # --- Step 1: Multi-server config ---
    config = {
        "mcpServers": {
            "db": {
                "command": "python",
                "args": ["./db_server.py"],
            },
            "docs": {
                "url": "http://localhost:8001/mcp",
                "transport": "http",
            }
        }
    }

    # --- Step 2: Discover tools ---
    async with Client(config) as mcp_client:
        mcp_tools = await mcp_client.list_tools()

        # Tools will be: db_list_tables, db_query_table, docs_search_docs
        t0_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

        # --- Step 3: Agent loop (identical to Option A) ---
        async with await AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000"
        ) as t0:
            messages = []
            episode_id = None

            while True:
                if not messages or not isinstance(messages[-1].content, list):
                    user_input = input("\n[User]\n")
                    if user_input.lower() == "quit":
                        break
                    messages.append(Message(role="user", content=user_input))

                response = await t0.inference(
                    function_name="multi_server_agent",
                    episode_id=episode_id,
                    input={"messages": messages},
                    additional_tools=t0_tools,
                )
                episode_id = response.episode_id
                messages.append(
                    Message(role="assistant", content=response.content)
                )

                tool_results = []
                for block in response.content:
                    if isinstance(block, Text):
                        print(f"\n[Assistant]\n{block.text}")
                    elif isinstance(block, ToolCall):
                        if block.name is None:
                            tool_results.append(ToolResult(
                                name=block.raw_name,
                                result="Error: tool not found",
                                id=block.id,
                            ))
                            continue

                        mcp_result = await mcp_client.call_tool(
                            block.name, block.arguments
                        )
                        result_text = "\n".join(
                            [b.text for b in mcp_result.content]
                        )
                        tool_results.append(ToolResult(
                            name=block.name,
                            result=result_text,
                            id=block.id,
                        ))

                if tool_results:
                    messages.append(
                        Message(role="user", content=tool_results)
                    )


asyncio.run(run_agent())
```

---

## 9. Transforms and Advanced Namespacing

### 9.1 The Namespace Transform (Under the Hood)

When you call `mount(server, namespace="api")`, FastMCP creates a `Namespace` transform automatically. You can also apply it directly:

```python
from fastmcp import FastMCP
from fastmcp.server.transforms import Namespace

mcp = FastMCP("Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Namespace all components
mcp.add_transform(Namespace("api"))

# Tool is now: api_greet
```

*(source: servers-transforms.md > "Namespace Transform", Original URL: https://gofastmcp.com/servers/transforms/namespace.md)*

### 9.2 Provider-Level Transforms (Post-Mount)

When using `mount()`, the returned provider reference lets you add transforms directly:

```python
main = FastMCP("Main")
mount = main.mount(sub_server, namespace="api")
mount.add_transform(ToolTransform({...}))
```

*(source: servers-transforms.md > "Provider-Level Transforms")*

### 9.3 Tool Transformation (Renaming, Reshaping)

Use `ToolTransform` to modify tool schemas after namespacing:

```python
from fastmcp import FastMCP
from fastmcp.server.transforms import ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

main = FastMCP("Main")
mount = main.mount(sub_server, namespace="api")
mount.add_transform(ToolTransform({
    "api_verbose_internal_name": ToolTransformConfig(
        name="search",
        description="Search the database.",
    )
}))
```

*(source: servers-transforms.md > "Tool Transformation")*

### 9.4 Transform Order

Transforms stack in the order they're added. The first transform added is innermost (closest to the provider):

```python
from fastmcp.server.providers import FastMCPProvider
from fastmcp.server.transforms import Namespace, ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

provider = FastMCPProvider(server)
provider.add_transform(Namespace("api"))           # Applied first
provider.add_transform(ToolTransform({             # Sees namespaced names
    "api_verbose_name": ToolTransformConfig(name="short"),
}))

# Flow: "verbose_name" -> "api_verbose_name" -> "short"
```

When a client requests "short", the transforms reverse the mapping: ToolTransform maps "short" to "api_verbose_name", then Namespace strips the prefix to find "verbose_name" in the provider.

*(source: servers-transforms.md > "Transform Order")*

---

## 10. Docker Compose Setup (Reference)

Based on the official MCP example:

```yaml
services:
  clickhouse:
    image: clickhouse:lts
    environment:
      CLICKHOUSE_USER: "chuser"
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: "1"
      CLICKHOUSE_PASSWORD: "chpassword"
    ports:
      - "8123:8123"
    healthcheck:
      test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      TENSORZERO_CLICKHOUSE_URL: "http://chuser:chpassword@clickhouse:8123/tensorzero"
      OPENAI_API_KEY: ${OPENAI_API_KEY:?Must be set}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000/health"]
      start_period: 1s
      start_interval: 1s
      timeout: 1s
    depends_on:
      clickhouse:
        condition: service_healthy
```

*(source: examples/mcp-model-context-protocol/docker-compose.yml)*

---

## 11. Checklist Mapping: Phase 4 Items → Implementation Details

### F4.1 — Option A (mount)

**FastMCP:** `main.mount(server_a, namespace="a")` + `main.mount(server_b, namespace="b")` is the documented, supported API. The connection is live and dynamic.

**mount() signature:**
```python
mount(self, server: FastMCP, namespace: str | None = None, ...) -> None
```

**TensorZero side:** No changes needed. TensorZero is agnostic to how tools are composed.

*(source: servers-composition.md, python-sdk-fastmcp-server-server.md > `mount()`)*

---

### F4.2 — Namespaced tool names

**FastMCP:** After mounting with namespaces, `list_tools()` returns names like `a_tool1`, `b_tool2`. The naming pattern is `{namespace}_{original_name}` with an underscore separator.

**Namespacing table (confirmed):**

| Component | Original   | With `namespace="a"` |
|-----------|-----------|----------------------|
| Tool      | `tool1`   | `a_tool1`            |
| Tool      | `tool2`   | `b_tool2`            |

**TensorZero side:** The conversion code passes `tool["name"]` (or `tool.name`) directly. Whatever FastMCP returns is what the LLM sees.

*(source: servers-composition.md > "How Namespacing Works")*

---

### F4.3 — T0 receives all combined tools

Pass the full `t0_tools` list to `additional_tools`:

```python
additional_tools=t0_tools  # All tools from all mounted servers
```

TensorZero forwards all of them to the LLM provider. The LLM can invoke any of them.

---

### F4.4 — Routing via namespaced name

When the LLM returns a `ToolCall`, the `name` / `raw_name` fields contain the namespaced name (e.g., `a_list_tables`). Pass it directly to FastMCP's `call_tool()`, which routes to the correct server via the Namespace transform's reverse mapping.

**Important distinction in T0 response:**
- `raw_name`: The exact string the model generated (might be invalid)
- `name`: The validated name (set to `null` if the name doesn't match any tool in `additional_tools`)

Use `block.name` for routing (it's validated). If `name` is `None`, the model hallucinated a tool name — handle as error.

*(source: gateway-api-reference.md > Response > `tool_call` content block)*

**FastMCP routing:** The `Namespace` transform's `get_tool()` method receives the namespaced name, strips the prefix, and calls `call_next()` with the original name, which reaches the correct mounted provider.

*(source: python-sdk-fastmcp-server-transforms-namespace.md > `get_tool`)*

---

### F4.5 — Option B (Client config dict)

**FastMCP:**

```python
from fastmcp import Client

client = Client({
    "mcpServers": {
        "a": {"command": "python", "args": ["./server_a.py"]},
        "b": {"url": "http://localhost:8001/mcp", "transport": "http"}
    }
})

async with client:
    tools = await client.list_tools()
    # Returns: a_tool1, a_tool2, b_tool3, b_tool4
    result = await client.call_tool("a_tool1", {"arg": "value"})
```

The `MCPConfigTransport` creates a composite client by mounting all servers on a single FastMCP instance, with each server's name used as its namespace.

> **Edge case:** If the config contains exactly **one** server, a direct connection is established **without** prefixing.

**TensorZero side:** Identical to Option A — the tool discovery mechanism is transparent. Convert with the same `{name, description, parameters}` pattern.

*(source: python-sdk-fastmcp-client-transports-config.md > `MCPConfigTransport`, clients.md > "Multi-Server Configuration")*

---

### F4.6 — No name collisions

**FastMCP guarantees:** Namespace prefixes prevent collisions. `a_search` vs `b_search` are distinct tools.

**TensorZero enforces uniqueness:**
> At inference time, the gateway ensures that an inference request doesn't have multiple tools with the same name.

If collisions somehow occur, the TensorZero gateway will reject the request.

**FastMCP conflict behavior (without namespaces):** When mounting without namespaces, the most recently mounted server takes precedence. **Always use namespaces** in multi-server scenarios.

*(source: gateway-configuration-reference.md > `[tools.tool_name]` > `name`, servers-composition.md > "Conflict Resolution")*

---

## 12. Gotchas and Edge Cases

### 12.1 — `name` vs `raw_name` on ToolCall

The model might hallucinate a tool name that doesn't exist. Always check `block.name` (validated) before routing. If `name is None`, the model produced an invalid tool name stored in `raw_name`.

*(source: gateway-api-reference.md > Response > `tool_call` content block)*

### 12.2 — `tool_choice` defaults

If you pass `additional_tools` but no `tool_choice`, TensorZero defaults to `"auto"`. If you want to force tool use, set `tool_choice="required"`.

*(source: gateway-configuration-reference.md > `tool_choice`)*

### 12.3 — `parallel_tool_calls` provider support

> Most model providers do not support parallel tool calls. At the moment, only Fireworks AI and OpenAI support parallel tool calls.

*(source: gateway-api-reference.md > `parallel_tool_calls`)*

### 12.4 — Dynamic tools vs static tools

> You should only use dynamic tools if your use case requires it. Otherwise, it's recommended to define tools in the configuration file.

For the MCP multi-server pattern, dynamic tools are the correct approach because tool definitions come from external servers at runtime.

*(source: gateway-guides.md > "Defining tools dynamically at inference time")*

### 12.5 — Tool result message structure

Tool results must be sent as `role: "user"` messages. Each `ToolResult` must include the `id` that matches the original `ToolCall.id`. Failing to match IDs will cause errors.

*(source: gateway-api-reference.md > `input.messages` > content block type `tool_result`)*

### 12.6 — Multiple tool results per message

When the model calls multiple tools in parallel, collect all results and send them in a single user message:

```python
if tool_results:
    user_message = Message(role="user", content=tool_results)  # list of ToolResult
    messages.append(user_message)
```

*(source: examples/mcp-model-context-protocol/main.py)*

### 12.7 — Single-server Client config skips namespacing

If your `mcpServers` config dict contains exactly **one** server, the `Client` establishes a **direct** connection without prefixing tool names. Make sure you have multiple entries when testing Phase 4.

*(source: python-sdk-fastmcp-client-transports-inference.md > `infer_transport`)*

### 12.8 — FastMCP Client raises on tool errors by default

`client.call_tool()` raises a `ToolError` on failure by default (`raise_on_error=True`). Wrap calls in try/except or pass `raise_on_error=False` for graceful handling:

```python
try:
    mcp_result = await mcp_client.call_tool(block.name, block.arguments)
except Exception as e:
    result_text = f"Error calling tool: {e}"
```

*(source: python-sdk-fastmcp-client-mixins-tools.md > `call_tool`)*

### 12.9 — `prefix` parameter is deprecated

Use `namespace` instead of `prefix` in `mount()`:

```python
# Deprecated
main.mount(subserver, prefix="api")

# Correct
main.mount(subserver, namespace="api")
```

*(source: getting-started.md > "Deprecated Features")*

### 12.10 — Proxy latency impact

When mounting proxy servers (remote HTTP or subprocess), latency affects all operations on the parent server:

| Operation      | Local | Proxied (HTTP) |
| -------------- | ----- | -------------- |
| `list_tools()` | 1-2ms | 300-400ms      |
| `call_tool()`  | 1-2ms | 200-500ms      |

Consider session reuse for stateless backends or caching for component lists.

*(source: servers-providers.md > "Performance Considerations")*

### 12.11 — Provider ordering

When a client requests a tool, FastMCP queries providers in registration order. The first provider that has the tool handles the request. `LocalProvider` is always first.

*(source: servers-providers.md > "Provider Order")*

---

## 13. Source References

| Topic                        | File                                          | Section / URL                                                     |
|------------------------------|-----------------------------------------------|-------------------------------------------------------------------|
| Composition (mount)          | `servers-composition.md`                      | https://gofastmcp.com/servers/composition.md                      |
| Namespace transform          | `servers-transforms.md`                       | https://gofastmcp.com/servers/transforms/namespace.md             |
| Transforms overview          | `servers-transforms.md`                       | https://gofastmcp.com/servers/transforms/transforms.md            |
| Proxy provider               | `servers-providers.md`                        | https://gofastmcp.com/servers/providers/proxy.md                  |
| Client overview              | `clients.md`                                  | https://gofastmcp.com/clients/overview.md                         |
| Client transports config     | `python-sdk-fastmcp-client-transports-config.md` | https://gofastmcp.com/python-sdk/fastmcp-client-transports-config.md |
| Client call_tool             | `python-sdk-fastmcp-client-mixins-tools.md`   | https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools.md  |
| mount() API                  | `python-sdk-fastmcp-server-server.md`         | https://gofastmcp.com/python-sdk/fastmcp-server-server.md        |
| Namespace class              | `python-sdk-fastmcp-server-transforms-namespace.md` | https://gofastmcp.com/python-sdk/fastmcp-server-transforms-namespace.md |
| MCP integration (T0)         | `gateway-guides.md`                           | "Integrating with Model Context Protocol (MCP) servers"           |
| Official MCP example (T0)    | `examples/mcp-model-context-protocol/main.py` | Full source                                                       |
| `additional_tools` API (T0)  | `gateway-api-reference.md`                    | `POST /inference` > `additional_tools`                            |
| Tool config reference (T0)   | `gateway-configuration-reference.md`          | `[tools.tool_name]`                                               |
| ToolCall response format (T0)| `gateway-api-reference.md`                    | Response > Chat Function > `tool_call`                            |
| `allowed_tools` (T0)         | `gateway-guides.md`                           | "Restricting allowed tools at inference time"                     |
| Deprecations (FastMCP)       | `getting-started.md`                          | https://gofastmcp.com/getting-started/migration.md                |
