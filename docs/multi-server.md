# Multi-Server Composition

## Why compose

Real agents need tools from multiple domains (databases, APIs, file systems). FastMCP handles composition natively with automatic namespace prefixes, so `run_agent` requires no changes to work with tools from multiple servers.

## Pattern A: Server-side mount()

```python
from fastmcp import FastMCP

composed = FastMCP("ComposedServer")
composed.mount(math_server, namespace="math")
composed.mount(text_server, namespace="text")
# Tool names become: math_add, math_multiply, text_word_count, text_reverse
```

`run_agent` treats namespaced names as opaque strings. The adapter converts `math_add` the same way it converts `add` -- no configuration needed.

## Pattern B: Client-side config dict

When servers run as separate processes or remote services, use FastMCP's config dict to connect to multiple servers from a single client.

```python
from fastmcp import Client

client = Client({
    "mcpServers": {
        "math": {"url": "http://math-server:8000/mcp"},
        "text": {"url": "http://text-server:8000/mcp"},
    }
})
```

The client auto-prefixes tool names with the server key (`math_`, `text_`). This happens at the Client level before tools reach `run_agent`.

## Namespace rules

| Scenario | Prefix behavior |
|----------|----------------|
| Single server, no mount | No prefix. Tool names used as-is. |
| Multiple servers via mount() | Prefix = namespace argument (`math_add`) |
| Multiple servers via config dict | Prefix = server key in the dict (`math_add`) |
| Name collision across servers | Use distinct namespace/key values to disambiguate |

## TensorZero configuration

Dynamic tools via `additional_tools` means tensorzero.toml does not need static tool definitions. A minimal configuration:

```toml
[models.my_model]
routing = ["my_model_provider"]

[models.my_model.providers.my_model_provider]
type = "openai"
model_name = "llama-3.3-70b-versatile"

[functions.agent]
type = "chat"

[functions.agent.variants.baseline]
type = "chat_completion"
model = "my_model"
system_template = "system.minijinja"
```

All tools are discovered at runtime via `mcp_client.list_tools()` and passed as `additional_tools` to each inference call.

## Examples

### Easy: two in-memory servers

```python
composed = FastMCP("Composed")
composed.mount(math, namespace="math")
composed.mount(text, namespace="text")

async def main() -> None:
    gateway = await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000")
    async with gateway as t0, Client(composed) as client:
        result = await run_agent(t0, client, "Multiply 6 by 7, then reverse 'hello'", "agent")
        print(result.response)
```

### Intermediate: mixed transports (in-memory + remote)

```python
client = Client({
    "mcpServers": {
        "local": {"url": "http://localhost:8001/mcp"},
        "remote": {"url": "https://mcp.example.com/mcp"},
    }
})
# Both servers' tools available with local_ and remote_ prefixes
```

### Advanced: dynamic server discovery

```python
async def build_client(server_urls: dict[str, str]) -> Client:
    """Build a client from a dynamically discovered set of MCP servers."""
    config = {"mcpServers": {name: {"url": url} for name, url in server_urls.items()}}
    return Client(config)

registry = {"billing": "http://billing:8000/mcp", "crm": "http://crm:8000/mcp"}
client = await build_client(registry)
```

See `examples/04_multi_server/` for a runnable example with Docker Compose configuration.
