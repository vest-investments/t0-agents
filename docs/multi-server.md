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

## Pattern C: glow `servers=` (the MCPRuntime router)

On the glow API, declare servers once and let the App route them:

```python
from t0_agents.glow import App, MCPServer

class Math(MCPServer, url="http://math-server:8000/mcp", namespace="math"): ...

app = App(
    t0=t0,
    servers=[
        Math,
        MCPServer(server=text_fastmcp, namespace="text"),  # in-memory instance
    ],
)

@app.agent("helper")
async def helper(question: str) -> str:
    """Answers with math_* and text_* tools."""
```

Behind `servers=` the App builds an `MCPRuntime` — one long-lived
fastmcp `Client` per server, shared by every agent and conversation in
the process.  Compared to composing servers by hand:

- **Warm connections.**  The handshake happens once per server on first
  use; a warm tool call is a single POST.  (The pre-2.2 composite proxy
  reconnected on every remote call.)
- **Mount-parity naming.**  `namespace="math"` prefixes tools as
  `math_add`; `prefix=False` presents names verbatim.
- **Fail-loud collisions.**  Two mounts producing the same final tool
  name raise `ConfigurationError` at build instead of silently shipping
  duplicates to inference.
- **Cached inventories.**  Remote `list_tools` results cache for 60 s per
  server, invalidated early by `tools/list_changed`; in-memory servers
  stay live-linked.
- **Bounded fan-out.**  Each server gets `tool_list_timeout` seconds
  (default 10) of the inventory fan-out.  A server that cannot answer
  degrades instead of blocking the merge: its last-known inventory is
  served (a tool added while it is unreachable stays invisible until it
  answers again), and a server that has never answered is skipped with
  a warning.
- **Session-loss recovery.**  A server rollout that terminates warm
  sessions is absorbed: the runtime rebuilds the client and re-sends the
  rejected call exactly once (safe — the server rejected it before
  dispatch).

Authenticated mounts (`headers=`/`auth=`) and per-agent `tools=`
filtering are covered in
[typed-mcp-and-guardrails](glow/typed-mcp-and-guardrails.md).

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

See `examples/agent/14_multi_server/` for a runnable example with Docker Compose configuration.
