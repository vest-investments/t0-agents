# t0-agents

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).

`t0-agents` bridges [TensorZero](https://www.tensorzero.com/) (LLM inference gateway) and [FastMCP](https://gofastmcp.com/) (MCP server framework), enabling LLM agents to discover and execute tools exposed by MCP servers through TensorZero's inference gateway.

## Quick start

### Explore the adapter layer (no Docker)

```python
from fastmcp import Client, FastMCP
from t0_agents.mcp import convert_tools, serialize_tool_result

mcp = FastMCP("Demo")

@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        t0_tools = convert_tools(tools)       # FastMCP → TensorZero format
        result = await client.call_tool("add", {"a": 3, "b": 7})
        print(serialize_tool_result(result))   # "10.0" (always str)
```

### Run the full agent loop (Docker required)

```python
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from t0_agents import run_agent

async def main():
    async with await AsyncTensorZeroGateway.build_http("http://localhost:3000") as t0:
        async with Client(mcp) as client:
            result = await run_agent(t0, client, "What's the weather?", "agent")
            print(result.response)       # Final text from the LLM
            print(result.turns)          # Number of inference turns
            print(result.episode_id)     # TensorZero episode ID for feedback
```

See [examples/](examples/) for the full progression from adapter basics to feedback loops.

## Install

```bash
pip install t0-agents
# or
uv add t0-agents
```

## Documentation

- [Getting Started](docs/getting-started.md) — Quickstart with and without Docker
- [Concepts & Architecture](docs/concepts.md) — Agent loop lifecycle and design rationale
- [API Reference](docs/api-reference.md) — Full reference for all public functions
- [Error Handling](docs/error-handling.md) — Error contract and recovery patterns
- [Multi-Server Composition](docs/multi-server.md) — Combining tools from multiple MCP servers
- [Observability & Feedback](docs/observability.md) — Logging, tracking, and TensorZero feedback
- [Production Guide](docs/production.md) — Deployment, system prompts, and monitoring

## Requirements

- Python >= 3.11
- Docker + Docker Compose (for TensorZero gateway)
- An LLM provider API key (Groq, OpenAI, etc.)

## Development

```bash
uv sync                            # Install dependencies
uv run pytest -m "not integration" # Offline tests (no gateway needed)
uv run pytest -m integration       # Requires docker compose up
uv run ruff check src/             # Lint
uv run mypy src/                   # Type check (strict)
```

## License

MIT
