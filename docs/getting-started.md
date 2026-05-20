# Getting Started

> **Prefer the four-line decorator path?** Start at
> [`docs/glow/getting-started.md`](glow/getting-started.md) — same
> install, one decorator, no `AgentConfig` boilerplate.

## Install

```bash
pip install t0-agents
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add t0-agents
```

## Quickstart: Adapter layer (no Docker)

The adapter converts FastMCP tools into the `additional_tools` dict format that TensorZero expects at inference time, and serializes tool execution results back to plain strings. No gateway or API keys needed.

```python
import asyncio
from fastmcp import Client
from t0_agents.mcp import convert_tools, serialize_tool_result

async def main():
    async with Client(mcp) as client:
        # Discover tools from the MCP server
        mcp_tools = await client.list_tools()

        # Convert to TensorZero format
        t0_tools = convert_tools(mcp_tools)
        # => [{"name": "add", "description": "...", "parameters": {...}}, ...]

        # Call a tool and serialize the result to str
        result = await client.call_tool("add", {"a": 3, "b": 7})
        serialized = serialize_tool_result(result)
        # => "10" (always str, as TensorZero requires)

asyncio.run(main())
```

`convert_tools` maps each `mcp.types.Tool` to a dict with `name`, `description`, and `parameters` (JSON Schema). `None` descriptions are coerced to `""` because TensorZero treats the field as required. `serialize_tool_result` converts a FastMCP `CallToolResult` to a plain string, handling errors, JSON data, and text content blocks.

## Quickstart: Agent loop (Docker required)

The agent loop handles the full cycle: tool discovery, LLM inference, tool execution, and result feedback until the model produces a final text response.

**Prerequisites:**

1. Set `GROQ_API_KEY` in your environment (or in a `.env` file)
2. Start the TensorZero gateway:
   ```bash
   docker compose up -d
   ```

```python
import asyncio
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from t0_agents import run_agent

async def main():
    gateway = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    )
    mcp_client = Client(mcp)

    async with await gateway as t0, mcp_client:
        result = await run_agent(
            t0,
            mcp_client,
            "What is the weather in Tokyo?",
            "agent",  # function name in tensorzero.toml
        )
        print(result.response)       # Final text from the LLM
        print(result.turns)           # Number of inference turns taken
        print(result.episode_id)      # TensorZero episode ID
        print(result.inference_ids)   # One inference ID per turn
        print(result.response_source) # "text" or "thought"

asyncio.run(main())
```

`AsyncTensorZeroGateway` must be entered as an async context manager **before** calling `run_agent`. The optional `tool_provider` (typically a `fastmcp.Client`) is only required when the agent dispatches tools; pass `tool_provider=None` (or omit it) for a chat-only inference loop — no FastMCP server is allocated and `list_tools()` is never called. The caller owns the lifecycle of whichever clients it provides.

## What just happened?

`run_agent` discovered tools from the MCP server via `list_tools()`, converted them with `convert_tools()`, then entered a loop: it called TensorZero's inference API with the user message and tool schemas, checked if the response contained tool calls, executed them against the MCP server, serialized the results back into a user message, and repeated until the LLM returned a text-only response. The final text, episode ID, inference IDs, turn count, and response source are returned as an `AgentResult`. See [Concepts](concepts.md) for the full architecture.

## Next steps

- [Agent](agent.md) -- the declarative, config-driven `Agent` class that wraps `run_agent` and the rest of the stack; this is the API most applications use day-to-day.
- [Concepts](concepts.md) -- architecture, loop lifecycle, design decisions.
- [Elicitation](elicitation.md) -- human-in-the-loop approval and structured input, with no changes to `run_agent`.
- [Error handling](error-handling.md) -- how the loop handles tool errors, hallucinated tool names, timeouts, and `max_turns`.
- [Observability](observability.md) -- episode IDs, inference IDs, feedback to ClickHouse.
- Examples: [`15_mcp_internals`](../examples/agent/15_mcp_internals/) (offline adapter basics), [`01_first_agent`](../examples/agent/01_first_agent/) (loop), [`02_agent_class`](../examples/agent/02_agent_class/) (error handling), [`04_elicitation`](../examples/agent/04_elicitation/).
