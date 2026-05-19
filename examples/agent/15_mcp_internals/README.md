# 15 — MCP internals (what `Agent` does under the hood)

Look inside the `t0_agents.mcp` submodule on its own — no Docker, no API
keys, no gateway. The same conversion + serialization helpers the
canonical `Agent` loop relies on, exposed and called directly so you can
see exactly what crosses the wire to TensorZero.

## What it demonstrates

1. **Tool discovery** — listing tools from a FastMCP server via the MCP
   client.
2. **Schema conversion** — `convert_tools` translates MCP tool metadata
   into the `additional_tools` payload that TensorZero expects.
3. **Result serialization** — `serialize_tool_result` turns FastMCP
   `CallToolResult` objects into the plain `str` that TensorZero
   accepts back.

## Prerequisites

- Python 3.11+
- Dependencies installed: `uv sync`

## Run

```bash
uv run python examples/agent/15_mcp_internals/main.py
```

## What to observe

- Two tools are discovered from the in-process FastMCP server (`add`,
  `get_weather`).
- Each tool's JSON schema is printed in the exact format that an
  `Agent` would forward to the TensorZero gateway as `additional_tools`
  on every inference.
- Tool calls return serialized strings, demonstrating the adapter's
  output contract.

This is the plumbing the rest of the framework is built on. If you ever
need to call TensorZero with MCP-discovered tools without going through
`Agent`, the three functions used here are the entire surface.
