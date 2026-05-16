# 01 - Adapter Basics

Explore how t0-agents converts FastMCP tools into TensorZero's JSON schema format -- no Docker, no API keys, no gateway needed.

## What it demonstrates

1. **Tool discovery** -- listing tools from a FastMCP server via the MCP client.
2. **Schema conversion** -- `convert_tools` translates MCP tool metadata into the `additional_tools` format that TensorZero expects.
3. **Result serialization** -- `serialize_tool_result` turns FastMCP `CallToolResult` objects into the plain `str` that TensorZero requires.

## Prerequisites

- Python 3.11+
- Dependencies installed: `uv sync`

## Run

```bash
uv run python examples/15_adapter_low_level/main.py
```

## What to observe

- Two tools are discovered from the in-process FastMCP server (`add`, `get_weather`).
- Each tool's JSON schema is printed in the exact format that `run_agent` would send to the TensorZero gateway as `additional_tools`.
- Tool calls return serialized strings, demonstrating the adapter's output contract.
