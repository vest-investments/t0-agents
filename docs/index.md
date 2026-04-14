# t0-agents Documentation

t0-agents is a Python library that connects [TensorZero](https://www.tensorzero.com/docs/) (an LLM inference gateway with built-in observability and analytics) to [FastMCP](https://gofastmcp.com/) (an MCP server framework for exposing tools). Together, they enable LLM agents that can discover and execute tools through MCP while retaining TensorZero's structured inference, episode tracking, and ClickHouse-backed analytics.

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `t0_agents.mcp` | Stable | MCP adapter (`convert_tools`, `serialize_tool_result`, `build_tool_result`) and agent loop (`run_agent`, `execute_tool_calls`, `execute_tool_calls_parallel`, `has_tool_calls`) |

## Documentation

- [Getting Started](getting-started.md) -- Install, adapter quickstart (no Docker), agent loop quickstart (Docker required)
- [Concepts](concepts.md) -- Architecture, agent loop lifecycle, data flow, design decisions
- [Elicitation](elicitation.md) -- Human-in-the-loop approval and structured input via MCP Elicitation

## Public API

```python
from t0_agents import run_agent, AgentResult, MaxTurnsExceededError, ToolChoice
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker + Docker Compose (for the TensorZero gateway, required by the agent loop)
