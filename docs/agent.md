# Agent Class

The `Agent` class provides a declarative wrapper around the lower-level
`run_agent` function. Configure once with `AgentConfig`, then call
`agent.run(message)` with no boilerplate.

## Quick Start

```python
import asyncio
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from t0_agents import Agent, AgentConfig

async def main():
    config = AgentConfig(
        name="my-assistant",
        function_name="agent",        # must match tensorzero.toml
    )

    gateway = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    )
    mcp_client = Client(my_mcp_server)

    async with gateway as t0, mcp_client:
        agent = Agent(config, t0=t0, tool_provider=mcp_client)
        result = await agent.run("What is 2 + 3?")
        print(result.response)

asyncio.run(main())
```

## AgentConfig

All configuration lives in the `AgentConfig` dataclass:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Human-readable agent name (logging, tags) |
| `function_name` | `str` | *(required)* | TensorZero function name |
| `max_turns` | `int` | `20` | Max inference turns before error |
| `tool_timeout` | `float \| None` | `30.0` | Per-tool timeout in seconds |
| `parallel_tools` | `bool` | `False` | Dispatch tool calls concurrently |
| `dryrun` | `bool` | `False` | Forward `dryrun=True` to T0 |
| `tool_choice` | `ToolChoice \| None` | `None` | Tool-choice directive |
| `tags` | `dict[str, str] \| None` | `None` | Observability tags |
| `namespace` | `str \| None` | `None` | Experimentation namespace |

Validation runs at construction time. Invalid values (empty name, negative
max_turns, etc.) raise `ConfigurationError`.

## Agent

### Constructor

```python
Agent(
    config: AgentConfig,
    *,
    t0: AsyncTensorZeroGateway,
    tool_provider: ToolProvider,
)
```

The caller owns the lifecycle of both `t0` and `tool_provider`. The `Agent`
does not open or close connections.

### run()

```python
async def run(
    self,
    user_message: str,
    *,
    episode_id: str | None = None,
) -> AgentResult
```

Runs the agent loop and returns `AgentResult` with the final response text,
episode metadata, and turn count.

### config

Read-only property returning the `AgentConfig` instance.

## ToolProvider Protocol

`ToolProvider` is a runtime-checkable `Protocol` requiring two async methods:

- `list_tools() -> list[Any]`
- `call_tool(name, arguments, *, timeout=None) -> Any`

`fastmcp.Client` satisfies this protocol. Custom implementations can be used
for testing or non-MCP tool backends.

## Error Handling

| Exception | When |
|---|---|
| `ConfigurationError` | Invalid `AgentConfig` or `tool_provider` |
| `MaxTurnsExceededError` | Agent loop exceeds `max_turns` |

Both are subclasses of `T0AgentsError`, which extends `Exception`.

## When to Use Agent vs run_agent

Use `Agent` when you want clean one-liner calls and plan to add memory,
guardrails, or streaming in future phases. Use `run_agent` directly when you
need full control over every parameter on each call.

Both produce the same `AgentResult`. `Agent.run()` delegates to `run_agent`
internally.
