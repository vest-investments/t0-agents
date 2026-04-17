# Multi-Agent Composition

`t0-agents` supports the classic **agent-as-tool** pattern: one `Agent`
delegates to other `Agent`s by exposing each one as an MCP tool on the
orchestrator's FastMCP server.  The orchestrator's LLM sees the
sub-agents as ordinary tools; calling one invokes the sub-agent's full
`.run()` loop and returns its response text.

Shipped in Milestone **M7** (Phase G).  No new dependencies — built on
top of the existing `core` + `mcp` submodules.

## Installation

```bash
pip install t0-agents
```

No extras required.

## Quick start

```python
from contextlib import asynccontextmanager
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

researcher = Agent(
    AgentConfig(name="researcher", function_name="research"),
    t0=t0,
    tool_provider=arxiv_client,
)
writer = Agent(
    AgentConfig(name="writer", function_name="writing"),
    t0=t0,
    tool_provider=markdown_client,
)

orchestrator = Agent(
    AgentConfig(
        name="report_generator",
        function_name="orchestrator",
        sub_agents=[researcher, writer],
    ),
    t0=t0,
    tool_provider=orchestrator_client,
)

result = await orchestrator.run(
    "Write a report on RLHF advances in 2025",
)
```

The orchestrator's `tool_provider` is a `fastmcp.Client` wrapping a
plain `FastMCP("…")` instance.  When the orchestrator is constructed,
the framework registers a `researcher_ask` and a `writer_ask` tool on
that server; the orchestrator LLM sees them in every inference's
`additional_tools` list.

See `examples/14_multi_agent/` for a runnable demo.

## How delegation works

Each sub-agent is wrapped via `t0_agents.compose.agent_as_tool`:

```python
async def _handler(message: str) -> str:
    result = await sub_agent.run(message)
    return result.response
```

The handler is registered with FastMCP's public runtime API
(`Tool.from_function(...)` + `server.add_tool(tool)`).  Sub-agents
keep all of their usual collaborators — memory, guardrails, their own
`tool_provider` — so a sub-agent's `.run()` is indistinguishable from
calling it directly.

## Naming and collisions

By default each sub-agent becomes `f"{sub_agent.config.name}_ask"`.
You can override this per sub-agent via
`t0_agents.compose.agent_as_tool(sub_agent, tool_name="...")` if you
want to expose multiple facets of the same sub-agent or avoid a
collision.

Collisions are caught eagerly at construction time:

| Failure | Exception raised |
| --- | --- |
| Two sub-agents share `config.name`               | `ConfigurationError` |
| Sub-agent tool-name collides with an existing tool on the orchestrator's FastMCP server | `ConfigurationError` |
| `tool_provider` is not a `fastmcp.Client`        | `ConfigurationError` |
| The `fastmcp.Client`'s transport is not in-process | `ConfigurationError` |

The in-process requirement comes from FastMCP: only the in-process
`FastMCPTransport` exposes the underlying server instance we need to
call `add_tool(...)` on.  Remote MCP transports (HTTP, SSE, WebSocket)
cannot host runtime-registered tools from this framework.

## Episode propagation

Each sub-agent starts its own TensorZero episode.  This keeps
ClickHouse analytics clean — one episode per agent chain of
inferences — and avoids semantic confusion from mixing a researcher's
"read papers" messages with a writer's "emit markdown" ones in a
single conversation history.

For correlation, every sub-agent run triggered by the orchestrator
receives a `t0_agents.parent_episode_id` tag pointing back at the
orchestrator's episode id.  You can filter ClickHouse by this tag to
reconstruct the delegation tree:

```sql
SELECT *
FROM tensorzero.ChatInference
WHERE tags['t0_agents.parent_episode_id'] = '<orchestrator-episode-id>'
```

When the orchestrator's own episode id isn't yet known (for example
the very first inference of a fresh orchestrator run, called without
`episode_id=`), the tag value falls back to `"unknown"`.  Once the
first inference lands and populates the episode id, it remains stable
through the rest of the run.

## Error model

Sub-agent handlers follow a **hybrid-fatal** policy:

- `T0AgentsFatalError` subclasses — including `GuardrailBlockedError`
  raised inside a sub-agent's own guardrail stack — propagate
  unchanged through the orchestrator's tool-call boundary.  The
  orchestrator's `.run()` caller sees the real exception type, just
  as they would if the guardrail had been installed on the
  orchestrator directly.
- Every other exception is wrapped in `t0_agents.SubAgentError`
  (subclass of `T0AgentsError`).  The orchestrator's loop stringifies
  it as a tool error (same convention as any other
  `fastmcp.exceptions.ToolError`) so the orchestrator LLM can decide
  to retry, pick a different sub-agent, or give up gracefully.

```python
try:
    result = await orchestrator.run("Do the thing")
except GuardrailBlockedError as exc:
    # A sub-agent's guardrail said no.
    ...
```

`SubAgentError` always carries the original exception as `__cause__`,
so `logging.exception(...)` produces a useful traceback.

## Caveats

- **Tool-schema stability per run.** Sub-agents are registered at
  `Agent.__init__` — they cannot be added mid-conversation.  T0 Q2
  says the gateway is fine with this, but the LLM-side cost of
  seeing the tool list change between turns isn't worth it.  If you
  need conditional delegation, register every candidate up front and
  let the orchestrator decide via `tool_choice`.
- **No peer-to-peer mesh.**  Sub-agents cannot delegate to other
  sub-agents directly.  Deeper trees are achievable by nesting: an
  orchestrator can have sub-agents that are themselves orchestrators
  with their own sub-agents.
- **Elicitation inside sub-agents is possible but per-Client.** Every
  sub-agent owns its own `fastmcp.Client`, so elicitation flows
  reach the sub-agent's own handler — not the orchestrator's.  This
  is usually the right thing (the researcher's arXiv API key
  approval shouldn't leak to the writer's renderer) but it means
  sub-agents in a browser-served deployment need their own
  `ElicitationForwarder` wiring.  See
  [`streaming.md`](streaming.md).
