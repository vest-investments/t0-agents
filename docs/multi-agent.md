# Multi-Agent Composition

Until now every chapter has discussed a single `Agent` instance — one LLM, one tool set, one memory, one guardrail chain. The composition layer lets one agent delegate to others: register sub-agents on `AgentConfig.sub_agents` and they appear as `{name}_ask` tools on the orchestrator's FastMCP server, each invoking its own `invoke()` loop with its own configuration.

`t0-agents` supports the classic **agent-as-tool** pattern: one `Agent`
delegates to other `Agent`s by exposing each one as an MCP tool on the
orchestrator's FastMCP server. The orchestrator's LLM sees the
sub-agents as ordinary tools; calling one invokes the sub-agent's full
`invoke()` loop and returns its response text.

Ships in the base distribution — no extras, no new dependencies beyond
the `core` + `mcp` foundation.

> **Code-driven topologies.**  Alongside the LLM-routed
> `AgentConfig.sub_agents=[...]` path documented on this page, t0-agents
> supports code-driven topologies in plain Python — no special
> framework abstractions needed.  Three canonical examples:
> [Sequential](../examples/agent/12_concurrency/01_sequential.py) (await chain),
> [Parallel](../examples/agent/12_concurrency/02_parallel.py)
> (`asyncio.gather` + reducer), and
> [Orchestrator](../examples/agent/12_concurrency/03_orchestrator.py)
> (the same `sub_agents=` mechanism this page documents).  The
> glow-decorator counterparts live in
> [examples/glow/04_sequential.py](../examples/glow/04_sequential.py),
> [05_parallel.py](../examples/glow/05_parallel.py), and
> [06_orchestrator.py](../examples/glow/06_orchestrator.py).  For the
> decorator-style sub-agent injection pattern (`Depends(<wrapper>)`),
> see [`docs/glow/depends-and-tools.md`](glow/depends-and-tools.md).
>
> Use `sub_agents=` (this page) when the right delegate depends on the
> input and the LLM should decide; write the dispatch in code (the
> Sequential / Parallel patterns) when the topology is fixed.

## Requirements

- Python >= 3.11
- `fastmcp >= 3.2.3`
- `tensorzero >= 2026.4.0`
- A TensorZero gateway (Docker Compose + a provider API key) to run
  the examples end-to-end, same as any other agent.

```bash
pip install t0-agents
```

## Quickstart

```python
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

result = await orchestrator.invoke(
    "Write a report on RLHF advances in 2025",
)
```

The orchestrator's `tool_provider` is a `fastmcp.Client` wrapping a
plain `FastMCP("…")` instance. When the orchestrator is constructed,
the framework registers a `researcher_ask` and a `writer_ask` tool on
that server; the orchestrator LLM sees them in every inference's
`additional_tools` list.

See [`examples/08_multi_agent/`](../examples/08_multi_agent/) for a
runnable demo with an arXiv-flavoured researcher and a markdown
writer.

## Concepts

Each sub-agent is exposed to the orchestrator LLM as an MCP tool
through FastMCP's runtime-registration API (`Tool.from_function` +
`server.add_tool`). The generated tool has a single string argument:

```python
async def researcher_ask(message: str) -> str: ...
```

When the orchestrator LLM decides to call the tool, the framework
runs the sub-agent's full `invoke(message)` loop and returns its
response. Sub-agents keep all of their usual collaborators — memory,
guardrails, their own `tool_provider` — so a delegation is
indistinguishable from calling the sub-agent directly.

**Inline dispatch.** Each sub-agent is registered on the
orchestrator's FastMCP server so `list_tools()` advertises it to
anyone who connects, but when the orchestrator itself makes the
call, the framework dispatches it in the orchestrator's asyncio
task rather than through the server sub-task. That is the mechanism
that keeps the orchestrator's episode id visible to the sub-agent
run (see [Episode propagation](#episode-propagation) below).
External callers that reach the orchestrator's MCP server from
another process still use the server-side handler and see the
default correlation behaviour.

## API reference

Public symbols in `t0_agents.delegation`:

### `agent_as_tool(sub_agent, *, tool_name=None, description=None) -> Tool`

Wrap a sub-agent as a `fastmcp.tools.Tool` ready for
`server.add_tool(...)`. Use this directly when you want to customise
the tool name or description rather than taking the declarative
`AgentConfig.sub_agents` path.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `sub_agent` | `Agent` | — | The sub-agent to expose. |
| `tool_name` | `str \| None` | `None` | Overrides the default `{sub_agent.config.name}_ask`. |
| `description` | `str \| None` | `None` | Overrides the default tool description (the one the orchestrator LLM sees). |

Also re-exported at the package root:

```python
from t0_agents import agent_as_tool
```

### `AgentConfig.sub_agents: list[Agent]`

Declarative delegation slot. When non-empty, `Agent.__init__`
registers each sub-agent as a `{sub_agent.config.name}_ask` tool on
the orchestrator's in-process FastMCP server. Empty (the default)
preserves the single-agent behaviour unchanged.

The orchestrator's `tool_provider` must be a `fastmcp.Client` backed
by an in-process `FastMCPTransport`. Any other tool provider raises
`ConfigurationError` at construction time.

### `install_sub_agents(parent: Agent) -> None`

The hook `Agent.__init__` calls when `AgentConfig.sub_agents` is
non-empty. Normal code doesn't call this directly; it's exposed for
advanced integrations that bypass the declarative config. Raises
`ConfigurationError` on any of:

- A non-`fastmcp.Client` `tool_provider`.
- A `fastmcp.Client` whose transport is not in-process.
- Duplicate `config.name` across the sub-agent list.
- A tool name already present on the orchestrator's FastMCP server.

### `default_tool_name(sub_agent: Agent) -> str`

Returns `f"{sub_agent.config.name}_ask"`. The `_ask` suffix is a
convention that hints to the orchestrator LLM that the tool is a
question-answer delegation, not a side-effecting action. Override by
passing `tool_name=` to `agent_as_tool`, or by choosing a
`config.name` that yields the name you want.

### `assert_unique_names(sub_agents) -> None`

Raises `ConfigurationError` if any two sub-agents in the iterable
share a `config.name`. Called inside `install_sub_agents`; useful on
its own when you are assembling a sub-agent list dynamically and
want to fail early.

### Parent-episode helpers

Advanced: only needed when you write a custom tool wrapper that
wants to tag sub-agent inferences with the orchestrator's episode
id.

- `bind_parent_episode_id(episode_id: str | None)` — context
  manager that sets the current orchestrator episode id.
- `get_parent_episode_id() -> str | None` — reads the current
  binding; returns `None` if called outside a
  `bind_parent_episode_id` block.
- `update_parent_episode_id(episode_id: str) -> None` — overwrites
  the binding in place. Used by `Agent.invoke` / `Run.events` to
  refresh the id once TensorZero allocates the real episode id on
  the first inference.

### `SubAgentError`

Subclass of `T0AgentsError`. Wraps any non-fatal exception raised
by a sub-agent during delegation. The original exception is
preserved as `__cause__`.

```python
from t0_agents import SubAgentError
```

## Naming and collisions

By default each sub-agent becomes `f"{sub_agent.config.name}_ask"`.
Override either by picking a different `config.name`, or by calling
`agent_as_tool(sub_agent, tool_name="...")` directly and registering
the returned tool yourself.

Collisions are caught eagerly at construction time:

| Failure | Exception raised |
| --- | --- |
| Two sub-agents share `config.name` | `ConfigurationError` |
| Sub-agent tool-name collides with an existing tool on the orchestrator's FastMCP server | `ConfigurationError` |
| `tool_provider` is not a `fastmcp.Client` | `ConfigurationError` |
| The `fastmcp.Client`'s transport is not in-process | `ConfigurationError` |

The in-process requirement follows from the FastMCP runtime API —
`add_tool()` only works on a local `FastMCP` server instance. Remote
MCP transports (HTTP, SSE, WebSocket) cannot host runtime-registered
tools from this framework.

## Episode propagation

Each sub-agent starts its own TensorZero episode. This keeps
ClickHouse analytics clean — one episode per agent chain of
inferences — and avoids semantic confusion from mixing a
researcher's "read papers" messages with a writer's "emit markdown"
ones in a single conversation history.

For correlation, every sub-agent run triggered by the orchestrator
receives a `t0_agents.parent_episode_id` tag pointing back at the
orchestrator's episode id. You can filter ClickHouse by this tag to
reconstruct the delegation tree:

```sql
SELECT *
FROM tensorzero.ChatInference
WHERE tags['t0_agents.parent_episode_id'] = '<orchestrator-episode-id>'
```

On a fresh orchestrator run (called without `episode_id=`), the tag
picks up the gateway-allocated episode id as soon as the first
inference lands. The `"unknown"` fallback only surfaces when a
sub-agent is invoked outside an orchestrator context — for example
when the user connects to the orchestrator's MCP server from another
process and calls the `{name}_ask` tool directly.

See [`docs/observability.md`](observability.md) for the wider
analytics story (episode ids, inference ids, feedback signals,
ClickHouse query patterns).

## Error model

Sub-agent handlers follow a **hybrid-fatal** policy:

- `T0AgentsFatalError` subclasses — including `GuardrailBlockedError`
  raised inside a sub-agent's own guardrail stack — propagate
  unchanged through the orchestrator's tool-call boundary. The
  orchestrator's `invoke()` caller sees the real exception type, just
  as they would if the guardrail had been installed on the
  orchestrator directly.
- Every other exception is wrapped in `t0_agents.SubAgentError`
  (subclass of `T0AgentsError`). The orchestrator's loop stringifies
  it as a tool error (same convention as any other
  `fastmcp.exceptions.ToolError`) so the orchestrator LLM can decide
  to retry, pick a different sub-agent, or give up gracefully.

```python
try:
    result = await orchestrator.invoke("Do the thing")
except GuardrailBlockedError as exc:
    # A sub-agent's guardrail said no.
    ...
```

`SubAgentError` always carries the original exception as `__cause__`,
so `logging.exception(...)` produces a useful traceback.

See [`docs/error-handling.md`](error-handling.md) for the
library-wide exception taxonomy and recovery patterns.

## Examples

### Easy -- two agents, one tool call

A minimal orchestrator that delegates to a single sub-agent. The
sub-agent owns one FastMCP tool; the orchestrator's own FastMCP
server starts empty.

```python
import asyncio

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

tools_server = FastMCP("tools-server")

@tools_server.tool
def lookup_topic(query: str) -> str:
    """Return a stub answer for the query."""
    return f"Found 3 references for {query!r}."

orchestrator_server = FastMCP("orchestrator-server")


async def main() -> None:
    async with (
        await AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
        ) as t0,
        Client(tools_server) as tools_client,
        Client(orchestrator_server) as orch_client,
    ):
        researcher = Agent(
            AgentConfig(name="researcher", function_name="research"),
            t0=t0,
            tool_provider=tools_client,
        )
        orchestrator = Agent(
            AgentConfig(
                name="coordinator",
                function_name="orchestrator",
                sub_agents=[researcher],
            ),
            t0=t0,
            tool_provider=orch_client,
        )
        result = await orchestrator.invoke("Find references on RLHF")
        print(result.response)


asyncio.run(main())
```

The orchestrator sees exactly one tool on its server —
`researcher_ask(message: str) -> str` — and decides when to call it.

### Intermediate -- researcher + writer trio

[`examples/08_multi_agent/`](../examples/08_multi_agent/) ships a
runnable demo with a researcher (arXiv-flavoured MCP server), a
writer (markdown-formatting MCP server), and a report-generator
orchestrator. It requires `docker compose up` and a
`GROQ_API_KEY`-backed variant. After a run, ClickHouse shows three
episodes — one per agent — with sub-agent rows correlated to the
orchestrator via the `t0_agents.parent_episode_id` tag.

### Advanced -- custom tool names and nested orchestrators

Compose orchestrators recursively by using one orchestrator as
another orchestrator's sub-agent — each orchestrator keeps its own
in-process FastMCP server, so the registration paths don't collide:

```python
inner = Agent(
    AgentConfig(
        name="report_drafter",
        function_name="orchestrator",
        sub_agents=[researcher, writer],
    ),
    t0=t0,
    tool_provider=inner_client,
)
outer = Agent(
    AgentConfig(
        name="project_lead",
        function_name="orchestrator",
        sub_agents=[inner],
    ),
    t0=t0,
    tool_provider=outer_client,
)
```

The outer orchestrator sees one tool: `report_drafter_ask`. When it
delegates, the inner orchestrator in turn sees two tools:
`researcher_ask` and `writer_ask`. Each level is an independent
episode, correlated via `parent_episode_id` tags. A runnable
version of this pattern ships at
[`examples/08_multi_agent/advanced.py`](../examples/08_multi_agent/advanced.py).

Override the default `{name}_ask` by using `agent_as_tool` directly
and skipping the declarative `sub_agents=[…]` path:

```python
from fastmcp import FastMCP
from t0_agents import agent_as_tool

server = FastMCP("orchestrator-server")
tool = agent_as_tool(
    researcher,
    tool_name="lookup",
    description="Find and summarise references for a topic.",
)
server.add_tool(tool)
```

The manual registration path bypasses the declarative `sub_agents`
hook, so the orchestrator won't dispatch calls to it in its own
asyncio task — those calls cross the FastMCP server sub-task and
carry `parent_episode_id="unknown"`. Use `tool_name=` overrides only
when losing the episode correlation is acceptable, or pair each call
with an explicit [`bind_parent_episode_id`](#parent-episode-helpers)
context in your own tool handler.

## Caveats

- **Tool-schema stability per run.** Sub-agents are registered at
  `Agent.__init__` — they cannot be added mid-conversation. The
  gateway permits per-call tool lists, but the LLM-side cost of
  seeing the tool list change between turns isn't worth it. If you
  need conditional delegation, register every candidate up front and
  let the orchestrator decide via `tool_choice`.
- **No peer-to-peer mesh.** Sub-agents cannot delegate to other
  sub-agents directly. Deeper trees are achievable by nesting: an
  orchestrator can have sub-agents that are themselves orchestrators
  with their own sub-agents.
- **Elicitation inside sub-agents is possible but per-Client.** Every
  sub-agent owns its own `fastmcp.Client`, so elicitation flows
  reach the sub-agent's own handler — not the orchestrator's. This
  is usually the right thing (the researcher's arXiv API key
  approval shouldn't leak to the writer's renderer) but it means
  sub-agents in a browser-served deployment need their own
  `ElicitationForwarder` wiring. See
  [`streaming.md`](streaming.md).

## Production considerations

- **Single-process deployment.** Orchestrator and sub-agents live in
  the same Python process — the in-process FastMCP requirement means
  sub-agents cannot be scaled out as separate services without a
  substantial rewrite. If you need horizontal scaling, split the
  workload at the MCP layer (each sub-agent's internal MCP servers
  can be remote) and keep the orchestrator local.
- **Concurrency.** One orchestrator `Agent` plus its sub-agents is
  safe under concurrent `run()` calls only when no tool in the
  graph uses elicitation. Elicitation handlers are shared per
  `fastmcp.Client`, so concurrent callers need their own `Client`
  per session. The [`serve/`](serving.md) layer enforces one
  `Client` per session for exactly this reason.
- **Scaling by nesting.** When a delegation tree gets wide, prefer
  nested orchestrators over flat lists with many siblings: each
  orchestrator's LLM sees a smaller tool list per inference, which
  reduces prompt cost and schema-selection noise.

See [`docs/production.md`](production.md) for deployment guidance
that applies across the library (TensorZero routing, retry,
feedback, observability).

## Observability

Each orchestrator and each sub-agent writes one episode to
ClickHouse, with sub-agent inferences tagged
`t0_agents.parent_episode_id=<orchestrator-episode>` for
correlation. All of the standard TensorZero signals — episode id,
inference ids, variant, model, token counts, cost, feedback — are
available for every sub-agent independently.

See [`docs/observability.md`](observability.md) for the full signal
catalogue and [`docs/api-reference.md`](api-reference.md) for the
`AgentResult` shape returned by every `invoke()` call.

---

Composition gives you an orchestrator that can call sub-agents in-process. The next step is letting external callers — a browser, a mobile app, another service, or a different team's cron job — drive that orchestrator over the network. `t0_agents.serve` wraps any `Agent` (single or orchestrator, with or without memory and guardrails) in a FastAPI application with `/chat`, `/feedback`, `/health`, `/info`, and an opt-in streaming surface. See [Serving](serving.md).

## External references

- [FastMCP — Tools](https://gofastmcp.com/servers/tools) —
  `Tool.from_function` and `server.add_tool` are the runtime-
  registration primitives this submodule composes.
- [TensorZero — Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes)
  — the `parent_episode_id` tag only makes sense against TensorZero's
  episode model.
