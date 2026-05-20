# Agent (explicit surface)

Every collaborator visible: build the gateway, build the MCP
client, instantiate `Agent`, invoke.

```python
async with await gateway as t0, mcp_client:
    agent = Agent(
        AgentConfig(name="barista", function_name="agent"),
        t0=t0,
        tool_provider=mcp_client,
    )
    result = await agent.invoke("...")
```

16 numbered examples plus two advanced tiers; each adds one concept
on top of the previous. Pick a chapter from the table below.

## Reading path

| # | Folder | Feature (docs chapter) | Docker? |
|---|---|---|---|
| 00 | [`00_chat_only`](00_chat_only/)             | Toolless chat — `Agent(config, t0=t0)` without `tool_provider`. The shortest possible agent: one inference, one text response, zero MCP plumbing. | yes |
| 01 | [`01_first_agent`](01_first_agent/)         | The agent loop, top to bottom — one user message, two tool calls, one final answer. | yes |
| 02 | [`02_agent_class`](02_agent_class/)         | The declarative `Agent` class. Three tiers: bare config, baked-in system prompt, per-call system override. | yes |
| 03 | [`03_output_modes`](03_output_modes/)       | `OutputMode[T]` — extract typed JSON, run the recovery ladder offline, hot-swap a schema mid-run, decorate with Pydantic. | yes / no |
| 04 | [`04_elicitation`](04_elicitation/)         | Tools that pause to ask a human — approval gates, structured input, the elicitation handler. | yes |
| 05 | [`05_memory`](05_memory/)                   | Three tiers of `MemoryProvider`: an ephemeral dict, the SQLite backend, and a TZ-native history strategy across process restarts. | yes |
| 06 | [`06_guardrails`](06_guardrails/)           | `require_approval`, `max_cost`, `max_tool_calls`, `block_tools`, plus a custom `Guardrail` with an audit log. Advanced tier: a hand-written `Middleware`. | yes |
| 07 | [`07_streaming`](07_streaming/)             | The live event stream — `Run.events()` over SSE and WebSocket, with elicitation piped to a browser UI. Advanced tier: `Run.events()` / `.cancel()` / `.fork()` / `.replay()` without HTTP. | yes |
| 08 | [`08_multi_agent`](08_multi_agent/)         | `AgentConfig.sub_agents=[...]` — orchestrator + research/writer sub-agents, plus nested orchestrators. | yes |
| 09 | [`09_serving`](09_serving/)                 | `create_app(agent)` — drop the whole stack behind `/chat`, `/feedback`, `/health`. | yes |
| 10 | [`10_cli`](10_cli/)                         | The terminal TUI — `t0-agents run`, `serve`, `info` with a Rich renderer and elicitation-as-prompt. | yes |
| 11 | [`11_testing`](11_testing/)                 | `MockGateway` for deterministic unit tests, `EmbeddedGatewayHelper` for real-inference integration tests without Docker. | mixed |

## Appendices

| # | Folder | Feature (docs chapter) | Docker? |
|---|---|---|---|
| 12 | [`12_concurrency`](12_concurrency/)             | Sequential / parallel / orchestrator patterns in plain Python — wall-clock speedups quantified. | no / yes |
| 13 | [`13_observability`](13_observability/)         | Episode + inference correlation, `t0.feedback(...)`, ClickHouse-ready dashboards. | yes |
| 14 | [`14_multi_server`](14_multi_server/)           | Mount two FastMCP servers behind one agent with namespace prefixes. | yes |
| 15 | [`15_mcp_internals`](15_mcp_internals/)         | The `t0_agents.mcp` submodule on its own — tool conversion, no `Agent` class, no gateway. The plumbing the rest of the framework is built on. | no |

## Advanced tiers

Two scripts live inside their parent folders rather than as top-level
chapters because they extend the parent concept:

| File | Demonstrates |
|---|---|
| [`07_streaming/advanced_run_handle.py`](07_streaming/advanced_run_handle.py) | The `Run[T]` control plane — `.events()`, `.cancel()`, `.fork()`, `.replay()` — without FastAPI. |
| [`06_guardrails/advanced_middleware.py`](06_guardrails/advanced_middleware.py) | A user-authored `Middleware` (timing + cost trace) attached via `AgentConfig(middleware=[...])`. |

## How each example boots

Most folders carry their own `docker-compose.yml` and
`config/tensorzero.toml`; a few (05, 06, 10) reuse the gateway from
[`01_first_agent/`](01_first_agent/) because they need the same
chat function. Each folder's README has its exact command.

The shared environment template is at [`.env.example`](.env.example).
Copy to `.env` and add `GROQ_API_KEY` before the first compose-up.
