# t0-agents Documentation

t0-agents is a Python library that connects [TensorZero](https://www.tensorzero.com/docs/) (an LLM inference gateway with built-in observability and analytics) to [FastMCP](https://gofastmcp.com/) (an MCP server framework for exposing tools).  Together they enable LLM agents that can discover and execute tools through MCP while retaining TensorZero's structured inference, episode tracking, and ClickHouse-backed analytics.

The library ships **two surfaces, one engine**: [`t0_agents.glow`](glow/index.md) is the four-line decorator path (start here); the canonical `Agent` / `Run[T]` / `Middleware` / `OutputMode[T]` surface is the explicit construction path (the page set that follows).  Every `@app.agent` wrapper exposes `_agent` — the same v1.0 `Agent` — so the two surfaces are not parallel paths, they are layers.

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `t0_agents.glow` | Experimental | FastAPI-style decorator surface (`[glow]` extra): `App`, `@app.agent`, `@app.tool`, `Depends`, `FluentRun`, class-based `MCPServer`, seven typed event selectors (`Thought`, `ToolCall`, `Text`, `ToolResult`, `TurnEnd`, `Done`, `Error`), `regex_guardrail` re-export.  Four-line "hello agent" path that compiles down to a real v1.0 `Agent`.  See the [`docs/glow/`](glow/index.md) ladder. |
| `t0_agents.core` | Stable | Declarative agent primitives: `Agent`, `AgentConfig`, `AgentResult`, the collaborator Protocols (`ToolProvider`, `MemoryProvider`, `Guardrail`), the `T0AgentsError` exception hierarchy, the `ElicitationForwarder` remote-channel bridge, the v1.0 three-abstraction surface (`Run[T]`, `Middleware`, `OutputMode[T]`), and the JSON-function dispatch covered in [Output modes](output-modes.md) and the legacy [Structured outputs](structured-outputs.md) reference. |
| `t0_agents.mcp` | Stable | MCP adapter (`convert_tools`, `serialize_tool_result`, `build_tool_result`), agent loop (`run_agent`, `execute_tool_calls`, `execute_tool_calls_parallel`, `has_tool_calls`), and reference elicitation handlers (`auto_approve_handler`, `auto_decline_handler`) |
| `t0_agents.memory` | Stable | Cross-run persistence keyed by TensorZero's episode id: the `MemoryProvider` Protocol, `MemoryRecord` dataclass, `render_memory_summary` renderer, the ephemeral `InMemoryMemory`, the SQLite-backed `SqliteMemory`, and `TensorZeroMemory` for episode-scoped reads from TZ's observability log (with the `memory_strategy="messages"` native-history path that threads conversation history through `input.messages` without stringification) |
| `t0_agents.guardrails` | Stable | Declarative policy layer: `require_approval`, `max_cost`, `max_tool_calls`, `block_tools`, plus the `Guardrail` Protocol for custom policies and the `GuardedToolProvider` wrapper installed by `Agent` when guardrails are configured |
| `t0_agents.stream` | Stable | Typed streaming surface: the `AgentEvent` union and its ten variants (`TextDeltaEvent`, `ToolCallEvent`, `ToolResultEvent`, `ThoughtDeltaEvent`, `TurnEndEvent`, `DoneEvent`, `ErrorEvent`, …), the `StreamingAgentLoop` that drives `t0.inference(stream=True)`, and the `TurnStreamAssembler` that folds TensorZero chunks into those events |
| `t0_agents.delegation` | Stable | Multi-agent delegation: register sub-agents as `{name}_ask` MCP tools on an orchestrator's in-process FastMCP server via `AgentConfig.sub_agents=[...]` or the standalone `agent_as_tool`, with `SubAgentError` for non-fatal delegation failures and `t0_agents.parent_episode_id` tags for ClickHouse correlation |
| `t0_agents.serve` | Stable | HTTP serving layer: `create_app` / `mount_agent` wrap an `Agent` in a FastAPI app with `/chat`, `/feedback`, `/health`, `/info` (plus `/chat/stream` SSE and `/chat/ws` WebSocket when `streaming=True`); `SessionFactory` + the `SessionProvider` Protocol (and the `CallableSessionProvider` adapter) drive per-session `Client` wiring for auth, multi-tenant routing, and live elicitation |
| `t0_agents.cli` | Beta | Terminal UI behind the `[cli]` extra: the `t0-agents` console script (`run`, `serve`, `info`) plus public primitives `TUIRenderer`, `rich_elicitation_handler`, `make_rich_elicitation_handler`, `get_active_renderer`, and `load_agent_factory` for driving an agent factory through a Rich `Live` panel |
| `t0_agents.testing` | Beta | Test-authoring surface in the base distribution: `MockGateway` (chat: `script` / `canned` / `recording`; JSON: `script_json` / `canned_json` for agents whose `AgentConfig.output` is a `JsonOutput` / `StructuredOutput`), `MockToolProvider` with custom schemas, ten `assert_*` helpers (five chat + five JSON: `assert_parsed_equals`, `assert_parsed_matches_schema`, `assert_invalid_json`, `assert_parse_strategy`, `assert_retry_attempts`), a `scenarios` catalogue, `EmbeddedGatewayHelper` for real TensorZero inference without Docker, `MockStreamingGateway` + `fake_event_stream` for `Run.events()` consumers, and four pytest fixtures (`mock_t0`, `mock_tool_provider`, `embedded_t0`, `agent_test_factory`) |

## Reading path

### glow — the four-line path

Start here if you want the shortest distance from query to answer.  Each
chapter is short enough to read in one pass; the runnable companions
live under [`examples/glow/`](../examples/glow/).

1. **[Getting Started](glow/getting-started.md)** — toolless chat, structured outputs by return type, per-call overrides (`.json()` / `.text()` / `.as_()`), and the three multi-agent patterns.
2. **[Streaming and events](glow/streaming-and-events.md)** — `FluentRun.events(*selectors)`, seven typed selectors, `.where(pred)` / `.map(fn)`, the token-by-token `Selector(TextDeltaEvent)` recipe.
3. **[Depends and tools](glow/depends-and-tools.md)** — value-style injection inside `@app.tool`, sub-agent injection inside `@app.agent`, and `app.dependency_overrides` for tests.
4. **[Typed MCP and guardrails](glow/typed-mcp-and-guardrails.md)** — class-based `MCPServer` declarations, the `Annotated[...]` override, and the `Annotated[T, guardrail(...)]` metadata pattern.
5. **[Escape hatch](glow/escape-hatch.md)** — `wrapper._agent` IS the v1.0 `Agent`; lifespan / build / shutdown; drop-down recipes for middleware, custom `OutputMode`, hand-rolled event handling, and `cancel()` / `fork()` / `replay()`.
6. **[Known limitations](glow/limitations.md)** — what's deferred to v1.1 (streaming JSON, per-agent MCP subsets, AsyncIterator tools, sync API).

### Agent — the canonical, fully-typed surface

The explicit construction path.  Read it when you outgrow the decorator or
want the typed view of every collaborator.

1. **[Getting Started](getting-started.md)** — install the library and run your first agent in under five minutes via the canonical `Agent.invoke` surface.
2. **[Agent](agent.md)** — meet the declarative `Agent` class, its `AgentConfig`, and the collaborator Protocols (`ToolProvider`, `MemoryProvider`, `Guardrail`) the later chapters wire in.
3. **[Run](run.md)** — `agent.invoke(message)` returns a `Run[T]`: the first-class invocation handle.  Await it for the final `AgentResult`, iterate `events()` for live `AgentEvent`s, cancel, replay, fork, attach feedback.
4. **[Middleware](middleware.md)** — what runs around every inference: memory hydration, guardrail gating, cost tracking, sub-agent orchestration, JSON retry.  Add your own via the Starlette-style chain.
5. **[Output Modes](output-modes.md)** — the polymorphic surface for what an agent returns: chat strings, JSON dicts, Pydantic models, tool calls.  Custom modes for YAML / Instructor-style retry / SQLAlchemy plug in via the same protocol.
6. **[Concepts](concepts.md)** — pop the hood: what `agent.invoke()` actually does, turn by turn, through the `run_agent` loop — tool discovery, inference, execution, error mapping, episode correlation.
7. **[Elicitation](elicitation.md)** — once tools can call back to a human, the loop has to pause; here is how that round-trip works end-to-end without any change to `run_agent`.
8. **[Memory](memory.md)** — the loop is stateless between runs; wire a `MemoryProvider` into `AgentConfig` to give it a past — ephemeral in-process, persistent SQLite, or a backend of your own.
9. **[Guardrails](guardrails.md)** — the agent now remembers and is equipped with tools; guardrails decide what it may do with them (approval gates, cost caps, tool-call limits, deny-lists, custom policies).
10. **[Streaming](streaming.md)** — until now you only saw the final response; iterate `agent.invoke(message).events()` to narrate the loop live with typed events, SSE, and WebSocket.
11. **[Multi-Agent Composition](multi-agent.md)** — the declarative `AgentConfig.sub_agents=[...]` path (LLM-driven delegation), plus runnable canonical patterns (Sequential, Parallel, Orchestrator) under [`examples/agent/12_concurrency/`](../examples/agent/12_concurrency/) and [`examples/glow/`](../examples/glow/).
12. **[Serving](serving.md)** — drop the whole stack (single agent or orchestrator, with or without memory and guardrails) behind `/chat`, `/chat/stream`, and `/chat/ws` with `create_app`.
13. **[CLI](cli.md)** — the same stack, in a terminal, via the `t0-agents` console script: an interactive `run` REPL, a `serve` dev HTTP server, and `info` introspection.
14. **[Testing](testing.md)** — keep every layer above from regressing with `MockGateway` and `MockToolProvider` for deterministic unit tests, and `EmbeddedGatewayHelper` for real-inference integration tests without Docker.

## Appendices

Cross-cutting references that span every chapter above. Reach for them when you need detail on a specific axis instead of the narrative.

- **[API Reference](api-reference.md)** — library-wide symbol index: public signatures and the shortest import path for every exported class, function, and Protocol.
- **[Migration v0.12 → v1.0](migration-v0.12-to-v1.0.md)** — the two surfaces removed in v1.0 (`Agent.run` / `Agent.stream` → `Agent.invoke` / `Run.events`, and five JSON config kwargs → `output=JsonOutput(...)`), each a one-line migration per call site.
- **[Error Handling](error-handling.md)** — tool-error semantics, recoverable inference retries, `ToolError` vs `McpError`, and the full exception taxonomy.
- **[Observability](observability.md)** — episode and inference correlation, the `t0.feedback` API, structured logging conventions, and ClickHouse analytics hooks.
- **[Token usage](usage.md)** — the live per-turn footprint vs. cumulative billing, the caller-supplied context window, `ConversationUsage`, the serve `usage` block, and the TensorZero recipes for rate-limiting, cost, and ClickHouse history.
- **[Production](production.md)** — transport selection, deployment topologies, system-prompt tuning, timeout budgets, and the elicitation-session invariants.
- **[Multi-Server Composition](multi-server.md)** — mounting multiple FastMCP servers behind one agent with namespace prefixes, plus the TensorZero configuration to match.
- **[Concurrency](concurrency.md)** — the three places parallelism lives in the stack (intra-turn `parallel_tools`, batch tools, multi-agent dispatch), with the decision table and the patterns the library deliberately does not provide.
- **[Structured outputs](structured-outputs.md)** — TensorZero `json` functions, dynamic `output_schema` overrides, per-call `json_mode` swaps, the three-layer recovery ladder (strict → permissive → retry), the `require_parsed()` guardrail, and the JSON testing helpers (`MockGateway.script_json`, `assert_parsed_*`).

## Guides

Production patterns that compose t0-agents with adjacent tooling.  Each
guide is self-contained for the system it covers; they cross-link to
each other when a pattern spans both sides.

- **[Elicitation in your API](guides/elicitation-in-api.md)** — wiring `t0-agents` into a FastAPI + SSE backend that forwards elicitation prompts to a browser and resolves them back through the same channel.  Covers `ElicitationForwarder`, per-session lifecycle, sticky-session and timeout constraints, scripted tests, and the production checklist.
- **[Elicitation in your FastMCP server](guides/elicitation-in-mcp-server.md)** — the server-side companion: the four `ctx.elicit()` shapes, dynamic-survey topologies, schema constraints, decline / cancel handling, graceful degradation when no handler is wired, and isolation tests with no `t0-agents` import.

## Public API

```python
# The canonical surface:
from t0_agents import Agent, AgentConfig, run_agent, AgentResult, MaxTurnsExceededError, ToolChoice

# The glow decorator surface ([glow] extra):
from t0_agents.glow import App, Depends, FluentRun, MCPServer, Selector
from t0_agents.glow import Thought, ToolCall, Text, ToolResult, TurnEnd, Done, Error
from t0_agents.glow import regex_guardrail
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker + Docker Compose (for the TensorZero gateway, required by the agent loop)
