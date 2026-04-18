# t0-agents Documentation

t0-agents is a Python library that connects [TensorZero](https://www.tensorzero.com/docs/) (an LLM inference gateway with built-in observability and analytics) to [FastMCP](https://gofastmcp.com/) (an MCP server framework for exposing tools). Together, they enable LLM agents that can discover and execute tools through MCP while retaining TensorZero's structured inference, episode tracking, and ClickHouse-backed analytics.

The documentation is written as a progressive tour: start with [Getting Started](getting-started.md), follow the **Reading path** below in order — each chapter picks up the tension the previous one left open — and reach for the **Appendices** when you need a cross-cutting reference.

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `t0_agents.core` | Stable | Declarative agent primitives: `Agent`, `AgentConfig`, `AgentResult`, the collaborator Protocols (`ToolProvider`, `MemoryProvider`, `Guardrail`), the `T0AgentsError` exception hierarchy, and the `ElicitationForwarder` remote-channel bridge |
| `t0_agents.mcp` | Stable | MCP adapter (`convert_tools`, `serialize_tool_result`, `build_tool_result`), agent loop (`run_agent`, `execute_tool_calls`, `execute_tool_calls_parallel`, `has_tool_calls`), and reference elicitation handlers (`auto_approve_handler`, `auto_decline_handler`) |
| `t0_agents.memory` | Stable | Cross-run persistence keyed by TensorZero's episode id: the `MemoryProvider` Protocol, `MemoryRecord` dataclass, `render_memory_summary` renderer, plus the ephemeral `InMemoryMemory` and SQLite-backed `SqliteMemory` built-in backends |
| `t0_agents.guardrails` | Stable | Declarative policy layer: `require_approval`, `max_cost`, `max_tool_calls`, `block_tools`, plus the `Guardrail` Protocol for custom policies and the `GuardedToolProvider` wrapper installed by `Agent` when guardrails are configured |
| `t0_agents.stream` | Stable | Typed streaming surface: the `AgentEvent` union and its ten variants (`TextDeltaEvent`, `ToolCallEvent`, `ToolResultEvent`, `ThoughtDeltaEvent`, `TurnEndEvent`, `DoneEvent`, `ErrorEvent`, …), the `StreamingAgentLoop` that drives `t0.inference(stream=True)`, and the `TurnStreamAssembler` that folds TensorZero chunks into those events |
| `t0_agents.compose` | Stable | Multi-agent composition: register sub-agents as `{name}_ask` MCP tools on an orchestrator's in-process FastMCP server via `AgentConfig.sub_agents=[...]` or the standalone `agent_as_tool`, with `SubAgentError` for non-fatal delegation failures and `t0_agents.parent_episode_id` tags for ClickHouse correlation |
| `t0_agents.serve` | Stable | HTTP serving layer: `create_app` / `mount_agent` wrap an `Agent` in a FastAPI app with `/chat`, `/feedback`, `/health`, `/info` (plus `/chat/stream` SSE and `/chat/ws` WebSocket when `streaming=True`); `SessionFactory` + the `SessionProvider` Protocol (and the `CallableSessionProvider` adapter) drive per-session `Client` wiring for auth, multi-tenant routing, and live elicitation |
| `t0_agents.cli` | Beta | Terminal UI behind the `[cli]` extra: the `t0-agents` console script (`run`, `serve`, `info`) plus public primitives `TUIRenderer`, `rich_elicitation_handler`, `make_rich_elicitation_handler`, `get_active_renderer`, and `load_agent_factory` for driving an agent factory through a Rich `Live` panel |
| `t0_agents.testing` | Beta | Test-authoring surface in the base distribution: `MockGateway` (script / canned / recording modes), `MockToolProvider` with custom schemas, five `assert_*` helpers, a `scenarios` catalogue, `EmbeddedGatewayHelper` for real TensorZero inference without Docker, `MockStreamingGateway` + `fake_event_stream` for `Agent.stream()` consumers, and four pytest fixtures (`mock_t0`, `mock_tool_provider`, `embedded_t0`, `agent_test_factory`) |

## Reading path

1. **[Getting Started](getting-started.md)** — install the library and run your first agent in under five minutes, using the low-level `run_agent` API directly.
2. **[Agent](agent.md)** — meet the declarative `Agent` class, its `AgentConfig`, and the collaborator Protocols (`ToolProvider`, `MemoryProvider`, `Guardrail`) the later chapters wire in.
3. **[Concepts](concepts.md)** — pop the hood: what `Agent.run()` actually does, turn by turn, through the `run_agent` loop — tool discovery, inference, execution, error mapping, episode correlation.
4. **[Elicitation](elicitation.md)** — once tools can call back to a human, the loop has to pause; here is how that round-trip works end-to-end without any change to `run_agent`.
5. **[Memory](memory.md)** — the loop is stateless between runs; wire a `MemoryProvider` into `AgentConfig` to give it a past — ephemeral in-process, persistent SQLite, or a backend of your own.
6. **[Guardrails](guardrails.md)** — the agent now remembers and is equipped with tools; guardrails decide what it may do with them (approval gates, cost caps, tool-call limits, deny-lists, custom policies).
7. **[Streaming](streaming.md)** — until now you only saw the final response; switch to `Agent.stream()` to narrate the loop live with typed events, SSE, and WebSocket.
8. **[Multi-Agent Composition](multi-agent.md)** — one agent is not always enough; let an orchestrator delegate to others via the agent-as-tool pattern, with episode correlation across the tree.
9. **[Serving](serving.md)** — drop the whole stack (single agent or orchestrator, with or without memory and guardrails) behind `/chat`, `/chat/stream`, and `/chat/ws` with `create_app`.
10. **[CLI](cli.md)** — the same stack, in a terminal, via the `t0-agents` console script: an interactive `run` REPL, a `serve` dev HTTP server, and `info` introspection.
11. **[Testing](testing.md)** — keep every layer above from regressing with `MockGateway` and `MockToolProvider` for deterministic unit tests, and `EmbeddedGatewayHelper` for real-inference integration tests without Docker.

## Appendices

Cross-cutting references that span every chapter above. Reach for them when you need detail on a specific axis instead of the narrative.

- **[API Reference](api-reference.md)** — library-wide symbol index: public signatures and the shortest import path for every exported class, function, and Protocol.
- **[Error Handling](error-handling.md)** — tool-error semantics, recoverable inference retries, `ToolError` vs `McpError`, and the full exception taxonomy.
- **[Observability](observability.md)** — episode and inference correlation, the `t0.feedback` API, structured logging conventions, and ClickHouse analytics hooks.
- **[Production](production.md)** — transport selection, deployment topologies, system-prompt tuning, timeout budgets, and the elicitation-session invariants.
- **[Multi-Server Composition](multi-server.md)** — mounting multiple FastMCP servers behind one agent with namespace prefixes, plus the TensorZero configuration to match.

## Public API

```python
from t0_agents import Agent, AgentConfig, run_agent, AgentResult, MaxTurnsExceededError, ToolChoice
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker + Docker Compose (for the TensorZero gateway, required by the agent loop)
