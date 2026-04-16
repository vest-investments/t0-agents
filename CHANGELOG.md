# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — Unreleased

### Added

- **HTTP serving layer** (`from t0_agents.serve import create_app,
  mount_agent, SessionFactory`) — declarative FastAPI wrap of an
  `Agent`. Milestone M4: Hosted Agent. Requires the new `[serve]`
  extra (`pip install 't0-agents[serve]'`), which pulls
  `fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0`, and `pydantic>=2.0`.
- **`create_app(agent, *, client_factory=..., share_client=False,
  deployment_label=None)`** — builds a FastAPI app with `POST /chat`,
  `POST /feedback`, `GET /health`, `GET /info`. Validates that
  exactly one of `client_factory` / `share_client=True` is provided
  (no silent ambiguity). `deployment_label` is merged into every
  inference's tags as `t0_agents.deployment=<label>`.
- **`mount_agent(parent, agent, prefix, **kwargs)`** — mounts a
  per-agent sub-app under `prefix` on an existing FastAPI
  application. Same kwargs as `create_app`. Enables hosting several
  agents in one process on disjoint URL prefixes (`/mobile`,
  `/analytics`).
- **`SessionFactory`** — advanced: per-session `(Client, Agent)`
  builder that `create_app` uses internally. Exposes
  `session(extra_tags=...)` async context manager whose lifetime is
  the caller's `async with` block (one HTTP request in Phase D; Phase
  E's SSE / WS streams will reuse the same primitive unchanged).
  Enforces the FastMCP Q3 / contract §3.5 per-session-Client mandate
  by default.
- **`ElicitationForwarder`, `ElicitationRequest`**
  (`from t0_agents import ElicitationForwarder, ElicitationRequest`)
   — future-based elicitation bridge. Phase D ships the skeleton with
  a default `handler` that declines everything (HTTP has no return
  channel). Phase E will complete `forward()` / `resolve()` /
  `wait_for()` for SSE + WS piping.
- **`Agent.warmup()`** — async readiness hook. Default no-op;
  override in a subclass to front-load readiness probes. Called by
  `serve.build_lifespan` during FastAPI startup.
- Example `examples/11_http_agent/` — runnable single-agent demo with
  `server.py`, `agent_app.py`, `run.sh`, `client.sh`, and a README
  that covers the multi-agent `mount_agent` pattern and the
  `share_client=True` trade-off.
- Public documentation at `docs/serving.md` — quickstart, per-surface
  deployment recipe, concurrency model, `share_client` trade-off,
  GKE / cloud-LB deployment notes (sticky sessions planned for
  Phase E).

### Changed

- Phase D does not modify `AgentConfig`; the serving layer wraps an
  existing `Agent` instance. `run_agent` and the `mcp/` submodule are
  untouched (contract §2.1, §6.1).

## [0.4.0] — Unreleased

### Added

- **Guardrail subsystem** (`from t0_agents.guardrails import require_approval,
  max_cost, max_tool_calls, block_tools`) — declarative policy layer on top
  of `Agent`. Milestone M3: Trusted Agent.
- **`require_approval(tools=..., handler=...)`** — interactive approval gate
  riding FastMCP's `elicitation_handler` signature. Developers can reuse
  the exact handler they already pass to `Client(elicitation_handler=...)`.
- **`max_cost(per_episode=...)`** — cumulative cost ceiling per run. Hard
  aborts via `GuardrailBlockedError` when `response.usage.cost` crosses
  the cap.
- **`max_tool_calls(per_run=...)`** — per-run tool-call cap; counter
  incremented pre-dispatch (race-safe under `parallel_tools=True`).
- **`block_tools(names)`** — soft deny-list for tool names. Enforcement
  is at call time; does NOT filter `list_tools()`.
- **`Guardrail` Protocol** (`runtime_checkable`) — implement four optional
  hooks (`before_run`, `before_tool`, `after_tool`, `after_run`) to bring
  your own guardrail. Returning `str` from `before_tool` soft-blocks;
  raising `GuardrailBlockedError` hard-aborts.
- **`GuardrailContext`** — mutable per-run state (counter, cost, extra).
  Constructed inside `Agent.run()` so each run gets a fresh context.
- **`GuardrailLogEntry`** — one hook activation appended to
  `AgentResult.guardrail_log`. Attached to the result even on fatal abort
  via the `try/finally` around `run_agent`.
- **`GuardedToolProvider`** — the wrapper `Agent.run()` installs around
  the caller's `ToolProvider` when `config.guardrails` is non-empty.
  Exposed for advanced composition; normal users don't instantiate it.
- **`GuardrailBlockedError`** — hard-abort marker. Inherits from new
  `T0AgentsFatalError` base, which `_execute_single_tool_call` re-raises
  past its broad `except Exception` so fatal blocks propagate cleanly.
- **`T0AgentsFatalError`** — new base class for exceptions the agent
  loop must propagate (versus stringify into a tool error).
- **`on_inference` keyword argument on `run_agent`** —
  `Callable[[ChatInferenceResponse], Awaitable[None]] | None = None`.
  Fires after every `t0.inference()` response, BEFORE tool execution.
  `None` (the default) preserves behaviour byte-for-byte. Used internally
  by `Agent.run()` to track cumulative cost for `max_cost`.
- **`AgentResult.guardrail_log`** — new field (default empty list)
  carrying per-hook activations for audit and manual feedback emission.
- Example `examples/10_guardrails/` demonstrating all four built-ins
  composed on one agent with a CLI approval handler.
- Public documentation at `docs/guardrails.md`.

### Changed

- `AgentConfig` gained a `guardrails: list[Guardrail]` field (default
  empty). When empty, `Agent.run()` forwards the raw `tool_provider` and
  passes `on_inference=None` — byte-identical to Phase B.

## [0.3.0] — Unreleased

### Added

- **Memory subsystem** (`from t0_agents.memory import InMemoryMemory, SqliteMemory,
  MemoryProvider`) — repository-pattern memory for `Agent`. Configure
  `AgentConfig(memory=SqliteMemory("./db"))` to persist state across
  `.run()` calls. Milestone M2: Stateful Agent.
- **`InMemoryMemory`** — ephemeral, dict-backed backend for tests and
  single-process apps.
- **`SqliteMemory`** — persistent backend behind the `[memory-sqlite]`
  extra (`pip install 't0-agents[memory-sqlite]'`).
- **`MemoryProvider` Protocol** — implement `load`, `save`, `search` to
  bring your own backend (Redis, Postgres, etc.).
- **`memory_strategy`** field on `AgentConfig` — controls how memory is
  injected into the prompt (`"auto"`, `"prepended_message"`,
  `"system_template"`).
- `system_arguments` keyword argument on `run_agent` — forwards template
  arguments to `input.system.arguments` in TensorZero inference calls.
- Example `examples/09_memory/` demonstrating cross-session memory.
- Public documentation at `docs/memory.md`.

## [0.2.0] — Unreleased

### Added

- **Agent class** (`from t0_agents import Agent, AgentConfig`) — declarative
  wrapper around `run_agent`. Configure once with `AgentConfig`, then call
  `agent.run(message)` with no boilerplate. Milestone M1: Declarative Agent.
- **ToolProvider Protocol** — structural type matching `fastmcp.Client`'s
  `list_tools()` + `call_tool()` interface for testing and extensibility.
- **T0AgentsError** base exception — all library exceptions now inherit from
  `T0AgentsError` instead of bare `Exception`.
- **ConfigurationError** — raised when `AgentConfig` or `Agent` construction
  parameters are invalid (empty name, negative max_turns, etc.).
- `tags` and `namespace` keyword arguments on `run_agent` — forwarded to every
  `t0.inference()` call for observability and experimentation routing.
- Example `examples/08_agent_class/` demonstrating `Agent` usage.
- Public documentation at `docs/agent.md`.

### Changed

- `MaxTurnsExceededError` now inherits from `T0AgentsError` instead of
  `Exception`. Existing `except MaxTurnsExceededError` handlers are unaffected.

## [0.1.0] — 2026-04-01

Initial release. MCP-to-TensorZero bridge with `run_agent`, tool conversion,
sequential/parallel tool execution, elicitation support, and observability.
