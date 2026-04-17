# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] — Unreleased

### Added

- **CLI** (`from t0_agents.cli import TUIRenderer, rich_elicitation_handler,
  make_rich_elicitation_handler, load_agent_factory`) — Typer + Rich
  terminal UI. Milestone M6: TUI Agent. Ships behind the new `[cli]`
  extra (`pip install 't0-agents[cli]'`), which pulls `typer>=0.12.0`,
  `rich>=13.7.0`, plus the streaming + serving deps (`fastapi`,
  `uvicorn[standard]`, `pydantic`, `sse-starlette`).
- **`t0-agents` entry-point script** wired to `t0_agents.cli.app:main`.
  Exposes three subcommands:
  - `t0-agents run <module:attr>` — interactive streaming REPL with
    live deltas, tool-call spinners, and Rich-rendered elicitation
    prompts. `--verbose` surfaces reasoning content.
  - `t0-agents serve <module:attr>` — wraps
    `t0_agents.serve.create_app(..., share_client=True)` in uvicorn.
    Phase D non-streaming routes by default (`POST /chat`, `/feedback`,
    `/health`, `/info`); opt in to SSE + WS with `--streaming`. WS
    elicitation is incompatible with shared-Client mode (FastMCP Q3),
    so `--streaming` is only useful for tools that never call
    `ctx.elicit` — for full browser elicitation, call `create_app`
    directly with a `client_factory` + `elicitation_client_factory`
    (see `examples/12_streaming_http/agent_app.py`). `--host`,
    `--port`, `--deployment` flags.
  - `t0-agents info <module:attr>` — prints a Rich `Table` of the
    loaded `AgentConfig`.
- **Agent factory convention** — every subcommand's ``<module:attr>``
  argument must resolve to a zero-argument callable returning an async
  context manager yielding an :class:`Agent` (idiomatically
  `@asynccontextmanager async def assistant(): ...`). The loader raises
  `typer.BadParameter` for bare `Agent` instances with an inline snippet
  pointing users at the factory pattern, and for malformed specs /
  missing modules / missing attributes.
- **`TUIRenderer`** — Rich `Live` + `Panel` renderer that folds
  `AsyncIterator[AgentEvent]` into a live terminal frame. Exposes
  `pause()` / `resume()` so an elicitation handler can stop Live while
  it runs a prompt, and `last_episode_id` / `last_result` so the REPL
  can thread continuity across turns.
- **`rich_elicitation_handler`** + **`make_rich_elicitation_handler`** —
  FastMCP-compatible elicitation handler rendered through Rich prompts.
  Dispatches on the FastMCP Q5 schema subset: primitives + enum + const,
  description and default preserved, scalar-wrap (`{"value": ...}`)
  supported. Coerces enum answers back to the original Python type. Fails
  soft to `Prompt.ask(str)` for unknown types (non-FastMCP servers).
  Returns `ElicitResult(action="cancel", content=None)` on
  `KeyboardInterrupt`.
- **`load_agent_factory`** (`t0_agents.cli.loader`) — public resolver
  that distinguishes "malformed spec", "missing module", "missing
  attribute", "resolved to a bare Agent instance", and "resolved to a
  non-callable" so the error path is actionable.
- Example `examples/13_tui/` — runnable demo with a FastMCP server
  (`add`, `set_reminder` with `ctx.elicit(response_type=Reminder)`),
  a `@asynccontextmanager` factory, and a README walkthrough of the
  three subcommands.
- Public documentation at `docs/cli.md` covering install, the three
  commands, the factory convention, custom renderers, elicitation
  semantics, and troubleshooting.

### Changed

- Phase F does not modify `core/`, `mcp/`, `stream/`, `serve/`,
  `memory/`, or `guardrails/`. `run_agent`, `AgentResult`, `Agent`,
  and every existing public symbol are byte-identical to 0.6.0
  (contract §2.1, §6.1–§6.3). Pre-existing offline tests remain
  green (363 passed across Phase A–F).

## [0.6.0] — Unreleased

### Added

- **Streaming primitives** (`from t0_agents import AgentEvent, DoneEvent,
  TextDeltaEvent, TextEndEvent, ToolCallDeltaEvent, ToolCallEvent,
  ToolResultEvent, ThoughtDeltaEvent, ThoughtEvent, TurnEndEvent,
  ErrorEvent, StreamingAgentLoop, TurnStreamAssembler`) — typed event
  union emitted by `Agent.stream()`.  Milestone M5: Live Agent.  Ships
  from the base distribution (no extra needed).
- **`Agent.stream(message, *, episode_id=None) -> AsyncIterator[AgentEvent]`**
  — streaming analogue of `Agent.run`.  Same memory / guardrail
  scaffolding; the terminal event is always `DoneEvent(result=...)`
  carrying a full `AgentResult`.  Non-streaming `Agent.run` is
  byte-identical to Phase D.
- **`TurnStreamAssembler`** — folds `AsyncIterator[ChatChunk]` from
  `AsyncTensorZeroGateway.inference(stream=True)` into an
  `AsyncIterator[AgentEvent]` with stable correlation ids per T0 Q1.
- **`StreamingAgentLoop`** — streaming analogue of `mcp.loop.run_agent`;
  reuses `convert_tools`, `execute_tool_calls[_parallel]` and the
  recoverable-retry contract from `mcp/`.  `run_agent` and the `mcp/`
  submodule are untouched (contract §2.1, §6.1).
- **`ElicitationForwarder` full completion** (`from t0_agents import
  ElicitationForwarder`) — Phase D skeleton grows `on_request` +
  `default_timeout` constructor kwargs, a Future-based `handler`, and
  public `resolve(elicitation_id, payload)` / `cancel_all(action=...)`
  methods.  Empty-constructor behaviour (always decline) is preserved
  for Phase D compatibility.
- **`SessionProvider` Protocol + `CallableSessionProvider` adapter**
  (`from t0_agents.serve import SessionProvider,
  CallableSessionProvider`) — advanced extension point for per-session
  `Client` construction.  Implement the Protocol directly for auth,
  multi-tenant routing, custom elicitation wiring; the default
  `client_factory` / `elicitation_client_factory` kwargs wrap in a
  `CallableSessionProvider` automatically.
- **`create_app(..., streaming=True, session_provider=...,
  elicitation_client_factory=...)`** — three additive kwargs.
  `streaming=True` mounts `GET /chat/stream` (SSE) and `WS /chat/ws`
  (bidirectional).  `session_provider` is the advanced WS wiring slot.
  `elicitation_client_factory: Callable[[ElicitationHandler],
  Client[Any]]` is the simple WS wiring slot — pair it with the existing
  `client_factory`.  Phase D callers that set only `client_factory` or
  `share_client=True` keep working unchanged.
- **WS wire protocol** (`from t0_agents.serve.ws_protocol import
  WSUserMessage, WSElicitationResponse, WSErrorFrame,
  WSElicitationRequest`) — Pydantic v2 envelopes for every frame in each
  direction.  Documented in `docs/streaming.md`.
- **`sse-starlette>=2.0`** added to the `[serve]` extra.
- Example `examples/12_streaming_http/` — runnable demo with a FastMCP
  server, `agent_app.py`, browser WS client (`static/index.html` +
  `static/app.js`) covering every `AgentEvent.type` and the elicitation
  round-trip.
- Public documentation at `docs/streaming.md` — simple +
  `SessionProvider`-advanced cookbooks, event-type reference, browser
  recipe, elicitation diagram, backpressure contract.

### Changed

- `SessionFactory` now accepts `session_provider=...` alongside
  `client_factory=...` and `shared_tool_provider=...` (three-way XOR).
  Phase D XOR validation error wording is preserved when exactly
  `client_factory` + `share_client` are both set (test compat).  The
  existing `session(extra_tags=...)` context manager gains an optional
  `elicitation_handler=...` kwarg that threads through to the
  provider's `build_client`.
- **`Agent.run(message, *, episode_id=None, tags=None)`** and
  **`Agent.stream(message, *, episode_id=None, tags=None)`** accept a
  per-call `tags` override that shallow-merges onto `AgentConfig.tags`
  (override wins on key collisions, configured deployment label is
  preserved).  Critical for the WebSocket path where one `Agent` is
  reused across frames — `AgentConfig.tags` alone cannot vary per
  frame.  Default `tags=None` leaves behaviour byte-identical for
  callers that never set it.
- **`WSUserMessage.tags`** is now honoured by the WebSocket route — the
  field existed in the wire protocol and was documented in
  `docs/streaming.md`, but `_drive_stream` dropped it before the
  gateway could observe it.  Now propagates through `agent.stream(tags=...)`.
- **`GET /chat/stream`** (SSE) accepts an optional `tags=<json>` query
  parameter (JSON-encoded `dict[str, str]`) for parity with `POST /chat`.
  Invalid JSON or non-object values surface as the first SSE frame — an
  `ErrorEvent` with `detail.kind == "invalid_tags"` — and the stream
  terminates.
- **`Guardrail.after_run`** Protocol signature tightened from
  `result: Any` to `result: AgentResult | None` and the docstring now
  documents explicitly that `result=None` is a valid value when the
  run aborted before producing an `AgentResult`.  Runtime behaviour is
  unchanged — `Agent.run` / `Agent.stream` have always invoked
  `after_run` in their `finally` blocks regardless of the result, per
  the `docs/guardrails.md` contract.

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
