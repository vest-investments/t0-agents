# Phase E Implementation Log

Phase: E (Streaming — SSE + WebSocket + Agent.stream)
Milestone: M5 — Live Agent
Date: 2026-04-16

## Divergences from Spec

### E1: Assembler consolidation emitted at stream-end, not per-chunk diff

**Spec (§4.2)** sketches a per-chunk diff strategy for consolidation
events: track `previous_seen_ids`, emit `TextEndEvent` / `ToolCallEvent`
/ `ThoughtEvent` the moment a block's id no longer appears in the
current chunk.

**Actual:**  `TurnStreamAssembler` emits delta events immediately as
chunks arrive, keeps per-block accumulators in insertion-order dicts,
and **flushes consolidation events once** after the input iterator is
exhausted — in the order blocks first appeared.

**Reason.**  The diff strategy assumes chunks interleave deltas for
multiple blocks such that a block's absence in a chunk signals
completion.  The installed TensorZero SDK (v2026.4.0) in practice emits
one block delta per chunk, and the terminal chunk is the chunk with
`usage is not None` (`.venv/.../tensorzero/types.py:417-428`).  Diffing
`seen_ids_in_chunk` across adjacent chunks would fire consolidation
events after *every* chunk for blocks that happen not to appear in that
specific chunk — adding noise without improving UX.

The stream-end flush preserves:

* The public `AgentEvent` union shape declared in the spec (§4.1).
* The contract's correlation-key guarantee (contract §3.3) — every
  consolidation event carries the original `id`.
* The `StreamingAgentLoop` dispatch contract (tool_calls are collected
  before the next turn's inference is issued).

Consumers that want per-block completion as soon as the block finishes
are not blocked by this choice — all consolidation events fire before
`TurnEndEvent`.

**Impact.**  None on the contract (§3.3 wording is about the
correlation key, not the emission cadence).  Documented in
`src/t0_agents/stream/assembler.py` docstring; test suite covers the
happy path and the invalid-JSON + non-object + JsonChunk + UnknownChunk
corner cases.

### E2: `SessionProvider` Protocol introduced

**Spec (§4.6)** sketches WS elicitation via
`agent_factory.session(elicitation_forwarder=...)` but leaves the
`Client`-construction step implicit — Phase D's `ClientFactory` is
`Callable[[], Client[Any]]` and cannot accept an elicitation handler.

**Actual:**  Phase E introduces a **public Protocol**
`t0_agents.serve.SessionProvider` with one method
`build_client(*, elicitation_handler=None) -> AbstractAsyncContextManager[Client[Any]]`.
The existing `client_factory` + new `elicitation_client_factory` kwargs
on `create_app` are sugar — the framework wraps them in a
`CallableSessionProvider` adapter.  `SessionFactory` now accepts three
mutually-exclusive modes (`session_provider` / `client_factory` /
`shared_tool_provider`); Phase D callers that use only `client_factory`
or `share_client=True` keep working unchanged.

**Reason.**  The spec's shape (pass a forwarder object into
`session()`) would make the WS route responsible for both constructing
the forwarder and construing how the Client gets wired with it — that
responsibility splits awkwardly across the route and the session
factory.  A single Protocol gives three audiences one well-documented
extension point:

* **Simple apps** — `create_app(client_factory=...,
  elicitation_client_factory=...)`.
* **Advanced apps** — implement `SessionProvider` directly for
  auth / multi-tenant / bespoke elicitation wiring.
* **Tests** — implement `SessionProvider` with a fake `Client`; no
  `unittest.mock` required.

Protocol choice (not ABC) aligns with existing
`ToolProvider` / `MemoryProvider` / `Guardrail` idioms in
`core/types.py`.  `@runtime_checkable` makes `isinstance` guards at
`SessionFactory` construction time fail fast.

**Impact.**  Widens the public surface of `t0_agents.serve` by two
symbols (`SessionProvider`, `CallableSessionProvider`) and one alias
(`ElicitationClientFactory`).  Contract §3 (Agent public surface) is
untouched — the forwarder, session factory, and streaming routes all
live in `core/` / `serve/` which are free to grow.  Phase D error
wording for the `client_factory + share_client=True` collision is
preserved verbatim for test compatibility.

Documented:

* User-facing: `docs/streaming.md` "Advanced: `SessionProvider`"
  section with a concrete auth / telemetry recipe.
* Developer-facing: `docs/internal/agentic-feature/developer-guide.md`
  §4 (public API surface) + §12 (customization knobs).

### E3: `StreamingAgentLoop.stream` tracks the final chunk via a caller-provided sink

**Plan §3.5** anticipates a class-level `_last_chunk` attribute.

**Actual:**  the `_tap_final_chunk` generator takes a single-element
list (`sink: list[ChatChunk | None]`) from the caller and stashes the
last observed chat chunk there.  This keeps `StreamingAgentLoop`
re-entrant across concurrent `stream()` calls on the same instance —
class-level state would corrupt under concurrency.

**Reason.**  Minor implementation detail surfaced while writing the
type-check-clean version.  The spec (§4.3) sketched the runner as a
class with a method, implying it could be instantiated per-call, but
in practice a shared instance is convenient (matches how `Agent`
holds a single `_t0` / `_tool_provider`).  The sink pattern is cheap
and keeps the method's signature unchanged.

**Impact.**  None on the public surface; `StreamingAgentLoop.stream`'s
signature matches the spec exactly.

## Resolved Open Questions

No open questions from `02-open-questions.md` were newly answered
during Phase E.  The three questions the spec referenced (T0 Q1,
FastMCP Q2, FastMCP Q3) were already resolved:

- **T0 Q1** — Per-block delta chunks carry stable ids; final chunk has
  `usage` populated.  The assembler relies on this invariant.
- **FastMCP Q2** — No built-in remote elicitation abstraction; the
  framework's `ElicitationForwarder` is the only path.  Phase E
  completed its Future-based resolver.
- **FastMCP Q3** — One elicitation handler per Client mandates
  per-session construction.  The `elicitation_client_factory` slot
  (and the broader `SessionProvider` Protocol) make this explicit.

## Contract Amendments

None.  `00-integration-contract.md` is unchanged:

- §2.1 (`mcp/` standalone): `mcp/` untouched.  `from t0_agents.mcp import
  run_agent` continues to work without the `[serve]` extra.
  `stream/` depends on `core/` and `mcp/` (permitted by §2.2).
- §2.4 (TensorZero-first): streaming uses
  `t0.inference(stream=True)` — no parallel transport, no duplicated
  responsibilities.  Elicitation stays the only HITL primitive.
- §3.3 (`Agent.stream` shape + correlation key): implemented exactly
  as specified.  Every delta event carries the chunk's `id`; consumers
  pair `TextDeltaEvent` with `TextEndEvent`, accumulate
  `ToolCallDeltaEvent` into `ToolCallEvent`, etc.
- §3.5 (concurrency): per-session `Client` invariant preserved —
  `SessionFactory.session(elicitation_handler=...)` routes through the
  `SessionProvider`, which raises `ConfigurationError` in
  shared-tool-provider mode when elicitation is requested.
- §6.1 (frozen `run_agent`): unchanged.
- §6.2 (frozen `AgentResult`): unchanged — `DoneEvent` carries the
  existing dataclass verbatim.
- §6.3 (frozen `mcp/` exports): unchanged.

## Release Notes Snippet

> **v0.6.0 — Live Agent (M5)**
>
> Two-level streaming for `Agent`: `from t0_agents import AgentEvent`
> plus `Agent.stream(...)` yields typed events (`TextDeltaEvent`,
> `ToolCallEvent`, `ToolResultEvent`, `ThoughtEvent`, `DoneEvent`, …)
> with stable correlation ids per T0 Q1.  The serving layer grows two
> routes behind `create_app(streaming=True, ...)`: `GET /chat/stream`
> (SSE, one-way) and `WS /chat/ws` (bidirectional, required for
> elicitation HITL).  The new `SessionProvider` Protocol is the
> public extension point for auth / multi-tenant / bespoke
> WebSocket wiring; the simple path (`client_factory` +
> `elicitation_client_factory`) wraps into a `CallableSessionProvider`
> automatically.  `ElicitationForwarder` is now fully functional —
> Future-based `handler` + `resolve(elicit_id, payload)` +
> `cancel_all(action=...)`.  The `[serve]` extra grows one dep
> (`sse-starlette>=2.0`); no base dep changes.  Phase A `run()`
> behaviour is byte-identical (regression covered).  See
> `docs/streaming.md` and `examples/12_streaming_http/`.
