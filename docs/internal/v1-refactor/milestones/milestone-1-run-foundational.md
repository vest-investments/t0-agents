# Milestone 1 — `Run` Foundational

> **Roadmap:** v1.0 four-abstraction refactor.
> **Deps:** none.  M1 is the foundation for M2, M3, M4.
> **Dev-log:** `docs/internal/v1-refactor/logs/milestone-1-dev-log.md` (always-write at close).

---

## §1 — Goal

Introduce `Run[T]` as the first-class invocation handle and `Agent.invoke(...) -> Run[T]` as the **canonical** entry point for v1.0.  Rewrite `agent.run()` and `agent.stream()` as deprecated shims that emit `DeprecationWarning` (once per process) pointing at `agent.invoke(...)`.  Shims keep working byte-identical to v0.12 modulo the warning — they are removed in v1.1 (see contract §6).

The other three milestones (M2 Middleware, M3 OutputMode, M4 Composer) consume `Run` — without it, they cannot be specified cleanly.

---

## §2 — Scope

### In scope
- `core/run.py` — `Run[T]`, `RunInput` (frozen dataclass), `RunState` enum, `RunCancelledError`, `RunFailedError`.
- `core/_runtime.py` — internal async driver (asyncio.Task + event buffer + Condition).
- `core/agent.py` — add `invoke()`; rewrite `run()` and `stream()` as shims.
- `__init__.py` — export new symbols.
- Unit tests under `tests/test_run/`.
- Stream runner adaptation: `stream/runner.py` emits events to Run buffer instead of yielding directly.

### Out of scope
- Middleware integration (M2).  `Agent.invoke()` accepts the `middleware=` kwarg but stores it for M2 use; the loop logic is unchanged in M1.
- OutputMode integration (M3).  `Agent.invoke()` accepts the `output=` kwarg but stores it for M3; behaviour is unchanged.
- Composers (M4).
- Streaming JSON (deferred — see Q11 in open questions).

### Non-goals
- Cross-process Run persistence (e.g., serialize Run to Redis for resume).  Run is in-process.
- Replay against a DIFFERENT gateway (cross-deployment).  `replay(agent=...)` accepts any Agent but does not orchestrate distributed gateways.

---

## §3 — Constraints

Inherited from `00-integration-contract.md`:
- §2.1 `mcp/` standalone — zero diff in `src/t0_agents/mcp/`.
- §2.2 TensorZero-first — `Run.episode_id`, `inference_ids`, `feedback()` map to TZ primitives.
- §2.3 Chat byte-identity — 18 examples run byte-identical to v0.12 modulo `DeprecationWarning` emissions to stderr.
- §2.4 Tier 1 migration — `agent.run()` / `agent.stream()` emit `DeprecationWarning` once per process; remain functional until v1.1.
- §2.5 Async-first, mypy strict.

Milestone-specific:
- `Run` holds the Agent via `weakref.ref(agent)` — never extends gateway lifetime.
- `RunInput` is frozen (`@dataclass(frozen=True, slots=True)`) — immutable snapshot.
- Eager by default (Q1 resolution).  `eager=False` opt-in.
- Deprecation warnings are deduplicated by Python's default `warnings` filter (one emission per source line, per process).  Implementation uses `warnings.warn(..., DeprecationWarning, stacklevel=2)`.

---

## §4 — Module additions

### New: `src/t0_agents/core/run.py`

Top-level shape (target ~400 LoC):

```python
from __future__ import annotations
import asyncio
import weakref
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from t0_agents.core.exceptions import T0AgentsError
from t0_agents.core.types import AgentResult
from t0_agents.stream import AgentEvent

T = TypeVar("T")

class RunCancelledError(T0AgentsError):
    """Raised by `await run.result()` when run was cancelled."""

class RunFailedError(T0AgentsError):
    """Raised by `await run.result()` when the underlying run failed."""
    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"Run failed: {cause!r}")
        self.cause = cause

RunState = Literal["pending", "running", "succeeded", "failed", "cancelled"]

@dataclass(frozen=True, slots=True)
class RunInput:
    user_message: str
    function_name: str
    function_kind: Literal["chat", "json"]
    episode_id: str | None
    tags: Mapping[str, str] | None
    system: str | None
    system_arguments: Mapping[str, Any] | None
    output_schema: Mapping[str, Any] | None
    json_mode: Literal["off", "on", "strict", "tool"] | None
    prior_messages: tuple[Mapping[str, Any], ...] = ()

class Run(Generic[T]):
    def __init__(self, agent: "Agent", input_: RunInput, *,
                 eager: bool = True, context: dict[str, Any] | None = None) -> None:
        ...

    # ---- properties ----
    @property
    def input(self) -> RunInput: ...
    @property
    def context(self) -> dict[str, Any]: ...
    @property
    def state(self) -> RunState: ...
    @property
    def episode_id(self) -> str | None: ...
    @property
    def inference_ids(self) -> tuple[str, ...]: ...
    @property
    def agent(self) -> "Agent | None": ...   # weakref dereferenced

    # ---- consumption ----
    async def result(self) -> AgentResult[T]: ...
    async def events(self, *, replay: bool = False) -> AsyncIterator[AgentEvent]: ...
    async def stream_text(self) -> AsyncIterator[str]: ...
    def __await__(self) -> Generator[Any, None, AgentResult[T]]:
        return self.result().__await__()

    # ---- control ----
    async def cancel(self) -> None: ...
    async def feedback(self, name: str, value: float | str | bool, *,
                       target: Literal["episode", "inference"] = "episode",
                       inference_id: str | None = None) -> None: ...

    # ---- composition ----
    async def replay(self, *, agent: "Agent | None" = None,
                     variant: str | None = None,
                     tags: Mapping[str, str] | None = None,
                     episode_id: str | None = None) -> "Run[T]": ...
    async def fork(self, extra_message: str, *,
                   agent: "Agent | None" = None) -> "Run[T]": ...

    # ---- introspection ----
    def transcript(self) -> list[Mapping[str, Any]]: ...
    def cost(self) -> float: ...
    def usage(self) -> Any: ...   # aggregated Usage shape

    # ---- internal ----
    def _start(self) -> None: ...
    async def _drive(self) -> None: ...   # the async task body
```

### New: `src/t0_agents/core/_runtime.py`

Internal driver helpers (~150 LoC):

```python
from __future__ import annotations
import asyncio
from collections import deque
from t0_agents.stream import AgentEvent

class EventBuffer:
    """Bounded buffer + Condition for replayable events.

    - Up to MAX_EVENTS (10_000) buffered.
    - `truncated: bool` flag set if older events were dropped.
    - `notify_all` on append; consumers await Condition.
    """
    MAX_EVENTS = 10_000

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []
        self._cond = asyncio.Condition()
        self._done = asyncio.Event()
        self._truncated = False
    # ...

async def drive_chat(run: "Run", agent: "Agent", input_: "RunInput") -> "AgentResult":
    """Internal driver for chat-mode runs.  Delegates to mcp.run_agent."""
    ...

async def drive_json(run: "Run", agent: "Agent", input_: "RunInput") -> "AgentResult":
    """Internal driver for JSON-mode runs.  Delegates to core.structured.run_structured_inference."""
    ...
```

### Modified: `src/t0_agents/core/agent.py`

Add `invoke()` method.  Keep `run()` and `stream()` as shims.

```python
def invoke(self, user_message: str, *,
           episode_id: str | None = None,
           tags: dict[str, str] | None = None,
           system_arguments: dict[str, Any] | None = None,
           system: str | None = None,
           # M2 placeholder (accepted, threaded into RunContext, not yet used):
           middleware: list["Middleware"] | None = None,
           # M3 placeholder (accepted, threaded, ChatOutput default at M1; OutputMode formalised in M3):
           output: "Any | None" = None,
           # M1 lifecycle:
           eager: bool = True,
           context: dict[str, Any] | None = None,
          ) -> Run[Any]:
    """Canonical entry point.  Returns Run[T] for any mode."""
    input_ = RunInput(...)
    return Run(self, input_, eager=eager, context=context)

async def run(self, user_message: str, **kwargs: Any) -> AgentResult[Any]:
    warnings.warn(
        "Agent.run() is deprecated and will be removed in t0-agents 1.1. "
        "Use `await agent.invoke(msg)` instead — Run is awaitable and gives the same AgentResult.",
        DeprecationWarning,
        stacklevel=2,
    )
    return await self.invoke(user_message, **kwargs).result()

async def stream(self, user_message: str, **kwargs: Any) -> AsyncIterator[AgentEvent]:
    warnings.warn(
        "Agent.stream() is deprecated and will be removed in t0-agents 1.1. "
        "Use `async for ev in agent.invoke(msg).events()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    async for ev in self.invoke(user_message, **kwargs).events():
        yield ev
```

### Modified: `src/t0_agents/__init__.py`

Add exports:
- `Run`
- `RunInput`
- `RunState` (export the Literal alias)
- `RunCancelledError`
- `RunFailedError`

### Modified: `src/t0_agents/stream/runner.py`

The `StreamingAgentLoop` adapts to push events into the Run's `EventBuffer` instead of yielding directly.  External `agent.stream()` shim wraps this by consuming from the buffer.

---

## §5 — Tests

### `tests/test_run/test_lifecycle.py`
- Eager start: `Run(...)` task is running immediately after construction.
- Lazy start: `Run(..., eager=False)` task is not running until `await run.result()` or `async for ev in run.events()`.
- State transitions: pending → running → succeeded.
- `await run.cancel()`: state → cancelled; subsequent `await run.result()` raises `RunCancelledError`.
- Failed run: `await run.result()` raises `RunFailedError` wrapping the underlying exception.
- `__await__`: `result = await run` works as alias.

### `tests/test_run/test_replay.py`
- Same agent: `await run.replay()` produces a new Run with same input, new episode_id (Q7).
- `replay(episode_id=original.episode_id)` preserves correlation.
- Different agent: `await run.replay(agent=other_agent)` runs against the other gateway.
- Gateway closed: replay against a closed-gateway agent raises `RuntimeError` with clear message.
- Weakref GC: if the original agent is garbage-collected and no explicit `agent=` is provided, `replay()` raises `RuntimeError`.

### `tests/test_run/test_fork.py`
- Fork preserves transcript up to fork point.
- Fork adds the extra_message and continues.
- Forked Run is independent (cancelling original does not cancel fork; new episode_id).

### `tests/test_run/test_two_channel.py`
- Coroutine A: `result = await run.result()`.
- Coroutine B: `async for ev in run.events()` concurrently.
- Both observe the same run; events arrive in order; result is consistent with what events emitted.

### `tests/test_run/test_events_buffer.py`
- Events bufferized; `events(replay=False)` starts at live cursor; `events(replay=True)` replays from event 0.
- Calling `events()` twice without `replay=True` resumes from cursor.
- Buffer overflow at 10_000 events → `truncated = True` flag + WARN log.

### `tests/test_run/test_feedback.py`
- `await run.feedback("user_thumbs_up", True)` calls `t0.feedback(...)` with `episode_id`.
- `target="inference"` requires `inference_id` (or uses latest); calls `t0.feedback(...)` with `inference_id`.

### `tests/test_run/test_compat_run_stream.py`
- `agent.run("hi")` returns `AgentResult` byte-identical to v0.12 (same fields, same values).  At M1, `AgentConfig` still accepts the legacy JSON kwargs (`function_kind=`, etc.) — those are removed in M3.
- `agent.stream("hi")` yields events byte-identical to v0.12 (same event types, same order).
- `agent.run("hi")` emits `DeprecationWarning` exactly once per source line (verified via `pytest.warns(DeprecationWarning)`).  Same for `agent.stream("hi")`.
- The warning message mentions `agent.invoke()` as the replacement.
- Runs the 18 examples under `examples/00..18` and verifies byte-identical stdout (stderr contains the deprecation warning from the shim).  In M1 the legacy JSON example still runs as in v0.12 — it gets migrated in M3.

### Integration tests (`tests/test_run/test_integration.py`)
- Marked `@pytest.mark.integration`.  Requires `docker compose up`.
- Real TZ gateway + real FastMCP server + Run lifecycle end-to-end.

---

## §6 — Example

No new example file in M1 (Run is a foundational shift; the canonical Run-centric example lands in M5 as `examples/20_run_compose/`).  Validation in M1 is via the 18 existing examples running byte-identical on stdout, with `DeprecationWarning` to stderr from the shims.

---

## §7 — Documentation

- Update `docs/agent.md` with a short forward reference: "v1.0 introduces `agent.invoke(...)` returning a `Run[T]`.  `agent.run()` and `agent.stream()` continue to work and are now thin shims."  Full doc lands in M5 (`docs/run.md`).
- Add `Run`, `RunInput`, `RunState`, `RunCancelledError`, `RunFailedError` to `docs/api-reference.md`.

---

## §8 — Checklist (Definition of Done)

### Pre-flight
- [ ] Read `00-integration-contract.md` (invariants §2).
- [ ] Read `01-implementation-plan.md` (cross-cutting checklist §3).
- [ ] Read `02-open-questions.md` (Q1, Q2, Q7 are relevant).
- [ ] Read prior dev-logs (none on M1).
- [ ] Read project memory (`MEMORY.md`).
- [ ] Inspect SDK source at `.venv/lib/python*/site-packages/tensorzero/` for `feedback()` signature.

### Code
- [ ] `core/run.py` implements all `Run[T]` methods listed in §4.
- [ ] `core/_runtime.py` implements `EventBuffer`, `drive_chat`, `drive_json`.
- [ ] `core/agent.py` adds `invoke()`; `run()` and `stream()` are deprecated shims emitting `DeprecationWarning` (stacklevel=2, message points at `invoke()`).
- [ ] `Run` holds Agent via `weakref.ref`.
- [ ] `RunInput` is frozen dataclass.
- [ ] `RunState` is `Literal`.
- [ ] Eager default; `eager=False` works.
- [ ] `__init__.py` exports updated.
- [ ] `stream/runner.py` adapts to EventBuffer.

### Tests
- [ ] `tests/test_run/test_lifecycle.py` — pass.
- [ ] `tests/test_run/test_replay.py` — pass.
- [ ] `tests/test_run/test_fork.py` — pass.
- [ ] `tests/test_run/test_two_channel.py` — pass.
- [ ] `tests/test_run/test_events_buffer.py` — pass.
- [ ] `tests/test_run/test_feedback.py` — pass.
- [ ] `tests/test_run/test_compat_run_stream.py` — pass.
- [ ] `tests/test_run/test_integration.py` — pass with docker.

### Example
- [ ] (No new example in M1.)

### Docs
- [ ] `docs/agent.md` short forward reference added.
- [ ] `docs/api-reference.md` lists new symbols.

### Style gates
- [ ] `uv run ruff check src/` clean.
- [ ] `uv run mypy src/` strict pass.

### Implementation log
- [ ] `docs/internal/v1-refactor/logs/milestone-1-dev-log.md` written with all 5 sections.

### CHANGELOG
- [ ] `## [1.0.0-M1] — Unreleased` entry added.

### Open questions to resolve
- [ ] Q1 (eager default) — resolved.
- [ ] Q2 (events buffer size) — resolved.
- [ ] Q7 (replay episode_id semantics) — resolved.

### Invariant checks (gate)
- [ ] `git diff src/t0_agents/mcp/` → empty.
- [ ] 18 examples (`examples/00..18`) run byte-identical on stdout (stderr may contain `DeprecationWarning` from M1 shims).
- [ ] Pre-existing tests across all submodules pass unchanged.

---

## §9 — Acceptance criteria

- `from t0_agents import Run, RunInput, RunState, RunCancelledError, RunFailedError` works.
- `run = agent.invoke("hi"); result = await run` works (awaitable Run).
- `async for ev in run.events()` works.
- `await run.cancel()` cancels in-flight; `await run.result()` after cancel raises `RunCancelledError`.
- `await run.replay()` produces new Run with same input, new episode_id (Q7 resolution).
- `await run.fork("extra")` produces new Run continuing from transcript.
- `agent.run("hi")` and `agent.stream("hi")` keep working byte-identical to v0.12 in shape and values, BUT emit `DeprecationWarning` once per source line pointing at `agent.invoke(...)`.
- The 18 existing examples run end-to-end; their stderr contains the deprecation warning; their stdout is byte-identical to v0.12.

---

## §10 — Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `weakref` GC causes surprise in user code | Test the weakref case explicitly; `replay(agent=)` opt-in for cross-Agent usage; document. |
| Eager start races with consumer setup | Eager Run buffers events from t=0; consumer that arrives "late" replays via `events(replay=True)` or accepts missed events via cursor-based `events()`. |
| `events(replay=False)` race when called before first event | Buffer is created in `__init__`; Condition wait is robust against pre-start subscription. |
| Streaming path becomes two-tier (EventBuffer + downstream stream) and develops lag | Profile under heavy event volume; bounded buffer mitigates memory but throughput should not regress vs v0.12. |
| Compat shim subtle behaviour drift | Parallel test suite — old tests run against shims, new tests run against `invoke`. |

---

## §11 — Estimated surface

- `core/run.py`: ~400 LoC.
- `core/_runtime.py`: ~150 LoC.
- `core/agent.py` modifications: ~+50 LoC (invoke + shims).
- `core/exceptions.py` modifications: ~+30 LoC (2 new exception classes).
- `stream/runner.py` modifications: ~+50 LoC.
- `tests/test_run/`: ~600 LoC across 7 files.
- Docs delta: ~30 lines in `docs/agent.md` + ~30 in `docs/api-reference.md`.

**Total: ~1300 LoC.  Estimated 3 days full-time.**
