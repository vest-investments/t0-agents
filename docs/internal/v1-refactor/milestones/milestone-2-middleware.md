# Milestone 2 — `Middleware` Extension Surface

> **Roadmap:** v1.0 four-abstraction refactor.
> **Deps:** M1 (`Run` exists as public type).
> **Dev-log:** `docs/internal/v1-refactor/logs/milestone-2-dev-log.md` (always-write at close).

---

## §1 — Goal

Replace hand-coded cross-cutting concerns (memory, guardrails, cost-tracking, sub-agents) in `core/agent.py` with a composable Starlette-style middleware chain.  `core/agent.py` drops from ~1413 LoC to ~250 LoC (orchestrator + dispatch only).  `AgentConfig.__post_init__` materialises the default middleware chain from existing kwargs (`memory=`, `guardrails=`, `sub_agents=`) — full v0.12 compat.

Power users gain a first-class extension surface: rate-limiters, PII redactors, custom observability, recovery policies, etc. — all impossible today without forking.

---

## §2 — Scope

### In scope
- `core/middleware/__init__.py` — re-exports.
- `core/middleware/base.py` — `Middleware` protocol, `RunContext`, `build_pipeline()`.
- `core/middleware/builtins.py` — 6 built-in middlewares.
- `core/agent.py` refactor — loop body becomes `build_pipeline(self._middleware, terminal).execute(ctx)`.
- `core/config.py` — add `middleware: list[Middleware] | None = None` field; `__post_init__` materialises default chain.
- `__init__.py` — export `Middleware`, `RunContext`, 6 built-ins.
- Unit tests under `tests/test_middleware/`.

### Out of scope
- Industrial middleware (M5): Bulkhead, CircuitBreaker, TenantIsolation, PiiRedactor, RedisRateLimit.
- OutputMode integration (M3): `JsonRetryMiddleware` ships as a no-op placeholder in M2; it consults OutputMode in M3.
- Per-tool middleware (currently no use case; future work).

### Non-goals
- Async-iterator middleware (consume events).  Middleware operates on the run, not on each event.
- Distributed middleware coordination (e.g., cross-process rate limiting persistence — that's a M5 concern via Redis).

---

## §3 — Constraints

Inherited from `00-integration-contract.md`:
- §2.1 `mcp/` standalone — middleware lives entirely in `core/`.
- §2.2 TensorZero-first — middlewares wrap TZ calls; never replace TZ features.
- §2.3 Chat byte-identity — 18 examples run byte-identical.
- §2.4 Additive compat — `AgentConfig(memory=, guardrails=, sub_agents=)` materialises the same chain as v0.12 hardcoded.

Milestone-specific:
- Default chain materialisation triggers only when `middleware is None`.  `middleware=[]` is an explicit "no middleware" override (semantically distinct).
- Order invariant: `SubAgentMiddleware` wraps `tool_provider` **before** `GuardrailsMiddleware` (guardrails see delegations).  Tests enforce this.
- `RunContext` is mutable inside a middleware but middlewares MUST NOT call `next_()` twice (raises clear error).

---

## §4 — Module additions

### New: `src/t0_agents/core/middleware/base.py`

```python
from __future__ import annotations
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tensorzero import AsyncTensorZeroGateway
    from t0_agents.core.agent import Agent
    from t0_agents.core.config import AgentConfig
    from t0_agents.core.run import Run
    from t0_agents.core.types import AgentResult, ToolProvider

Next = Callable[["RunContext"], Awaitable["AgentResult"]]

@runtime_checkable
class Middleware(Protocol):
    name: str
    async def __call__(self, ctx: "RunContext", next_: Next) -> "AgentResult": ...

@dataclass
class RunContext:
    # ---- inputs (mutable carefully) ----
    config: "AgentConfig"
    user_message: str
    episode_id: str | None
    tags: dict[str, str] | None
    system: str | None
    system_arguments: dict[str, Any] | None
    output: "Any"   # OutputMode | None (M3)

    # ---- gateway + tools ----
    t0: "AsyncTensorZeroGateway"
    tool_provider: "ToolProvider | None"

    # ---- per-run scratch ----
    on_inference: Callable[..., Awaitable[None]] | None = None
    prior_messages: list[Mapping[str, Any]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ---- output ----
    result: "AgentResult | None" = None
    run: "Run | None" = None

    def chain_on_inference(self, cb: Callable[..., Awaitable[None]]) -> None:
        """Compose `cb` after existing `on_inference` so multiple middlewares can both observe inferences."""
        ...

    def replace_tool_provider(self, tp: "ToolProvider") -> None:
        """Replace ctx.tool_provider, recording the original for cleanup."""
        ...

def build_pipeline(
    middleware: list[Middleware],
    terminal: Next,
) -> Next:
    """Compose middleware list around terminal via right-fold.

    Returns a callable that, when invoked with a RunContext, runs the
    full chain.  Order: middleware[0] is outermost.
    """
    chain = terminal
    for mw in reversed(middleware):
        nxt = chain   # capture current chain before reassignment
        async def wrapped(ctx: "RunContext", _mw=mw, _next=nxt) -> "AgentResult":
            return await _mw(ctx, _next)
        chain = wrapped
    return chain
```

### New: `src/t0_agents/core/middleware/builtins.py`

Six middlewares, each ~100 LoC.  Functional parity with v0.12 inline code:

```python
class MemoryMiddleware:
    name = "memory"
    def __init__(self, provider: "MemoryProvider", strategy: str) -> None: ...
    async def __call__(self, ctx, next_):
        records = await self._load(ctx)
        # inject summary / system_template / prior_messages based on strategy
        try:
            result = await next_(ctx)
        finally:
            await self._save(ctx, result)
        return result

class GuardrailsMiddleware:
    name = "guardrails"
    def __init__(self, guardrails: list["Guardrail"]) -> None: ...
    async def __call__(self, ctx, next_):
        guard_ctx = GuardrailContext(...)
        for g in self._guardrails:
            await g.before_run(guard_ctx)
        ctx.replace_tool_provider(GuardedToolProvider(ctx.tool_provider, self._guardrails, guard_ctx))
        try:
            result = await next_(ctx)
            for g in self._guardrails:
                await g.after_run(guard_ctx, result)
            return result
        except Exception:
            for g in self._guardrails:
                await g.after_run(guard_ctx, None)
            raise

class CostTrackingMiddleware:
    name = "cost_tracking"
    async def __call__(self, ctx, next_):
        def tracker(response):  # ctx.chain_on_inference appends this
            ...
        ctx.chain_on_inference(tracker)
        return await next_(ctx)

class SubAgentMiddleware:
    name = "sub_agents"
    def __init__(self, sub_agents: list["Agent"]) -> None: ...
    async def __call__(self, ctx, next_):
        ctx.replace_tool_provider(_OrchestrationToolProvider(ctx.tool_provider, self._sub_agents))
        return await next_(ctx)

class JsonRetryMiddleware:
    name = "json_retry"
    # M2: no-op placeholder.  M3: delegates to OutputMode.recover().
    async def __call__(self, ctx, next_):
        return await next_(ctx)

class ObservabilityMiddleware:
    name = "observability"
    def __init__(self, tracer: "Any" = None) -> None: ...
    async def __call__(self, ctx, next_):
        with tracer.start_as_current_span(f"agent.run/{ctx.config.name}"):
            return await next_(ctx)
```

### Modified: `src/t0_agents/core/agent.py`

Replace the ~1100 LoC of inline orchestration with:

```python
async def _execute_run(self, ctx: RunContext) -> AgentResult:
    terminal = self._make_terminal()  # returns async (ctx) -> AgentResult
    pipeline = build_pipeline(self._middleware, terminal)
    return await pipeline(ctx)
```

`_make_terminal()` returns the async callable that invokes `run_agent()` (or `run_structured_inference()` for JSON in M3).  The terminal is the only place that knows about the chat vs JSON branch.

### Modified: `src/t0_agents/core/config.py`

```python
@dataclass(slots=True)
class AgentConfig:
    # ... existing fields ...
    middleware: list["Middleware"] | None = None  # NEW

    def __post_init__(self) -> None:
        # ... existing validations ...

        if self.middleware is None:
            self.middleware = self._build_default_chain()

    def _build_default_chain(self) -> list["Middleware"]:
        from t0_agents.core.middleware import (
            MemoryMiddleware, GuardrailsMiddleware, CostTrackingMiddleware,
            SubAgentMiddleware, JsonRetryMiddleware,
        )
        chain: list[Middleware] = []
        if self.memory:
            chain.append(MemoryMiddleware(self.memory, self.memory_strategy))
        if self.guardrails:
            chain.append(GuardrailsMiddleware(list(self.guardrails)))
        chain.append(CostTrackingMiddleware())
        if self.sub_agents:
            chain.append(SubAgentMiddleware(list(self.sub_agents)))
        chain.append(JsonRetryMiddleware())  # no-op in M2; activates in M3
        return chain

    @staticmethod
    def default_middleware_for(*, memory=None, guardrails=None, sub_agents=None) -> list["Middleware"]:
        """Helper for power users: build the default chain explicitly."""
        ...
```

### Modified: `src/t0_agents/__init__.py`

Add exports: `Middleware`, `RunContext`, `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`.

---

## §5 — Tests

### `tests/test_middleware/test_chain_order.py`
- Three test middlewares append their name to `ctx.extras["log"]` before and after calling `next_()`.  Verify outermost-first order.
- Short-circuit: middleware that returns without calling `next_()` skips inner layers.
- Calling `next_()` twice raises `T0AgentsError`.

### `tests/test_middleware/test_default_chain.py`
- `AgentConfig(memory=InMemoryMemory())` produces middleware list containing `MemoryMiddleware` and `CostTrackingMiddleware`.
- `AgentConfig(guardrails=[X, Y])` produces chain with `GuardrailsMiddleware([X, Y])` and `CostTrackingMiddleware`.
- `AgentConfig(sub_agents=[a, b])` produces chain with `SubAgentMiddleware([a, b])`.
- Default chain order: `Memory → Guardrails → CostTracking → SubAgents → JsonRetry`.
- `middleware=[]` overrides to "no middleware".
- `middleware=[Custom()]` overrides default entirely.

### `tests/test_middleware/test_compat_v0_12.py`
- Build agent with `AgentConfig(memory=..., guardrails=[...], sub_agents=[...])`.  Run.  Verify `AgentResult` byte-identical to v0.12.
- Run the 18 examples.  Verify byte-identical output shape.

### `tests/test_middleware/test_custom_middleware.py`
- Custom `PiiRedactor` middleware that lowercases `ctx.user_message`.  Verify the inference sees the lowercased version.
- Custom `RateLimiter` middleware that uses an `asyncio.Semaphore` — verify concurrent calls are serialized.

### `tests/test_middleware/test_tool_provider_order.py`
- Agent with both `sub_agents` and `guardrails`.  Custom guardrail records the tool_provider it sees in `before_tool`.  Verify it sees the `_OrchestrationToolProvider` wrapping (i.e., guardrails are outside sub-agents).

### `tests/test_middleware/test_observability.py`
- Custom tracer captures span names.  Verify `agent.run/<name>` span is created.

---

## §6 — Example

No new example in M2 (the new extension surface lands as an example in M5).  Validation is via `tests/test_middleware/` and the 18 existing examples running byte-identical.

---

## §7 — Documentation

- `docs/agent.md` — short note: "v1.0 exposes the middleware chain via `AgentConfig.middleware`.  See `docs/middleware.md` for the full guide" (placeholder reference; full doc lands in M5).
- `docs/api-reference.md` — list new symbols.
- Internal: update `02-open-questions.md` Q4 / Q5 / Q6 with current decision state if M2 forces them.

---

## §8 — Checklist (Definition of Done)

### Pre-flight
- [ ] Read `00-integration-contract.md`.
- [ ] Read `01-implementation-plan.md` (cross-cutting §3).
- [ ] Read `02-open-questions.md` (Q4, Q5, Q6 may be touched).
- [ ] Read **M1 dev-log** (mandatory — always-write policy).
- [ ] Read project memory (`MEMORY.md`).

### Code
- [ ] `core/middleware/base.py` implements `Middleware` Protocol, `RunContext`, `build_pipeline()`.
- [ ] `core/middleware/builtins.py` implements 6 middlewares with v0.12 parity.
- [ ] `RunContext.chain_on_inference()` and `replace_tool_provider()` work correctly.
- [ ] `build_pipeline()` composes outermost-first; short-circuit works.
- [ ] `core/agent.py` orchestrates via pipeline (`_execute_run`).  `agent.py` <= 300 LoC.
- [ ] `core/config.py` — `middleware: list[Middleware] | None`; `__post_init__` materialises default; `default_middleware_for()` helper.
- [ ] `middleware=None` vs `middleware=[]` semantics distinct.
- [ ] Order invariant: `SubAgentMiddleware` inside `GuardrailsMiddleware`.
- [ ] `__init__.py` exports updated.

### Tests
- [ ] `tests/test_middleware/test_chain_order.py` — pass.
- [ ] `tests/test_middleware/test_default_chain.py` — pass.
- [ ] `tests/test_middleware/test_compat_v0_12.py` — pass.
- [ ] `tests/test_middleware/test_custom_middleware.py` — pass.
- [ ] `tests/test_middleware/test_tool_provider_order.py` — pass.
- [ ] `tests/test_middleware/test_observability.py` — pass.

### Example
- [ ] (No new example in M2.)

### Docs
- [ ] `docs/agent.md` short note.
- [ ] `docs/api-reference.md` updated.

### Style gates
- [ ] `uv run ruff check src/` clean.
- [ ] `uv run mypy src/` strict pass.

### Implementation log
- [ ] `docs/internal/v1-refactor/logs/milestone-2-dev-log.md` written.

### CHANGELOG
- [ ] `## [1.0.0-M2] — Unreleased` entry added.

### Open questions to resolve
- [ ] Q4 (`BulkheadMiddleware` base vs extra) — early decision; deferred to M5 if not blocking M2.
- [ ] Q13 (`JsonRetryMiddleware` placement in chain) — must be resolved here because M3 will activate the recovery.

### Invariant checks (gate)
- [ ] `git diff src/t0_agents/mcp/` → empty.
- [ ] 18 examples run byte-identical.
- [ ] All pre-existing tests pass unchanged.

---

## §9 — Acceptance criteria

- `from t0_agents import Middleware, RunContext, MemoryMiddleware, GuardrailsMiddleware, CostTrackingMiddleware, SubAgentMiddleware, JsonRetryMiddleware, ObservabilityMiddleware` works.
- `AgentConfig(memory=..., guardrails=[...], sub_agents=[...])` produces functionally equivalent runs to v0.12 (byte-identical examples).
- `AgentConfig(middleware=[Custom()])` overrides default chain entirely.
- `agent.py` is < 300 LoC.
- Power user can construct `AgentConfig(..., middleware=[MyCustom(), *AgentConfig.default_middleware_for(memory=m, guardrails=g), MyAfter()])`.

---

## §10 — Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Order invariant (SubAgents inside Guardrails) silently violated | Explicit test (`test_tool_provider_order.py`) + comment in `_build_default_chain()`. |
| `__post_init__` causes circular import (config → middleware → types → config) | Lazy import inside `_build_default_chain()`. |
| Pipeline composition has subtle bugs (closure capture) | Unit tests with 3 middlewares; verify each is called exactly once outermost-first. |
| `RunContext.chain_on_inference()` composes badly with multiple middlewares | Test with both Cost + custom observability; verify both callbacks fire per inference. |
| Default chain materialisation drifts over time as new kwargs are added | Document the helper `default_middleware_for(...)` as the single source of truth. |
| Compat shim breaks for users that pass `middleware=[]` (no middleware) — they get no memory, no guardrails, no cost-tracking | Document explicitly that `middleware=[]` is "no middleware" — power user opt-in. |

---

## §11 — Estimated surface

- `core/middleware/base.py`: ~180 LoC.
- `core/middleware/builtins.py`: ~600 LoC (6 middlewares × ~100 each).
- `core/agent.py` modifications: net ~−900 LoC (down from ~1413 to ~250).
- `core/config.py` modifications: ~+50 LoC.
- `tests/test_middleware/`: ~800 LoC across 6 files.
- Docs delta: ~30 lines.

**Total: ~+700 LoC src, ~−900 LoC src refactor, +800 LoC tests.  Estimated 5 days full-time.**
