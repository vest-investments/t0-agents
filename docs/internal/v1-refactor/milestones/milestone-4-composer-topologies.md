# Milestone 4 — `Composer` + Topologies

> **Roadmap:** v1.0 four-abstraction refactor.
> **Deps:** M1 (`Run`), M2 (`Middleware`), M3 (`OutputMode`).
> **Dev-log:** `docs/internal/v1-refactor/logs/milestone-4-dev-log.md` (always-write at close).

---

## §1 — Goal

Introduce `Composer` as the fourth abstraction: topologies of `Run`s.  `Composer.invoke(input) → Run[T]` is uniform with `Agent.invoke()`, enabling arbitrary nesting (pipeline of races of consensus, etc.).

Ship 9 built-in topologies covering industrial/military patterns: `sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`.

Industrial properties: cancellation propagation via asyncio task tree, audit via `parent_episode_id`, isolation via per-sub-Run middleware chains.

---

## §2 — Scope

### In scope
- `compose/topology.py` — `Composable` protocol, `Composer` protocol, abstract base.
- `compose/builtins.py` — 9 topology factory functions.
- `compose/_runtime.py` — internal driver helpers for each topology.
- `compose/__init__.py` — export builders + preserve existing `agent_as_tool`.
- `__init__.py` — re-export top-level selectors.
- Unit tests under `tests/test_compose/v1/` (new subdirectory to avoid clobbering legacy `tests/test_compose/`).

### Out of scope
- Industrial middleware integration (M5): `BulkheadMiddleware`, `CircuitBreakerMiddleware` as middleware (not as Composer).
- Distributed Composers (cross-process orchestration).  Composers run in one process.
- Persistence of Composer state across process restarts.
- Declarative DSL (YAML/JSON definition of a Composer).
- Visualisation / debugging UI.

### Non-goals
- Replacing `AgentConfig.sub_agents` — that legacy path stays as M2's `SubAgentMiddleware`.  Composers are a strict additive feature.
- Changing the existing `compose/agent_as_tool.py` primitive.

---

## §3 — Constraints

Inherited from `00-integration-contract.md`:
- §2.1 `mcp/` standalone — Composers live in `compose/`, never import from `mcp/`.
- §2.2 TensorZero-first — `parent_episode_id` correlation uses TZ-native mechanism (already supported by v0.12).
- §2.3 Chat byte-identity — `examples/14_multi_agent/` runs byte-identical.
- §2.4 Additive compat — `AgentConfig.sub_agents=[...]` unchanged.

Milestone-specific:
- `Composer.invoke()` returns `Run[T]` (same type as `Agent.invoke()`) for uniform composition.
- Cancellation: `await root_run.cancel()` propagates to all sub-Runs.  Default timeout 30s (Q3 resolution); per-topology override.
- Audit: each sub-Run carries `parent_episode_id` set to the root Run's `episode_id`.
- Order of execution within a topology must be deterministic where it matters (`sequential`, `saga`); intentionally non-deterministic where parallelism is the point (`parallel`, `race`).

---

## §4 — Module additions

### New: `src/t0_agents/compose/topology.py`

```python
from __future__ import annotations
from collections.abc import Awaitable
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from t0_agents.core.run import Run

T = TypeVar("T", covariant=True)
Input = TypeVar("Input", contravariant=True)

@runtime_checkable
class Composable(Protocol[Input, T]):
    """An object that can be invoked to produce a Run.

    Both Agent and Composer satisfy this protocol.
    """
    def invoke(self, input: Input, **kwargs: Any) -> Run[T]: ...

class Composer(Composable[Any, Any]):
    """A topology of Composables.  Itself a Composable."""
    children: tuple[Composable, ...]
    name: str

    def invoke(self, input: Any, **kwargs: Any) -> Run[Any]: ...

class ComposerError(T0AgentsError):
    """Raised when a sub-Run inside a Composer fails and the topology's error_policy demands propagation."""
    def __init__(self, node: Composable, cause: BaseException) -> None:
        super().__init__(f"Composer node {node!r} failed: {cause!r}")
        self.node = node
        self.cause = cause
```

### New: `src/t0_agents/compose/builtins.py`

9 factory functions.  Each returns a `Composer`:

```python
def sequential(*nodes: Composable, chain: Callable[[AgentResult, Any], Any] | None = None) -> Composer:
    """Pipeline: each node's result feeds the next.

    `chain(prev_result, original_input)` produces the next node's input.
    Default chain: `prev_result.value` if str, else `json.dumps(prev_result.value)`.
    """

def parallel(*nodes: Composable,
             reducer: Callable[[list[AgentResult]], Any] | None = None,
             error_policy: Literal["fail_fast", "continue"] = "fail_fast") -> Composer:
    """Fan-out: all nodes on same input; optional reducer aggregates."""

def race(*nodes: Composable, cancel_others: bool = True) -> Composer:
    """First-completed wins; others cancelled if cancel_others=True."""

def consensus(*nodes: Composable, merge: Callable[[list[Any]], Any]) -> Composer:
    """N propose; merge picks the answer.  Useful for voting/ensemble."""

def map_reduce(node: Composable, *,
               reducer: Callable[[list[AgentResult]], Any],
               concurrency: int | None = None) -> Composer:
    """For each item in input list, invoke node in parallel (bounded by concurrency); reduce."""

def conditional(*, router: Callable[[Any], str],
                routes: dict[str, Composable],
                default: Composable | None = None) -> Composer:
    """Router picks the route key; sub-Composable runs."""

def saga(*steps: tuple[Composable, Composable | None]) -> Composer:
    """Sequential with compensation: on failure at step N, run compensate(N-1..0)."""

def retry_chain(*nodes: Composable,
                on_error: Callable[[BaseException], bool] | None = None) -> Composer:
    """First node; on failure, try next; etc.  `on_error` filters which errors trigger fallback."""

def circuit_breaker(node: Composable, *,
                    threshold: int = 5,
                    reset_timeout: float = 30.0) -> Composer:
    """Wraps a node with open/half-open/closed state machine."""
```

### New: `src/t0_agents/compose/_runtime.py`

Internal drivers — one per topology.  Each driver returns a Run; the Run's `_drive()` orchestrates sub-Runs.

```python
async def _drive_sequential(nodes, input, chain, kwargs) -> AgentResult:
    current = input
    accumulated_inference_ids: list[str] = []
    for node in nodes:
        sub_run = node.invoke(current, **kwargs)
        sub_result = await sub_run.result()
        accumulated_inference_ids.extend(sub_run.inference_ids)
        current = chain(sub_result, input)
    return _build_composite_result(current, accumulated_inference_ids, ...)

async def _drive_parallel(nodes, input, reducer, error_policy, kwargs) -> AgentResult: ...
async def _drive_race(nodes, input, cancel_others, kwargs) -> AgentResult: ...
async def _drive_consensus(nodes, input, merge, kwargs) -> AgentResult: ...
async def _drive_map_reduce(node, input, reducer, concurrency, kwargs) -> AgentResult: ...
async def _drive_conditional(router, routes, default, input, kwargs) -> AgentResult: ...
async def _drive_saga(steps, input, kwargs) -> AgentResult: ...
async def _drive_retry_chain(nodes, on_error, input, kwargs) -> AgentResult: ...
async def _drive_circuit_breaker(node, threshold, reset_timeout, state, input, kwargs) -> AgentResult: ...
```

Each driver:
- Sets `parent_episode_id` on each sub-Run (via the `invoke(..., tags={"parent_episode_id": root.episode_id, ...})`).
- Propagates cancellation: if root Run is cancelled, all sub-Runs receive `await sub.cancel()`.
- Aggregates `inference_ids` from all sub-Runs into the composite result.

### Modified: `src/t0_agents/compose/__init__.py`

```python
from t0_agents.compose.agent_as_tool import agent_as_tool   # existing, unchanged
from t0_agents.compose.topology import Composable, Composer, ComposerError
from t0_agents.compose.builtins import (
    sequential, parallel, race, consensus, map_reduce,
    conditional, saga, retry_chain, circuit_breaker,
)
```

### Modified: `src/t0_agents/__init__.py`

Re-export top-level selectors: `sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`, `Composable`, `Composer`, `ComposerError`.

---

## §5 — Tests

### `tests/test_compose/v1/test_sequential.py`
- Pipeline of 3 agents: A → B → C.  Each receives the previous's output.
- Default chain handles str → str.
- Custom chain handles dict → str (e.g., `lambda r, _: r.value["query"]`).
- Cancellation mid-pipeline cancels current and skips remaining.

### `tests/test_compose/v1/test_parallel.py`
- 3 agents on same input; reducer collects results.
- `error_policy="fail_fast"`: one sub-Run fails → `ComposerError` propagates immediately.
- `error_policy="continue"`: one fails → reducer sees list with `Exception` for the failed slot.

### `tests/test_compose/v1/test_race.py`
- 3 agents; first to complete wins.  Other Runs are cancelled (verify `state == "cancelled"`).
- `cancel_others=False`: losers complete in background; result is the first.

### `tests/test_compose/v1/test_consensus.py`
- 3 agents vote on classification; merge = `Counter(votes).most_common(1)[0][0]`.
- Edge case: tie → merge function's responsibility (caller-defined).

### `tests/test_compose/v1/test_map_reduce.py`
- Scatter over a list of 10 items; concurrency=3 → at most 3 in flight.
- Reducer aggregates the 10 sub-results.
- `concurrency=None` → all in flight at once.

### `tests/test_compose/v1/test_conditional.py`
- Router based on input.label routes to one of 3 sub-Composables.
- Unknown label → default (if provided) or `ComposerError`.

### `tests/test_compose/v1/test_saga.py`
- 3 steps, each with a compensate.  Steps 0 + 1 succeed; step 2 fails.  Verify compensate(1), compensate(0) run in reverse order.
- Step with `None` compensate (no rollback for that step) is skipped silently.

### `tests/test_compose/v1/test_retry_chain.py`
- First node fails (e.g., raises `TZ rate limit`).  Second node succeeds.  Verify second's result is returned.
- `on_error` filter: certain exception types pass through, others retry.

### `tests/test_compose/v1/test_circuit_breaker.py`
- Threshold=3; 3 consecutive failures → state opens; next invocation raises `ComposerError` immediately without calling node.
- After `reset_timeout`, state half-opens; next call attempts; success → closed; failure → re-opens.

### `tests/test_compose/v1/test_nested_composers.py`
- 3-level nesting: `sequential(race(a, b), conditional(routes={"x": consensus([c, d, e], merge=majority), "y": agent_f}))`.
- Verify all sub-Runs carry `parent_episode_id` of the root.
- Verify cancel from root propagates to all 4-5 levels.

### `tests/test_compose/v1/test_cancellation.py`
- Long-running pipeline.  `await root.cancel()` mid-flight.
- All sub-Runs transition to `cancelled` within 30s (default timeout).
- Resource cleanup: tool calls in-flight receive `asyncio.CancelledError`.

### `tests/test_compose/v1/test_compat_sub_agents.py`
- `AgentConfig(sub_agents=[a, b])` (legacy) runs `examples/14_multi_agent/` byte-identical.
- No Composers required; the legacy path is preserved by `SubAgentMiddleware` from M2.

---

## §6 — Example

No new example in M4 (consolidates in `examples/22_industrial_topology/` in M5).  Validation: `examples/14_multi_agent/` runs byte-identical.

---

## §7 — Documentation

- `docs/agent.md` — note that Composers exist; full doc lands in M5 (`docs/composition.md`).
- `docs/api-reference.md` — list new symbols.
- `docs/multi-agent.md` — update to mention Composers as the v1.0 way to build topologies; legacy `sub_agents` still works.
- Internal: update `02-open-questions.md` Q3, Q8, Q9, Q10.

---

## §8 — Checklist (Definition of Done)

### Pre-flight
- [ ] Read `00-integration-contract.md`.
- [ ] Read `01-implementation-plan.md`.
- [ ] Read `02-open-questions.md` (Q3, Q8, Q9, Q10).
- [ ] Read **M1 + M2 + M3 dev-logs** (mandatory).
- [ ] Read project memory.
- [ ] Inspect existing `compose/agent_as_tool.py` to understand the primitive.

### Code
- [ ] `compose/topology.py` — protocol + base + error.
- [ ] `compose/builtins.py` — 9 topology builders.
- [ ] `compose/_runtime.py` — 9 internal drivers.
- [ ] `compose/__init__.py` — re-exports.
- [ ] `__init__.py` exports updated.
- [ ] `parent_episode_id` correctly propagated to all sub-Runs.
- [ ] Cancellation propagated to sub-Runs with 30s default timeout.
- [ ] Composers nest arbitrarily.

### Tests
- [ ] All 11 test files in `tests/test_compose/v1/` pass.
- [ ] Legacy `tests/test_compose/` tests pass unchanged.

### Example
- [ ] (No new example in M4.)
- [ ] `examples/14_multi_agent/` runs byte-identical.

### Docs
- [ ] `docs/agent.md` updated.
- [ ] `docs/api-reference.md` updated.
- [ ] `docs/multi-agent.md` updated.

### Style gates
- [ ] `uv run ruff check src/` clean.
- [ ] `uv run mypy src/` strict pass.

### Implementation log
- [ ] `docs/internal/v1-refactor/logs/milestone-4-dev-log.md` written.

### CHANGELOG
- [ ] `## [1.0.0-M4] — Unreleased` entry.

### Open questions to resolve
- [ ] Q3 (Composer.cancel timeout) — decided.
- [ ] Q8 (parallel reducer signature) — decided.
- [ ] Q9 (conditional router signature) — decided.
- [ ] Q10 (saga compensation idempotency) — decided (user responsibility, documented).

### Invariant checks (gate)
- [ ] `git diff src/t0_agents/mcp/` → empty.
- [ ] 18 examples run byte-identical.
- [ ] All pre-existing tests pass unchanged.

---

## §9 — Acceptance criteria

- `from t0_agents import sequential, parallel, race, consensus, map_reduce, conditional, saga, retry_chain, circuit_breaker, Composable, Composer` works.
- `sequential(a, b, c).invoke(input)` returns a `Run[T]`.
- Composers nest: `sequential(race(a, b), conditional(routes={...}))` works.
- `await root_run.cancel()` propagates to all sub-Runs in cascade.
- ClickHouse query (manually verified at M5 smoke): `SELECT * FROM ChatInference WHERE parent_episode_id = '<root>'` shows full hierarchy.

---

## §10 — Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Cancellation incomplete leaves resources hanging | Explicit `test_cancellation.py` with asyncio task tree inspection; 30s timeout with diagnostic log on overshoot. |
| `sequential` chain semantics confusing for dict→str inputs | Document default chain explicitly; provide examples in `docs/composition.md` (M5). |
| `circuit_breaker` state shared across invocations of the same `Composer` instance | Document explicitly: state is per-Composer-instance, not per-call.  Users can build a per-tenant breaker by constructing one per tenant. |
| `parent_episode_id` not threaded correctly under deep nesting | Test with 3+ levels; verify ClickHouse via integration test (or manual smoke at M5). |
| Reducer / merge user code raises mid-aggregation | Wrap in try/except; surface as `ComposerError(node=..., cause=...)` with the topology context. |
| Recursive composers (composer A invokes composer A) — infinite loop | No detection in M4; documented as user responsibility.  Future: add depth check. |

---

## §11 — Estimated surface

- `compose/topology.py`: ~120 LoC.
- `compose/builtins.py`: ~400 LoC (9 factories + signatures).
- `compose/_runtime.py`: ~600 LoC (9 drivers).
- `compose/__init__.py`: ~+20 LoC.
- `__init__.py`: ~+15 LoC.
- `tests/test_compose/v1/`: ~1100 LoC across 11 files.
- Docs delta: ~40 lines.

**Total: ~+1200 LoC src + ~1100 LoC tests.  Estimated 5 days full-time.**
