# Milestone 2 — Middleware Extension Surface — Dev Log

> **Date closed:** 2026-05-16
> **Audience:** the implementer of M3 (OutputMode) and any future contributor revisiting the v1.0 refactor.
> **Authoritative sources:** `docs/internal/v1-refactor/00-integration-contract.md`, `docs/internal/v1-refactor/milestones/milestone-2-middleware.md`, `docs/internal/v1-refactor/logs/milestone-1-dev-log.md`.

---

## §1 — Summary

### Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `src/t0_agents/core/middleware/base.py` | 261 | `Middleware` Protocol, `Next` type alias, `OnInferenceCallback`, `RunContext` dataclass with `chain_on_inference` / `replace_tool_provider`, `_NextGuard`, `build_pipeline`. |
| `src/t0_agents/core/middleware/builtins.py` | 561 | Six built-in middlewares: `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware` (M2 no-op), `ObservabilityMiddleware`.  Plus `_CumulativeUsage` accumulator helper. |
| `src/t0_agents/core/middleware/__init__.py` | 56 | Re-exports for the submodule. |
| `tests/test_middleware/__init__.py` | 0 | Package marker. |
| `tests/test_middleware/test_chain_order.py` | 198 | `build_pipeline` direct tests: order, empty list, short-circuit, double-call raise, `chain_on_inference` composition, `replace_tool_provider` stack semantics. |
| `tests/test_middleware/test_default_chain.py` | 226 | `AgentConfig.__post_init__` materialisation tests, `middleware=[]` override, `middleware=[Custom()]` override, `default_middleware_for(...)` helper round-trip. |
| `tests/test_middleware/test_compat_v0_12.py` | 209 | End-to-end byte-identity scenarios (bare chat, chat+memory, chat+guardrails, chat+memory+guardrails) using FakeT0 + FakeMCPClient. |
| `tests/test_middleware/test_custom_middleware.py` | 211 | User-authored middleware ergonomics: `PiiLowercaser`, inference observer, asyncio.Semaphore serialiser; protocol duck-type check. |
| `tests/test_middleware/test_tool_provider_order.py` | 175 | Order invariant: `GuardedToolProvider(_OrchestrationToolProvider(client))`.  Stack inspector validates the wrapping. |
| `tests/test_middleware/test_observability.py` | 132 | `ObservabilityMiddleware` span name, tracer=None no-op, layering with other middlewares. |
| `tests/test_middleware/test_cost_and_usage.py` | 200 | `Run.cost()` / `Run.usage()` populated single-inference, no-guardrails, missing-pricing, `middleware=[]` opt-out, multi-turn accumulation, `RunUsage` frozen-dataclass check. |
| `tests/test_middleware/test_pipeline_short_circuit.py` | 121 | Agent-integrated short-circuit + double-call + exception propagation. |

### Files modified

- `src/t0_agents/core/agent.py` — replaced `_execute_chat_legacy` / `_execute_chat_streaming_legacy` / `_execute_json_legacy` (≈1100 LoC inline orchestration) with `_drive(ctx, *, is_streaming, middleware)` + three slim terminals (`_chat_terminal`, `_chat_streaming_terminal`, `_json_terminal`).  Helper methods retained.  Imports trimmed (`T0AgentsFatalError`, `GuardrailLogEntry`, `_make_cost_tracker`, `_make_json_cost_tracker`, `bind_parent_episode_id`, `_OrchestrationToolProvider` no longer imported from agent.py; the corresponding middlewares import them).  Net: ~1670 → 900 LoC.
- `src/t0_agents/core/config.py` — new `middleware: list[Middleware] | None = None` field.  `__post_init__` extended to call `_build_default_chain()` when `middleware is None`.  `default_middleware_for(...)` static method added with lazy import to dodge cycles.  Net: +103 LoC.
- `src/t0_agents/core/run.py` — `_drive()` builds a `RunContext` via the new `_build_run_context(agent, *, is_streaming)` helper and dispatches to `agent._drive(ctx, is_streaming=..., middleware=self._middleware)`.  Cost/usage placeholders rewired: `_set_cost` / `_set_usage` private setters added; `Run.cost()` / `Run.usage()` now read from new private fields `_cost: float` / `_usage: RunUsage | None`.
- `src/t0_agents/core/types.py` — added `RunUsage` frozen dataclass (input_tokens, output_tokens, cost).
- `src/t0_agents/__init__.py` — exported 8 new symbols: `Middleware`, `RunContext`, `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`.  Plus `RunUsage`.
- `src/t0_agents/compose/delegation.py:69` — `sub_agent.run(...)` → `sub_agent.invoke(...).result()`.
- `src/t0_agents/compose/orchestrator.py:294` — same migration.
- `src/t0_agents/serve/routes.py:62` — `agent.run(...)` → `agent.invoke(...).result()`.
- `src/t0_agents/serve/streaming_routes.py:180, 290` — `agent.stream(...)` → `agent.invoke(...).events()`.
- `tests/test_compose/test_delegation.py` — replaced `sub.run = AsyncMock(...)` with `sub.invoke = _mock_invoke_returning(result)` / `_mock_invoke_raising(exc)` helpers (added inline).  4 mock sites migrated.
- `tests/test_compose/test_orchestrator.py` — replaced `sub.run = AsyncMock(...)` with `sub.invoke = MagicMock(return_value=mock_run)` pattern.  Updated `TestParentEpisodeIdRefresh` "wrapper-not-installed" tests to reflect v1.0 always-on `CostTrackingMiddleware` (assertion replaced with behavioural check on `get_parent_episode_id()`).  9 sites migrated.
- `tests/test_compose/test_integration.py:132` — `_SpyAgent` subclass migrated from overriding `run` to overriding the new `_chat_terminal` (recommended v1.0 extension point per M1 D-2 carry-forward).
- `tests/test_serve/test_chat.py` — 5 subclass-override tests migrated from `async def run` to `def invoke` returning a Run-shaped MagicMock.  Helper `_mock_run_raising` / `_mock_run_returning` added inline.
- `tests/test_serve/test_session.py` — `CapturingAgent` subclass migrated to override `invoke`.
- `CHANGELOG.md` — `## [1.0.0-M2] — Unreleased` block.
- `docs/agent.md` — paragraph announcing the v1.0 middleware preview.
- `docs/api-reference.md` — new `Middleware` and `RunUsage` sections (~110 lines).
- `docs/internal/v1-refactor/02-open-questions.md` — Q13 marked **resolved**.

### Public API delta (added to `t0_agents.__init__.__all__`)

`Middleware`, `RunContext`, `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`, `RunUsage`.

### Tests

- 52 new tests under `tests/test_middleware/` (8 files).
- Existing suite: 697 → 749 non-integration tests passing (+52 net).
- Test migrations across `tests/test_compose/` (14 mock-pattern sites) and `tests/test_serve/` (6 subclass-override sites) — covered by the existing test count.

### Docs

- `docs/agent.md`: 1-paragraph v1.0 middleware preview.
- `docs/api-reference.md`: ~110-line `Middleware` + `RunUsage` section.
- Internal: Q13 resolved with citation to this dev-log.

---

## §2 — Deviations from the milestone spec

### D-1.  Default chain order changed: `SubAgents` precedes `Guardrails`.

**What changed.**  Spec §4 example puts the default chain as `Memory → Guardrails → CostTracking → SubAgents → JsonRetry`.  Implementation uses `Memory → SubAgents → Guardrails → CostTracking → JsonRetry` — `SubAgents` is moved up.

**Why.**  Spec §3.2 states the load-bearing order invariant: *`SubAgentMiddleware` wraps `tool_provider` inside `GuardrailsMiddleware` (guardrails see delegations).*  The v0.12 `_execute_chat_legacy` at lines 664–694 confirms this:

```python
if self._config.sub_agents:                   # Step 3a — INNER wrap.
    tool_provider = _OrchestrationToolProvider(tool_provider, ...)
if self._config.guardrails:                   # Step 3b — OUTER wrap.
    tool_provider = GuardedToolProvider(tool_provider, ...)
```

In v0.12 `GuardedToolProvider` is the OUTER wrap because the guardrail block runs AFTER the sub-agent block.  When the terminal calls `tool_provider.call_tool(name)`, control hits `GuardedToolProvider.call_tool` first (firing `before_tool` hooks) and then `_OrchestrationToolProvider.call_tool` (which either dispatches a sub-agent or forwards to the bare client).  This is what "guardrails see delegations" means: guardrails fire for EVERY tool call including sub-agent invocations.

In the v1.0 middleware model, each middleware contributes one `replace_tool_provider` call in chain order.  The wrap that happens LAST becomes the OUTERMOST (each `replace_tool_provider` stacks on top of the current `ctx.tool_provider`).  For `GuardedToolProvider` to end up as the OUTER wrap, `GuardrailsMiddleware` must run AFTER `SubAgentMiddleware` in chain order.

The spec §4 example is therefore internally inconsistent with the §3.2 invariant.  The invariant wins.

**Impact on subsequent milestones.**  None at the API level — `default_middleware_for(...)` and `_build_default_chain()` are both updated consistently.  M3 and M5 must keep the same ordering rule: any middleware that wraps `tool_provider` and wants its wrap OUTER must run later in chain order than any other middleware that wraps `tool_provider`.

### D-2.  `agent.py` did not shrink to ≤ 300 LoC (acceptance criterion §9).

**What changed.**  Spec §9 acceptance criterion: *"`agent.py` is < 300 LoC."*  Final size: 900 LoC.

**Why.**  Two contributors to the residual weight:

1. **`Agent.invoke` docstring** — ~140 lines of inline kwarg documentation copied verbatim from M1.  Removing the docstring would shrink the file ~140 LoC but degrade the in-source documentation surface significantly.
2. **`_json_terminal` body** — ~85 LoC of retry-loop code (lifted verbatim from `_execute_json_legacy:1294-1395`).  In M2 the `JsonRetryMiddleware` is a no-op placeholder; the retry logic must live somewhere, and the terminal is the natural place until M3 migrates it.

In M3, when `JsonRetryMiddleware` activates and consults `OutputMode.recover()`:
- The retry loop moves into the middleware.
- `_json_terminal` reduces to ~10 LoC (single `run_structured_inference` call).
- `agent.py` shrinks by another ~75 LoC to ~825 LoC.

To hit ≤ 300 LoC the `Agent.invoke` docstring would still need to move (e.g. into `docs/run.md`).  This is a M5 polish item.

**Impact on M3+.**  M3 should expect `agent.py` to start at 900 LoC and drop by ~75 once the JSON retry loop moves to middleware.  M5 should consider relocating the long `Agent.invoke` docstring.

### D-3.  `RunUsage` added to public API (not in spec symbol list).

**What changed.**  Spec §3.3 mentions `Usage` informally on `Run.usage() -> Usage` but never defines it.  Implementation adds `RunUsage` (frozen dataclass with `input_tokens`, `output_tokens`, `cost`) as a new public type in `core/types.py`.

**Why.**  The user resolved M1 D-5 carry-forward by asking M2 to wire `Run.cost()` / `Run.usage()` to real data.  `Run.cost() -> float` is straightforward; `Run.usage() -> ?` needed a return type.  Options considered:

- (A) Return `tensorzero.types.Usage` — but it doesn't expose `cost`, only tokens.
- (B) Return `dict[str, Any]` — uglier for callers that want typed access.
- (C) Define `RunUsage` (frozen dataclass) — typed, exported, additive surface — chosen.

**Impact on subsequent milestones.**  M5 documents `RunUsage` in `docs/run.md` alongside the `Run` class.  No other milestone needs to interact with it directly — it's a leaf type.

### D-4.  `Run.cost()` / `Run.usage()` placeholders disabled when `middleware=[]`.

**What changed.**  Spec is silent on the interaction between `middleware=[]` (explicit "no middleware") and `Run.cost()` / `Run.usage()`.  Implementation: when `middleware=[]`, `CostTrackingMiddleware` does not run, so `Run.cost()` stays at `0.0` and `Run.usage()` stays at `None`.

**Why.**  `middleware=[]` is documented as the explicit "no middleware" override — power users opt out of EVERY layer including cost tracking.  Auto-enabling cost tracking even when the user said "no middleware" would silently violate the override contract.

**Impact on subsequent milestones.**  M5's `docs/middleware.md` should explicitly call out this trade-off so users know what they're giving up when they set `middleware=[]`.

### D-5.  Test migration patterns recorded (M1 D-2 carry-forward).

**What changed.**  M1 D-2 noted that ~15 tests would need migration in M2: `tests/test_compose/test_delegation.py` (`sub.run = AsyncMock(...)`), `tests/test_compose/test_orchestrator.py` (same), `tests/test_serve/test_chat.py` (subclass-override `async def run`), `tests/test_serve/test_session.py` (same), `tests/test_compose/test_integration.py:140` (`super().run(...)`).  All migrated.  Documented here for the record:

- **Mock pattern** (compose/): `sub.run = AsyncMock(return_value=result)` → `sub.invoke = MagicMock(return_value=mock_run)` where `mock_run.result = AsyncMock(return_value=result)`.  Helpers `_mock_invoke_returning` / `_mock_invoke_raising` are duplicated inline in each test file (rather than a shared `conftest.py`) to keep each test file self-contained.
- **Subclass-override pattern** (serve/): `class FailingAgent(Agent): async def run(...)` → `def invoke(...)` returning a Run-shaped MagicMock.  For tests that observe the inner terminal (e.g. `tests/test_compose/test_integration.py::_SpyAgent`), override the new `_chat_terminal` method directly — this is the recommended v1.0 extension point.

**Impact on subsequent milestones.**  M3's `OutputMode` refactor changes the JSON path; any tests that currently mock `agent.run` for JSON agents should follow the same pattern.

### D-6.  Internal call-site migration (M1 carry-forward 4) — 4 sites in scope, all migrated.

**What changed.**  Per M1 dev-log §4 carry-forward 4, four internal call-sites still emitted `DeprecationWarning` because they called `agent.run(...)` / `agent.stream(...)` internally.  All migrated in M2: `compose/delegation.py:69`, `compose/orchestrator.py:294`, `serve/routes.py:62`, `serve/streaming_routes.py` (SSE + WS handlers at lines 180 and 290).

**Why.**  The user opted for the "migrate all 4 + ~15 tests" scope (3-question interview, answer 1).  Aligns with M2 already touching `compose/orchestrator.py` via `SubAgentMiddleware`.

**Impact on subsequent milestones.**  M3 has no remaining internal call-sites to migrate.  M5 should verify zero `DeprecationWarning` emissions from library code during the final test run (the user code's `agent.run` / `agent.stream` shim warnings continue to be emitted as documented).

### D-7.  `system_arguments={}` collapses to "no key" in chat terminals.

**What changed.**  v0.12 `_merge_caller_system_arguments` collapsed an empty merged dict to "no `system_arguments` key" so the gateway never received `input.system={}`.  Initial implementation forwarded the empty dict, breaking `tests/test_core/test_agent.py::test_run_system_arguments_empty_dict_no_op`.  Fixed in `_chat_terminal` and `_chat_streaming_terminal` with `if ctx.system_arguments:` (truthy check — `None` and `{}` both skip).  `_json_terminal` uses `system_arguments=ctx.system_arguments or None`.

**Why.**  Byte-identity invariant §2.3 requires the chat path to produce the same `t0.inference` input shape as v0.12.  An empty dict is structurally different from "no key", so the collapse must happen at the terminal boundary.

**Impact on subsequent milestones.**  M3's JSON terminal (which becomes a single-attempt call after `JsonRetryMiddleware` activates) must preserve the same `or None` collapse.

### D-8.  `GuardrailsMiddleware` runs `before_run` INSIDE the try/finally.

**What changed.**  Initial implementation put the `before_run` loop OUTSIDE the `try:` block wrapping `next_(ctx)`.  This broke `tests/test_guardrails/test_agent_with_guardrails.py::TestAfterRunWithFatalAbort::test_after_run_called_when_before_run_raises_fatal` — when `before_run` raised `T0AgentsFatalError`, the `after_run` `finally` block never ran.

**Why.**  v0.12 `_execute_chat_legacy:706-806` wraps BOTH `before_run` AND `run_agent` in a single `try:` so `after_run` fires once around the entire run-level lifecycle, including the fatal-abort-in-before_run edge case.  M2 must preserve this — the `docs/guardrails.md` contract is "after_run fires once around every run".  Fix: moved the `before_run` loop inside the same `try:` block as `next_(ctx)`.

**Impact on subsequent milestones.**  None — the fix is contained in `GuardrailsMiddleware.__call__`.

### D-9.  `TestParentEpisodeIdRefresh` tests updated for always-on `on_inference`.

**What changed.**  Two tests in `tests/test_compose/test_orchestrator.py` asserted `captured == [None]` for the `on_inference` kwarg when no guardrails were configured.  These were testing the v0.12 invariant "no observer wires up `on_inference` unless explicitly configured".  In v1.0 with always-on `CostTrackingMiddleware`, `on_inference` is always set (never `None`).

**Why.**  The intent of the original tests was "verify the parent_episode_id refresh wrapper is NOT installed when episode_id is provided".  v1.0 replaces the assertion shape: invoke the on_inference with a fake response and verify `get_parent_episode_id()` does NOT change (because no refresh wrapper was installed, even though the cost tracker callback exists).

**Impact on subsequent milestones.**  None — the test fix is local.

---

## §3 — Open questions raised or resolved

### Q13 — `JsonRetryMiddleware` placement in default chain → **innermost**

**Decision (citing the spec recommendation).**  `JsonRetryMiddleware` is the last entry in the default chain, just before the terminal.  Concrete chain shape after `__post_init__`:

```
Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal
```

In M2 `JsonRetryMiddleware.__call__` is a no-op pass-through (`return await next_(ctx)`).  The retry loop logic stays inline in `_json_terminal` until M3 activates the middleware.

When M3 activates `JsonRetryMiddleware`:
1. The middleware body inverts.  It calls `next_(ctx)` and inspects the result against `ctx.output.recover(...)`.
2. If recovery returns `Recovery.retry(hint)`, the middleware re-emits `next_(ctx)` with a corrective user-turn message.
3. `_json_terminal` reduces to a single `run_structured_inference` call.

Audit of retries surfaces via TensorZero tags (`retry_attempt=N`) and the M3 `result.diagnostics.attempts` field — NOT via outer middleware layers.  Memory and guardrails see only the final successful inference, keeping their semantic boundary clean.

Updated `02-open-questions.md` Q13 to `Status: resolved`.

### No new questions raised at M2.

The interview at the start of M2 (3 questions: carry-forward 4 scope, Q13, Run.cost/usage wiring) closed every ambiguity that surfaced during planning.  Implementation surfaced D-1 (spec example error) but the fix was forced by the §3.2 invariant — no question to raise.

---

## §4 — Carry-forward notes for subsequent milestones

### For M3 (OutputMode)

1. **`JsonRetryMiddleware` activation.**  The middleware exists as a no-op placeholder at `core/middleware/builtins.py:JsonRetryMiddleware`.  M3 should:
   - Move the retry loop body from `Agent._json_terminal` (currently ~85 LoC inline) into `JsonRetryMiddleware.__call__`.
   - Replace `_json_terminal` with a single-attempt body (~10 LoC).
   - Consume `ctx.output.recover(...)` to drive the retry — the M2 `_build_retry_hint` helper in `core/agent.py` moves into `JsonOutput.recover()` and becomes one of the recovery hint sources.
   - Update `_RETRY_HINT_BASE` / `_build_retry_hint` cite to point at the new `JsonOutput` location.

2. **Remove M1-deprecated `invoke()` kwargs.**  `output_schema`, `json_mode`, `max_json_retries` are still on `Agent.invoke()` and `RunInput`.  Drop them in M3 alongside the corresponding `AgentConfig` fields (Tier 2 removal).  Update `Run.replay()` / `Run.fork()` to stop threading them.  Delete `tests/test_run/test_compat_run_stream.py::TestLegacyKwargsForwarded`.

3. **`Run[T]` becomes generic.**  M2 keeps `T` as a phantom type parameter on `Run`.  M3 makes `AgentResult[T]` generic and `Run.result() -> AgentResult[T]`.  The `Run[T]` type narrowing flows from the `output=` kwarg at `Agent.invoke(...)` time.

4. **Migrate `examples/17_structured/`.**  The example imports `run_structured_inference` directly; M3 should migrate to `Agent(...).invoke(..., output=JsonOutput(...))`.

5. **JSON `events()` raises.**  M1's gate on `function_kind="json"` at `Run.events()` (raises `ConfigurationError`) can be relaxed in M3 when `StreamingOutput(base=JsonOutput(...))` becomes possible — gate on `output.supports_streaming` instead.

6. **`Run.cost()` / `Run.usage()` semantics with retries.**  In M3 with JsonRetry-activated, every retry attempt increments `CostTrackingMiddleware`'s accumulator (because each attempt fires `on_inference`).  This means a JSON run with 2 retries reports cumulative cost of 3 inferences — same as v0.12 (which also charged for every JSON retry inference).  M3 should NOT change this; the cost is the user's economic reality.

7. **`ctx.output` field.**  `RunContext.output` is currently `Any` (M2 placeholder).  M3 types it as `OutputMode[T] | None`.

### For M4 (Composer)

1. **Composer middleware composition.**  Each sub-Run in a Composer (sequential, parallel, race, ...) runs with its own middleware chain materialised from its own `AgentConfig`.  Composers should not inject middleware into sub-runs — the user composes that explicitly.

2. **Custom middleware introspection.**  `ctx.extras["_tool_provider_stack"]` is the canonical audit trail.  Composers that need to verify "all sub-agents ran with guardrails" can introspect `Run.context["middleware_names"]` (TODO: M4 adds a snapshot at Run start).  For M2 there is no such snapshot — only the inspector pattern shown in `tests/test_middleware/test_tool_provider_order.py`.

3. **Cancellation through middleware.**  When `await root_run.cancel()` propagates through Composers to sub-Runs, each sub-Run's middleware chain unwinds.  `MemoryMiddleware.save` (in `finally`) and `GuardrailsMiddleware.after_run(result=None)` (in `finally`) fire as expected.  M4 cancellation tests should verify this.

### For M5 (Industrial + docs)

1. **`docs/middleware.md`.**  The canonical user-facing doc for the middleware chain.  Promote the M2 `docs/api-reference.md` "Middleware" section into a dedicated page.  Document:
   - The Starlette-style protocol.
   - The default chain (with the post-D-1 ordering: `Memory → SubAgents → Guardrails → CostTracking → JsonRetry`).
   - Custom middleware authoring (PII redactor, rate limiter, observability).
   - The `middleware=[]` opt-out — explicitly note that `Run.cost()` stops working in this mode (per D-4).

2. **`Agent.invoke` docstring relocation.**  Per D-2, the long docstring keeps `agent.py` ≥ 900 LoC.  Move it to `docs/run.md` and slim the in-source docstring to ~20 lines.  Brings `agent.py` toward the ≤ 300 LoC acceptance criterion (the JSON terminal body in M3 also helps).

3. **Industrial middleware.**  `BulkheadMiddleware`, `CircuitBreakerMiddleware`, `TenantIsolationMiddleware`, `PiiRedactorMiddleware`, `RedisRateLimitMiddleware` extend the M2 surface.  They follow the same protocol; ship under appropriate extras (`[industrial]`, `[ratelimit]`).

4. **CHANGELOG consolidation.**  The M2 `## [1.0.0-M2] — Unreleased` block folds into the single `## [1.0.0] — Unreleased` consolidated entry at M5.  M2's "Added" + "Changed" + "Validated" + "Open questions resolved" sub-sections should be reorganised under the M5 consolidated `## [1.0.0]` Added/Changed/Deprecated/Removed/Fixed sub-sections.

5. **Migration guide example.**  `docs/migration-v0.12-to-v1.0.md` (M5) should include a one-liner showing how a user adds a custom middleware to an existing v0.12 codebase:

```python
# v0.12: forking core/agent.py to inject a PII redactor.
# v1.0:
agent = Agent(
    AgentConfig(
        name="...",
        function_name="...",
        memory=...,
        guardrails=[...],
        middleware=[
            PiiRedactor(),  # outermost — runs before memory/guardrails
            *AgentConfig.default_middleware_for(memory=m, guardrails=g),
        ],
    ),
    ...,
)
```

---

## §5 — Verification

Commands run and outcomes:

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/ tests/test_middleware/ tests/test_compose/ tests/test_serve/` | **passed** — all checks passed (M2-touched files only; 6 pre-existing ruff errors in `tests/test_cli/`, `tests/test_core/test_agent_integration.py`, `tests/test_core/test_agent_parity.py`, `tests/test_memory/test_agent_memory.py`, `tests/test_memory/test_agent_memory_integration.py` are pre-M2 and not introduced by this milestone). |
| `uv run mypy src/` | **passed** — no issues found in 60 source files. |
| `uv run pytest -m "not integration"` | **passed** — 749 passed, 2 skipped, 57 deselected, 128 deprecation warnings (down from 151 in M1 because internal call-sites no longer emit shims). |
| `git diff src/t0_agents/mcp/` | **empty** — invariant §2.1 holds. |
| `python -c "from t0_agents import Middleware, RunContext, MemoryMiddleware, GuardrailsMiddleware, CostTrackingMiddleware, SubAgentMiddleware, JsonRetryMiddleware, ObservabilityMiddleware, RunUsage"` | **passed** — all 9 new public symbols import cleanly. |
| `python -c "AgentConfig(name='x', function_name='y').middleware"` | **passed** — returns `[CostTrackingMiddleware, JsonRetryMiddleware]` (default chain materialised). |
| `AgentConfig(middleware=[]).middleware == []` | **passed** — explicit override preserved. |

User-side verification (pending — deferred to user):

- `uv run pytest -m integration` (with `docker compose up -d` + `GROQ_API_KEY` in `.env`) — to confirm the 57 integration tests still pass with the middleware chain wiring.
- 18-examples byte-identity smoke (`examples/00..16, 18`) — Tier 1 byte-identity invariant.  Subprocess capture of stdout against a v0.12 baseline.
- `examples/17_structured/` byte-identity — runs via `run_structured_inference` directly until M3 migrates it.

Numbers for the regression baseline:

- Pre-M2 non-integration test count: 697.
- Post-M2 non-integration test count: 749 (+52 tests under `tests/test_middleware/`).
- New code: ~1670 lines across `core/middleware/{base,builtins,__init__}.py`, `core/run.py` cost-setters, `core/types.py` RunUsage, `core/config.py` middleware field, `core/agent.py` `_drive` + terminals.
- Deleted code: ~1100 LoC inline orchestration from `core/agent.py` (the three `_execute_*_legacy` bodies).
- Modified code: `core/agent.py` shrank from 1670 → 900 LoC.  Net repo size change: ~+1700 src, ~+1500 tests, ~+130 docs.

LoC distribution of new middleware surface:
- `core/middleware/base.py`: 261 LoC.
- `core/middleware/builtins.py`: 561 LoC.
- `core/middleware/__init__.py`: 56 LoC.
- Total `core/middleware/`: 878 LoC.
