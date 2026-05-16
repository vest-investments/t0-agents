# t0-agents v1.0 — Implementation Plan

> **Status:** Authoritative roadmap.  Read alongside `00-integration-contract.md`.
> **Read order during `/v1-milestone <N>`:** contract → THIS file → open questions → prior dev-logs → milestone spec.

---

## §1 — Milestone table

| #  | Title                                  | Deps    | Estimated surface             |
|----|----------------------------------------|---------|-------------------------------|
| M1 | `Run` foundational                     | —       | 2 new files, ~600 LoC + tests |
| M2 | `Middleware` extension surface         | M1      | 3 new files, ~900 LoC + tests |
| M3 | `OutputMode[T]` polymorphism           | M1, M2  | 6 new files, ~800 LoC + tests |
| M4 | `Composer` + topologies                | M1, M2, M3 | 3 new files, ~1200 LoC + tests |
| M5 | Industrial middleware + examples + docs + polish | M1–M4 | 1 new file + 4 examples + 5 docs |

**Dependency DAG:**

```
M1 (Run) ────► M2 (Middleware) ────► M3 (OutputMode) ────► M4 (Composer) ────► M5 (polish)
                                              │                  ▲
                                              └──────────────────┘
                                              (M4 reuses OutputMode in topology terminals)
```

The dependency column is **advisory** — the user may choose to start out of order.  The `/v1-milestone` command does not gate on prior-log existence, but it does require reading every prior log that exists before planning the current milestone (always-write policy).

---

## §2 — Milestone summaries

### M1 — `Run` foundational

Introduce `Run[T]` as the first-class invocation handle.  `Agent.invoke(input) → Run[T]` becomes the **canonical** entry point.  `agent.run()` and `agent.stream()` are rewritten as deprecated shims emitting `DeprecationWarning` (once per source line) and delegating to `invoke()`.  Shim stdout output is byte-identical to v0.12.

**Why first:** the other three milestones return or consume `Run`.  Without it as a stable public type, M2/M3/M4 cannot be specified cleanly.

**New symbols:** `Run`, `RunInput`, `RunState`, `RunCancelledError`, `RunFailedError`.

### M2 — `Middleware` extension surface

Replace hardcoded cross-cutting concerns with a composable Starlette-style chain.  `core/agent.py` drops from ~1413 to ~250 LoC.  `AgentConfig.__post_init__` materialises the default chain from existing kwargs to preserve compat.

**Why second:** Composer (M4) reuses middleware per sub-Run.  OutputMode (M3) consults the chain for the retry middleware.

**New symbols:** `Middleware`, `RunContext`, `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`.

### M3 — `OutputMode[T]` polymorphism

Collapse the chat/JSON dichotomy into a polymorphic protocol.  `AgentResult` becomes `AgentResult[T]` generic with `value: T` and `@property` aliases for v0.12 fields.  The ~80 lines of cross-field validation in `config.py` collapse to ~10.

**Why third:** Composer (M4) needs `AgentResult[T]` to be clean — `parallel` and `consensus` reducers operate on typed values.

**New symbols:** `OutputMode`, `ChatOutput`, `JsonOutput`, `StructuredOutput`, `ToolCallOutput`, `StreamingOutput`, `OutputDiagnostics`, `InferenceRequest`, `Attempt`, `ParseFailure`, `Recovery`.

### M4 — `Composer` + topologies

Introduce the fourth abstraction: topologies of Runs.  9 built-in topology builders.  Industrial properties: cancellation propagation, audit via `parent_episode_id`, isolation via per-sub-Run middleware chains.

**Why fourth:** assembles M1+M2+M3 into the composition primitive.

**New symbols:** `Composable`, `Composer`, `sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`, `ComposerError`.

### M5 — Industrial middleware + examples + docs + polish

Industrial middleware (Bulkhead, CircuitBreaker, TenantIsolation, PiiRedactor, RedisRateLimit), 4 new examples (19–22), 4 new docs (run, middleware, output-modes, composition), migration guide, CHANGELOG consolidation, optional version bump to 1.0.0.

**Why last:** consolidates the four abstractions into a coherent narrative for users.

**New symbols:** `BulkheadMiddleware`, `CircuitBreakerMiddleware`, `TenantIsolationMiddleware`, `PiiRedactorMiddleware`, `RedisRateLimitMiddleware`.

---

## §3 — Cross-cutting checklist (every milestone applies)

Before declaring a milestone done, every item below must hold.  These are in addition to the milestone-specific Definition of Done in the spec.

### §3.1 — Code

- [ ] Public API surface in `src/t0_agents/__init__.py` updated; new symbols listed in `__all__`.
- [ ] `mypy --strict` passes on `src/`.  No new `Any` introduced.
- [ ] `ruff check src/` passes with rules `E, F, I, N, UP, RUF` and line length 100.
- [ ] All new code is async-first.  No synchronous wrappers.
- [ ] Python ≥ 3.11 features only.  Targets 3.11/3.12/3.13.

### §3.2 — Tests

- [ ] Every new module has a unit test file under `tests/`.
- [ ] `uv run pytest -m "not integration"` 100% green.
- [ ] `uv run pytest -m integration` 100% green with `docker compose up -d`.
- [ ] Existing tests under `tests/test_mcp/`, `tests/test_core/`, `tests/test_stream/`, `tests/test_serve/`, `tests/test_compose/`, `tests/test_guardrails/`, `tests/test_memory/`, `tests/test_cli/`, `tests/test_testing/` stay green **unchanged** (except when a test is intentionally updated for new behaviour — recorded in dev-log §2).

### §3.3 — Examples

- [ ] All 17 chat-mode examples (`examples/00..16, 18`) run byte-identical on stdout (stderr may include `DeprecationWarning` from M1 shims).
- [ ] The canonical JSON example (`examples/17_structured/`) is migrated as part of M3; from M3 onward it uses `output=JsonOutput(...)`.
- [ ] Any new example for the milestone runs end-to-end.

### §3.4 — Docs

- [ ] If a new public symbol is added, it appears in `docs/api-reference.md` (or its equivalent — verify path before adding).
- [ ] If the symbol introduces a new user-facing capability, a dedicated doc page exists (or is queued for M5).
- [ ] Internal-only files (`docs/internal/v1-refactor/*`) updated when contract / open-question state changes.

### §3.5 — CHANGELOG

- [ ] M1–M4: append a new `## [1.0.0-M<N>] — Unreleased` sub-section under the existing v0.12.0 entry.
- [ ] M5: consolidate the four sub-sections into a single `## [1.0.0] — Unreleased` with Added / Changed / Deprecated (empty) / Removed (empty) / Fixed (empty).

### §3.6 — Invariant checks (gate)

- [ ] `git diff src/t0_agents/mcp/` shows **zero changes**.
- [ ] `mcp/__init__.py` exports unchanged.
- [ ] `run_agent` public signature unchanged.

### §3.7 — Dev-log (always-write)

- [ ] `docs/internal/v1-refactor/logs/milestone-$N-dev-log.md` exists.
- [ ] Has all five sections.
- [ ] §2 (Deviations) may be "No deviations from spec." but §1, §3, §4, §5 must be populated.
- [ ] If §2 documents a deviation that amends `00-integration-contract.md`, the contract is amended in the same commit.

---

## §4 — Estimated effort

- **Solo full-time:** ~3–4 weeks total (M1: 3d, M2: 5d, M3: 5d, M4: 5d, M5: 3d, buffer 2–3d).
- **Solo part-time (50%):** ~6–8 weeks.
- **Single PR per milestone:** recommended.

---

## §5 — Risk register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Compat shim for `agent.run()` diverges sutilmente | M | H | Parallel test suite: old tests against shims, new tests against `invoke`.  Byte-compare stdout on 17 chat-mode examples each milestone; verify stderr contains the expected `DeprecationWarning`. |
| `weakref` to Agent causes surprise GC | L | M | Document explicitly; tests cover the case; `replay()` accepts explicit `agent=`. |
| Default middleware chain materialisation introduces circular imports | M | M | Lazy import inside `__post_init__`. |
| Generic `AgentResult[T]` breaks mypy inference upstream | L | M | Default `Agent` (untyped) = `Agent[str]`.  Document migration. |
| Composer cancellation incomplete leaves resources hanging | M | H | Explicit cleanup tests; aiomonitor-style debugging hooks. |
| `mcp/` accidentally touched | L | H | Hard CI gate: `git diff src/t0_agents/mcp/` must be empty. |
| Industrial deps (Redis, OTel) bloat base install | L | M | Behind extras with `ImportError` guards. |

---

## §6 — Verification end-to-end (final, M5)

After M5 closes the roadmap:

1. Run all 22 examples (18 old + 4 new) — all pass.
2. `uv run pytest -m integration` with docker compose up — all pass.
3. Smoke test PaaS plan (`~/Descargas/prompt-as-a-service-mvp-plan-with-t0-agents.md`) reproducible with v1.0 API — code is cleaner, not noisier.
4. Smoke test industrial Composer (triage tickets):
   - `await root.cancel()` propagates to all sub-Runs.
   - ClickHouse query `WHERE parent_episode_id = '<root>'` reconstructs hierarchy.
   - `CircuitBreakerMiddleware` opens after N consecutive failures.
   - `BulkheadMiddleware` caps concurrency.
5. `docs/migration-v0.12-to-v1.0.md` walked through with a sample v0.12 codebase — every snippet works.

When all five hold, the v1.0 refactor is done.
