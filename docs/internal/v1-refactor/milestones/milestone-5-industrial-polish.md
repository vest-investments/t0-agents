# Milestone 5 — Industrial Middleware, Examples, Docs, Polish

> **Roadmap:** v1.0 four-abstraction refactor.
> **Deps:** M1, M2, M3, M4 — all four abstractions exist.
> **Dev-log:** `docs/internal/v1-refactor/logs/milestone-5-dev-log.md` (always-write at close).

---

## §1 — Goal

Close the v1.0 roadmap.  Ship 5 industrial middleware (Bulkhead, CircuitBreaker, TenantIsolation, PiiRedactor, RedisRateLimit) that complete the "industrial/military-grade composition" promise.  Add 4 new examples (19–22), 4 new docs (run, middleware, output-modes, composition), migration guide.  Consolidate CHANGELOG to a single `## [1.0.0]` entry.  Optionally bump version to 1.0.0 (user-approved).

Smoke test the PaaS plan and a real industrial topology to verify v1.0 is ready for release.

---

## §2 — Scope

### In scope
- `middleware/industrial.py` — 5 industrial middlewares.
- `pyproject.toml` — new extras: `[industrial]`, `[ratelimit]`, `[otel]`.
- `examples/19_middleware_custom/` — PII redactor + Redis rate-limit.
- `examples/20_run_compose/` — replay + fork + two-channel.
- `examples/21_output_modes/` — Pydantic + custom YAML mode.
- `examples/22_industrial_topology/` — triage ticket pipeline.
- `docs/run.md`, `docs/middleware.md`, `docs/output-modes.md`, `docs/composition.md` — new public docs.
- `docs/migration-v0.12-to-v1.0.md` — migration guide.
- `README.md` — feature matrix + hello-world updated.
- `docs/agent.md`, `docs/index.md` — cross-references.
- `CHANGELOG.md` — consolidate M1–M4 entries into `## [1.0.0]`.
- Smoke tests: PaaS plan + industrial Composer.

### Out of scope
- New abstractions beyond M1–M4.
- Cross-process distribution.
- Performance optimisations beyond what naturally falls out.
- Deprecation warnings (none in v1.0 — every old API still works).

### Non-goals
- Building the PaaS itself.  Only smoke-testing that the plan is reproducible with v1.0.
- 100% test coverage of every middleware combination — the new middlewares are tested individually; combinatorial coverage is best-effort.

---

## §3 — Constraints

Inherited from `00-integration-contract.md`:
- §2.1 `mcp/` standalone — industrial middleware lives in `middleware/`, never imports from `mcp/`.
- §2.2 TensorZero-first — `TenantIsolationMiddleware` enforces TZ tags conventions, doesn't invent a separate tenant system.
- §2.3 Chat byte-identity — all 18 + 4 new examples run.
- §2.4 Additive compat — every v0.12 API still works.
- §2.6 Opt-in per layer — industrial middleware that pulls deps goes behind extras.

Milestone-specific:
- Each industrial middleware with heavy dependencies (Redis, OTel) hidden behind `pyproject.toml` extra + `ImportError` guard at import time.
- Bulkhead + CircuitBreaker (pure-stdlib) ship in base (Q4 resolution).
- `[ratelimit]` extra pulls `redis` (Q5 resolution).
- `[otel]` extra pulls `opentelemetry-api` + `opentelemetry-sdk` (Q6 resolution).
- The migration guide must walk through real v0.12 snippets and show that they continue to work; explicitly document the new patterns.

---

## §4 — Module additions

### New: `src/t0_agents/middleware/industrial.py`

5 middlewares, each ~100 LoC:

```python
class BulkheadMiddleware:
    """Bound concurrent runs to `max_concurrent` via asyncio.Semaphore.

    Base install (no extras).  Q4 resolution.
    """
    name = "bulkhead"
    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
    async def __call__(self, ctx, next_):
        async with self._sem:
            return await next_(ctx)

class CircuitBreakerMiddleware:
    """Open the circuit after N consecutive failures; reset after timeout.

    Base install.
    """
    name = "circuit_breaker"
    def __init__(self, threshold: int = 5, reset_timeout: float = 30.0) -> None: ...
    async def __call__(self, ctx, next_):
        if self._state == "open" and time.monotonic() - self._opened_at < self._reset_timeout:
            raise T0AgentsFatalError(f"Circuit breaker open ({self._consecutive_failures} consecutive failures)")
        try:
            result = await next_(ctx)
            self._consecutive_failures = 0
            self._state = "closed"
            return result
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
            raise

class TenantIsolationMiddleware:
    """Enforce that ctx.tags contains a tenant_id, and that memory operations are scoped.

    Base install.
    """
    name = "tenant_isolation"
    def __init__(self, tenant_tag: str = "tenant_id") -> None: ...
    async def __call__(self, ctx, next_):
        if not ctx.tags or self._tenant_tag not in ctx.tags:
            raise ConfigurationError(f"TenantIsolationMiddleware requires ctx.tags['{self._tenant_tag}']")
        # Optionally scope memory provider by prepending tenant to episode_id
        return await next_(ctx)

class PiiRedactorMiddleware:
    """Redact PII patterns from ctx.user_message before inference.

    Base install (uses stdlib regex).  Caller provides patterns.
    """
    name = "pii_redactor"
    def __init__(self, patterns: list[tuple[re.Pattern[str], str]]) -> None: ...
    async def __call__(self, ctx, next_):
        for pat, repl in self._patterns:
            ctx.user_message = pat.sub(repl, ctx.user_message)
        return await next_(ctx)

class RedisRateLimitMiddleware:
    """Token-bucket rate limit, keyed by a tag value, backed by Redis.

    Extra: `t0-agents[ratelimit]`.  Q5 resolution.
    """
    name = "redis_rate_limit"
    def __init__(self, redis_client, *, per: str, rate: float, burst: int) -> None: ...
    async def __call__(self, ctx, next_):
        key = ctx.tags.get(self._per) if ctx.tags else None
        if key is None:
            raise ConfigurationError(f"RedisRateLimitMiddleware requires ctx.tags['{self._per}']")
        # Lua-script token bucket; raises if exhausted
        await self._consume(key)
        return await next_(ctx)
```

### Modified: `src/t0_agents/__init__.py`

Add exports for the 5 industrial middlewares (with conditional imports for the extra-gated ones).

### Modified: `pyproject.toml`

```toml
[project.optional-dependencies]
# ... existing extras ...
industrial = []  # nothing extra; Bulkhead + CircuitBreaker + TenantIsolation + PiiRedactor are pure-stdlib
ratelimit = ["redis>=5.0"]
otel = ["opentelemetry-api>=1.20", "opentelemetry-sdk>=1.20"]
```

### New: `examples/19_middleware_custom/`

Self-contained example:
- `main.py` — agent with `PiiRedactorMiddleware` (regex patterns for email, phone) + `RedisRateLimitMiddleware` (`per="user_id"`, rate=10/min).
- `README.md` — narrates the use case.

### New: `examples/20_run_compose/`

- `main.py` — `run = agent.invoke(...)`, demonstrates `await run.result()`, `async for ev in run.events()` (two-channel), `await run.replay(agent=other_agent)`, `await run.fork("Consider scenario B")`.
- `README.md` — explains lifecycle, replay, fork.

### New: `examples/21_output_modes/`

- `main_pydantic.py` — `StructuredOutput[Customer](Customer)`; extracts a Pydantic model from raw text.
- `main_custom_yaml.py` — custom `YamlOutput(OutputMode)` showing third-party extensibility (parses YAML instead of JSON).
- `README.md`.

### New: `examples/22_industrial_topology/`

- `main.py` — full ticket triage pipeline:
  ```python
  topology = sequential(
      classifier_agent,
      conditional(
          router=lambda res: res.value.category,
          routes={
              "technical": sequential(
                  tech_researcher,
                  consensus([senior_a, senior_b, senior_c], merge=majority),
                  writer,
              ),
              "billing": retry_chain(billing_groq, billing_claude),
              "general": race(general_a, general_b),
          },
          default=human_escalation,
      ),
  )
  topology = with_middleware(topology, [
      ObservabilityMiddleware(tracer),
      BulkheadMiddleware(max_concurrent=100),
      CircuitBreakerMiddleware(threshold=5),
      TenantIsolationMiddleware(),
  ])
  run = topology.invoke(ticket)
  async for ev in run.events(): audit_log.write(ev)
  final = await run.result()
  ```
- `README.md` — narrates the industrial pattern.

### New: `docs/run.md`

Public guide:
- Run lifecycle: pending → running → succeeded/failed/cancelled.
- Eager vs lazy.
- Replay vs fork.
- Two-channel consumption.
- Feedback API.
- Code examples drawn from `examples/20_run_compose/`.

### New: `docs/middleware.md`

Public guide:
- The protocol.
- Built-in middlewares (table).
- Order matters (sub-agents inside guardrails).
- Writing custom middleware (pattern + example).
- Industrial middlewares (with extras explained).
- Code examples from `examples/19_middleware_custom/`.

### New: `docs/output-modes.md`

Public guide:
- The protocol.
- Built-in modes (table).
- Writing a custom mode (YAML walkthrough).
- Recovery ladder.
- Code examples from `examples/21_output_modes/`.

### New: `docs/composition.md`

Public guide:
- 9 topologies (table with use cases).
- Nesting Composers.
- Cancellation propagation.
- Audit via `parent_episode_id` + ClickHouse query.
- Industrial properties (cancellation, isolation, audit, backpressure).
- Code examples from `examples/22_industrial_topology/`.

### New: `docs/migration-v0.12-to-v1.0.md`

Migration guide.  Honest, two-tier story:

**Promise:**
- Every v0.12 snippet keeps **functioning** in v1.0.
- Tier 1 (entry points): `agent.run()` and `agent.stream()` emit `DeprecationWarning` but work.  Migrate at your own pace, must be done before v1.1.
- Tier 2 (JSON config kwargs): removed in v1.0 — must migrate immediately.  One-line change per `AgentConfig` construction.
- Tier 0 (everything else): unchanged.

**Sections:**

1. **What changes (and what doesn't).**  TL;DR table.
2. **Migrating `Agent.run()` → `Agent.invoke()`.**  Side-by-side, including the new awaitable `Run` semantics (`result = await agent.invoke(msg)` is the most direct rename).
3. **Migrating `Agent.stream()` → `agent.invoke().events()`.**  Side-by-side.
4. **Migrating JSON config kwargs → `output=OutputMode`.**  Full mapping table:
   - `function_kind="chat"` (default) → omit / `output=ChatOutput()`.
   - `function_kind="json", json_mode="strict", output_schema=S` → `output=JsonOutput(schema=S, mode="strict")`.
   - …
   - Plus the new `StructuredOutput[M]` recipe for Pydantic users.
5. **What's the same.**  `AgentResult.response`, `.parsed`, `.parse_strategy`, etc. — same names, same values.
6. **What's new and opt-in.**  Middleware, Composers, Run replay/fork.  Not required to use; available when you want.
7. **Deprecation timeline.**  v1.0 = warnings on Tier 1.  v1.1 = removal of Tier 1.  Plan accordingly.
8. **PaaS-specific notes.**  Reference the `~/Descargas/prompt-as-a-service-mvp-plan-with-t0-agents.md` plan: the changes needed (mostly one-line renames per route handler).

### Modified: `README.md`

- Hello-world uses `agent.invoke(...)`.
- Feature matrix adds the 4 abstractions.
- Reading path includes the 4 new docs.

### Modified: `docs/agent.md`

Update with cross-references to the 4 new docs; mark obsolete sections.

### Modified: `docs/index.md`

Reading path now includes:
- Run → Middleware → Output Modes → Composition (between existing entries).

### Modified: `CHANGELOG.md`

Consolidate `[1.0.0-M1..M4]` entries into a single `## [1.0.0] — Unreleased`:

```markdown
## [1.0.0] — Unreleased

### Added
- `Run[T]` — first-class invocation handle.  `agent.invoke(input)` returns a `Run` that exposes `result()`, `events()`, `replay()`, `fork()`, `cancel()`, `feedback()`, `transcript()`, `cost()`, `usage()`.
- `Middleware` chain — Starlette-style `async (ctx, next_) -> AgentResult` composition.  6 built-ins: `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`.
- `OutputMode[T]` polymorphism — 5 built-ins: `ChatOutput`, `JsonOutput`, `StructuredOutput[M]` (Pydantic), `ToolCallOutput`, `StreamingOutput`.
- `AgentResult[T]` generic with `value: T`.  v0.12 fields preserved as `@property` aliases.
- `Composer` topologies — 9 builders: `sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`.
- 5 industrial middlewares: `BulkheadMiddleware`, `CircuitBreakerMiddleware`, `TenantIsolationMiddleware`, `PiiRedactorMiddleware`, `RedisRateLimitMiddleware`.
- 4 new examples (19–22) and 4 new docs (run, middleware, output-modes, composition).
- Migration guide (`docs/migration-v0.12-to-v1.0.md`).
- New extras: `[industrial]`, `[ratelimit]`, `[otel]`.

### Changed
- `Agent` internals refactored to middleware pipeline.  `agent.py` drops from ~1413 to ~250 LoC.
- `AgentResult` is now `AgentResult[T]` generic.  Read-side API preserved.

### Deprecated
- `Agent.run(msg, **kw)` — emits `DeprecationWarning` once per source line.  Migrate to `await agent.invoke(msg, **kw)` (Run is awaitable).  Will be removed in 1.1.
- `Agent.stream(msg, **kw)` — emits `DeprecationWarning` once per source line.  Migrate to `async for ev in agent.invoke(msg, **kw).events()`.  Will be removed in 1.1.

### Removed
- `AgentConfig.function_kind` — replaced by `AgentConfig.output: OutputMode = ChatOutput()`.
- `AgentConfig.json_mode` — replaced by `JsonOutput.json_mode` field.
- `AgentConfig.output_schema` — replaced by `JsonOutput.schema` field.
- `AgentConfig.permissive_json_parsing` — replaced by `JsonOutput.permissive` field.
- `AgentConfig.max_json_retries` — replaced by `JsonOutput.max_retries` field.

  Migration: `AgentConfig(function_kind="json", json_mode="strict", output_schema=S)` → `AgentConfig(output=JsonOutput(schema=S, mode="strict"))`.  See migration guide for the full table.

### Fixed
- *(none specific to v1.0; bug fixes shipped under v0.12.x as usual)*
```

---

## §5 — Tests

### `tests/test_middleware/industrial/test_bulkhead.py`
- 10 concurrent invokes with `BulkheadMiddleware(max_concurrent=3)`; verify at most 3 in flight (via inspect on the sem).

### `tests/test_middleware/industrial/test_circuit_breaker.py`
- Threshold=3 failures → state opens; subsequent call raises immediately.
- After `reset_timeout` → state half-opens; success → closed; failure → re-opens.

### `tests/test_middleware/industrial/test_tenant_isolation.py`
- Missing `tenant_id` in tags → `ConfigurationError`.
- Present → run proceeds; verify episode_id is scoped (if scoping enabled).

### `tests/test_middleware/industrial/test_pii_redactor.py`
- Regex pattern for email → user message has redactor applied before inference.

### `tests/test_middleware/industrial/test_redis_rate_limit.py`
- Mocked Redis client (or `fakeredis`).  Rate=2/sec, burst=2; verify 3rd call within a second is rejected.
- Missing `per` tag → `ConfigurationError`.

### Smoke tests (manual or scripted at M5 close)
- `examples/22_industrial_topology/` runs against `docker compose up`:
  - Verify `await root.cancel()` cancels all sub-Runs.
  - Verify ClickHouse `WHERE parent_episode_id = '<root>'` returns the hierarchy.
  - Verify CircuitBreaker opens after N failures.
  - Verify Bulkhead caps concurrency.
- PaaS reproduction: walk through `~/Descargas/prompt-as-a-service-mvp-plan-with-t0-agents.md` snippets; verify each compiles and runs (or maps cleanly to v1.0 patterns).

---

## §6 — Example

The four new examples ARE the deliverable of M5.  Each must run end-to-end.

---

## §7 — Documentation

Listed in §4.  The four new docs are the primary public-facing deliverable.

---

## §8 — Checklist (Definition of Done)

### Pre-flight
- [ ] Read `00-integration-contract.md`.
- [ ] Read `01-implementation-plan.md`.
- [ ] Read `02-open-questions.md` (Q4, Q5, Q6 to be officially resolved here).
- [ ] Read **M1 + M2 + M3 + M4 dev-logs** (mandatory).
- [ ] Read project memory.

### Code
- [ ] `middleware/industrial.py` implements 5 middlewares.
- [ ] `pyproject.toml` extras: `[industrial]`, `[ratelimit]`, `[otel]`.
- [ ] Import guards: `RedisRateLimitMiddleware` raises `ImportError` with guidance if `redis` not installed.
- [ ] `__init__.py` exports new symbols (with conditional imports).

### Tests
- [ ] 5 unit tests in `tests/test_middleware/industrial/` pass.
- [ ] Smoke tests: PaaS plan and industrial topology examples verified end-to-end with docker.

### Examples
- [ ] `examples/19_middleware_custom/` runs.
- [ ] `examples/20_run_compose/` runs.
- [ ] `examples/21_output_modes/` runs.
- [ ] `examples/22_industrial_topology/` runs.

### Docs
- [ ] `docs/run.md` published.
- [ ] `docs/middleware.md` published.
- [ ] `docs/output-modes.md` published.
- [ ] `docs/composition.md` published.
- [ ] `docs/migration-v0.12-to-v1.0.md` published.
- [ ] `README.md` updated.
- [ ] `docs/agent.md` updated.
- [ ] `docs/index.md` reading path updated.

### Style gates
- [ ] `uv run ruff check src/` clean.
- [ ] `uv run mypy src/` strict pass.

### Implementation log
- [ ] `docs/internal/v1-refactor/logs/milestone-5-dev-log.md` written.

### CHANGELOG
- [ ] `## [1.0.0-M1..M4]` consolidated into `## [1.0.0] — Unreleased`.
- [ ] Sections: Added (full list), Changed, Deprecated (empty), Removed (empty), Fixed.

### Version bump
- [ ] `pyproject.toml` version → `1.0.0` (with user approval).

### Open questions to resolve
- [ ] Q4 (Bulkhead packaging) — confirmed: base.
- [ ] Q5 (RedisRateLimit extra name) — confirmed: `[ratelimit]`.
- [ ] Q6 (Observability packaging) — decided.

### Invariant checks (gate)
- [ ] `git diff src/t0_agents/mcp/` → empty.
- [ ] All 18 + 4 new examples run.
- [ ] All pre-existing tests pass unchanged.
- [ ] Migration guide walks through with real v0.12 codebase — no surprises.

---

## §9 — Acceptance criteria

- `from t0_agents.middleware.industrial import BulkheadMiddleware, CircuitBreakerMiddleware, TenantIsolationMiddleware, PiiRedactorMiddleware` works.
- `from t0_agents.middleware.industrial import RedisRateLimitMiddleware` works when `redis` installed; clear `ImportError` otherwise.
- All 4 new examples run end-to-end.
- All 4 new docs published and cross-linked from `docs/index.md`.
- Migration guide explicitly states: "every v0.12 codebase upgrades to v1.0 without changing a line.  Adopt new abstractions when it pays off."
- `CHANGELOG.md` has a single, complete `## [1.0.0]` entry.
- The PaaS plan and the industrial topology both reproduce on v1.0.
- If user approves: `pyproject.toml` reports version `1.0.0`.

---

## §10 — Risks & mitigations

| Risk | Mitigation |
|------|------------|
| New docs contradict legacy docs | Review ritual: re-read every doc page; update or cross-link as needed. |
| Version bump to 1.0 implies stability commitment we can't keep | Validate the 4 abstractions cover the PaaS scenarios before bumping; gate bump behind user explicit approval. |
| Industrial middleware bugs surface only under real load | Smoke test with the industrial topology example; document known limitations in `docs/composition.md`. |
| `redis` import errors confuse users | Lazy import + clear message: "Install with `pip install 't0-agents[ratelimit]'`". |
| Migration guide misses an edge case | Walk through real v0.12 codebase (the user's PaaS plan is the test case). |

---

## §11 — Estimated surface

- `middleware/industrial.py`: ~600 LoC.
- 4 new examples: ~600 LoC + 4 READMEs.
- 4 new docs (`run`, `middleware`, `output-modes`, `composition`): ~1500 lines total.
- `docs/migration-v0.12-to-v1.0.md`: ~400 lines.
- `README.md` update: ~50 lines changed.
- `CHANGELOG.md` consolidation: ~100 lines.
- `pyproject.toml`: ~10 lines (extras).
- `tests/test_middleware/industrial/`: ~500 LoC across 5 files.

**Total: ~+1100 LoC src + ~600 LoC examples + ~2000 lines docs + ~500 LoC tests.  Estimated 3 days full-time.**
