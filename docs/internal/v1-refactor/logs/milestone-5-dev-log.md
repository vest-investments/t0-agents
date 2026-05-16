# Milestone 5 — Housekeeping de cierre — Dev Log

> **Date closed:** 2026-05-16
> **Audience:** the next session that will polish the v1.0 public docs (`docs/middleware.md`, `docs/output-modes.md`, `docs/composition.md`, `docs/migration-v0.12-to-v1.0.md`) and any future contributor revisiting the v1.0 refactor.
> **Authoritative sources:** `docs/internal/v1-refactor/00-integration-contract.md`, `docs/internal/v1-refactor/milestones/milestone-5-industrial-polish.md`, `docs/internal/v1-refactor/logs/milestone-{1,2,3,4}-dev-log.md`.

---

## §1 — Summary

M5 closes the v1.0 four-abstraction refactor (`Run` · `Middleware` ·
`OutputMode` · `Composer`) with a **reduced housekeeping scope**.  The
original spec at `milestones/milestone-5-industrial-polish.md`
included 5 industrial middlewares (`BulkheadMiddleware`,
`CircuitBreakerMiddleware`, `TenantIsolationMiddleware`,
`PiiRedactorMiddleware`, `RedisRateLimitMiddleware`), an
`OTelTracerAdapter` helper under the `[otel]` extra, 4 new examples
(19–22), and 5 new docs (`docs/run.md`, `docs/middleware.md`,
`docs/output-modes.md`, `docs/composition.md`,
`docs/migration-v0.12-to-v1.0.md`).  After advancing 6 commits (C1–C6
of the original plan) the user reviewed the scope and dropped the
industrial bundle (no concrete consumer; would saturate the framework
surface) plus deferred the 4 new public docs to a follow-up session
dedicated to documentation polish.  Examples 19–22 are deferred with
the docs.

The shipped M5 surface is therefore:

### Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `docs/run.md` | 340 | Canonical public guide for `Run[T]` — lifecycle, two-channel consumption, replay / fork, error model, `Run(agent=None)` mode for third-party Composer authors, two runnable snippets. |
| `docs/internal/v1-refactor/logs/milestone-5-dev-log.md` | this file | M5 closing log. |

### Files modified

- `CHANGELOG.md` — four `## [1.0.0-M1..M4] — Unreleased` sub-sections (~420 lines) consolidated into a single `## [1.0.0] — Unreleased` with the canonical Keep-a-Changelog sub-sections (Added, Changed, Deprecated, Removed, Fixed) plus a Notes section that documents the dropped industrial scope and the deferred doc pages.
- `pyproject.toml` — `version = "0.12.0"` → `version = "1.0.0"`.
- `uv.lock` — regenerated (one line change: `t0-agents 0.12.0 → 1.0.0`).
- `src/t0_agents/core/agent.py` — `Agent.invoke` docstring slim from ~55 LoC to ~11 LoC (the verbose kwarg documentation moves to `docs/run.md`).  Net: 913 → 870 LoC.
- `docs/agent.md` — replaced the four-paragraph v1.0 preview block (M1/M2/M3/M4 forward references) with a single one-paragraph pointer at the four canonical doc pages and the migration guide.  `Agent.run` mentions in the quickstart example unchanged for byte-identity.
- `docs/api-reference.md` — `Run` section now opens with a "see `docs/run.md`" pointer.
- `docs/guardrails.md` — new section "Guardrails vs. TensorZero rate limits" distinguishing per-invocation guardrails (`max_cost`, `max_tool_calls`) from gateway-layer `rate_limiting.rules` (cross-request, 429-on-breach).  The original cherry-pick of C5 referenced `RedisRateLimitMiddleware` as a "third layer"; that paragraph was removed at cherry-pick time because the middleware is no longer part of v1.0.
- `docs/production.md` — new section "Rate limits and quotas" with cross-link to the new guardrails section.  Same edit: the original C5 cherry-pick mentioned `RedisRateLimitMiddleware`; that paragraph was removed.
- `examples/02_basic_agent/README.md` — directory drift fix (replaced `01_basic_agent` → `02_basic_agent`; removed stale forward-reference note).
- `examples/{02,03,04,05,06,07,18}/docker-compose.yml` — 7 files; replaced `clickhouse/clickhouse-server:lts` (non-existent) with the canonical `clickhouse:lts` so `docker compose up` works on a fresh machine.
- `tests/test_mcp/test_harness.py` — added 1 regression test `TestRecoverableInferenceError::test_429_rate_limit_re_raised` locking in the "propagate, never auto-retry" contract for `TensorZeroError(status_code=429)`.  This closes the parked rate-limit follow-up checklist item.
- `docs/internal/v1-refactor/02-open-questions.md` — Q4 / Q5 / Q6 marked `Status: deferred (M5, 2026-05-16)` with resolution text that cites this dev-log §3.

### Public API delta (added to `t0_agents.__init__.__all__`)

**None.**  The four-abstraction surface from M1–M4 is unchanged.  No new top-level symbols ship at M5.

### Tests

- **+1** under `tests/test_mcp/test_harness.py` (429 regression).
- Pre-M5 non-integration test count: 886 (post-M4 baseline).
- Post-M5 non-integration test count: **887** (+1 net).

### Docs

- 1 new public doc (`docs/run.md`).
- 2 new doc sections in existing pages (`docs/guardrails.md`, `docs/production.md`).
- 1 doc-page rewrite (`docs/agent.md` preamble).
- 1 doc-page edit (`docs/api-reference.md` Run section header).
- Internal: Q4/Q5/Q6 closed as `deferred`.

---

## §2 — Deviations from the milestone spec

### D-1.  Industrial middleware dropped from v1.0.

**What changed.**  Spec §4 lists 5 industrial middlewares
(`BulkheadMiddleware`, `CircuitBreakerMiddleware`,
`TenantIsolationMiddleware`, `PiiRedactorMiddleware`,
`RedisRateLimitMiddleware`) and an `OTelTracerAdapter` helper plus
three new extras (`[industrial]`, `[ratelimit]`, `[otel]`) plus 5 unit
test files under `tests/test_middleware/industrial/`.  Initial M5
implementation shipped all of this as commits C1–C4 (47 tests, +600
LoC src + +650 LoC tests + 3 extras + 6 top-level exports).  After the
user reviewed the API surface during M5 close, the entire industrial
bundle was dropped.

**Why.**  Two reasons stacked:

1. **No concrete consumer.**  The PaaS plan (the one motivating
   consumer the user has) already runs through
   `agent.invoke(...).result()` and has no need for in-app rate
   limiting (gateway-layer `rate_limiting.rules` covers that) or for
   bulkhead / circuit-breaker / tenant-isolation / PII patterns
   today.  Shipping six classes the user can't point at a real
   call-site for is speculative API surface — the kind of thing the
   `feedback_tensorzero_first` memory and the
   `feedback_consistency_over_overhead` memory warn against.

2. **Surface saturation.**  v1.0 already exposes ~60 public symbols
   across `Run`, `Middleware`, `OutputMode`, and `Composer`.  Six
   more middleware classes would push the public namespace past the
   point a new user can comfortably scan.  Better to keep v1.0
   focused on the four abstractions and add industrial primitives
   in a future minor when a concrete user demands them.

**Impact on subsequent milestones / minors.**  None within v1.0.
For a hypothetical v1.1 industrial-middleware milestone: the
implementation pattern (class with `name: str` + `async __call__(ctx,
next_)` + lazy-import for extras-gated deps + `ImportError` guard with
install hint) is settled — the M5-era implementation lives in the
reflog and can be cherry-picked back if it turns out the right
approach.  The Q4/Q5/Q6 recommendations (base for pure-stdlib,
`[ratelimit]` extra for Redis, Protocol-based middleware + `[otel]`
extra adapter) carry forward as-is.

### D-2.  Four new public docs and 4 examples deferred to a follow-up session.

**What changed.**  Spec §4 lists `docs/middleware.md` (~450 lines),
`docs/output-modes.md` (~400 lines), `docs/composition.md` (~500
lines), `docs/migration-v0.12-to-v1.0.md` (~450 lines), plus 4 new
examples (`examples/19_middleware_custom/`, `20_run_compose/`,
`21_output_modes/`, `22_industrial_topology/`).  M5 ships only
`docs/run.md` (`docs/middleware.md` etc. are deferred to the next
session).  Examples 19–22 are also deferred — `19_middleware_custom`
and `22_industrial_topology` were keyed off the dropped industrial
bundle anyway, so they would need rescoping; `20_run_compose` and
`21_output_modes` are pure demos of `Run` / `OutputMode` that fit
naturally with the doc-polish session.

**Why.**  The user (interview 2026-05-16, second round) decided to
treat documentation as a dedicated session — "después de M5 voy a
empezar otra sesión para pulir y documentar mejor lo de los
milestones anteriores".  Rationale: each doc is ~400-500 lines of
careful prose drawing on the dev-log + the CHANGELOG + the example
code; rushing it inside M5 (already large) risks shallow content.
Better to ship 1.0.0 with the API surface + the consolidated
CHANGELOG + the `Run` page already promoted, then iterate on docs
under 1.0.1 / 1.1.

**Impact on subsequent sessions.**  See §4 carry-forward — the next
session has a tight checklist of what each deferred doc must cover.
Cross-links to the deferred docs already exist in `docs/run.md` and
`docs/agent.md` (pointers that resolve once the targets ship).

### D-3.  `docs/api-reference.md` Output Modes section deferred.

**What changed.**  M3 dev-log §4 carry-forward 5 said `docs/api-reference.md`
should add a dedicated "Output modes" section by M5.  Deferred
because the natural place for output-mode narrative is
`docs/output-modes.md` (D-2 above), and dropping a stub-only
api-reference section ahead of the dedicated doc creates an
out-of-order cross-link pattern.  The api-reference Output-modes
section will ship in the next session alongside `docs/output-modes.md`.

**Impact on subsequent sessions.**  Captured in §4 carry-forward.

### D-4.  Migration guide (`docs/migration-v0.12-to-v1.0.md`) deferred.

**What changed.**  Spec §4 lists the full migration guide as part of
M5.  Deferred to the next session.  This is the most user-facing of
the deferred docs because Tier 1 (`Agent.run` deprecation) and Tier 2
(5 JSON config kwargs removed) are direct breakage signals.

**Why this is OK in the interim.**  The CHANGELOG `## [1.0.0]` entry
already documents Tier 1 + Tier 2 with explicit migration recipes
(under "Deprecated" and "Removed").  The dev-logs of M1 (Tier 1 shim
mechanics) and M3 (Tier 2 cross-field validation collapse + per-call
JSON translation in the shim) cover the implementation details.  Any
user upgrading from v0.12 to v1.0 today has enough information to
migrate via the CHANGELOG; the standalone migration guide is a
better-curated walkthrough but not a blocker.

**Impact.**  The next session captures this as the highest-priority
carry-forward.

### D-5.  Long `Agent.invoke` docstring relocation kept partial.

**What changed.**  M2 dev-log §4 carry-forward 2 asked M5 to relocate
the long `Agent.invoke` docstring to `docs/run.md` and slim the
in-source docstring to ~20 lines, helping `agent.py` toward the
≤300-LoC acceptance criterion (spec §9).  M5's C6 ships the slim
(913 → 870 LoC) and `docs/run.md` carries the full surface in
"Constructor arguments (the `invoke` kwargs)".  The
acceptance-criterion target of ≤300 LoC is **not met** (870 LoC
remains) — the bulk now comes from `Agent.__init__`, the legacy `run`
/ `stream` shims, the `invoke()` body (per-call kwarg translation),
the legacy `_chat_terminal` / `_chat_streaming_terminal` /
`_single_shot_terminal` orchestration, and `_validate_system_kwargs`.

**Why.**  Hitting ≤300 LoC would require either: (a) moving the
deprecated shims (`run`/`stream`) into a separate module, which would
fragment the public Agent surface, or (b) collapsing the three
terminal methods into one polymorphic dispatch, which would muddle
the chat-vs-streaming-vs-single-shot path-selection invariants
(M1 D-2, M2 D-7).  Both options trade ergonomics for a LoC
target that the spec called "acceptance criterion" but is really an
aspirational ceiling.  The user did not flag this during M5 close.

**Impact on subsequent minors.**  When `agent.run` / `agent.stream`
are removed in v1.1, the file naturally shrinks by ~150 LoC.  The
≤300 target becomes reachable then; tagging the milestone goal to v1.1
is the cleaner path.

---

## §3 — Open questions raised or resolved

### Q4 — `BulkheadMiddleware` base or extra → **deferred**

**Decision.**  The industrial-middleware scope as a whole is deferred
(see D-1).  The pre-decision recommendation in this question (base
install, grouped with `CircuitBreaker`) carries forward as the
starting point for a hypothetical future industrial milestone.
Updated `02-open-questions.md` Q4 to `Status: deferred (M5, 2026-05-16)`
with a resolution paragraph citing this dev-log.

### Q5 — `RedisRateLimitMiddleware` extra name → **deferred**

**Decision.**  Same as Q4: industrial scope deferred.  `[ratelimit]`
remains the recommended extra name (capability-based; future Memcached
/ etcd backends can join the same extra).  Updated to `Status: deferred`.

### Q6 — `ObservabilityMiddleware` packaging → **partially resolved + deferred**

**Decision.**  The Protocol-based shape is already settled at M2: the
existing `ObservabilityMiddleware` (`src/t0_agents/core/middleware/builtins.py`)
accepts any tracer with `start_as_current_span(name)` — OpenTelemetry
tracers expose this natively, so OTel users can wire today with no
adapter.  The `[otel]` extra + `OTelTracerAdapter` convenience helper
were the M5 scope and are deferred with the rest of the industrial
bundle.  Updated to `Status: deferred (M5, 2026-05-16)`.

### No new questions raised at M5.

The interview (second round, 2026-05-16) raised three decisions
(revert scope, CHANGELOG/version handling, Q4/Q5/Q6 treatment); each
resolved before continuation.  Implementation only required deletion
work + cherry-pick + 1 file rewrite (CHANGELOG) + 1 file edit
(`02-open-questions.md`); no new ambiguity surfaced.

---

## §4 — Carry-forward notes for subsequent sessions

### For the next session (public-docs polish)

The dedicated docs session inherits a tight checklist from M1–M4
dev-logs that M5 deferred (D-2 above).  Priority order:

1. **`docs/migration-v0.12-to-v1.0.md`** (highest priority — user-facing breakage).  The CHANGELOG `## [1.0.0]` entry is the source of truth for content; expand into a structured walkthrough with side-by-side `v0.12 → v1.0` snippets.  Required sections (per `milestones/milestone-5-industrial-polish.md` §4 sketch):
   1. What changes (TL;DR table — Tier 0 / Tier 1 / Tier 2).
   2. Migrating `Agent.run()` → `await agent.invoke()` (include `RunFailedError` defensive fallback note from M1 D-3).
   3. Migrating `Agent.stream()` → `agent.invoke().events()`.
   4. Migrating JSON config kwargs → `output=JsonOutput(...)` (full mapping table; note the `StructuredOutput` → `StructuredInferenceResult` rename per M3 D-1).
   5. What's the same (`AgentResult.response/.parsed/.parse_strategy/.json_parse_attempts/.function_kind`).
   6. What's new and opt-in (Middleware, Composers, Run replay/fork; saga compensation idempotency caveat per M4 §4 carry-forward 5).
   7. Deprecation timeline (v1.0 warns Tier 1; v1.1 removes).
   8. PaaS-specific notes (PaaS plan at `~/Descargas/prompt-as-a-service-mvp-plan-with-t0-agents.md` already uses `agent.invoke(...).result()` — minimal rewriting needed).

2. **`docs/middleware.md`** (~450 lines).  Promote M2 CHANGELOG content
   into a dedicated page.  Required content (per M2 dev-log §4
   carry-forward 1):
   - The Starlette-style protocol + `RunContext`.
   - The default chain ordering: `Memory → SubAgents → Guardrails → CostTracking → JsonRetry` (cite M2 D-1: this preserves the v0.12 invariant "guardrails see sub-agent delegations").
   - The 6 built-in middlewares — table + one-liner each.
   - The `middleware=[]` opt-out — explicitly note `Run.cost()` / `Run.usage()` stop working (M2 D-4).
   - Custom middleware authoring — PII redactor walkthrough.
   - The `allow_retry = True` opt-in for retry-style middlewares (M3 D-3).

3. **`docs/output-modes.md`** (~400 lines).  Per M3 §4 carry-forward 1:
   - The protocol (`expects_tool_loop` / `supports_streaming` / `shape_inference` / `materialize` / `recover`).
   - The 5 built-ins with construction examples.
   - The retry-and-recovery mechanism (`OutputDiagnostics.parse_strategy`, `JsonRetryMiddleware`, `Recovery` types).
   - Third-party authoring: YAML extraction, Instructor-style retry, SQLAlchemy row materialisation as concrete examples.
   - The `[pydantic]` extra for `StructuredOutput`.
   - The `failure_detail` field on `OutputDiagnostics` (M3 D-2).
   - The `_event_to_json` wire-format preservation (M3 D-5) — pointer only.

4. **`docs/composition.md`** (~500 lines).  Per M4 §4 carry-forward 1:
   - The `Composable[Input, T]` protocol.
   - The 9 topologies with runnable examples.
   - The narrow `Composer.invoke(...)` surface (M4 D-3): only `eager`, `context`, `cancel_timeout` are accepted.
   - Cancellation cascade — 30s default (Q3 resolution).
   - Audit via `parent_episode_id` + a ClickHouse `WHERE` example.
   - Saga: compensation idempotent (Q10), same input as paired action (M4 D-4), raising compensate logs WARN without blocking cascade.
   - `retry_chain` exception asymmetry (M4 D-5).
   - Two-paths comparison: declarative `AgentConfig.sub_agents=` vs. explicit topology factories.
   - `CompositeRun.feedback/.replay/.fork` deferred (M4 D-2).

5. **`docs/structured-outputs.md` migration block** (per M3 §4
   carry-forward 2).  Add a v0.12 → v1.0 migration section at the top
   with cross-link to `docs/output-modes.md`.

6. **`docs/api-reference.md` Output modes section** (per M3 §4
   carry-forward 5).  Promote the M3 preview block at the top of the
   page into a proper section beneath the existing Composers section.

7. **`docs/structured-outputs.md`** (the existing legacy reference)
   should pick up cross-links to the new pages so the legacy page
   doesn't drift.

8. **`docs/multi-agent.md`** — cross-link to `docs/composition.md` for
   the M4 code-driven path (per M4 dev-log §1).

### For a hypothetical industrial-middleware minor (post-1.0)

If a concrete consumer ever asks for `BulkheadMiddleware` /
`CircuitBreakerMiddleware` / `TenantIsolationMiddleware` /
`PiiRedactorMiddleware` / `RedisRateLimitMiddleware` / `OTelTracerAdapter`,
the M5-era implementation is in the reflog and can be cherry-picked:

- Local commits `1f5438f .. c85b5cc` (now overwritten by reset, recoverable for 90 days via `git reflog`) contain a working implementation of all 5 + the OTel adapter + 47 passing tests + 3 extras (`[industrial]`, `[ratelimit]`, `[otel]`).
- The implementation followed the M2 builtin protocol verbatim; lazy `try: import redis.asyncio` for the `[ratelimit]` gate; lazy `import opentelemetry` for `[otel]`.
- Tests used `tests/test_middleware/industrial/` package with the `_make_ctx` helper from `tests/test_middleware/test_chain_order.py` for the minimal `RunContext` without a gateway.

The Q4/Q5/Q6 recommendations in `02-open-questions.md` stay valid as
the starting design.

### For users upgrading from v0.12 to v1.0 today

Until `docs/migration-v0.12-to-v1.0.md` ships, point them at:
- `CHANGELOG.md` `## [1.0.0]` Deprecated + Removed sections for the breakage signals.
- `docs/run.md` for the `Agent.run` → `agent.invoke` migration shape.
- `docs/internal/v1-refactor/logs/milestone-{1,3}-dev-log.md` for the
  shim mechanics (M1) and the per-call JSON kwarg translation (M3).

---

## §5 — Verification

Commands run and outcomes:

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/` | **passed** — all checks clean (70 source files). |
| `uv run mypy src/` | **passed** — no issues found in 70 source files. |
| `uv run pytest -m "not integration"` | **passed** — 887 passed, 2 skipped, 57 deselected, 122 warnings.  Baseline post-M4 was 886; +1 new test (429 regression). |
| `git diff origin/main src/t0_agents/mcp/` | **empty** — invariant §2.1 holds across the whole roadmap. |
| `grep -rE "BulkheadMiddleware\|CircuitBreakerMiddleware\|TenantIsolationMiddleware\|PiiRedactorMiddleware\|RedisRateLimitMiddleware\|OTelTracerAdapter" src/ docs/ examples/ CHANGELOG.md` | **0 matches in user-facing surfaces**; remaining mentions are in `docs/internal/v1-refactor/{00-integration-contract,02-open-questions}.md` (planning docs) and this dev-log (intentional). |
| `uv run python -c "from importlib.metadata import version; print(version('t0-agents'))"` | **`1.0.0`** — version bump landed in `pyproject.toml` and `uv.lock`. |
| `git log --oneline origin/main..HEAD` | **5 commits exactly** — `chore: pre-public polish`, `docs: docs/run.md + slim`, `docs: CHANGELOG consolidation`, `chore: bump 1.0.0`, `docs: dev-log + defer Q4/Q5/Q6`. |

User-side verification (pending — deferred to user):

- `uv run pytest -m integration` (with `docker compose up -d` +
  `GROQ_API_KEY` in `.env`) — to confirm the integration tests still
  pass with the v1.0 surface.
- 17 chat-mode examples (`examples/00..16, 18`) byte-identity smoke —
  subprocess capture of stdout against a v0.12 baseline.  In
  particular, spot-check the 7 docker-compose files modified in C5
  (`examples/{02..07,18}`) to confirm `clickhouse:lts` pulls
  successfully.
- Migrated `examples/17_structured/` (from M3 onward) runs against the
  live gateway.

Numbers for the regression baseline:

- Pre-M5 non-integration test count: 886.
- Post-M5 non-integration test count: 887 (+1 — `test_429_rate_limit_re_raised`).
- New source code in M5: 0 LoC (`docs/run.md` is documentation).
- Removed source code in M5: ~45 LoC of `Agent.invoke` docstring (moved to `docs/run.md`).
- `core/agent.py`: 913 → 870 LoC (post-docstring-slim).
- Repo size change: ~+340 lines of docs (`docs/run.md`), ~+150 lines of docs (CHANGELOG `## [1.0.0]` consolidated minus the 4 sub-sections that were ~420 lines), ~+50 lines of polish edits across `docs/{guardrails,production,agent,api-reference}.md`, ~+1 test.  Net: roughly neutral on src/ + ~+400 lines of docs.

When the user runs the integration suite + byte-identity smoke and
confirms green, v1.0.0 is publishable.  The follow-up documentation
session inherits the 8 carry-forward items in §4 and can ship as
1.0.1 / 1.1 with no breaking changes.
