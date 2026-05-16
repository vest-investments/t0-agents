# Milestone 4 — Composer + Topologies — Dev Log

> **Date closed:** 2026-05-16
> **Audience:** the implementer of M5 (industrial polish + docs) and any future contributor revisiting the v1.0 refactor.
> **Authoritative sources:** `docs/internal/v1-refactor/00-integration-contract.md`, `docs/internal/v1-refactor/milestones/milestone-4-composer-topologies.md`, `docs/internal/v1-refactor/logs/milestone-{1,2,3}-dev-log.md`.

---

## §1 — Summary

### Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `src/t0_agents/compose/topology.py` | 296 | `Composable[Input, T]` Protocol, `Composer` base, `CompositeRun(Run)` subclass, `ComposerError(T0AgentsError)`, `DEFAULT_CANCEL_TIMEOUT = 30.0` module constant. |
| `src/t0_agents/compose/_runtime.py` | 514 | Nine `_drive_*` driver coroutines (`sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`) + three helpers (`_default_sequential_chain`, `_invoke_with_parent`, `_build_composite_result`) + `_run_compensations`. |
| `src/t0_agents/compose/builtins.py` | 581 | Nine `@dataclass` Composer subclasses + nine matching factory functions.  Each factory wires its composer instance into a `CompositeRun` whose driver coroutine calls the corresponding `_drive_*`. |
| `tests/test_compose/v1/__init__.py` | 0 | Package marker. |
| `tests/test_compose/v1/_helpers.py` | 130 | Shared `MockRun(Run)` + `MockComposable` + `CountingComposable` fakes.  Built on the M4 `Run.__init__(agent=None)` widening so tests don't need a TZ gateway. |
| `tests/test_compose/v1/test_sequential.py` | 91 | 8 tests: default chain, custom chain, JSON coercion, inference_id aggregation, first-step failure wrapping, cancel mid-pipeline, empty raises, episode_id propagation. |
| `tests/test_compose/v1/test_parallel.py` | 87 | 7 tests: default reducer, custom reducer, fail_fast, continue policy, reducer-exception wrapping, inference_id aggregation, empty raises. |
| `tests/test_compose/v1/test_race.py` | 67 | 5 tests: first-wins, loser cancellation, `cancel_others=False`, winner failure wrapping, empty raises. |
| `tests/test_compose/v1/test_consensus.py` | 64 | 5 tests: majority vote, tie-break by merge, failure propagation, merge-exception wrapping, empty raises. |
| `tests/test_compose/v1/test_map_reduce.py` | 95 | 7 tests: scatter+reduce, bounded concurrency, unbounded concurrency, str rejection, non-iterable rejection, empty iterable, failure wrapping. |
| `tests/test_compose/v1/test_conditional.py` | 66 | 5 tests: router routing, default fallback, unknown-no-default → ComposerError, router exception wrapping, target failure wrapping. |
| `tests/test_compose/v1/test_saga.py` | 110 | 6 tests: all-success, reverse compensation on failure, None-compensate skipped, compensate input == action input, failing compensate logged non-blocking, empty raises. |
| `tests/test_compose/v1/test_retry_chain.py` | 56 | 6 tests: first-fails-second-succeeds, all-fail reraises original, on_error=False stops immediately, on_error=True continues, empty raises, no-fall-through-on-success. |
| `tests/test_compose/v1/test_circuit_breaker.py` | 86 | 5 tests: closed pass-through, threshold opens, open refuses without invoking, half_open success closes, half_open failure re-opens. |
| `tests/test_compose/v1/test_nested_composers.py` | 65 | 4 tests: three-level nest resolution, inference_id aggregation across layers, Composer satisfies Composable protocol, three-level provenance. |
| `tests/test_compose/v1/test_cancellation.py` | 84 | 5 tests: sequential cancel, parallel cascade, nested cascade, lazy cancel direct transition, cancel_timeout overshoot WARN. |
| `tests/test_compose/v1/test_compat_sub_agents.py` | 65 | 4 tests: legacy `sub_agents=` path still importable, Agent satisfies Composable, factories return Composer instances, Composer.invoke rejects Agent-level kwargs (narrow surface). |

### Files modified

- `src/t0_agents/core/run.py` (lines 28, 102–138) — widened `agent: Agent` to `Agent | None`; replaced `self._agent_ref: weakref.ref[Agent] = weakref.ref(agent)` with `Callable[[], Agent | None]` guard.  Documented in §2 D-1.  +1 line of import (Callable), +5 LoC body.
- `src/t0_agents/compose/__init__.py` — re-exports for 9 factory functions, 9 Composer subclasses, 5 protocol/error symbols, plus the M2 declarative-path symbols.  ~+30 LoC.
- `src/t0_agents/__init__.py` — added 9 topology factories + `Composable`, `Composer`, `ComposerError` to top-level `__all__` and import block.  ~+15 LoC.
- `docs/agent.md` — appended a 1-paragraph forward reference describing the M4 Composers preview, alongside the existing M1/M2/M3 callouts.
- `docs/api-reference.md` — new "Composers *(v1.0)*" section: protocol types, the 9 builders with one-line descriptions, the `DEFAULT_CANCEL_TIMEOUT` constant, a small end-to-end example.  ~+60 lines.
- `docs/multi-agent.md` — added a 1-paragraph callout positioning Composers as the orthogonal code-driven path alongside the LLM-driven `AgentConfig.sub_agents=` path.
- `CHANGELOG.md` — appended `## [1.0.0-M4] — Unreleased` block (Added / Changed / Deprecated / Removed / Fixed + "Open questions resolved" sub-section).
- `docs/internal/v1-refactor/02-open-questions.md` — Q3, Q8, Q9, Q10 transitioned `Status: open` → `Status: resolved (M4, 2026-05-16)` with the resolution paragraph + citation to the implementation file.

### Public API delta (added to `t0_agents.__init__.__all__`)

`Composable`, `Composer`, `ComposerError`, `sequential`, `parallel`, `race`, `consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`, `circuit_breaker`.  Twelve new symbols.  `CompositeRun` and `DEFAULT_CANCEL_TIMEOUT` live on the `compose` submodule namespace only — implementation details users rarely touch.

### Tests

- 69 new tests under `tests/test_compose/v1/` (12 test files + `__init__.py` + `_helpers.py`; counts 67 at initial PR plus 2 fail-fast regression tests added in the D-7 follow-up commit).
- Existing suite: 811 → 886 non-integration tests (+75 net — 69 new from v1/ + 6 from upstream of M3 that I hadn't counted in the planning baseline).  No existing test was modified beyond ruff auto-fixes (import ordering, raw-string patterns).
- mypy strict: clean on 70 source files.
- ruff: clean on `src/` + `tests/test_compose/v1/`.

### Docs

- `docs/agent.md`: +1 paragraph forward reference.
- `docs/api-reference.md`: +60-line "Composers" section.
- `docs/multi-agent.md`: +1 paragraph orthogonal-path callout.
- Internal: Q3/Q8/Q9/Q10 resolved.

---

## §2 — Deviations from the milestone spec

### D-1.  `Run.__init__` widened to accept `agent: Agent | None`.

**What changed.**  Spec §4 lists `compose/{topology,builtins,_runtime}.py` and `compose/__init__.py` + top-level `__init__.py` as the only modified modules in M4.  Implementation widens `src/t0_agents/core/run.py:102-132` (one-line type annotation change + weakref guard) so `CompositeRun` can satisfy the nominal `Run[T]` return type that `Composer.invoke()` is contracted to produce (§3.4).

**Why.**  Three options were considered:

1. **Subclass `Run` and manually replicate ~30 lines of state init in `CompositeRun.__init__`** (no Run modification).  Rejected because the duplicated init drifts the moment a future Run field is added — every CompositeRun construction silently misses the new state.  Subclassing-without-super is a known maintenance hazard.
2. **Synthesise a stub Agent** that just satisfies `weakref.ref(agent)` and a few read-only properties.  Rejected because Run inspects the agent in several places (`feedback()`, `replay()`, `fork()`, `_drive`) and the stub leaks into every code path that touches the composite, fragile and hard to assert tests against.
3. **Widen `agent: Agent` to `agent: Agent | None` in `Run.__init__`** (chosen).  Single-line type widening + a 3-line weakref guard in the constructor body.  Forward-compatible: every existing call site passes a non-None Agent; mypy strict accepts the change cleanly across 70 source files and 884 tests.  The four agent-reading methods on Run (`_drive`, `feedback`, `replay`, `fork`) already handle the GC'd-Agent path (`self._agent_ref()` may return `None`) with a clear `RuntimeError`, so the agent-less path adds zero new branches in code that already exists.

**Impact on subsequent milestones.**  None at the API level — the widening is invisible to public callers.  M5's `docs/run.md` should mention the `agent=None` mode briefly when documenting Run's lifecycle, primarily so third-party Composer authors know the same path is available (analogous to how third-party `OutputMode` authors use the M3 OutputMode protocol).

### D-2.  ``CompositeRun.feedback`` / ``.replay`` / ``.fork`` raise ``NotImplementedError``.

**What changed.**  M1 contract §3.1 lists `feedback`, `replay`, `fork` as part of the public `Run` surface.  M4's `CompositeRun(Run[Any])` inherits the signatures but overrides each with an explicit `NotImplementedError` documenting the deferral.

**Why.**  Semantics on a composite are non-trivial:

- **`feedback`** — a Composer has no single TensorZero `episode_id` of its own; the root episode is the first sub-Run's.  Forwarding `feedback(metric, value)` to "the right episode" requires choosing which sub-Run (the first?  the last?  the failing one?).  Each topology has a different natural answer.  Cleaner to defer until users complain — pointing them at `composite._sub_runs[i].feedback(...)` in the meantime.
- **`replay`** — re-running a Composer is `composer.invoke(input)`, which is more explicit and idiomatic than `.replay()`.  The asymmetry with Agent runs is acceptable because Composers are constructed declaratively (no in-flight state to capture beyond their config).
- **`fork`** — "continue this composition with an extra user turn" doesn't map cleanly when the topology isn't a chat agent (e.g., a `parallel` of 5 voters — what does "extra message" mean?).  Cleaner to compose explicitly: `sequential(composer, follow_up_agent)`.

**Impact on subsequent milestones.**  M5 should NOT document `composite.feedback`/`.replay`/`.fork` in `docs/composition.md`.  If a future minor adds these, the dev-log entry should specify per-topology semantics explicitly.  Each `NotImplementedError` message points at the right escape hatch already (sub-Run access, re-invocation, explicit composition).

### D-3.  Composer `invoke()` surface is narrow, not broad.

**What changed.**  Spec §4 sketches `Composer.invoke(self, input: Any, **kwargs: Any) -> Run[Any]` without enumerating which kwargs are accepted.  Implementation locks the public surface to three: `eager`, `context`, `cancel_timeout`.  Caller-supplied `tags`, `episode_id`, `system`, `system_arguments`, `prior_messages`, `output`, `middleware` are NOT accepted — passing any of those raises `TypeError` at call time.

**Why.**  Decided 2026-05-16 in the planning interview.  The narrow surface eliminates per-topology ambiguity:

- For `parallel(a, b, c)` with caller-supplied `tags={"tenant": "acme"}`, do all three children inherit the tags, only the first, or none?  Each topology would need its own answer; documentation cost compounds with every kwarg the Composer accepts.
- The M2/M3 dev-log carry-forwards ("composers should not inject into sub-runs") are explicit: each sub-Run uses its own `AgentConfig.middleware` / `.output`.  A narrow `invoke` surface is the natural enforcement.
- Users who want per-Composer-level tags can bake them into each child `AgentConfig.tags=` before constructing the composer.  Users who want runtime tagging compose explicitly: `sequential(tagger, ...)` where `tagger` is an Agent whose `AgentConfig.tags=` differs.

**Impact on subsequent milestones.**  M5's `docs/composition.md` must spell out the narrow surface explicitly and point at the recipe for per-tenant tagging.  If a future minor widens the surface (e.g., to forward `tags` via shallow-merge), the change is additive (existing callers keep working).

### D-4.  Saga compensations receive the chained input, not the original.

**What changed.**  Spec §4 lists `saga(*steps: tuple[Composable, Composable | None])` without specifying the compensate input.  Implementation passes each compensate the same input its paired action received — i.e., the chained value flowing into `action_N`, not the original `saga.invoke(input)` value.

**Why.**  Decided 2026-05-16 in the planning interview.  Locally consistent with the action: a compensate that "undoes what the action did with this input" mirrors the action's view of the world.  Encoding uniqueness (resource ids, transaction handles, etc.) into the action's input is the standard saga pattern.

**Impact on subsequent milestones.**  M5's `docs/composition.md` migration recipe for users adopting saga should call this out.  The contract §3.4 description does not mention compensate input shape; consider amending in M5 alongside the doc.

### D-5.  `retry_chain` re-raises original exception un-wrapped (not `ComposerError`).

**What changed.**  Spec §10 mentions reducer / merge errors being wrapped in `ComposerError`.  `retry_chain` follows the opposite convention: when every node fails, the **last** node's original exception is re-raised un-wrapped.

**Why.**  Provider escalation is the canonical retry_chain pattern: `retry_chain(primary_llm, backup_llm, on_error=lambda e: isinstance(e, RateLimitError))`.  Callers naturally expect `except RateLimitError` to catch the eventual failure — wrapping in `ComposerError` would force every retry_chain user to write `except ComposerError as ce: raise ce.cause` boilerplate.  The dev-log records the choice; the docstring spells out the asymmetry; tests validate it (`test_retry_chain_all_fail_reraises_last_unwrapped`).

**Impact on subsequent milestones.**  M5 documenting the topology table should note the `retry_chain` exception explicitly.

### D-6.  `tests/test_compose/v1/test_compat_sub_agents.py` is a smoke-only test.

**What changed.**  Spec §5 calls for a `test_compat_sub_agents.py` that asserts `examples/14_multi_agent/` runs byte-identical.  Implementation downgrades it to a 4-test smoke that verifies the legacy public API surface is unchanged (`SubAgentMiddleware` importable, `agent_as_tool` callable, `Agent.invoke` exists, Composer.invoke rejects Agent-level kwargs).

**Why.**  Byte-identity of `examples/14_multi_agent/` is an invariant gate (`§3 Cross-cutting checklist`) — verified by the user via subprocess capture.  Reproducing that test in the v1/ unit-test directory would require either a live TZ gateway (then the test is integration-marked and out of scope for the v1/ unit suite) or a mocked gateway that recreates the example's exact orchestration — equivalent to the legacy integration test in `tests/test_compose/test_integration.py` which is already integration-marked.  Smoke test in `v1/` + user-side example verification cover the same surface without duplication.

**Impact on subsequent milestones.**  None — `examples/14_multi_agent/` byte-identity is the user's verification responsibility per the cross-cutting checklist.

### D-7.  Post-PR-review fix: `parallel`/`consensus` fail-fast no longer waits for slow siblings.

**What changed.**  The initial M4 commit shipped `_drive_parallel` (`error_policy="fail_fast"`) and `_drive_consensus` using `asyncio.gather(..., return_exceptions=True)` followed by a post-hoc scan for failed slots.  Because `gather(return_exceptions=True)` blocks until every task settles, the "cancel siblings on first failure" contract in the plan and CHANGELOG was structurally broken: a 0.01s failure with two 5s sibling agents would still take ~5s to surface.  The follow-up commit switches both drivers to `asyncio.wait(..., return_when=asyncio.FIRST_EXCEPTION)` and cancels siblings (both the result-wrapping tasks and the underlying sub-Runs via `composite._cancel_sub_runs`) the moment the first exception surfaces.  Two regression tests (`test_parallel_fail_fast_does_not_wait_for_slow_siblings`, `test_consensus_fail_fast_does_not_wait_for_slow_voters`) lock the latency: the cascade must surface within `cancel_timeout`, not within `max(child_delay)`.

**Why.**  Flagged by the `gemini-code-assist[bot]` PR #33 review (two HIGH-priority comments) and confirmed against the M4 plan ("fail_fast: first sub-Run exception cancels others under `cancel_timeout` and raises `ComposerError`") and the CHANGELOG entry ("`fail_fast` cancels siblings on first failure").  The dead-code path (`composite._cancel_sub_runs()` after `gather(return_exceptions=True)`) was the visible smell — by the time the post-hoc scan ran, every sub-Run was already terminal so the cascade was a no-op.

**Impact on subsequent milestones.**  None at the API level — observable behaviour for users matches the plan/CHANGELOG more closely.  M5's `docs/composition.md` should reuse the regression tests' setup (mix of fast-failing + slow-succeeding mock composables) as a documentation example of the cascade in action.

---

## §3 — Open questions resolved

### Q3 — `Composer.cancel()` timeout configurable → **30s default + per-topology kwarg**

**Decision.**  Every topology factory accepts `cancel_timeout: float = 30.0`.  Module constant `DEFAULT_CANCEL_TIMEOUT = 30.0` exported from `t0_agents.compose.topology`.  Per-call override via `composer.invoke(input, cancel_timeout=…)` takes precedence for that specific Run.  Overshoot logs a single WARN on `t0_agents.compose.topology` and lets sub-Run cancellation continue best-effort; the parent transitions to `cancelled` regardless.  Implemented in `CompositeRun._cancel_sub_runs` (`src/t0_agents/compose/topology.py`).  Validated by `tests/test_compose/v1/test_cancellation.py::test_cancel_timeout_overshoot_logs_warning`.

### Q8 — `parallel` reducer signature → **list[AgentResult[T]] (fail_fast) or list[AgentResult[T] | BaseException] (continue)**

**Decision.**  `Callable[[list[AgentResult[T]]], U]` under `error_policy="fail_fast"` (default).  Under `error_policy="continue"` the type widens to `Callable[[list[AgentResult[T] | BaseException]], U]` — failed slots surface as the raw exception so the reducer can `isinstance(item, BaseException)` to detect them.  Reducer exceptions are wrapped in `ComposerError(node=parent_composer, cause=...)`.  Pythonic, explicit, fail-loud.  User confirmed this design (interview 2026-05-16) over synthetic-AgentResult and two-reducer alternatives.

### Q9 — `conditional` router signature → **sync `Callable[[Input], str]`**

**Decision.**  Synchronous router on raw input.  No automatic LLM-driven routing — users compose explicitly: `sequential(classifier_agent, conditional(router=lambda r: r.value.label, routes=...))`.  Router exceptions wrap in `ComposerError(node=conditional_composer, cause=...)`; unknown route keys with no `default` set raise `ComposerError(node=conditional_composer, cause=ValueError(...))`.  Implemented in `_drive_conditional`.

### Q10 — `saga` compensation idempotency → **user responsibility, documented**

**Decision.**  No runtime enforcement.  The `saga` docstring spells out that compensations must be idempotent because a partial failure (e.g., a raising compensate mid-cascade) may invoke them more than once.  M5's `docs/composition.md` will repeat the caveat.  Compensations receive the same input action received (D-4).  A raising compensate logs a WARN but does not block earlier compensations in the cascade — implemented in `_run_compensations`.

### No new questions raised at M4.

The planning interview surfaced four design questions (Composer.invoke kwarg surface, parallel error semantics, saga compensate input, cancel timeout); each was resolved before implementation began.  Implementation surfaced D-1 (the `Run.__init__` widening), D-2 (NotImplementedError on composite feedback/replay/fork), D-3 (narrow invoke surface), D-4 (saga compensate input), D-5 (retry_chain un-wrapped re-raise), and D-6 (smoke-only compat test) but each is a direct consequence of M4 design rather than an open question requiring an entry in `02-open-questions.md`.

---

## §4 — Carry-forward notes for subsequent milestones

### For M5 (Industrial + docs)

1. **`docs/composition.md`** — the canonical user-facing doc for the M4 surface.  Promote the CHANGELOG entry into a dedicated page.  Required content:
   - The `Composable[Input, T]` protocol and how it lets Agents and Composers nest uniformly.
   - The 9 built-in topologies with one runnable example each (mock or live).
   - The cancellation cascade and the 30s default timeout — when to override.
   - The `parent_episode_id` audit and the ClickHouse query that reconstructs the hierarchy.
   - The two paths comparison: declarative `AgentConfig.sub_agents=` (LLM-driven delegation) vs explicit `sequential(...)` / `parallel(...)` (code-driven topology).
   - The saga compensation contract: compensations are idempotent, receive the same input action did, can fail without blocking earlier compensations.
   - `retry_chain`'s asymmetric error handling (re-raises last original exception un-wrapped) — distinct from every other topology's `ComposerError` wrapping.

2. **`examples/22_industrial_topology/`** — M5's M4 example deliverable per the spec §6 deferral.  Showcase three patterns end-to-end: a `sequential(researcher, sentiment, writer)` pipeline, a `parallel(provider_a, provider_b, consensus=...)` ensemble, and a `saga` with compensation that's runnable against a live gateway.

3. **CHANGELOG consolidation** — M4's `## [1.0.0-M4] — Unreleased` block folds into the single `## [1.0.0] — Unreleased` consolidated entry at M5.  M4's "Added", "Changed", and "Open questions resolved" sub-sections should be reorganised under the M5 consolidated Added/Changed/Deprecated/Removed/Fixed sections.

4. **`Run.__init__(agent=None)` mode** — M5's `docs/run.md` should mention this as a recognised mode used by `CompositeRun`.  Third-party Composer authors who want to build their own topology can subclass `CompositeRun` (or directly construct a `Run(None, ...)` with a custom `_drive` override).

5. **Migration guide saga caveat** — `docs/migration-v0.12-to-v1.0.md` (M5) should mention saga's compensation idempotency requirement, since users adopting saga from scratch may not have encountered the pattern's classical constraints.

6. **`CompositeRun.feedback` / `.replay` / `.fork`** are deliberate `NotImplementedError`s.  If user feedback after v1.0 asks for them, the right design has per-topology semantics — recorded in §2 D-2.  Defer until demand surfaces.

### For testing infrastructure / future work

7. **Mock infrastructure pattern.**  `tests/test_compose/v1/_helpers.py` exposes a `MockRun(Run[Any])` that exploits the M4 `Run.__init__(agent=None)` widening to produce a Run-shaped object without a live gateway.  Future Composer-adjacent test work (e.g., M5's industrial middleware tests for `BulkheadMiddleware` if it consumes Run handles) can reuse the same pattern.

8. **`circuit_breaker` state mutation in tests.**  Tests that exercise the breaker's state machine (`tests/test_compose/v1/test_circuit_breaker.py`) reach into `breaker._state`, `breaker._consecutive_failures`, `breaker._opened_at` directly because the state is intentionally private.  If we add a public introspection method in a future minor (e.g., `breaker.state -> Literal[...]`), update those tests to use the public surface.

---

## §5 — Verification

Commands run and outcomes:

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/ tests/test_compose/v1/` | **passed** — all checks clean. |
| `uv run mypy src/` | **passed** — no issues found in 70 source files (with the pre-existing `mcp/` `disallow_any_generics = false` override from M3). |
| `uv run pytest -m "not integration"` | **passed** — 886 passed, 2 skipped, 57 deselected, 122 deprecation warnings (expected M1 shim emissions in pre-existing tests). |
| `uv run pytest tests/test_compose/v1/` | **passed** — 69 passed in 1.6s (67 at PR-open + 2 fail-fast latency regression tests added in D-7). |
| `git diff src/t0_agents/mcp/` | **empty** — invariant §2.1 holds. |
| `python -c "from t0_agents import sequential, parallel, race, consensus, map_reduce, conditional, saga, retry_chain, circuit_breaker, Composable, Composer, ComposerError"` | **passed** — all 12 new symbols import cleanly. |
| Nested smoke construction (`sequential(race(a, b), conditional(routes={...}))`) | **passed** — tested in `test_nested_composers.py`. |

User-side verification (pending — deferred to user):

- `uv run pytest -m integration` (with `docker compose up -d` + `GROQ_API_KEY` in `.env`) — to confirm the existing integration suite still passes with M4 wiring.
- 18-example byte-identity smoke (`examples/00..16, 18, 17_structured`) — subprocess capture of stdout against a v0.12 baseline.  M4 makes no changes to the chat or single-shot paths.
- `examples/14_multi_agent/` byte-identity — the legacy `AgentConfig.sub_agents=` declarative path is unchanged by M4; spot-check that the example runs end-to-end.

Numbers for the regression baseline:

- Pre-M4 non-integration test count: 811 (M3 close).
- Post-M4 non-integration test count: 886 (+75 net — 69 new under `tests/test_compose/v1/` + a 6-test drift from upstream tests I had not counted in the planning baseline).
- New code: ~1391 lines across `compose/{topology,builtins,_runtime}.py`.
- Modified code: `core/run.py` +6 LoC (the `agent: Agent | None` widening + guard).
- New tests: ~1116 lines across 12 files + `_helpers.py`.
- Net repo size change: ~+1400 src, ~+1170 tests, ~+140 docs.

LoC distribution of new compose surface:

- `compose/topology.py`: 296 LoC.
- `compose/_runtime.py`: 514 LoC.
- `compose/builtins.py`: 581 LoC.
- Total `compose/` (M4 additions only): 1391 LoC.
