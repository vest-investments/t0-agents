# t0-agents v1.0 — Open Questions

> **Status:** Living document.  Each milestone resolves or raises questions.  Resolutions move questions to `Status: resolved` with a citation (file:line for source-validated answers).
> **Read order during `/v1-milestone <N>`:** after the contract + implementation plan, before prior dev-logs and the milestone spec.

---

## Q1 — `Run` eager vs lazy default

**Status:** resolved (M1, 2026-05-15).

**Context.** `agent.invoke(...)` returns a `Run`.  Eager means the asyncio task starts immediately upon construction; lazy means it starts when `.result()` or `.events()` is first awaited.

**Trade-offs:**
- **Eager** matches v0.12 mental model (`agent.run()` returns a thing that "is running").  Cancellation before consumption is trivial.  Two-channel consumption works without races.
- **Lazy** matches futures/promises pattern.  Essential for `eager=False` queueing (worker pops Run and `await`s).

**Resolution.** Eager by default; `eager=False` opt-in.  `asyncio.create_task` fires in `Run.__init__` when `eager=True`; for chat agents the task body waits on `_mode_chosen` until `result()` or `events()` is called (lazy *dispatch* with eager task scheduling, see [milestone-1-dev-log.md] §3).

---

## Q2 — Events buffer size

**Status:** resolved (M1, 2026-05-15).

**Context.** `Run.events()` buffers events for replay (`replay=True`).  Unbounded buffer leaks memory on long runs.

**Resolution.** Bounded queue at 10_000 events (drop-oldest on overflow) with a single WARN log per `EventBuffer` instance and a `truncated` property the user can introspect.  Implemented in `core/_runtime.EventBuffer`.  Tune later if real usage exceeds (Q2 may re-open if telemetry shows widespread truncation).

---

## Q3 — `Composer.cancel()` timeout configurable

**Status:** open.

**Context.** `await root_run.cancel()` propagates to all sub-Runs.  Sub-Runs may be in `await tool_call()` and need time to clean up.

**Trade-offs:**
- Wait indefinitely → caller hangs.
- Hard kill → tools may not run cleanup, leaving DB transactions / file handles open.
- 30s default with kwarg override → pragmatic.

**Recommendation.** 30s default; per-topology kwarg (`saga(..., cancel_timeout=60.0)`); global override via `BulkheadMiddleware`-style config.

**Resolution target:** M4 dev-log §3.

---

## Q4 — `BulkheadMiddleware` base or extra

**Status:** open.

**Context.** Bulkhead = `asyncio.Semaphore` limiting concurrency.  No third-party dep needed.  Industrial pattern that every prod deployment wants.

**Trade-offs:**
- Base: ergonomic, no extra install step.
- Extra (`[industrial]`): cleaner separation.

**Recommendation.** Base, since it's pure-stdlib and ubiquitous.  Group with `CircuitBreaker` (also stdlib).

**Resolution target:** M5 dev-log §3.

---

## Q5 — `RedisRateLimitMiddleware` extra name

**Status:** open.

**Context.** Requires `redis` (or `redis.asyncio`).  Other middleware that may follow: `MemcachedRateLimit`, `EtcdLeader`.

**Trade-offs:**
- `[redis]` — specific to dependency.
- `[ratelimit]` — capability-based; one extra installs whatever ratelimit backend we ship.
- `[distributed]` — broader bucket for any networked-state middleware.

**Recommendation.** `[ratelimit]` (pulls `redis` for now; future backends add to same extra).

**Resolution target:** M5 dev-log §3.

---

## Q6 — `ObservabilityMiddleware` packaging

**Status:** open.

**Context.** OTel is the de-facto standard.  Datadog has its own SDK.  Some users want both.

**Trade-offs:**
- Bundled `[otel]` extra installs `opentelemetry-api` + `opentelemetry-sdk` — pulls ~5 deps.
- Standalone middleware that accepts any tracer (Protocol) — user installs their own tracing lib.

**Recommendation.** Standalone Protocol-based middleware that accepts a tracer.  Optional `[otel]` extra provides `OTelTracerAdapter` for OpenTelemetry users.

**Resolution target:** M5 dev-log §3.

---

## Q7 — `Run.replay()` preserves episode_id or forks new

**Status:** resolved (M1, 2026-05-15).

**Context.** TZ `episode_id` is a correlation key in ClickHouse, not a persistent state object.  Two interpretations:
- Replay = "re-execute under same correlation" → keep episode_id.
- Replay = "execute again, possibly with different model/variant" → new episode_id (analytics distinguishes original from replay).

**Resolution.** New episode_id by default; `replay(..., episode_id=original.episode_id)` opt-in for "same correlation" use case.  `Run.replay` also accepts `agent=` to retarget (defaults to the original via weakref) and `variant=` (forwarded as a TZ tag `t0_agents.variant=<value>` so ClickHouse can audit replay provenance without an SDK change).

---

## Q8 — `parallel` reducer signature

**Status:** open.

**Context.** `parallel(*nodes, reducer=...)` runs nodes, collects results, applies reducer.

**Signatures considered:**
- `Callable[[list[AgentResult[T]]], U]` — reducer sees full results (cost, episode_id, diagnostics, etc.).
- `Callable[[list[T]], U]` — reducer sees only values; simpler but loses metadata.

**Trade-offs:**
- Full AgentResult: more expressive, enables "pick by lowest cost" or "filter by parse_strategy".
- Just values: simpler API, matches `consensus(merge=...)` natural shape.

**Recommendation.** `Callable[[list[AgentResult[T]]], U]` for `parallel`, `Callable[[list[T]], T]` for `consensus.merge` (which conceptually operates on agreed values).

**Resolution target:** M4 dev-log §3.

---

## Q9 — `conditional` router signature

**Status:** open.

**Context.** `conditional(*, router=..., routes={...})` routes input to a sub-node based on a key.

**Signatures considered:**
- `Callable[[Input], str]` — sync router on raw input; no inference yet.
- `Callable[[AgentResult[T]], str]` — async router after a classifier run.  Requires preceding classifier — but that's just `sequential(classifier, conditional(router=lambda res: res.value.label, ...))`.

**Recommendation.** Sync `Callable[[Input], str]`.  If user needs an LLM-driven router, they explicitly wrap with `sequential(classifier, conditional(...))`.

**Resolution target:** M4 dev-log §3.

---

## Q10 — `saga` compensation idempotency

**Status:** open.

**Context.** Saga pattern: each step has a compensate step.  On failure at step N, run compensate(N-1), compensate(N-2), ..., compensate(0).

**Trade-offs:**
- Enforce idempotency in code (e.g., dedupe by ID).  Adds complexity to every compensation.
- Document as user responsibility.  Saga is an advanced pattern; users adopting it understand distributed transactions.

**Recommendation.** Document as user responsibility.  Add a note in `docs/composition.md`.

**Resolution target:** M4 dev-log §3.

---

## Q11 — `StreamingOutput` activation when TZ supports streaming JSON

**Status:** open (deferred to post-v1.0).

**Context.** TZ currently does not emit `output.parsed` mid-stream (per guide §7.1).  `StreamingOutput` ships as a placeholder.

**Trade-offs:**
- Implement now with stdlib JSON-stream parser (`ijson`) → adds dependency.
- Wait for TZ upstream → cleaner architecture.

**Recommendation.** Defer.  `StreamingOutput(base=ChatOutput())` works for chat streaming.  `StreamingOutput(base=JsonOutput(...))` raises `ConfigurationError` with a "not yet supported" message + link to upstream issue.

**Resolution target:** M3 dev-log §3.  Re-open when TZ ships streaming JSON.

---

## Q12 — `AgentResult[T]` type narrowing helpers

**Status:** open.

**Context.** With `AgentResult[T]` generic, users access `.value: T`.  Old code accesses `.response: str` or `.parsed: dict | None`.  Discrimination by `.function_kind` is awkward.

**Trade-offs:**
- TypeGuards (`is_json(result) -> TypeGuard[AgentResult[dict]]`) — mypy-friendly, requires PEP 647.
- `match`-style pattern in user code — explicit but verbose.
- Just rely on `output: OutputMode[T]` for static typing — `agent = Agent[Customer](config=..., output=StructuredOutput(Customer))` gives `Agent.invoke(...) -> Run[Customer]`.

**Recommendation.** Rely on `Agent[T]` typing.  Add `is_json()`, `is_chat()` TypeGuard helpers as bonus.

**Resolution target:** M3 dev-log §3.

---

## Q13 — `JsonRetryMiddleware` placement in default chain

**Status:** open.

**Context.** When `OutputMode.recover()` returns `Recovery.retry(hint)`, the retry middleware re-emits inference.  Where does it live in the chain?

**Trade-offs:**
- Innermost (right before terminal) → memory/guardrails see only the final successful inference.  Cleanest.
- Outermost → memory/guardrails see every retry.  Better audit, more noise.

**Recommendation.** Innermost.  Audit of retries goes via TZ tags (`retry_attempt=N`) and `result.diagnostics.attempts`.

**Resolution target:** M3 dev-log §3.

---

## Q14 — Deprecation timing for `agent.run` and `agent.stream`

**Status:** resolved (user decision 2026-05-15).

**Context.** v1.0 introduces `agent.invoke()` as the canonical entry point.  Two coexistence options were considered:
- Keep `agent.run()` / `agent.stream()` forever as thin shims (full additive compat).
- Deprecate them for one minor; remove in v1.1.

**Resolution.** Deprecate in v1.0 with `DeprecationWarning`; remove in v1.1.

- v1.0: `agent.run(...)` and `agent.stream(...)` keep working byte-identical to v0.12, but emit `DeprecationWarning` exactly once per process pointing at the canonical replacement (`agent.invoke(...)` / `agent.invoke(...).events()`).
- v1.1: removed entirely.

**Rationale.** A single canonical entry point is cleaner.  Users with the PaaS plan that drive every inference through `agent.run` get a grace period (one minor) to migrate, then the surface clears.

**Impact:** M1's `agent.run` / `agent.stream` shim emits warnings.  M5's migration guide documents the rename.  M5's CHANGELOG has a non-empty `Deprecated` section.

---

## Q15 — Legacy JSON config kwargs policy

**Status:** resolved (user decision 2026-05-15).

**Context.** v1.0 introduces `AgentConfig(output=OutputMode)` as the canonical config surface.  The legacy kwargs (`function_kind`, `json_mode`, `output_schema`, `permissive_json_parsing`, `max_json_retries`) need a fate.

**Resolution.** **Removed in v1.0** (no deprecation path).  Migration is one line per `AgentConfig` construction.

- v1.0: `AgentConfig.__init__` no longer accepts the 5 legacy kwargs.  Passing any raises `TypeError` at construction (standard dataclass behaviour).
- Migration documented in M3 + M5 migration guide:
  `AgentConfig(function_kind="json", json_mode="strict", output_schema=S, permissive_json_parsing=True, max_json_retries=2)` → `AgentConfig(output=JsonOutput(schema=S, mode="strict", permissive=True, max_retries=2))`.

**Rationale.** These kwargs only affect JSON-mode users (a smaller % of the surface than `run`/`stream`).  The migration is mechanical and bounded.  A deprecation path would carry forward dead-shape validation in `__post_init__` for an entire minor.  Removing directly keeps the v1.0 config surface clean.

**Impact:**
- M3 spec: `AgentConfig.__post_init__` no longer migrates legacy kwargs to `OutputMode`.  The 5 fields are simply deleted.
- M3 deliverable: migrate `examples/17_structured/` to the new shape as part of M3.
- M5 migration guide: explicit before/after for the JSON config kwargs.
- M5 CHANGELOG: non-empty `Removed` section listing the 5 kwargs.

---

## How to update this file

When a milestone resolves a question:

1. Change `Status: open` → `Status: resolved`.
2. Add a `**Resolution:**` block with the decision + reasoning.
3. If source-validated, include a `file:line` citation.
4. Reference the dev-log §3 where it was decided.

When a milestone raises a NEW question:

1. Add it at the bottom with a new `Q<N>` number (continue the sequence).
2. Set `Status: open`.
3. Reference the dev-log §3 where it was raised.

Never delete a question.  Resolved questions stay as a paper trail.
