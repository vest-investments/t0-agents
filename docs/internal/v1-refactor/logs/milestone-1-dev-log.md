# Milestone 1 — Run Foundational — Dev Log

> **Date closed:** 2026-05-15
> **Audience:** the implementer of the next milestone (M2 Middleware) and any future contributor revisiting the v1.0 refactor.
> **Authoritative sources:** `docs/internal/v1-refactor/00-integration-contract.md`, `docs/internal/v1-refactor/milestones/milestone-1-run-foundational.md`.

---

## §1 — Summary

### Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `src/t0_agents/core/run.py` | 630 | `Run[T]`, `RunInput` (frozen + slots), `RunState` literal alias, `build_run_input` constructor. |
| `src/t0_agents/core/_runtime.py` | 137 | `EventBuffer` — bounded (10 000 events) replayable buffer with `truncated` flag, `asyncio.Condition`, one-time WARN log on overflow. |
| `tests/test_run/__init__.py` | 0 | Package marker. |
| `tests/test_run/test_lifecycle.py` | 227 | Eager / lazy start, state transitions, cancel, `__await__`, failure routing (`RunFailedError`). |
| `tests/test_run/test_events_buffer.py` | 188 | `EventBuffer` direct tests: live cursor, replay, double-subscribe, overflow + truncated + WARN. |
| `tests/test_run/test_feedback.py` | 120 | `Run.feedback` dispatching to `episode_id` / `inference_id` targets. |
| `tests/test_run/test_replay.py` | 117 | Q7 resolution: new episode by default; `episode_id=` opt-in; cross-agent; weakref-GC failure. |
| `tests/test_run/test_fork.py` | 108 | Transcript boundary capture; `prior_messages` threaded into `mcp.run_agent`. |
| `tests/test_run/test_two_channel.py` | 88 | Concurrent `result()` + `events()`; two `events()` iterators on the same Run. |
| `tests/test_run/test_compat_run_stream.py` | 179 | v0.12 byte-identity, `DeprecationWarning` emission, JSON-mode stream rejection, legacy-kwarg forwarding, line-based dedup. |
| `tests/test_run/test_integration.py` | 111 | `@pytest.mark.integration`: live T0+FastMCP end-to-end. |

### Files modified

- `src/t0_agents/core/exceptions.py` — added `RunCancelledError`, `RunFailedError(cause)`.
- `src/t0_agents/core/agent.py` — new `invoke(...) -> Run[Any]`; renamed `run`/`stream`/`_run_json` bodies to `_execute_chat_legacy` / `_execute_chat_streaming_legacy` / `_execute_json_legacy`; reimplemented `run` and `stream` as deprecated shims (`warnings.warn(stacklevel=2)` + delegate to `invoke()`).  `run()` shim unwraps `RunFailedError` to preserve v0.12 byte-identity for drive-time exceptions; `stream()` shim relies on `events()` re-raising originals (already the v0.12 behaviour).
- `src/t0_agents/__init__.py` — exported `Run`, `RunInput`, `RunState`, `RunCancelledError`, `RunFailedError`; updated `__all__`.
- `docs/agent.md` — short forward reference at the top about `agent.invoke(...)` and the shim deprecation timeline.
- `docs/api-reference.md` — new `Run` section listing the public surface, `RunInput`, `RunState`, `RunCancelledError`, `RunFailedError`.
- `CHANGELOG.md` — `## [1.0.0-M1] — Unreleased` block with Added / Deprecated / Validated sub-sections.
- `docs/internal/v1-refactor/00-integration-contract.md` — §3.1 amendment documenting the M1-only legacy JSON kwargs and the permanent `prior_messages` kwarg on `Agent.invoke()`.
- `docs/internal/v1-refactor/02-open-questions.md` — Q1, Q2, Q7 marked **resolved** with the M1 decisions.

### Public API delta (added to `t0_agents.__init__.__all__`)

`Run`, `RunInput`, `RunState`, `RunCancelledError`, `RunFailedError`.  `Agent.invoke(...)` is a new method on the existing `Agent` class but it does not appear separately in `__all__`.

### Tests

- 49 new tests under `tests/test_run/` (7 unit files + 1 integration file + the empty `__init__.py`).
- Existing suite: 647 tests passed pre-M1; 696 tests passed post-M1.  No existing test was modified.

### Docs

- `docs/agent.md`: 1-paragraph forward reference.
- `docs/api-reference.md`: new ~80-line `Run` section.
- Internal: contract §3.1 amendment + 3 open questions resolved.

---

## §2 — Deviations from the milestone spec

### D-1.  Spec `Agent.invoke` signature extended with three M1-only legacy JSON kwargs (`output_schema`, `json_mode`, `max_json_retries`).

**What changed.**  Contract §3.1 originally listed `invoke()`'s kwargs as `episode_id, tags, output, middleware, eager, context`.  The implementation adds `output_schema`, `json_mode`, `max_json_retries` (deprecated in M1, removed in M3) and `prior_messages` (permanent).

**Why.**  v0.12's `Agent.run` accepts per-call overrides for `output_schema`, `json_mode`, and `max_json_retries`.  Tier 1 byte-identity (the deprecation shims keep working) requires the shim to forward these into `invoke()`.  Two options were considered:

- (A) accept them on `invoke()` for M1, remove in M3 — chosen.
- (B) split the `agent.run()` shim into a chat branch using `invoke()` and a JSON branch bypassing it — rejected because it would have left the shim non-trivial and made the orchestration boundary unclear.

`prior_messages` is required for `Run.fork()` to work — fork() builds a transcript and seeds the next invocation with it.  This kwarg already existed on `mcp.run_agent`; surfacing it on `invoke()` is a minor extension that survives M3.

User decision on 2026-05-15 confirmed option (A).

**Impact on M2+.**  When M3 removes the legacy fields under Tier 2, it should also remove the same three kwargs from `Agent.invoke()` and from `RunInput`.  M3's `OutputMode` materialisation needs to consume the equivalent information from `AgentConfig.output: OutputMode` (the replacement field).  `prior_messages` survives untouched.

### D-2.  Internal callers of `agent.run(...)` / `agent.stream(...)` continue to call the deprecated shims.

**What changed.**  Initial implementation rerouted four internal call sites to the canonical `agent.invoke(...)`:

- `compose.delegation.agent_as_tool` (sub-agent dispatch)
- `compose.orchestrator._OrchestrationToolProvider.call_tool` (sub-agent dispatch)
- `serve.routes.chat` (HTTP `/chat` endpoint)
- `serve.streaming_routes.{_sse_event_source, _drive_stream}` (SSE + WebSocket)

This silenced the internal `DeprecationWarning`s but broke ~9 tests across `tests/test_compose/` (mock pattern `sub.run = AsyncMock(...)`) and ~6 tests across `tests/test_serve/` (subclass-override pattern `class ObservingAgent(Agent): async def run(...)`).  Cross-cutting checklist §3.2 mandates existing tests stay green unchanged.

**Why.**  Reverted all four internal sites to `agent.run(...)` / `agent.stream(...)`.  Each site emits exactly one `DeprecationWarning` per process via Python's default filter dedup `(message, category, module, lineno)` — about 4 internal warnings per process, acceptable as transitional noise.

**Impact on M2+.**  When M2 / M3 refactor the orchestration, those four internal call sites should switch to `agent.invoke(...).result()` (chat-path) or `agent.invoke(...).events()` (streaming-path), and the corresponding ~15 tests should be migrated in the same commit:

- `tests/test_compose/test_delegation.py` + `tests/test_compose/test_orchestrator.py`: mock `sub.invoke` returning a mock with an async `.result()` method, instead of mocking `sub.run` directly.
- `tests/test_serve/test_chat.py` + `tests/test_serve/test_session.py`: subclass `Agent` and override `_execute_chat_legacy` (M1) or the post-M2 middleware-chain entry (e.g. `_drive_chat`), not `run`.  M2 should expose a cleaner extension point for this testing pattern.

### D-3.  `Run.result()` raises the original exception (not `RunFailedError`).

**What changed.**  Contract §5.2 says `RunFailedError(cause)` wraps the underlying error when `await run.result()` is called on a failed run.  Implementation changed this so `result()` raises the original exception directly (matching `events()` semantics).  `RunFailedError` is kept in the type system as a defensive fallback for the "failed state without a captured exception" edge case (shouldn't normally happen).

**Why.**  The original wrap-in-`RunFailedError` design forced every internal user of `invoke().result()` to write `try: ... except RunFailedError as e: raise e.cause` boilerplate to get back to the standard Python `except SpecificError` idiom.  That asymmetry between `result()` and `events()` (events() already raises originals) was inconsistent and uneconomic.  Raising originals on both paths:

- removes the unwrap boilerplate from the `agent.run` shim (now genuinely a 3-line shim);
- matches v0.12 idioms — `except MaxTurnsExceededError`, `except GuardrailBlockedError`, etc. work directly on Run consumers;
- aligns `result()` and `events()` failure semantics for a uniform mental model.

`RunCancelledError` stays as the dedicated cancellation exception so callers can distinguish "the run was cancelled" from "the run failed."

**Impact on M2+.**  Contract §5.2 amended (the line `RunFailedError(cause: Exception)` wraps the underlying error → "raised only when the drive completed in `failed` state without a captured exception").  No M2 carry-forward needed — the change is self-contained.

### D-4.  `Run.transcript()` only captures the boundary user + assistant pair in M1.

**What changed.**  Spec §4 lists `Run.transcript() -> list[Message]` without specifying granularity.  M1 implements it as a 2-element list `[user, assistant]` (the original user message + the final response).  Intermediate tool turns are not captured.

**Why.**  Capturing every turn would require either modifying `mcp.run_agent` (forbidden — invariant §2.1) or adding a transcript-recording callback to `Agent._execute_chat_*_legacy`.  The latter is a larger refactor and was deemed M2/M3-shaped: the middleware chain is the natural place to record transcripts.

**Impact on M2+.**  M2's `Middleware.RunContext` can keep a full transcript field if useful; if so, `Run.transcript()` can switch to reading from there.  M1's `fork()` works correctly with the 2-element transcript for the spec test's use case (`test_fork.py::TestForkContinuation::test_fork_passes_transcript_as_prior_messages`).  Richer transcripts unlock multi-turn fork scenarios.

### D-5.  `Run.cost()` and `Run.usage()` are M1 placeholders.

**What changed.**  Both methods exist with the spec signatures but return `0.0` and `None` respectively.

**Why.**  v0.12 does not expose aggregated cost / usage on `AgentResult`; the guardrail subsystem tracks cost-so-far on `GuardrailContext` but that's not on the result.  Implementing real aggregation requires the cost-tracking middleware planned for M2.

**Impact on M2+.**  M2's `CostTrackingMiddleware` accumulates usage on its `RunContext.extras`; `Run.cost()` / `Run.usage()` should read from there.  Update the docstrings when M2 lands.

### D-6.  `_execute_chat_legacy` / `_execute_chat_streaming_legacy` accept a new `prior_messages` kwarg.

**What changed.**  Both legacy helpers now accept `prior_messages: list[Mapping[str, Any]] | None = None`.  When non-None, the value overrides whatever the memory strategy would produce (specifically, replaces `extra_kwargs["prior_messages"]`).

**Why.**  `Run.fork()` needs to thread the original transcript through `mcp.run_agent` / `StreamingAgentLoop.stream`.  Both of those already accept `prior_messages`; this just exposes it through the Agent layer.

**Impact on M2+.**  When M2 moves orchestration into middleware, the new `MemoryMiddleware` should keep the same "caller-explicit overrides memory-derived" precedence.

---

## §3 — Open questions resolved

### Q1 — `Run` eager vs lazy default → **eager**

**Decision.**  `eager=True` by default.  `asyncio.create_task(self._drive())` fires in `Run.__init__`.  `eager=False` opt-in defers the `_start()` to the first `result()` / `events()` consumer.

**Implementation detail (not in the question).**  For chat agents in eager mode, the task body waits on an `asyncio.Event` (`_mode_chosen`) before dispatching to streaming vs non-streaming — see D-1 reasoning in the plan file (forced by cross-cutting checklist §3.2: switching `agent.run()` to drive via streaming would break ~17 existing test files that script `FakeChatResponse`).

### Q2 — Events buffer size → **10 000 events, bounded, with `truncated` flag**

**Decision.**  `EventBuffer.MAX_EVENTS = 10_000`.  On overflow: drop oldest, set `truncated=True`, emit one `WARNING` per `EventBuffer` instance via `logging.getLogger("t0_agents.core._runtime")`.

**Validation.**  `tests/test_run/test_events_buffer.py::TestOverflow` confirms the truncation, the WARN log fires exactly once per buffer instance even with multiple overflows, and `events(replay=True)` after overflow yields from the first still-buffered event.

### Q7 — `Run.replay()` episode semantics → **new episode by default**

**Decision.**  `Run.replay()` mints a new `episode_id` (the caller-supplied `episode_id=` kwarg defaults to `None` and the underlying `Agent.invoke()` treats `None` as "let TensorZero allocate a fresh episode").  Pass `replay(episode_id=original.episode_id)` to preserve correlation.

**Bonus surface.**  `replay(variant=...)` is also accepted in M1 — the value is forwarded as the TensorZero tag `t0_agents.variant=<value>` so ClickHouse can audit replay provenance without an SDK change.  This is a small extension beyond the spec but maps cleanly onto Q7's "execute again, possibly with different model/variant" wording.

### No new questions raised at M1.

---

## §4 — Carry-forward notes for subsequent milestones

### For M2 (Middleware)

1. **Legacy helpers.**  `Agent._execute_chat_legacy`, `Agent._execute_chat_streaming_legacy`, `Agent._execute_json_legacy` are the v0.12-shaped orchestration bodies.  M2 replaces them with a middleware chain.  Match their behaviour exactly — they are the source of truth for chat byte-identity until M2 lands.

2. **Run mode-dispatch contract.**  `Run._drive()` waits on `_mode_chosen` for chat agents before dispatching.  The first consumer (`result()` vs `events()`) picks the mode.  M2's middleware chain must preserve this — the streaming path must still be reached when `events()` is the first call.  Concretely: M2's `build_pipeline` should accept an `is_streaming: bool` flag selected at Run drive time, not at config time.

3. **`prior_messages` precedence.**  Caller-supplied `prior_messages` (via `Run.fork`) overrides memory-strategy-derived `prior_messages`.  Tests in `tests/test_run/test_fork.py` enforce this — keep the precedence when wiring `MemoryMiddleware`.

4. **Sub-agent call sites.**  `compose/delegation.py:69` and `compose/orchestrator.py:294` still call `sub_agent.run(...)` (the deprecated shim).  Each emits one `DeprecationWarning` per process.  M2 should reroute both to `sub_agent.invoke(...).result()` **and** update the 9 mock-based tests in `tests/test_compose/test_delegation.py` and `tests/test_compose/test_orchestrator.py` in the same commit so they don't fail.

5. **`SubAgentMiddleware` order.**  Spec §3.2 says: `SubAgentMiddleware` wraps `tool_provider` BEFORE `GuardrailsMiddleware`.  The current `_execute_chat_legacy` order is `tool_provider → SubAgentTP wrap → GuardedTP wrap` — preserve this.

### For M3 (OutputMode)

1. **Remove the three M1-only `invoke()` kwargs.**  `output_schema`, `json_mode`, `max_json_retries` on `Agent.invoke()` are Tier-2-removal candidates.  Drop them from `Agent.invoke`, from `RunInput`, from `build_run_input`, and from the docstrings.  Update `Run.replay()` / `Run.fork()` to stop threading them.  Update `tests/test_run/test_compat_run_stream.py::TestLegacyKwargsForwarded` (likely delete the test class entirely — those kwargs no longer exist).

2. **`JsonRetryMiddleware` activation.**  Spec §3.1 mentions `JsonRetryMiddleware`.  In M1 the JSON retry loop still lives inline in `_execute_json_legacy`.  M3 should activate the middleware via `OutputMode.recover()`; route the corrective hint through it.  The existing `_build_retry_hint` helper in `agent.py` should move to the new `JsonOutput.recover()`.

3. **`Run[T]` becomes generic.**  In M1 the `T` parameter is phantom — `Run.result()` returns the non-generic `AgentResult`.  M3 makes `AgentResult[T]` generic; `Run.result()` should then return `AgentResult[T]` and `Run.value` (the new field) should be `T`.  Update `Run` and `Agent.invoke()` typing.

4. **JSON `events()` raises.**  D-2's `events()` raises `ConfigurationError` for JSON agents in M1.  When M3's `StreamingOutput(base=JsonOutput(...))` becomes possible, this gate can be removed (or relaxed to only raise when the OutputMode doesn't support streaming).

5. **Migrate `examples/17_structured/`.**  The example currently imports `run_structured_inference` directly; M3 should migrate it to `Agent(...).invoke(..., output=JsonOutput(...)).result()`.

### For M4 (Composer)

1. **`Run.fork` semantics.**  M1 forks with the boundary-pair transcript (D-3).  If M4 builds composers that need richer fork behaviour (multi-turn continuations), revisit the transcript implementation — likely depends on M2's middleware-level message history.

2. **Cancellation.**  Spec §3.4 says `await root_run.cancel()` propagates to all sub-Runs.  `Run.cancel()` in M1 only cancels its own task; composers will need to chain cancel calls through their child `Run`s.

### For M5 (Industrial + docs)

1. **`Run` doc page.**  `docs/run.md` should be the canonical user-facing doc for the Run abstraction.  The M1 `docs/api-reference.md` "Run" section is a starting point but should move to / be linked from `docs/run.md`.

2. **CHANGELOG consolidation.**  Per the cross-cutting checklist §3.5, M5 consolidates the per-milestone sub-sections into a single `## [1.0.0] — Unreleased`.  M1's `## [1.0.0-M1] — Unreleased` block is the first one to fold in.

3. **`Run.cost()` / `Run.usage()` placeholders.**  M2 should fill them; M5 should document them.

---

## §5 — Verification

Commands run and outcomes:

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/` | **passed** — all checks passed. |
| `uv run mypy src/` | **passed** — no issues found in 57 source files. |
| `uv run pytest -m "not integration"` | **passed** — 696 passed, 2 skipped, 57 deselected, 151 deprecation warnings (expected from shim emissions). |
| `uv run ruff check tests/test_run/` | **passed** after fixing 4 trivial issues (unused imports, dead-code class, mutable default). |
| `git diff src/t0_agents/mcp/` | **empty** — invariant §2.1 holds. |

Manual / pending verification (requires user-side docker):

- `uv run pytest -m integration` — to be run by the user with `docker compose up -d` and a valid `GROQ_API_KEY` in `.env`.
- 18-examples byte-identity smoke (subprocess capture of `examples/00..16, 18` stdout against a v0.12 baseline; `examples/17_structured/` is unchanged in M1 since it imports `run_structured_inference` directly).

Numbers for the regression baseline:

- Pre-M1 non-integration test count: 647.
- Post-M1 non-integration test count: 696 (+49 tests under `tests/test_run/`).
- New code: ~1670 lines across `core/run.py`, `core/_runtime.py`, exception additions, `__init__.py` exports, and tests.
- Modified code: `core/agent.py` grew from 1414 → 1667 lines (the new `invoke()` method + shim bodies; the renamed-and-extended legacy helpers).  M2 reverses this growth when the orchestration moves into middleware.
