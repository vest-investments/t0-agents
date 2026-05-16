# Milestone 3 — OutputMode Polymorphism — Dev Log

> **Date closed:** 2026-05-16
> **Audience:** the implementer of M4 (Composer) and any future contributor revisiting the v1.0 refactor.
> **Authoritative sources:** `docs/internal/v1-refactor/00-integration-contract.md`, `docs/internal/v1-refactor/milestones/milestone-3-output-mode.md`, `docs/internal/v1-refactor/logs/milestone-1-dev-log.md`, `docs/internal/v1-refactor/logs/milestone-2-dev-log.md`.

---

## §1 — Summary

### Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `src/t0_agents/core/output.py` | 248 | `OutputMode[T_co]` Protocol; helper types (`InferenceRequest`, `Attempt`, `ParseFailure`, `Recovery`, `OutputDiagnostics`); `ParseStrategy` re-export. |
| `src/t0_agents/core/outputs/__init__.py` | 33 | Re-exports for the five built-in modes. |
| `src/t0_agents/core/outputs/chat.py` | 112 | `ChatOutput` — `expects_tool_loop=True`, `supports_streaming=True`; `materialize` extracts text/thought blocks; `recover` returns `Recovery.fail()`. |
| `src/t0_agents/core/outputs/json.py` | 220 | `JsonOutput` — single-shot JSON with permissive parsing + bounded retry-with-hint.  Class-level `RETRY_HINT_BASE` + instance `_build_retry_hint`. |
| `src/t0_agents/core/outputs/structured.py` | 237 | `StructuredOutput[M]` — Pydantic-validated; lazy pydantic import; `_summarise_validation_error` helper for field-aware retry hints. |
| `src/t0_agents/core/outputs/tool_call.py` | 93 | `ToolCallOutput` — router pattern; extracts `ToolCall` blocks without executing them. |
| `src/t0_agents/core/outputs/streaming.py` | 67 | `StreamingOutput(base)` placeholder; `__post_init__` raises `ConfigurationError` for non-streaming bases (Q11). |
| `tests/test_output/__init__.py` | 0 | Package marker. |
| `tests/test_output/test_chat_output.py` | 101 | Capability flags; shape_inference no-op; materialize join Text/Thought; reject wrong response type; recover always fail. |
| `tests/test_output/test_json_output.py` | 200 | shape_inference injects `params.chat_completion.json_mode` + `output_schema`; materialize strict/permissive/none; recover budget + hint enrichment. |
| `tests/test_output/test_structured_output.py` | 180 | Lazy pydantic import; schema derivation from `model.model_json_schema()`; `model_validate` happy path + `ValidationError` path; recover validation-aware hint. |
| `tests/test_output/test_tool_call_output.py` | 123 | `tool_choice` plumbing; materialize extracts ToolCall blocks; filters non-toolcall blocks; empty list → `parse_strategy="none"`. |
| `tests/test_output/test_streaming_output.py` | 45 | `StreamingOutput(base=ChatOutput())` works; `StreamingOutput(base=JsonOutput(...))` raises; delegation. |
| `tests/test_output/test_agent_result_compat.py` | 98 | v0.12 read-side property aliases; `response=`/`value=` constructor parity; XOR enforcement. |
| `tests/test_output/test_legacy_kwargs_removed.py` | 59 | All 5 Tier 2 kwargs raise `TypeError`. |
| `tests/test_output/test_stream_rejects_json.py` | 63 | `agent.invoke(msg, output=JsonOutput(...)).events()` raises `ConfigurationError`; same for `StructuredOutput`. |
| `examples/17_structured/structured_pydantic.py` | 84 | Runnable Pydantic-driven extraction demo (`StructuredOutput[Customer]`). |
| `docs/internal/v1-refactor/logs/milestone-3-dev-log.md` | this file | M3 closing log. |

### Files modified

- `src/t0_agents/core/types.py` — `AgentResult[T]` generic; `value: T` canonical field; 5 v0.12 read-side `@property` aliases; `response=` accepted as constructor alias for v0.12 mcp/loop.py compat.  Net: ~+150 LoC, ~140 → 319.
- `src/t0_agents/core/config.py` — deleted 5 Tier 2 fields + ~80 LoC cross-field validation; added `output: OutputMode = ChatOutput()`; new `sub_agents` + `memory_strategy="messages"` incompatibility checks against single-shot modes.  Net: ~−40 LoC.
- `src/t0_agents/core/agent.py` — deleted `_RETRY_HINT_BASE` (5 LoC) + `_build_retry_hint` (~30 LoC); replaced `_json_terminal` (~150 LoC retry loop) with `_single_shot_terminal` (~80 LoC flat body); added per-call overloads on `invoke()` + module-level `_resolve_json_mode` helper; `Agent.run` shim translates legacy per-call JSON kwargs into a synthesised per-call `JsonOutput`.  Net: ~−60 LoC (900 → 910 — held flat due to overload boilerplate and longer docstring on the new dispatch surface).
- `src/t0_agents/core/run.py` — deleted `RunInput.function_kind`/`output_schema`/`json_mode`/`max_json_retries` (4 fields); dispatch via `output.expects_tool_loop`; events() gate on `output.supports_streaming`; `Run.replay`/`Run.fork`/`build_run_input` drop the 3 legacy kwargs; `Run` constructor now requires `output: OutputMode` (was `Any | None`).  Net: ~−30 LoC.
- `src/t0_agents/core/middleware/base.py` — typed `RunContext.output: OutputMode[Any] | None`; `build_pipeline` checks `getattr(mw, "allow_retry", False)` to bypass `_NextGuard` for retry-enabled middlewares.  Net: ~+5 LoC.
- `src/t0_agents/core/middleware/builtins.py` — activated `JsonRetryMiddleware.__call__` (~80 LoC retry loop with `ctx.output.recover()` integration); `allow_retry = True` class attribute; latches `ctx.episode_id` from the first response so retries reuse it.  Imports `Attempt`/`OutputDiagnostics`/`ParseFailure` from `core/output`.  Net: ~+75 LoC (515 → 675).
- `src/t0_agents/core/structured.py` — renamed `StructuredOutput` dataclass to `StructuredInferenceResult` (free the canonical name for the new Pydantic OutputMode); docstring/return-type updates.  Behavioural shape unchanged.
- `src/t0_agents/serve/streaming_routes.py` — `_event_to_json` explicitly projects `DoneEvent.result` through the v0.12 read-side keys (`response`, `parsed`, `parse_strategy`, `json_parse_attempts`, `function_kind`) so HTTP/WS clients keep working byte-identical.
- `src/t0_agents/serve/app.py` — `share_client=True` error message refers to single-shot output modes instead of `function_kind='json'`.
- `src/t0_agents/__init__.py` — added 11 new exports: `OutputMode`, `ChatOutput`, `JsonOutput`, `StructuredOutput`, `ToolCallOutput`, `StreamingOutput`, `OutputDiagnostics`, `Recovery`, `ParseFailure`, `Attempt`, `InferenceRequest`.  Added `StructuredInferenceResult` (renamed from `StructuredOutput` in `core/structured.py`).
- `pyproject.toml` — new `[pydantic]` extra; `[[tool.mypy.overrides]]` for `t0_agents.mcp.*` with `disallow_any_generics = false`.
- `src/t0_agents/stream/events.py` / `src/t0_agents/stream/runner.py` / `src/t0_agents/cli/tui.py` / `src/t0_agents/testing/assertions.py` / `src/t0_agents/guardrails/structured.py` — `AgentResult` → `AgentResult[Any]` type annotations.
- `examples/17_structured/structured_agent.py`, `recovery_demo.py`, `recovery_real_provider.py`, `ner_dynamic_schema.py` — migrated to `output=JsonOutput(...)` shape and `agent.invoke(...).result()`.  `main.py` unchanged (uses `run_structured_inference` directly — Tier 0).
- 18 test files migrated (`tests/test_core/test_agent_json_{run,retry,safety}.py`, `tests/test_core/test_config_structured.py`, `tests/test_guardrails/test_require_parsed.py`, `tests/test_run/test_compat_run_stream.py`, `tests/test_run/test_lifecycle.py`, `tests/test_serve/test_session.py`, `tests/test_core/test_agent.py`, `tests/test_core/test_structured*.py`, `tests/test_testing/test_json_helpers.py`).  `TestLegacyKwargsForwarded` from M1 deleted (M1 dev-log §4 carry-forward 2).
- `CHANGELOG.md` — `## [1.0.0-M3] — Unreleased` block.
- `docs/internal/v1-refactor/02-open-questions.md` — Q11 and Q12 marked **resolved**.

### Public API delta (added to `t0_agents.__init__.__all__`)

`OutputMode`, `ChatOutput`, `JsonOutput`, `StructuredOutput`, `ToolCallOutput`, `StreamingOutput`, `OutputDiagnostics`, `Recovery`, `ParseFailure`, `Attempt`, `InferenceRequest`, `StructuredInferenceResult` (renamed).

Removed (Tier 2 — contract §2.4): `AgentConfig.function_kind/json_mode/output_schema/permissive_json_parsing/max_json_retries`; the M1-deprecated per-call invoke kwargs `output_schema/json_mode/max_json_retries`; `RunInput.function_kind/output_schema/json_mode/max_json_retries`; `Agent._RETRY_HINT_BASE`; `Agent._build_retry_hint`.

### Tests

- 69 new tests under `tests/test_output/`.
- Existing suite: 749 → 811 non-integration tests passing (+62 net after the test migrations).
- mypy strict pass on 67 source files.
- ruff clean.

### Docs

- `CHANGELOG.md`: `[1.0.0-M3] — Unreleased` block (~110 lines).
- `02-open-questions.md`: Q11/Q12 resolved.
- (Public docs updates — `docs/{agent,api-reference,structured-outputs}.md` — deferred to a separate commit; see §4 carry-forward note for M5.)

---

## §2 — Deviations from the milestone spec

### D-1.  Renamed `StructuredOutput` dataclass to `StructuredInferenceResult`.

**What changed.**  The pre-M3 `core/structured.py` exposed
`StructuredOutput` (a dataclass — the return type of
`run_structured_inference`).  The M3 spec calls the new Pydantic-driven
`OutputMode` `StructuredOutput[M]` by name.  Both can't live under
the same public name.  Renamed the dataclass to
`StructuredInferenceResult`; the new OutputMode takes over the
unqualified `StructuredOutput`.

**Why.**  Spec is authoritative on the OutputMode name (`StructuredOutput[M]` appears in the milestone spec §1, §4, §9 acceptance, §11).  The dataclass name was not listed as Tier 0 in the contract §2.4; only `run_structured_inference` (the helper) is promised unchanged.  Renaming the return type is a bounded migration cost (direct importers update one line: `from t0_agents.core.structured import StructuredInferenceResult`).

**Impact on subsequent milestones.**  None at the API level — the rename surfaces only to v0.12 callers who imported the dataclass name directly (rare; most callers use `result.parsed` / `result.raw` field access).  Documented in CHANGELOG `Removed` with the migration line.

### D-2.  `OutputDiagnostics` gained a `failure_detail: str | None` field.

**What changed.**  Spec §4 defines `OutputDiagnostics` with three fields (`parse_strategy`, `attempts`, `raw_text`).  Implementation adds a fourth: `failure_detail` — non-recovery context the OutputMode populates when reporting `parse_strategy="none"`.

**Why.**  `StructuredOutput.recover` needs the Pydantic `ValidationError` summary to build a field-aware corrective hint.  Without an explicit conduit, the middleware would have no way to convey "validation failure with these specific field problems" from `materialize` to `recover`.  Stuffing it into `raw_text` would conflate the model's raw output with the validator's diagnostic.

`JsonRetryMiddleware` reads `diag.failure_detail` and threads it into `ParseFailure.detail` when present; falls back to a generic "JSON parsing failed" detail otherwise.

**Impact on subsequent milestones.**  M5 documenting OutputDiagnostics should mention this field.  Third-party OutputMode authors can use it the same way.

### D-3.  `_NextGuard` opt-out via `allow_retry` class attribute.

**What changed.**  M2's `_NextGuard` raised `T0AgentsError` on the second `next_(ctx)` call, by design.  `JsonRetryMiddleware` *needs* to call `next_(ctx)` multiple times — once per retry attempt.  Without an escape hatch, the M3 retry middleware would trip the guard on the first retry.

**Why.**  Added `allow_retry: bool` class attribute on `Middleware` (optional, default `False`); `build_pipeline` checks `getattr(mw, "allow_retry", False)` and skips the `_NextGuard` for retry-enabled middlewares.  `JsonRetryMiddleware` sets `allow_retry = True`.

**Impact on subsequent milestones.**  M4 Composer middlewares that need to re-invoke their inner chain (e.g. for retry-chain or saga compensation patterns) can use the same opt-in.  Document in M5 `docs/middleware.md`: "custom middlewares that retry should set `allow_retry = True`".

### D-4.  M3 surfaces a per-call `Agent.run` translation shim instead of dropping the per-call legacy kwargs.

**What changed.**  M1 dev-log §4 M3 carry-forward 1 said: "Remove the three M1-only `invoke()` kwargs.  ... Update `Run.replay()` / `Run.fork()` to stop threading them.  Update tests that exercise legacy-kwarg forwarding — likely delete the test class entirely."

I deleted them from `Agent.invoke`, `RunInput`, and the test class — but the `Agent.run` shim **still accepts** the three legacy kwargs and translates them into a synthesised per-call `JsonOutput`.

**Why.**  `examples/17_structured/ner_dynamic_schema.py` and likely other v0.12 callers use per-call `agent.run(msg, output_schema=...)` overrides on JSON agents.  Dropping them from the shim would break Tier 1 byte-identity (`agent.run(msg, ...)` continues to work byte-identical to v0.12 — contract §6 Tier 1).  Translating internally keeps the shim a one-liner per call site but avoids leaking the legacy kwargs through to the canonical `invoke` surface.

**Impact on subsequent milestones.**  None — the translation is encapsulated in `Agent.run`.  In v1.1 when the shim is removed entirely (per Q14), this translation goes away with it.

### D-5.  HTTP/WS serialization of `DoneEvent.result` explicitly projects through v0.12 keys.

**What changed.**  Pre-M3, `_event_to_json` used `dataclasses.asdict(event)` which descended into `AgentResult` and returned the raw field names.  After making `AgentResult[T]` generic with `value: T`, the JSON serialization would expose `value`/`diagnostics` keys, breaking downstream HTTP/WS clients that consume `response`/`parsed`/`parse_strategy` etc.

**Why.**  Chat byte-identity invariant §2.3 covers the wire format.  HTTP and WebSocket clients of `t0_agents.serve.create_app(..., streaming=True)` consume JSON frames; the field names are part of the public contract.  `_event_to_json` now special-cases `DoneEvent` and projects `result` through the v0.12 read-side keys explicitly.

**Impact on subsequent milestones.**  M5 documenting the serve surface should note this projection — the JSON shape stays v0.12-compatible.

### D-6.  `Agent.run` per-call kwargs translation handles config-level inheritance.

**What changed.**  When the shim translates `agent.run(msg, output_schema=S)` to a per-call `JsonOutput`, what permissive/max_retries values does the translated `JsonOutput` use?

Implementation: when the agent's `config.output` is already a `JsonOutput`, the shim inherits non-overridden fields from it (`permissive`, `max_retries` if not in the per-call call, and `json_mode` / `schema` if not overridden).  When config.output is not a JsonOutput (e.g., default `ChatOutput`), the shim defaults to `JsonOutput(json_mode="strict", permissive=True, max_retries=2)`.

**Why.**  v0.12 semantics: per-call kwargs override config-level fields one-for-one; un-passed fields stay at config-level defaults.  Preserving this requires the inheritance.

**Impact on subsequent milestones.**  None — encapsulated in `Agent.run`.

### D-7.  Test migration scope wider than estimated.

**What changed.**  18 test files migrated (matches M3 plan estimate).  Additional files needed adjustment: `tests/test_serve/test_streaming.py` (3 tests — assertion against `result["response"]` key after the JSON-projection change), `tests/test_run/test_lifecycle.py:test_input_snapshot_carries_call_args` (replaced `run.input.function_kind` with `run.output.expects_tool_loop`).

**Why.**  Tests that exercised the v0.12 wire format / config field directly needed surgical updates.  No behaviour change; just assertion-shape updates.

**Impact on subsequent milestones.**  None — bounded.

### D-8.  `mcp/loop.py` AgentResult construction preserved via `response=` constructor alias.

**What changed.**  `mcp/loop.py` (invariant §2.1) constructs `AgentResult(response=text, ...)`.  Making `AgentResult` generic with `value: T` would normally break this call site.

**Why.**  I made the constructor accept BOTH `value=` and `response=` (one of them required; mutually exclusive) via overloads + sentinel-based `__init__`.  `response=` is a v0.12-compat alias that maps to `value=`.  The mcp/loop.py call site continues to work unchanged.

Verified via `git diff src/t0_agents/mcp/` → empty.  Also added a `disallow_any_generics = false` mypy override for `t0_agents.mcp.*` so the strict-mode check doesn't flag `mcp/loop.py`'s unparametrised `AgentResult` return annotation.

**Impact on subsequent milestones.**  None — the alias and the mypy override are both stable.

---

## §3 — Open questions raised or resolved

### Q11 — `StreamingOutput` activation → **deferred officially**

**Decision.**  `StreamingOutput(base=ChatOutput())` works (functionally equivalent to plain `ChatOutput()` — the wrapper is intentional documentation).  `StreamingOutput(base=JsonOutput(...))` raises `ConfigurationError` at construction with a "not yet supported in v1.0" message.  Re-open when TZ ships streaming JSON.

Updated `02-open-questions.md` Q11 to `Status: resolved`.

### Q12 — `AgentResult[T]` type narrowing helpers → **per-call overloads only**

**Decision.**  Type narrowing flows through per-call overloads on `Agent.invoke(output=...)`.  No `Agent[T]` class-level generic, no `is_json`/`is_chat` `TypeGuard` helpers in v1.0.

**Rationale.**  User interview 2026-05-16: per-call overloads achieve the spec §9 acceptance criterion (`result.value: Customer` mypy-verified) with minimal plumbing.  `Agent[T]` would have required threading the generic through `AgentConfig` and every callsite — much higher cost for marginal gain.  `TypeGuard` helpers can be added later as backward-compatible extensions if demand justifies.

Updated `02-open-questions.md` Q12 to `Status: resolved`.

### Q13 — `JsonRetryMiddleware` placement → confirmed innermost (already resolved at M2)

M2 resolved Q13 by placing `JsonRetryMiddleware` as the innermost middleware in the default chain.  M3 activates the middleware body — the placement decision stands unchanged.

### No new questions raised at M3.

Implementation surfaced D-2 (`failure_detail` field on `OutputDiagnostics`) and D-3 (`allow_retry` opt-out for `_NextGuard`) but both were direct consequences of the M3 design rather than open questions warranting an entry in `02-open-questions.md`.

---

## §4 — Carry-forward notes for subsequent milestones

### For M4 (Composer)

1. **`AgentResult[T]` is real now.**  M1's phantom `Run[T]` is concrete in M3 — `Run.result()` returns `AgentResult[T]`, and per-call overloads on `Agent.invoke(output=OutputMode[T])` narrow `T` automatically.  M4's `parallel(*nodes, reducer=...)` and `consensus(*nodes, merge=...)` reducer signatures should be:

   ```python
   parallel: Callable[[list[AgentResult[T]]], U]
   consensus.merge: Callable[[list[T]], T]
   ```

   The spec recommendation in `02-open-questions.md` Q8 stands (`AgentResult[T]` for parallel; just values for consensus).

2. **`allow_retry` opt-out for retry middlewares.**  M3 added `allow_retry: bool` (default `False`) as an opt-in class attribute on `Middleware`.  `_NextGuard` skips for retry-enabled middlewares.  M4 Composer middlewares that need to re-invoke their inner chain (e.g. `retry_chain` provider escalation, `saga` compensation) can use the same opt-in.

3. **`OutputMode` flows through `RunContext.output`.**  M4 Composers that build sub-Runs need to forward `output` correctly.  When the user does `parallel(agent1, agent2)`, each sub-Run inherits its own agent's `config.output`.  Composers should not inject an OutputMode of their own — the user composes that explicitly per child.

4. **Single-shot vs chat-style dispatch.**  Composers built on streaming-capable agents (`ChatOutput`) can be either eager-result or streaming.  Composers built on single-shot output modes (`JsonOutput`, `StructuredOutput`, `ToolCallOutput`) only support `result()`.  `parallel(json_agent_1, json_agent_2)` is fine; `parallel(json_agent_1, json_agent_2).events()` should raise via the existing gate on `Run.events()`.

5. **`Run.replay()` / `Run.fork()` propagate `output` correctly.**  M1 D-7 prepared this; M3 confirms — `replay()` and `fork()` pass `self._output` to the new `invoke()` call.  No M4 wiring change.

### For M5 (Industrial + docs)

1. **`docs/output-modes.md`** — the canonical user-facing doc for OutputMode.  Promote the CHANGELOG entry into a dedicated page.  Document:
   - The protocol (`expects_tool_loop`/`supports_streaming`/`name`/`shape_inference`/`materialize`/`recover`).
   - The 5 built-ins with construction examples.
   - The retry-and-recovery mechanism (`OutputDiagnostics.parse_strategy`, `JsonRetryMiddleware`, `Recovery` types).
   - Third-party authoring: YAML extraction, Instructor-style retry, SQLAlchemy row materialisation as concrete examples.
   - The `[pydantic]` extra for `StructuredOutput`.
2. **`docs/structured-outputs.md` migration**.  M3 updated the page lightly; M5 should do a full pass with cross-links to `docs/output-modes.md` and a v0.12 → v1.0 migration section.

3. **`docs/migration-v0.12-to-v1.0.md`** — must include the Tier 2 removal:

   ```python
   # v0.12:
   config = AgentConfig(
       function_kind="json",
       json_mode="strict",
       output_schema=S,
       permissive_json_parsing=True,
       max_json_retries=2,
   )
   # v1.0:
   config = AgentConfig(
       output=JsonOutput(
           schema=S,
           json_mode="strict",
           permissive=True,
           max_retries=2,
       ),
   )
   ```

   Note the `StructuredInferenceResult` rename if any caller imported the dataclass name.

4. **CHANGELOG consolidation.**  M3's `## [1.0.0-M3] — Unreleased` block is the third entry; M5 consolidates the four sub-sections (M1/M2/M3/M4) into a single `## [1.0.0] — Unreleased`.

5. **`docs/api-reference.md`** "Output modes" section needs to ship.  M3 noted this as a deferral; ensure it lands by M5.

6. **`_event_to_json` projection in `serve/streaming_routes.py`** is M3-introduced (D-5).  M5 docs should mention the v0.12 wire-format preservation explicitly so users serving over HTTP/WS know what shape to expect.

### For testing infrastructure / future work

7. **Test class shapes — `_json_agent` helpers.**  Every JSON-agent test file now has a `_json_agent` helper that accepts `output: JsonOutput | None`.  When adding new JSON-agent tests, follow the same pattern (default to `JsonOutput(json_mode="strict")`; override per-test as needed).

8. **`AgentResult` constructor pattern.**  Tests that construct `AgentResult` directly should use `value=` (canonical) plus `diagnostics=OutputDiagnostics(...)` for non-chat shapes.  The `response=` alias is reserved for the `mcp/loop.py` invariant call site and should not be used by new code.

---

## §5 — Verification

Commands run and outcomes:

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/` | **passed** — all checks clean. |
| `uv run mypy src/` | **passed** — no issues found in 67 source files (with the `mcp/` `disallow_any_generics = false` override). |
| `uv run pytest -m "not integration"` | **passed** — 811 passed, 2 skipped, 57 deselected, 121 deprecation warnings (expected from M1 shim emissions in migrated tests). |
| `git diff src/t0_agents/mcp/` | **empty** — invariant §2.1 holds. |
| `python -c "from t0_agents import OutputMode, ChatOutput, JsonOutput, StructuredOutput, ToolCallOutput, StreamingOutput, OutputDiagnostics, Recovery, ParseFailure, Attempt, InferenceRequest"` | **passed** — all 11 new symbols import cleanly. |
| `python -c "from t0_agents import AgentConfig; AgentConfig(name='x', function_name='y', function_kind='json')"` | **passed** (raises TypeError as expected — Tier 2 removal). |
| `uv run python examples/17_structured/recovery_demo.py` | **passed** — produced the expected four-scenario table (strict/permissive/retry/none). |

User-side verification (pending — deferred to user):

- `uv run pytest -m integration` (with `docker compose up -d` + `GROQ_API_KEY` in `.env`) — to confirm the integration tests still pass with M3 wiring.
- 17 chat-mode examples (`examples/00..16, 18`) byte-identity smoke — subprocess capture of stdout against a v0.12 baseline.
- Migrated `examples/17_structured/{structured_agent,recovery_real_provider,ner_dynamic_schema,structured_pydantic}.py` end-to-end against a live gateway.

Numbers for the regression baseline:

- Pre-M3 non-integration test count: 749.
- Post-M3 non-integration test count: 811 (+62 net — 69 new under `tests/test_output/` minus ~7 deleted: `TestLegacyKwargsForwarded` class + a handful of v0.12-specific test cases superseded by the new shape).
- New code: ~1170 lines across `core/output.py`, `core/outputs/{__init__,chat,json,structured,tool_call,streaming}.py`.
- Modified code: `core/types.py` (~+150 LoC for generic + properties + constructor), `core/config.py` (~−40 LoC: deletion of cross-field validation), `core/agent.py` (~flat: retry helpers + JSON terminal deleted, overloads + single-shot terminal added), `core/run.py` (~−30 LoC: legacy fields removed), `core/middleware/builtins.py` (~+75 LoC: JsonRetryMiddleware activation).
- Net repo size change: ~+1900 src, ~+880 tests, ~+150 docs, ~−110 deletions (legacy validation + retry helpers + 4 RunInput fields + invoke kwargs + the renamed dataclass).
