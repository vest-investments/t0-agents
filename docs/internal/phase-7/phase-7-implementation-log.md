# Phase 7 Implementation Log

> Bridges the design in `docs/phase-7/phase-7-harness-engineering-notes.md`
> (the priority-ordered catalog of improvements) with what we actually shipped.
> Scope of this phase is **harness engineering for tool-use reliability**:
> closing logging micro-gaps from Phase 6, handling Thought-only responses,
> exposing `tool_choice`, catching malformed tool names in-loop, building
> developer scripts, and adding a system prompt example.

## What we built

| File | Purpose |
|---|---|
| `src/t0_agents/mcp/loop.py` (edit) | F7.1: `time.monotonic()` timing around `call_tool()` with `duration_ms` logged at DEBUG. F7.2: tool names list in dispatch log. F7.3: `content_types=` breakdown at INFO per turn. F7.4: `Thought.text` logged at DEBUG. F7.5: Thought-only fallback with `response_source`. F7.6: `tool_choice` parameter forwarded to `t0.inference()`. F7.7: `RECOVERABLE_INFERENCE_PATTERNS` + try/except `TensorZeroError` in-loop with corrective message injection. F7.8: `dryrun` constraint documented in docstring. |
| `src/t0_agents/core/types.py` (edit) | F7.5: Added `response_source: Literal["text", "thought"] = "text"` field to `AgentResult`. |
| `src/t0_agents/__init__.py` (edit) | F7.6: Re-exported `ToolChoice` from `tensorzero` for caller convenience. |
| `tests/_fake_t0.py` (edit) | F7.7: `FakeT0.__init__` now accepts `list[FakeChatResponse \| Exception]`; `inference()` raises items that are exceptions. |
| `tests/test_mcp/test_harness.py` | **NEW.** 21 Phase-7 unit tests across 5 classes: `TestLoggingMicrogaps` (F7.1-F7.4), `TestThoughtOnlyHandling` (F7.5), `TestToolChoice` (F7.6), `TestRecoverableInferenceError` (F7.7), `TestDryrunDocstring` (F7.8). |
| `scripts/flake_benchmark.py` | **NEW.** F7.9: Standalone script — invokes `uv run pytest` N times, parses summary line, reports aggregate pass/skip/fail rates. |
| `scripts/trace_episode.py` | **NEW.** F7.10: Standalone script — queries ClickHouse HTTP API for `ChatInference` + feedback tables, prints human-readable per-turn trace. |
| `examples/02_system_prompt/` | **NEW.** F7.11: System prompt example with `config/tensorzero.toml` (demonstrates `system_template` directive), `config/functions/agent/baseline/system.txt` (tool-use system prompt), `server.py`, `main.py`. |
| `docs/implementation-plan/02-validation-checklist-v3-final.md` (edit) | Added Phase 7 section (F7.1-F7.12) with inline test-function references. Updated progress summary. |

## Decisions taken during planning

| Topic | Options | We chose | Why |
|---|---|---|---|
| §3.1 system prompt | Include in Phase 7 / defer / minimal infrastructure only | **Defer with example** | System prompt tuning is a separate concern. Phase 7 adds `examples/02_system_prompt/` as infrastructure for future tuning, without iterating on prompt quality. User-confirmed. |
| §2.1 channel catch | Always-on / feature flag / defer | **Always-on, extensible** | Feature flag adds API surface for little benefit. The catch is purely defensive. Extensibility via `RECOVERABLE_INFERENCE_PATTERNS` module-level list — new patterns require only `append()`. User requested extensibility for other models. |
| §2.4 Thought flag | `thought_only: bool` / `response_source: Literal` | **`response_source: Literal["text", "thought"]`** | More extensible if a third source appears. User-confirmed. |
| Developer scripts | Both / trace only / benchmark only / defer | **Both** | Benchmark needed for model evaluation in future phases. Trace utility has immediate debugging value. User-confirmed. |
| F7.7 corrective message | Synthetic ToolResult / plain user message | **Plain user message** | Synthetic ToolResult with `name="__error__"` risks T0 rejecting it (T0 may validate ToolResult names against registered tools). Plain text user message is safer and model-agnostic. Uses `Message(role="assistant", content=[])` + `Message(role="user", content=correction)` pattern. |

## Divergences discovered during implementation

1. **`Message` from tensorzero is a TypedDict (dict), not a dataclass.** The
   initial F7.7 tests used `m.content` (attribute access). Dict-based `Message`
   requires `m["content"]` and `isinstance(m, dict)` checks instead of
   `hasattr(m, "content")`. Fixed by adjusting the test assertions to use dict
   access patterns.

2. **No changes to existing test modules were needed.** All 81 pre-existing
   unit tests pass without modification. The `FakeT0` change (accepting
   exceptions in the response queue) is backward-compatible because
   `list[FakeChatResponse]` is a subtype of `list[FakeChatResponse | Exception]`.

3. **The `Thought` type is already imported in `test_loop.py`.** Phase 3's
   tests already import `Thought` from tensorzero and use it for the
   `test_filters_thought_blocks_from_final_response` test. Phase 7 adds the
   import to `loop.py` itself (not just tests).

4. **`ToolChoice` type has two definitions in tensorzero.** `tensorzero.types`
   defines it as `Union[Literal["auto", "required", "off"], Dict[...]]`, while
   `tensorzero.generated_types` uses `Literal["none", "auto", "required"] |
   ToolChoiceSpecific`. Both are exported from `tensorzero.__init__`. We import
   from `tensorzero` (the public API) and let the SDK resolve which definition
   to use.

5. **`TensorZeroError.__init__` uses `status_code` kwarg, not positional.**
   The constructor is `TensorZeroError(status_code=400, text="...")`. Tests
   that construct it for the error queue must use keyword arguments.

## Final status

| Item | Validation | Result |
|---|---|---|
| F7.1 (duration_ms) | Unit: `TestLoggingMicrogaps::test_f7_1_duration_ms_logged_for_tool_call` + `::test_f7_1_duration_ms_logged_on_error` | checkmark |
| F7.2 (tool names) | Unit: `TestLoggingMicrogaps::test_f7_2_tool_names_in_dispatch_log` | checkmark |
| F7.3 (content_types) | Unit: `TestLoggingMicrogaps::test_f7_3_content_type_breakdown_logged` | checkmark |
| F7.4 (Thought DEBUG) | Unit: `TestLoggingMicrogaps::test_f7_4_thought_text_logged_at_debug` | checkmark |
| F7.5 (response_source) | Unit: 5 tests in `TestThoughtOnlyHandling` (extraction, None fallback, Text priority, multi-thought join, fallback log) | checkmark |
| F7.6 (tool_choice) | Unit: 3 tests in `TestToolChoice` (forwarding, None default, literal values) | checkmark |
| F7.7 (error catch) | Unit: 7 tests in `TestRecoverableInferenceError` (catch + retry, available tools, re-raise non-matching, re-raise non-400, warning log, sentinel ID, extensibility) | checkmark |
| F7.8 (dryrun docs) | Unit: `TestDryrunDocstring::test_f7_8_dryrun_feedback_constraint_documented` | checkmark |
| F7.9 (flake benchmark) | Manual: script imports cleanly, CLI parses | checkmark |
| F7.10 (trace episode) | Manual: script imports cleanly, CLI parses (requires docker compose for full test) | checkmark |
| F7.11 (system prompt example) | Manual: directory structure and config created with system_template directive | checkmark |
| F7.12 (regression) | 102 unit tests passed. `ruff check src/ tests/ scripts/`: clean. `mypy src/`: clean. | checkmark |

Test totals after Phase 7: **126 total** (102 unit + 24 integration). Breakdown:
- 102 unit tests: 81 pre-existing + 21 new Phase-7 (`test_harness.py`)
- 24 integration tests: unchanged (Phase 7 adds no new integration tests)

Phase 7 closed.
