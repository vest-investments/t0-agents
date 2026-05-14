# Milestone 4 — Dev Log

> Implementation log for `M4 — Provider Safety, Streaming Guard,
> Quirks`.  Written 2026-05-14 at the end of the milestone, per the
> always-write policy.  Subsequent milestones must read this log (and
> the M1–M3 logs) before planning.

---

## §1 Summary — what was done

### Files added (4)

- `src/t0_agents/guardrails/structured.py` (86 lines) — new
  `require_parsed()` factory: returns a `Guardrail` whose `after_run`
  raises `GuardrailBlockedError` when a JSON run completes with
  `result.parsed is None`.  Chat runs are a strict no-op.
- `tests/test_core/test_agent_json_safety.py` (347 lines) — 10 unit
  tests across four classes: `TestToolModeCheck` (3),
  `TestStreamGuard` (2), `TestAnthropicWarning` (3),
  `TestAfterRunFatalPropagation` (2).  Carries a local `FakeJsonT0` /
  `_fake_json_response` (M5 will consolidate into `MockGateway`).
- `tests/test_guardrails/test_require_parsed.py` (144 lines) — 3
  unit tests for the new guardrail (passes on valid parse, aborts on
  `parsed=None`, no-op on chat runs).
- `tests/test_core/test_agent_json_safety_integration.py` (53 lines)
  — 1 integration test (`@pytest.mark.integration`) verifying the
  stream guard fires at first `__anext__` against the live gateway.

### Files modified (7)

- `src/t0_agents/core/agent.py` (+101 lines).  Five edit sites:
  - `Agent.__init__` — adds two instance flags
    (`_needs_tool_mode_check`, `_warned_anthropic_combination`).
  - `Agent.stream` docstring + a new "Step 0" guard at the top of
    the async-generator body that raises `ConfigurationError` when
    `config.function_kind == "json"`.
  - `Agent._run_json` — new "Step 0" safety block (Anthropic warning
    + tool-mode `list_tools()` check) runs before memory strategy
    resolution.
  - Three `after_run` exception handlers (chat `run`, `stream`,
    `_run_json`) gain `except T0AgentsFatalError: raise` above the
    bare `except Exception`, restoring the documented contract.
  - New private method `Agent._maybe_warn_anthropic_quirk()` in the
    private-helpers section.
- `src/t0_agents/guardrails/__init__.py` — re-exports
  `require_parsed`; module docstring built-ins list updated.
- `tests/_fake_mcp.py` — bug fix: `FakeMCPClient.__init__` honoured
  `tools=[]` as falsy (collapsed to the default one-tool list).
  Switched to `tools if tools is not None else [...]`.  No test had
  semantically relied on the previous behaviour — all callers either
  passed a non-empty list or did not invoke `list_tools()`.
- `docs/structured-outputs.md` — new "Safety & provider quirks"
  section with four subsections (configuration errors, why no
  streaming JSON, Anthropic warning, undetectable provider quirks,
  `require_parsed`).  Milestone summary block at the top updated to
  M1–M4.  "What's coming" trimmed to M5 only.
- `docs/guardrails.md` — new `### require_parsed` subsection under
  Built-in guardrails; `Factories` API-reference table gains the
  new row.
- `docs/internal/structured-outputs/02-open-questions.md` — Q10
  resolved with citations to the implementation and tests.
- `CHANGELOG.md` — new `## [0.10.0-M4] — Unreleased` block prepended
  above M3.  Sections: Added (`require_parsed`, stream guard,
  tool-mode conflict check, Anthropic warning), Changed (`after_run`
  fatal-error propagation realignment), Compatibility (chat
  byte-identical; `mcp/` byte-identical; opt-out path for callers
  who relied on the old swallow-everything behaviour).

### Public API delta

- `t0_agents.guardrails.require_parsed` — new factory.  Not
  re-exported at `t0_agents.*` (consistent with the other built-ins
  `block_tools`, `max_cost`, `max_tool_calls`).
- `Agent.run(...)` and `Agent.stream(...)` signatures unchanged.
- `AgentConfig` and `AgentResult` shapes unchanged.

### Tests added

| File | Count |
|------|------:|
| `tests/test_core/test_agent_json_safety.py` | 10 unit |
| `tests/test_guardrails/test_require_parsed.py` | 3 unit |
| `tests/test_core/test_agent_json_safety_integration.py` | 1 integration |
| **Total (M4)** | **14 new tests** |

Combined with M1's 19 + M2's 13 + M3's 31, the structured-outputs
surface now ships ~77 tests across `tests/test_core/` and
`tests/test_guardrails/`.  Offline suite total: 609 (M3 baseline
596 + 13 new offline tests).

### Docs added or modified

- New: 1 (this dev log).
- Modified: 4 (`docs/structured-outputs.md`, `docs/guardrails.md`,
  `docs/internal/structured-outputs/02-open-questions.md`,
  `CHANGELOG.md`).

---

## §2 Deviations from the milestone spec

### §2.1 `require_parsed` lives in `guardrails/structured.py`, not `guardrails/builtins.py`

The spec §4.2 calls for the factory to land in
`src/t0_agents/guardrails/builtins.py`, but no such file exists.
The repo organises built-ins one-file-per-family (`approval.py`,
`limits.py`, `validators.py`).  Creating `builtins.py` would have
broken the convention and forced future built-ins to choose between
"new family file" and "extend builtins.py".

**What changed:** new file `guardrails/structured.py` — name picked
to mirror the structured-outputs roadmap and to match the
"result-shape guardrail" family.

**Impact on subsequent milestones:** none load-bearing.  M5 may add
additional structured-shape guardrails (e.g. schema-conformance
checks) under the same module.  The re-export at
`t0_agents.guardrails.__init__` is the only public surface.

### §2.2 `Agent.stream` signature unchanged (no per-call kwargs added)

The M2 dev-log §4 floated the option of adding `output_schema` /
`json_mode` kwargs to `Agent.stream` to mirror `Agent.run`.  The M4
spec did not require it, and the user confirmed at planning time we
should leave the signature alone.

**What changed:** nothing — the stream signature is identical to
v0.9.0.  Adding kwargs that would be immediately rejected was
unreachable surface for no caller benefit and would have required
rejection tests on chat agents too.

**Impact on subsequent milestones:** none.  If TZ ever adds
mid-stream `parsed` emission, M5+ revisits.

### §2.3 Anthropic warning fires from `_run_json`, not `Agent.__init__`

Spec §4.1.2 allows the warning to fire from either `Agent.__init__`
or the first `run`.  Picked the lazy variant (top of `_run_json`)
for two reasons:

1. **Consistency with the tool-mode check** (also async-deferred to
   `_run_json`).  Both safety surfaces fire at the same point in the
   lifecycle.
2. **No warning for teardown agents.**  Setup code that constructs
   an `Agent`, performs a sanity check, then discards it does not
   emit the warning.  An init-time warning would have logged for
   every dropped instance.

**Impact on subsequent milestones:** none.  The latch
(`_warned_anthropic_combination`) makes it once-per-instance
regardless of trigger point.

### §2.4 Pre-existing `after_run` exception-handler bug fixed in scope

The Plan agent's pre-implementation validation surfaced that the
three `after_run` lifecycle handlers in `core/agent.py` (chat
`run`, `stream`, `_run_json`) all wrap the hook call in a bare
`except Exception` that silently swallows `T0AgentsFatalError`.
`docs/guardrails.md:154-158` documents the opposite contract: *"A
custom hook that needs to abort should raise a subclass of
`T0AgentsFatalError`, which bypasses the non-fatal trap."*  The
`before_run` analogue at line 421 already honoured the contract;
the three `after_run` sites did not.

**What changed:** added `except T0AgentsFatalError: raise` ahead of
the bare `except Exception` at all three sites.  Tied to M4 because
`require_parsed` is the first built-in to actually exercise the
`after_run` hard-abort path — shipping it without the fix would
have made the new guardrail effectively dead code.

**Why ship it now rather than punt:** the user confirmed during
planning.  Three-line change per site (one `except T0AgentsFatalError:
raise` clause), mechanical, mirrors `before_run`.  The CHANGELOG
documents the opt-out path for callers who depended on the
swallow-everything behaviour (replace the raise with a logger call
in their own implementation).

**Impact on subsequent milestones:** none — M5 / future milestones
inherit the corrected behaviour.  Other built-in guardrails
(`require_approval`, `max_cost`, `max_tool_calls`, `block_tools`)
do not raise fatal errors from `after_run`, so the alignment is
observationally compatible with every existing built-in.

### §2.5 Pre-existing `FakeMCPClient` falsy-coercion bug fixed in scope

`tests/_fake_mcp.py` constructed `_tools` via `tools or [<default>]`,
which collapsed `tools=[]` to the default one-tool list.  The bug
was harmless for chat-stream tests (they never inspected the empty
list) but broke the M4 tool-mode-with-empty-tools acceptance test.

**What changed:** `tools=[]` now produces an empty list; `tools=None`
still falls back to the default `[MCPTool(name="add", ...)]`.  Other
test files that passed `tools=[]` were either calling no tools or
not asserting on `list_tools()` output, so all 609 offline tests
still pass.

**Impact on subsequent milestones:** none.

### §2.6 14 tests instead of the spec's ~10

The spec called for ~10 tests.  M4 ships 14 because:

- `TestToolModeCheck` grew to 3 tests instead of 2 — added the
  caching assertion so the M4 "verify once, cache after" design is
  pinned (the spec's risks table calls this out as a follow-up).
- `TestAnthropicWarning` grew to 3 tests instead of 2 — split the
  "fires for json_mode in {on, tool}" case into two tests so each
  branch is independently verifiable.
- `TestAfterRunFatalPropagation` (2 tests, both new) covers the
  pre-existing-bug fix that landed alongside M4 (see §2.4).  Without
  these, the fix would be unprotected against regression.

No carry-forward implication.

---

## §3 Open questions raised or resolved

### Resolved

- **Q10** (provider auto-detection for safety warnings) — resolved
  exactly as hypothesised.  `Agent` emits a generic
  `logger.warning` once per instance when
  `config.json_mode in {"on", "tool"}`, no auto-detection.
  Implementation in `src/t0_agents/core/agent.py
  ::Agent._maybe_warn_anthropic_quirk`; test coverage in
  `tests/test_core/test_agent_json_safety.py::TestAnthropicWarning`
  (3 tests).

### Raised (new during M4)

None.  The single open follow-up question — whether to apply
`permissive_parse` on assembled streaming `raw` deltas — is already
deferred to M5+ per M3 dev-log §4 and M4 spec §2 "out of scope".

---

## §4 Carry-forward notes

### For M5 — Testing helpers + polish

- **`MockGateway` JSON consolidation.**  M1, M2, and M3 each
  shipped a local `FakeJsonT0` / `_fake_json_response` pair; M4
  adds two more copies (in `test_agent_json_safety.py` and
  `test_require_parsed.py`).  When M5 lands the JSON branch of
  `MockGateway`, four local fakes can be deleted in one pass.  Two
  of the M4 fakes are minimal (do not need `inference()` call
  recording); the others (M3's `test_agent_json_retry.py`, M4's
  `test_agent_json_safety.py::FakeJsonT0`) need the multi-call
  scripted shape with kwargs capture.  Mirror the more capable
  shape; the minimal users can ignore the extra surface.

- **Shape-contract test
  (`tests/test_testing/test_mock_shape.py`).**  Currently chat-only;
  M5 extends to JSON.  Confirm the eventual JSON branch covers
  `episode_id`, `inference_id`, `output.raw`, `output.parsed`,
  `usage`, and `variant_name` (every field read by `_run_json`).

- **Per-call retry-hint override hook.**  M3 dev-log §4 flagged
  this as a future i18n / DX improvement.  Still deferred — no M4
  caller hit the limitation.  Cleanest path: a new
  `Callable[[Mapping | None], str]` kwarg on `Agent.run`.

- **`require_parsed` + schema-conformance pairing recipe.**  The
  guardrail intentionally does not run JSON-Schema validation
  (Q12).  Add a one-paragraph recipe to
  `docs/structured-outputs.md` showing the Pydantic-validator
  combo (`Model.model_validate(result.parsed)`) when M5 lands the
  cross-link pass.

- **Stream + `permissive_parse` for JSON deltas.**  M3 dev-log §4
  floated buffering `raw` deltas and running `permissive_parse` on
  the assembled string before the terminal event.  M4's stream
  guard makes this conceptually a future
  positive-feature replacement for the current `ConfigurationError`,
  not a contract break.  Still out of scope through M5.

### For future milestones (post-M5)

- **JSON-Schema validation as a built-in `require_schema(model_or_schema)`
  guardrail.**  Pydantic is already pulled via `[serve]`; a small
  helper that wraps `model.model_validate(result.parsed)` in an
  `after_run` hook would be a natural sibling to `require_parsed`.
  Out of roadmap scope; flagging here so the design space is
  explicit.

- **Anthropic provider auto-detection.**  Q10 is now resolved as
  "generic warning, no detection".  If TensorZero adds a function-
  config introspection API in a future release, the warning could
  upgrade to a precise check.  Revisit then.

---

## §5 Verification

Commands run during M4 close-out (all from the repo root,
2026-05-14):

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/` | passed — `All checks passed!` |
| `uv run mypy src/` | passed — `Success: no issues found in 55 source files` |
| `uv run pytest -m "not integration"` | passed — 609 passed, 2 skipped, 53 deselected (2.74 s).  M3 baseline was 596 passed / 52 deselected; the +13 offline and +1 deselected match the 13 new offline tests and 1 new integration test added by M4. |
| `uv run pytest tests/test_core/test_agent_json_safety.py -v` | passed — 10/10 green in 0.09 s. |
| `uv run pytest tests/test_guardrails/test_require_parsed.py -v` | passed — 3/3 green in 0.08 s. |
| `git diff src/t0_agents/mcp/` | empty (contract §2.1 invariant holds). |
| `git diff tests/test_mcp/ tests/test_stream/ tests/test_serve/ tests/test_compose/ tests/test_memory/ tests/test_cli/ tests/test_testing/` | empty (chat-byte-identity invariant holds for every existing test surface). |
| `git diff --stat` | 7 files modified (+331 / -12); 4 new files added. |

### Integration tests

`uv run pytest -m integration` — **passed (user verification on
2026-05-14)**.  All 53 selected integration tests green in 51.20 s,
including the new `test_agent_json_safety_integration::test_stream_guard_against_real_gateway`
and the existing M1–M3 integration suites
(`test_agent_json_run_integration`, `test_agent_json_retry_integration`,
`test_structured_integration`) plus the full v0.9.0 suite
(`test_mcp`, `test_compose`, `test_agent_integration`,
`test_guardrails`, `test_memory`, `test_serve`, `test_cli`).

---

*End of Milestone 4 dev log.*
