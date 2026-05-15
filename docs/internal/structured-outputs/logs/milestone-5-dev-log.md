# Milestone 5 — Dev Log

> Implementation log for `M5 — Testing helpers, examples, polish`.
> Written 2026-05-14 at the end of the milestone, per the always-write
> policy.  Closes the structured-outputs roadmap.

---

## §1 Summary — what was done

### Files added (7)

- `tests/test_testing/test_json_helpers.py` (~290 lines, 12 unit tests).
  Two classes: `TestScriptedJsonGateway` (7 tests — basic dict, missing
  fields defaulted, auto-derive parsed, explicit parsed-None wins,
  `canned_json` kwargs capture, pre-built `JsonInferenceResponse`
  pass-through, exhausted-script `AssertionError`) and
  `TestJsonAssertHelpers` (5 tests covering every new assertion's
  positive + negative paths and the `jsonschema`-missing branch).
- `examples/17_structured/ner_dynamic_schema.py` (~175 lines).  Four
  scenarios: default strict + four-category schema; per-call
  `output_schema` override adding a `products` category; per-call
  `json_mode="tool"` swap (legal because no tools attached); recovery
  demo with `json_mode="off"` exercising the M3 permissive layer.
- `examples/17_structured/config/functions/extract_entities/output_schema.json`
  (~25 lines, four-category NER schema).
- `examples/17_structured/config/functions/extract_entities/system_template.minijinja`
  (~12 lines, minijinja prompt).
- `config/functions/extract_entities/output_schema.json` (mirror of the
  example schema at the repo root).
- `config/functions/extract_entities/system_template.minijinja` (mirror
  of the example template at the repo root).
- `docs/internal/structured-outputs/logs/milestone-5-dev-log.md` (this
  file).

### Files modified (15)

- `src/t0_agents/testing/mocks.py` — imports `JsonInferenceOutput`,
  `JsonInferenceResponse`; widens `_SourceReturn` and
  `MockGateway.inference()` return type to the
  `ChatInferenceResponse | JsonInferenceResponse` union; adds
  `MockJsonResponse` builder; adds `MockGateway.script_json` /
  `canned_json` classmethods; adds the private `_try_parse_json`,
  `_coerce_uuid`, and `_materialise_json_dict` helpers; widens
  `RecordingGateway` typing the same way; updates `__all__`.
- `src/t0_agents/testing/assertions.py` — imports `ParseStrategy`
  alongside `AgentResult`; filters JSON responses out of
  `_collect_gateway_tool_calls` (chat-only); adds five new helpers
  (`assert_parsed_equals`, `assert_parsed_matches_schema`,
  `assert_invalid_json`, `assert_parse_strategy`,
  `assert_retry_attempts`); soft-imports `jsonschema` inside the
  schema helper with a `# type: ignore[import-untyped]` for mypy
  strict mode; updates `__all__`.
- `src/t0_agents/testing/__init__.py` — re-exports `MockJsonResponse`
  and the five new asserts; extends the public-surface docstring and
  `__all__`.
- `tests/test_testing/test_mock_shape.py` — adds `_JSON_INFERENCE_KWARGS`
  list (kwargs the JSON dispatch passes to `t0.inference`) and
  `TestMockGatewayJsonShape` with three tests (script_json accepts
  every JSON loop kwarg; the real `inference` signature still accepts
  `output_schema`/`params`; the materialised `JsonInferenceResponse`
  exposes every field `Agent._run_json` reads).
- `tests/test_core/test_structured.py` (M1) — drops the local
  `FakeJsonT0` / `_fake_json_response` pair; rewires the six tests to
  `MockGateway.script_json` or the lower-level `MockGateway(source)`
  constructor (for the chat-response-on-JSON-call sniff test).
- `tests/test_core/test_agent_json_run.py` (M2) — same refactor across
  the 14 tests; introduces a single `_OK_RAW` constant to avoid
  repeating the customer-info JSON across happy-path scripts.
- `tests/test_core/test_agent_json_retry.py` (M3) — refactors the 9
  retry-orchestration tests; replaces the `t0._queue` length probe
  with `len(t0.calls)` (equivalent assertion, public surface).
- `tests/test_core/test_agent_json_safety.py` (M4) — refactors the
  13 safety tests; uses a single `_OK_ENTRY` constant for the
  default valid-JSON dict; escapes regex metacharacters in
  `pytest.raises(match=...)` patterns flagged by RUF043.
- `tests/test_guardrails/test_require_parsed.py` (M4) — refactors the
  three guardrail tests; smallest of the five refactors.
- `examples/17_structured/README.md` — extends the intro to five entry
  points; documents the NER example's four scenarios in a new "What
  `ner_dynamic_schema.py` demonstrates" section; updates the Run block
  and the What-to-observe block.
- `examples/17_structured/config/tensorzero.toml` — appends the
  `[functions.extract_entities]` block.
- `config/tensorzero.toml` — appends the same
  `[functions.extract_entities]` block at the repo root.
- `docs/structured-outputs.md` — replaces the M1-M4 milestone summary
  line with a five-milestone summary; expands the Pydantic recipe in
  "Schema validation in Python" with `ValidationError` handling and
  nested-model coverage; inserts a new "Testing your JSON agent"
  section before "Safety & provider quirks"; replaces "What's coming"
  with "Roadmap status" (closure note + deferred follow-ups); expands
  the See-also footer to link `agent.md`, `guardrails.md`, `memory.md`,
  `testing.md`, `streaming.md`, and `examples/17_structured/`.
- `docs/testing.md` — adds a "Testing JSON agents" subsection inside
  `## MockGateway` (after "What's recorded") and a "JSON assertions"
  subsection inside `## Assertions` (Five → Ten helpers).
- `docs/streaming.md` — adds a "Chat-only" callout block right after
  the intro pointing at the JSON-stream guard in
  `structured-outputs.md`.
- `docs/memory.md` — refines the `agent_response` / `turns` /
  `inference_ids` rows in the `MemoryRecord` field table with JSON-
  specific semantics and links to `structured-outputs.md`.
- `docs/index.md` — adds a Structured outputs entry to the Appendices
  list; mentions JSON dispatch in the `t0_agents.core` row of the
  module table; expands the `t0_agents.testing` row to mention
  `script_json` / `canned_json` and the five JSON asserts.
- `CHANGELOG.md` — consolidates the four sub-sections
  `## [0.10.0-M1]` through `## [0.10.0-M4]` plus M5 into a single
  `## [0.10.0] — Unreleased` block organised by area (Added /
  Changed / Compatibility).  Preserves every concrete bullet from
  the per-milestone entries.

### Public API delta

- `from t0_agents.testing import MockJsonResponse` (new).
- `from t0_agents.testing import (`
  `    assert_parsed_equals, assert_parsed_matches_schema,`
  `    assert_invalid_json, assert_parse_strategy, assert_retry_attempts,`
  `)` — five new helpers.
- `MockGateway.script_json(turns, *, episode_id=None)` classmethod.
- `MockGateway.canned_json(fn, *, episode_id=None)` classmethod.
- `MockGateway.inference()` return-type annotation widened to
  `ChatInferenceResponse | JsonInferenceResponse`.

### Tests added

| File | New tests |
|------|-----------|
| `tests/test_testing/test_json_helpers.py` | 12 unit |
| `tests/test_testing/test_mock_shape.py` (extended) | 3 unit |
| **Total (M5)** | **15 new offline tests** |

Combined with M1's 19 + M2's 13 + M3's 31 + M4's 17, the
structured-outputs surface ships **~95** tests across `tests/test_core/`,
`tests/test_guardrails/`, and `tests/test_testing/`.  Offline suite
total: **627** (M4 baseline 612 + 15 new).  No tests deleted or
renamed in the five-file FakeJsonT0 refactor.

### Docs added or modified

- New: 1 (this dev log).
- Modified: 6 (`docs/structured-outputs.md`, `docs/testing.md`,
  `docs/streaming.md`, `docs/memory.md`, `docs/index.md`,
  `CHANGELOG.md`).

---

## §2 Deviations from the milestone spec

### §2.1 `testing/asserts.py` → `testing/assertions.py`

The spec (§2 In-Scope, §4.2, §8) repeatedly says `testing/asserts.py`,
but the actual module file is `testing/assertions.py` (and has been
since at least v0.9.0).  M5 extends the existing file; introducing a
new `asserts.py` alongside it would have split the assertion surface
across two modules for no benefit.

**Impact on subsequent milestones:** N/A — M5 closes the roadmap.

### §2.2 `examples/16_structured/` → `examples/17_structured/`

The spec (§2, §6, §7, §8) says `examples/16_structured/`, but
`examples/16_parallel/` already exists.  Corrected by M1 dev-log §2.1
and inherited unchanged by M2, M3.  M5's NER example lands at
`examples/17_structured/ner_dynamic_schema.py`.

### §2.3 `JsonResponseLike` → `MockJsonResponse`

The spec (§4.1) names the response builder `JsonResponseLike` and
explicitly says it "does **not** import from `tensorzero.types`
(testing/ is base, no extra dependency)".  Both points were wrong:

1. **TensorZero is a base dependency** of t0-agents (declared in
   `pyproject.toml` `[project] dependencies`), so the testing module
   has always been free to import from it.  `testing/mocks.py:30-35`
   already imports `ChatInferenceResponse`, `ContentBlock`, `Text`,
   `Usage`.  M5 adds `JsonInferenceOutput` and `JsonInferenceResponse`
   the same way and materialises real SDK objects rather than ducks.
2. **Naming.**  `JsonResponseLike` would have been confusingly
   asymmetric with the existing `MockChatResponse` builder.  M5 ships
   `MockJsonResponse` for parity; both expose `to_response()`.

This is a contract clarification, not a behavioural change.  The
0-integration-contract is not amended; the spec's §3 third bullet is
the inaccuracy.

### §2.4 `jsonschema` strategy — soft import only

The spec (§3 Constraints) outlined two options for `jsonschema`
handling: soft import or a documented `pip install jsonschema`
recommendation.  User chose **soft import only** during planning.

**What changed:** `assert_parsed_matches_schema` imports `jsonschema`
inside the function body (not at module load), wrapped in a
`try / except ImportError`.  The raised `ImportError` carries the
install hint `pip install jsonschema`.  mypy strict mode is satisfied
with a single `# type: ignore[import-untyped]` on the import line.
`pyproject.toml` is unchanged — no `[validation]` extra was added.

**Impact:** none load-bearing.  Future milestones that want a stricter
contract (e.g. a `require_schema(model_or_schema)` built-in
guardrail) can choose to upgrade `jsonschema` to a regular dependency
or add an extra.  Both are non-breaking from this baseline.

### §2.5 `MockGateway.script_json` auto-derive when `parsed` key absent

The spec (§4.1) said dict entries carry `parsed: Any | None` without
specifying the absent-key semantic.  User chose during planning:
**auto-derive when the key is absent; honour explicit value (even
`None`) when present**.

**Why:** M1/M2 fakes auto-derived (terse for happy-path tests); M3/M4
fakes required explicit `parsed` (essential for retry scripts where
`raw` is parseable but TZ returned `parsed=None`).  The chosen
semantic supports both styles with no DSL.  Covered by
`test_script_json_auto_derives_parsed` and
`test_script_json_explicit_parsed_none_wins` in
`tests/test_testing/test_json_helpers.py`.

### §2.6 Full sweep of local `FakeJsonT0` fakes

The spec (§4.5 in spirit; §4 says "No other module changes") implied
M5 should add the new API without touching existing test files.  User
chose during planning: **full sweep — refactor all 5 test files**
(`test_structured.py`, `test_agent_json_run.py`,
`test_agent_json_retry.py`, `test_agent_json_safety.py`,
`test_require_parsed.py`) to use `MockGateway.script_json` and delete
every local `FakeJsonT0` / `_fake_json_response` pair.

**Why:** the M3 dev-log §4 carry-forward and M4 dev-log §4 carry-
forward both flagged this as future work; landing it in M5 closes the
loop and leaves zero locally-rolled JSON fakes in the test tree.

**Impact:** 5 test files were rewritten end-to-end.  No test deleted
or renamed; every assertion preserved.  Each refactor was followed
by a per-file `pytest -v` run to confirm 100 % preservation (12 + 14
+ 9 + 13 + 3 = **51 tests** across the five files, all green).

### §2.7 12 unit tests in `test_json_helpers.py` (spec called for 10)

The spec §5.1 listed exactly 10 tests.  The implementation ships 12:

- `test_script_json_passes_through_prebuilt` (extra) — confirms
  `JsonInferenceResponse` instances passed in the script bypass the
  dict materialiser entirely (identity check).
- `test_script_json_raises_when_exhausted` (extra) — pins the
  "no graceful empty turn" semantic the JSON queue ships with;
  important regression guard since the chat queue's empty-fallback
  silently returns `Text("")`.

Neither test alters the spec's contract; they harden two adjacent
edge cases.

### §2.8 `docs/adoption-guide.md` skipped

The spec (§7.3) leaves adoption-guide updates to the implementer.
User chose during planning: **skip**.  The adoption guide's audience
is orchestration-pattern adoption (chat-loop + tools, HTTP serving,
CLI); JSON-function coverage lives at `docs/structured-outputs.md`
where it is discoverable through the See-also footer.  A future
revision of the guide can add a `§7-bis` paragraph if the audience
changes.

Recorded here so a future reader does not re-investigate.

### §2.9 No version bump in `pyproject.toml`

The spec (§8 CHANGELOG consolidation) leaves the version bump to the
user.  Default decision: **defer to release time**.  `pyproject.toml`
remains at its current version; the consolidated CHANGELOG entry is
`## [0.10.0] — Unreleased`.  When the user is ready to cut a release,
they update both files together.

### §2.10 Pre-existing test-tree ruff warnings

`uv run ruff check tests/` surfaces 7 pre-existing failures in
unrelated files (`tests/test_cli/test_elicit_tui.py`,
`tests/test_core/test_agent.py`, `tests/test_core/test_agent_parity.py`,
`tests/test_memory/test_agent_memory.py`).  These pre-date M5 and
were noted in M1 dev-log §5.  Per CLAUDE.md the required gate is
`uv run ruff check src/`, which passes cleanly.  M5 fixed the four
RUF043 (regex-metacharacter in `pytest.raises(match=...)`) findings
that the refactor introduced (two in `test_agent_json_safety.py`'s
`§11.1` patterns and two in `test_json_helpers.py`'s
`result.parsed`-mentioning patterns), keeping the M5-touched files
clean.

---

## §3 Open questions raised or resolved

### Raised (new during M5)

None.  Every M5 design decision was either covered by the spec, the
contract, the prior dev-logs, or the user's planning answers.

### Resolved

No previously-open questions targeted M5.  Q12 (schema validation
post-permissive) remains decided as "out of scope" since M3, and M5
ships `assert_parsed_matches_schema` as a **testing-only** helper —
the production code path still does not run JSON-Schema validation
in Python.

---

## §4 Carry-forward notes

This is the end-of-roadmap log.  The items below are recorded for a
**future** structured-outputs roadmap or a v0.11.0 release cycle; they
are explicitly out of scope for v0.10.0.

### Streaming JSON with permissive recovery on assembled raw deltas

M3 dev-log §4 and M4 dev-log §4 both deferred this.  TensorZero
streams emit `raw` deltas without a terminal `parsed` field (guide
§7.1, §11.3).  A future milestone could buffer every chunk's `raw`,
concatenate, run `permissive_parse` once at the terminal event, and
emit a single `JsonReadyEvent` with the recovered `parsed`.  The
current `ConfigurationError` from `Agent.stream(...)` on a JSON agent
would become a fallback when the buffer-and-parse approach fails.
Requires extending `AgentEvent` with a new variant and revising the
contract §3.3.

### `require_schema(model_or_schema)` built-in guardrail

M4 dev-log §4 floated this.  Pydantic is already pulled by the
`[serve]` extra; a small helper that wraps
`model.model_validate(result.parsed)` (or `jsonschema.validate(...)`
when the caller passes a dict) in the `after_run` hook would be a
natural sibling to `require_parsed`.  Today the equivalent must be
written by every consumer; the Pydantic recipe at
`docs/structured-outputs.md#schema-validation-in-python` shows the
shape.  Out of v0.10.0 scope; flagged here so the design space is
explicit.

### Per-call retry-hint override hook

M3 dev-log §4 deferred an i18n / per-agent override for the
corrective retry hint.  Cleanest API today is a new kwarg on
`Agent.run`: `retry_hint: Callable[[Mapping[str, Any] | None], str] |
None = None`, the orchestrator swaps `_build_retry_hint(schema)` for
the callable's output when supplied.  No structural refactor needed.

### Provider auto-detection for the Anthropic warning

Q10 was resolved as "generic warning, no detection".  If a future
TensorZero release exposes a function-config introspection API (the
M1 Q1 resolution noted the SDK does not in v2026.4.0), the warning
could upgrade to a precise check that fires only when the configured
variant routes to an Anthropic extended-thinking model.  Revisit then.

### `MockGateway.script_json` queue introspection

The M3 refactor replaced `len(t0._queue) == 1` (private attr) with
`len(t0.calls) == 1` (public surface).  The migration is clean today,
but if a future test genuinely needs to inspect the unconsumed-script
tail, exposing a `MockGateway.pending_script` read-only property is a
two-line addition.  Not needed for any current test.

### Lifting the `memory_strategy="messages"` guard on JSON runs

M2 dev-log §2.4 introduced the strategy-level guard; M3 extended
`run_structured_inference` with `prior_messages` but reserved it for
the retry orchestration.  Lifting the guard requires translating
loaded `MemoryRecord`s into the helper's message shape — its own
design exercise.  Today's behaviour ("strict reject") is
intentionally conservative.

---

## §5 Verification

Commands run during M5 close-out (all from the repo root,
2026-05-14):

| Command | Outcome |
|---------|---------|
| `uv run ruff check src/` | passed — `All checks passed!` |
| `uv run mypy src/` | passed — `Success: no issues found in 55 source files` |
| `uv run pytest -m "not integration"` | passed — **627 passed**, 2 skipped, 53 deselected (2.93 s).  M4 baseline 612 + 15 new offline tests (12 in `test_json_helpers.py` + 3 in `test_mock_shape.py`).  No tests dropped by the five-file FakeJsonT0 refactor. |
| `uv run pytest tests/test_testing/test_json_helpers.py -v` | passed — 12/12 green in 0.10 s. |
| `uv run pytest tests/test_testing/test_mock_shape.py -v` | passed — 9/9 green (6 chat + 3 JSON). |
| `uv run pytest tests/test_core/test_structured.py tests/test_core/test_agent_json_run.py tests/test_core/test_agent_json_retry.py tests/test_core/test_agent_json_safety.py tests/test_guardrails/test_require_parsed.py -v` | passed — 45/45 green (refactored M1+M2+M3+M4 JSON tests). |
| `uv run ruff check src/ tests/test_testing/ tests/test_core/test_structured.py tests/test_core/test_agent_json_run.py tests/test_core/test_agent_json_retry.py tests/test_core/test_agent_json_safety.py tests/test_guardrails/test_require_parsed.py examples/17_structured/` | passed — `All checks passed!` |
| `uv run mypy src/ examples/17_structured/ner_dynamic_schema.py` | passed (NER example clean under strict mode). |
| `git diff src/t0_agents/mcp/` | empty (contract §2.1 invariant holds). |
| `git diff src/t0_agents/core/` | empty (M5 ships no `core/` changes; the runtime was complete at M4). |
| `git diff src/t0_agents/guardrails/` | empty. |
| `grep -c '^## \[0\.10\.0\]' CHANGELOG.md` | `1` (consolidation succeeded; the four `0.10.0-Mx` sub-sections collapsed into a single `0.10.0` entry). |
| `git diff --stat src/t0_agents/` | three files modified (`testing/__init__.py`, `testing/assertions.py`, `testing/mocks.py`); +392 / −25 lines. |

### Integration tests

`uv run pytest -m integration` — **pending user verification** (the
M5-touched surface is testing helpers + docs + examples; no
behavioural changes to the runtime).  The five M1-M4 integration test
files (`test_structured_integration.py`,
`test_agent_json_run_integration.py`,
`test_agent_json_retry_integration.py`,
`test_agent_json_safety_integration.py`) are unchanged and should
remain green when `docker compose up -d` is up with `GROQ_API_KEY`
configured.

`uv run python examples/17_structured/ner_dynamic_schema.py` —
**pending user verification** (requires `docker compose up -d` plus
`GROQ_API_KEY`).  The example imports cleanly and passes mypy/ruff;
the four scenarios exercise the M1-M4 runtime that the user
previously verified end-to-end.

---

*End of Milestone 5 dev log.  Roadmap closed.*
