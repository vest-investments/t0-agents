# Milestone 5 — Testing Helpers, Examples, Polish

> Closes the structured-outputs roadmap.  Adds the JSON-aware testing
> primitives (so consumers of `t0-agents` can write fast offline tests
> against their JSON agents), ships a multi-feature NER example,
> consolidates docs, and produces the final `## [0.10.0] — Unreleased`
> CHANGELOG entry from the per-milestone sub-sections.

## §1 Goal

Make every primitive landed in M1–M4 testable offline without docker,
provide one end-to-end example that exercises the full surface (dynamic
schema override, `json_mode` swap, retry layer, structured guardrail),
and leave the public docs in a state where a reader can adopt JSON
mode without reading the internal contract.

## §2 Scope

### In scope

- `MockGateway` learns to produce `JsonInferenceResponse`-shaped
  responses.  New construction modes: `script_json(...)` and
  `canned_json(...)`.
- `MockToolProvider` unchanged (no tools in JSON runs).
- New assertion helpers in `t0_agents.testing.asserts`:
  - `assert_parsed_equals(result, expected)`.
  - `assert_parsed_matches_schema(result, schema)` (uses
    `jsonschema` if available; falls back to a recipe in docs if
    not — see §3 constraints).
  - `assert_invalid_json(result)`.
  - `assert_parse_strategy(result, strategy)`.
  - `assert_retry_attempts(result, n)`.
- Cross-link `docs/structured-outputs.md` from `docs/index.md`,
  `docs/concepts.md`, `docs/agent.md`, `docs/streaming.md`,
  `docs/guardrails.md`, `docs/memory.md`, `docs/testing.md`.
- Example `examples/16_structured/ner_dynamic_schema.py` — NER
  using `extract_entities` (guide §12.2) with per-call schema
  override and a per-call `json_mode = "tool"` swap demo.
- Shape-contract test (`tests/test_testing/test_mock_shape.py`)
  extended to cover JSON inference shape.
- Final CHANGELOG consolidation: merge `## [0.10.0-M1..M5]` sub-
  sections into a single `## [0.10.0] — Unreleased` entry.

### Out of scope

- New runtime features (everything functional ships in M1–M4).
- New built-in guardrails beyond `require_parsed()`.
- Streaming JSON support (denied by M4).

### Non-goals

- Adding `jsonschema` as a hard dependency.  If available, use it for
  `assert_parsed_matches_schema`; otherwise, the helper raises a
  clear error suggesting `pip install jsonschema` or providing a
  manual validation recipe.

## §3 Constraints

- New testing helpers ship from the **base distribution** (`testing/`
  is base; no new extras).
- `jsonschema` is **not** a base dependency.  Use a soft import:

  ```python
  try:
      import jsonschema as _jsonschema
  except ImportError:
      _jsonschema = None

  def assert_parsed_matches_schema(result, schema):
      if _jsonschema is None:
          raise ImportError(
              "assert_parsed_matches_schema requires jsonschema; "
              "install with `pip install jsonschema`."
          )
      ...
  ```

  Or: document the helper as needing `pip install jsonschema` (which
  many projects already have).  The dev-log records the choice.

- `MockGateway.script_json(turns)` accepts a list of either
  `{"raw": "...", "parsed": ...}` dicts (deliberate) or pre-built
  `JsonInferenceResponse`-shaped objects.

- The cross-link pass must not break the existing tour through
  `docs/agent.md`, `docs/memory.md`, `docs/guardrails.md`, etc.  Verify
  by reading them top-to-bottom after the edits.

- mypy strict, ruff clean, line length 100.

## §4 Module additions

### §4.1 `src/t0_agents/testing/_gateway.py` (or wherever `MockGateway` lives) (modified)

Add `script_json` and `canned_json`:

```python
@classmethod
def script_json(cls, turns: list[dict[str, Any] | JsonResponseLike]) -> "MockGateway":
    """Build a MockGateway that pops one JSON response per call.

    Each entry is either:
    - {"raw": str, "parsed": Any | None, "episode_id": str | None,
      "inference_id": str | None, "usage": Usage | None}
    - A pre-built JsonResponseLike object.

    Missing fields are populated with sensible defaults (random UUIDs
    for ids; None for usage).
    """
    ...

@classmethod
def canned_json(
    cls, callback: Callable[..., dict[str, Any] | JsonResponseLike],
) -> "MockGateway":
    """Build a MockGateway whose inference() delegates to `callback`.

    The callback receives the full kwargs and returns a response dict
    or pre-built JsonResponseLike."""
    ...
```

`JsonResponseLike` is a duck-typed local dataclass with `.output.raw`,
`.output.parsed`, `.episode_id`, `.inference_id`, `.usage`.  It does
**not** import from `tensorzero.types` (testing/ is base, no extra
dependency).

### §4.2 `src/t0_agents/testing/asserts.py` (modified)

Add the five new assert helpers.  Each:

- Accepts an `AgentResult` (or `StructuredOutput` where reasonable).
- Raises `AssertionError` on mismatch with a message that explains
  what was expected and what was observed.

### §4.3 `src/t0_agents/testing/__init__.py` (modified)

Re-export the new helpers under the existing public surface.

### §4.4 `tests/test_testing/test_mock_shape.py` (modified)

Extend the shape-contract test to assert that:

- `MockGateway.script_json(...)` returns objects with `output.raw`,
  `output.parsed`, `episode_id`, `inference_id` (the kwargs the
  retry orchestration in `Agent._run_json` actually inspects).
- The `JsonResponseLike` is structurally compatible with whatever
  `run_structured_inference` expects from the SDK (decided in M1; see
  Q1).

### §4.5 No other module changes

`core/`, `mcp/`, `memory/`, `guardrails/`, `serve/`, `cli/`,
`compose/`, `stream/` are untouched.

## §5 Tests

### §5.1 Unit tests (`tests/test_testing/test_json_helpers.py`)

- `test_script_json_basic`: `MockGateway.script_json([{"raw": ...,
  "parsed": ...}])`; build an agent; call `run`; assert the agent
  observed the scripted output.
- `test_script_json_missing_fields_defaulted`: missing `episode_id`
  gets a UUID; missing `usage` is None.
- `test_canned_json_receives_kwargs`: callback captures the kwargs;
  assert it saw `function_name`, `messages`, etc.
- `test_assert_parsed_equals_pass`: passes when `result.parsed` equals
  the expected.
- `test_assert_parsed_equals_fail`: fails with a clear message when
  they differ.
- `test_assert_parsed_matches_schema_pass`: with `jsonschema` available,
  validates against a schema.
- `test_assert_parsed_matches_schema_missing_dep`: without
  `jsonschema`, raises `ImportError` (or the chosen alternative).
- `test_assert_invalid_json_pass`: result has `parsed is None` and
  `parse_strategy == "none"` — assertion passes.
- `test_assert_parse_strategy_pass_fail`: positive and negative cases.
- `test_assert_retry_attempts_pass_fail`: positive and negative cases.

### §5.2 Update to shape-contract test

Verify `MockGateway` covers every kwarg/branch the JSON dispatch path
in `Agent._run_json` uses, mirroring the M8 (agentic-feature M8)
shape contract.

### §5.3 Reuse M1–M4 integration tests

Re-run them to confirm nothing broke in the cross-link pass.

## §6 Example

`examples/16_structured/ner_dynamic_schema.py`:

- Uses the canonical `extract_entities` function from guide §12.2.
- Demonstrates three scenarios in one runnable script:
  1. Default `json_mode = "strict"` (config-level) returns the four
     entity categories.
  2. Per-call `output_schema=` override that adds a new category
     (`product`) — the dynamic schema replaces the static one for this
     call (guide §4).
  3. Per-call `json_mode = "tool"` swap to demonstrate provider
     compatibility (the function is configured with no tools, so this
     works; if tools were present, the M4 guard would error).
- Recovery demo: a fourth scenario that uses a deliberately weak
  prompt and shows `result.parse_strategy == "retry"`.
- README explains each scenario with a one-paragraph context.

## §7 Documentation

### §7.1 `docs/structured-outputs.md` (final polish)

- Add a "Testing your JSON agent" section with examples of every new
  helper.
- Ensure every M1–M4 section's "What's coming" placeholder is removed
  or replaced.
- Add a "Recipe: integrate with Pydantic" section showing the
  one-liner (`MyModel.model_validate(result.parsed)`) referenced in
  Q7.
- Add a final "See also" section linking to `docs/agent.md`,
  `docs/guardrails.md`, `docs/memory.md`, `docs/testing.md`,
  `docs/streaming.md` (with the explicit note: streaming + JSON is
  not supported).

### §7.2 Cross-links elsewhere

Add a "JSON functions" link at the appropriate point of each of:

- `docs/index.md` — feature roll-call.
- `docs/concepts.md` — point at the "Chat vs JSON" primer from M1.
- `docs/agent.md` — link from the `AgentConfig` reference table.
- `docs/guardrails.md` — link from the `require_parsed()` entry
  (already added in M4; just verify).
- `docs/memory.md` — link from the "what gets persisted" subsection.
- `docs/testing.md` — link from the `MockGateway` reference + new
  subsection on JSON-aware test patterns.
- `docs/streaming.md` — link from the "What's NOT supported" subsection
  (or add one); make explicit that streaming + JSON functions raises
  `ConfigurationError`.

### §7.3 `docs/adoption-guide.md` (modified — minor)

The adoption-guide already exists (uncommitted in the working tree).
Verify it does not need a JSON section; if it should mention JSON
mode, add a single paragraph in the "advanced usage" section.

## §8 Checklist

### Pre-flight
- [ ] Read `logs/milestone-1-dev-log.md` through
      `logs/milestone-4-dev-log.md` in full.  M5 is the only milestone
      that consolidates state across all prior ones — every deviation
      potentially matters.
- [ ] Re-read the cross-cutting checklist (`01-implementation-plan.md`
      §5).  M5 closes every item that prior milestones could leave
      half-finished (CHANGELOG consolidation, cross-links, etc.).

### Code
- [ ] Add `MockGateway.script_json` and `MockGateway.canned_json`.
- [ ] Add the 5 new assertion helpers in `testing/asserts.py`.
- [ ] Re-export from `testing/__init__.py`.
- [ ] Verify `core/`, `mcp/`, every other submodule is byte-identical
      vs M4 end-state.

### Tests
- [ ] `tests/test_testing/test_json_helpers.py` — 10 unit tests.
- [ ] `tests/test_testing/test_mock_shape.py` extended.
- [ ] Re-run M1–M4 integration tests; nothing regresses.
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `uv run pytest -m integration` green.

### Example
- [ ] `examples/16_structured/ner_dynamic_schema.py` runnable.
- [ ] `examples/16_structured/README.md` updated with the four
      scenarios (extract_customer_info from M1, structured_agent from
      M2, recovery_demo from M3, ner_dynamic_schema from M5).

### Docs
- [ ] `docs/structured-outputs.md` final polish: "Testing your JSON
      agent" section, Pydantic recipe, "See also" links.
- [ ] Cross-links added to all the docs listed in §7.2.
- [ ] Read `docs/index.md → docs/concepts.md → docs/agent.md →
      docs/structured-outputs.md → docs/guardrails.md →
      docs/memory.md → docs/streaming.md → docs/testing.md` end-to-end
      and confirm there are no broken links.

### Style gates
- [ ] `uv run ruff check src/`.
- [ ] `uv run mypy src/`.

### CHANGELOG consolidation
- [ ] Collapse `## [0.10.0-M1] — Unreleased` through
      `## [0.10.0-M5] — Unreleased` into a single
      `## [0.10.0] — Unreleased` entry, preserving every concrete
      bullet from the per-milestone sections.
- [ ] Bump version in `pyproject.toml` if appropriate (user decision —
      typically deferred to release time, so just record the decision
      in the dev-log).

### Implementation log
- [ ] Write `docs/internal/structured-outputs/logs/milestone-5-dev-log.md`.
      Note: this is the final log; record any roadmap-level
      observations and any items deferred to follow-up work.

## §9 Acceptance criteria

1. `MockGateway.script_json(...)` and
   `MockGateway.canned_json(...)` are public API.
2. Every new assert helper has at least one passing positive test and
   one failing negative test.
3. The shape-contract test verifies `MockGateway` covers the JSON
   dispatch path.
4. `examples/16_structured/ner_dynamic_schema.py` runs successfully
   with docker compose up.
5. `docs/structured-outputs.md` is complete: sections from M1
   (single-shot), M2 (Agent), M3 (Robust parsing), M4 (Safety),
   M5 (Testing + Pydantic recipe + See also).
6. Every doc page in `docs/` referenced from `docs/structured-outputs.md`
   has at least one inbound link back.
7. `CHANGELOG.md` has a single `## [0.10.0] — Unreleased` entry
   replacing the per-milestone sub-sections.
8. `mcp/` source is byte-identical to v0.9.0.
9. `uv run pytest -m "not integration"` is green; integration tests
   are green with docker up.
10. The final dev-log records every observation worth carrying into
    a follow-up roadmap.

## §10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `jsonschema` import handling differs across CI vs dev environments | Soft import with explicit error message; document the helper as needing `pip install jsonschema`; record in dev-log. |
| Cross-link pass introduces broken internal links | Mandatory end-to-end read-through of every modified doc; consider a `linkcheck` script if one exists in the project (none today). |
| The NER example requires a function configured with dynamic schema support, which the user's tensorzero.toml may not have | Ship a complete `examples/16_structured/config/tensorzero.toml` and `docker-compose.yml` so the example is self-contained. |
| CHANGELOG consolidation loses entries because the implementer skims the per-milestone sections | Use `grep` to enumerate every bullet line under the M1–M5 sub-sections before rewriting the section; verify the final entry mentions every public symbol introduced. |
| `MockGateway.script_json` shape diverges from what `Agent._run_json` actually needs (e.g. missing a field the M2/M3 code reads) | The shape-contract test is the safety net; pad it explicitly with every kwarg the dispatch reads. |

## §11 Estimated surface

- New files: 2 (`tests/test_testing/test_json_helpers.py`,
  `examples/16_structured/ner_dynamic_schema.py`).
- Modified files: ~10 (`testing/` modules + cross-link edits in many
  `docs/*.md`).
- New tests: ~10.
- Lines of code added: ~200 in `testing/`, ~150 in docs cross-link
  edits.
