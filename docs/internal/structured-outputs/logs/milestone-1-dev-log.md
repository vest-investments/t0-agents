# Milestone 1 — Dev Log

> Written 2026-05-14 at the end of the milestone, per the always-write
> policy.  Subsequent milestones must read this log (and any others
> that exist) before planning.

## §1 Summary — what was done

### Files added (line counts)

- `src/t0_agents/core/structured.py` (200 lines) — `ParseStrategy` type
  alias, `StructuredOutput` frozen-slots dataclass,
  `run_structured_inference()` async helper.
- `tests/test_core/test_structured.py` (210 lines) — six unit tests
  exercising the helper.  Local hand-rolled `FakeJsonT0` /
  `_fake_json_response` build real `JsonInferenceResponse` /
  `JsonInferenceOutput` / `Usage` instances from the SDK.
- `tests/test_core/test_config_structured.py` (170 lines) — eleven
  unit tests over three test classes:
  `TestFunctionKindValidation`, `TestFunctionKindJsonForbidsSubAgents`,
  `TestOutputSchemaPath`.
- `tests/test_core/test_structured_integration.py` (95 lines) — two
  integration tests under `pytestmark = pytest.mark.integration`
  (static schema; dynamic schema override).
- `examples/17_structured/main.py` (60 lines) — runnable demo.
- `examples/17_structured/docker-compose.yml` (34 lines) — gateway
  service bound to the example's config.
- `examples/17_structured/README.md` (60 lines) — usage instructions.
- `examples/17_structured/config/tensorzero.toml` (30 lines) — gateway
  config defining `extract_customer_info`.
- `examples/17_structured/config/functions/extract_customer_info/output_schema.json`
  (17 lines).
- `examples/17_structured/config/functions/extract_customer_info/system_template.minijinja`
  (22 lines).
- `config/functions/extract_customer_info/output_schema.json` (17
  lines) — copy at the repo-root config so integration tests resolve
  the same function name.
- `config/functions/extract_customer_info/system_template.minijinja`
  (22 lines) — same template, repo-root copy.
- `docs/structured-outputs.md` (115 lines) — new public doc.
- `docs/internal/structured-outputs/logs/milestone-1-dev-log.md` (this
  file).

### Files modified

- `src/t0_agents/core/config.py` — added `Mapping`, `Path`, `Any`, and
  `json` imports; three new fields on `AgentConfig` (`function_kind`,
  `output_schema`, `json_mode`); three new validation branches plus
  the `output_schema: Path` resolver in `__post_init__`.
- `src/t0_agents/core/types.py` — added `function_kind` and `parsed`
  fields to `AgentResult`; expanded the class docstring to call out
  the M1 surface.
- `src/t0_agents/__init__.py` — imported `StructuredOutput` and added
  to `__all__`.
- `config/tensorzero.toml` — appended the `extract_customer_info`
  JSON function block (variant `gpt_oss_20`, `json_mode = "strict"`).
- `docs/concepts.md` — appended a "Chat vs JSON functions" paragraph
  under "The three participants".
- `docs/agent.md` — added three rows to the `AgentConfig` reference
  table (`function_kind`, `output_schema`, `json_mode`).
- `docs/internal/structured-outputs/00-integration-contract.md` §3.4
  and §3.5 — recorded the `response_kind` → `function_kind`
  amendment and the `response_source` byte-identity note.
- `docs/internal/structured-outputs/01-implementation-plan.md` §5.4
  — corrected `16_structured` → `17_structured` (four occurrences).
- `docs/internal/structured-outputs/02-open-questions.md` — Q1, Q3,
  Q11 marked resolved with file-line citations.
- `CHANGELOG.md` — prepended `## [0.10.0-M1] — Unreleased` block.

### Public API delta

- New: `from t0_agents import StructuredOutput`.
- New (not re-exported at the package root, but importable):
  `from t0_agents.core.structured import run_structured_inference`,
  `ParseStrategy`.
- `AgentConfig` gains `function_kind`, `output_schema`, `json_mode`
  with defaults that preserve byte-identity.
- `AgentResult` gains `function_kind`, `parsed` with defaults that
  preserve byte-identity.

### Tests added

- 6 unit tests in `tests/test_core/test_structured.py`.
- 11 unit tests in `tests/test_core/test_config_structured.py`
  (the milestone spec called for 6 — the extra five cover the path-
  resolution failure modes that the spec implicitly demanded:
  absolute vs relative, missing file, invalid JSON, top-level array).
- 2 integration tests in
  `tests/test_core/test_structured_integration.py`.

Total: 17 unit + 2 integration = 19 new tests.

## §2 Deviations from the milestone spec

### §2.1 Example folder numbering (`16_structured` → `17_structured`)

The milestone spec consistently referenced `examples/16_structured/`,
but `examples/16_parallel/` already existed in the repo.  Two example
folders sharing the `16_` prefix would have been awkward to scan, so
the M1 example lives at `examples/17_structured/`.  Corrected the
forward references in
`docs/internal/structured-outputs/01-implementation-plan.md` §5.4
(four occurrences across the M1–M5 list).

**Impact on subsequent milestones:** M2/M3/M5 each ship an example
file inside `examples/17_structured/` — the directory name change is
already reflected in the corrected plan.  No other source touched
this path.

### §2.2 `AgentResult.response_kind` → `AgentResult.function_kind`

The contract (§3.4) named the new field `response_kind`, but
`AgentResult` already carries `response_source: Literal["text",
"thought"]` for which content block produced the response text.  A
sibling `response_kind: Literal["text", "json"]` would have been
confusingly close.  Renamed to `function_kind`, which mirrors
`AgentConfig.function_kind` the same way `episode_id` mirrors the
TZ-assigned id.  Contract §3.4 amended (table + amendment note).
Also added a §3.5 note that `response_source` stays `"text"` for JSON
runs (degenerate but kept for byte-identity).

**Impact on subsequent milestones:** M2's `Agent.run` JSON dispatch
must set `result.function_kind = "json"` (not `response_kind`).
M5's `MockGateway` JSON shape contract and `assert_parsed_*` helpers
will reference `result.function_kind` as well.  Both touchpoints are
new in those milestones — no rewrites of existing logic.

### §2.3 Helper signature scope — `output_schema: Mapping[str, Any] | None` only

The spec's helper signature accepts `output_schema: Mapping[str, Any]
| None`, deliberately omitting the `Path` variant that `AgentConfig`
accepts.  This matches the spec verbatim and the rationale (helper is
single-shot; if the caller needs a path, they read the file
themselves or build an `AgentConfig`).  Recorded here for visibility
since the difference between the config field type and the helper
kwarg type might confuse a future reader.

### §2.4 Path-resolution failure tests

The milestone spec's test list (§5.2) called for 6 tests, including
`test_output_schema_loads_from_path` (one happy-path test).  The
implementation grew that into five tests (absolute, relative,
missing, invalid JSON, non-object top-level) to lock down the
behaviour properly — `output_schema: Path` is a published surface and
any quirky failure mode here would surface in real configs months
from now.  Total `test_config_structured.py` test count: 11 instead
of 6.  No carry-forward implication.

### §2.5 Repo-root `config/tensorzero.toml` modified

The milestone spec's §5.3 integration test outline assumed an
`extract_email` function in some shared TOML — it did not specify
where.  The simplest path was adding `extract_customer_info`
(matching the example's function name) to the existing
`config/tensorzero.toml` that `docker compose up -d` (run from the
repo root) loads.  This is the same TOML the existing integration
tests in `tests/test_core/test_agent_integration.py` rely on, so
keeping JSON-mode coverage there is a natural fit.

**Impact on subsequent milestones:** M2/M3/M4/M5 integration tests
can reuse the same `extract_customer_info` function or define their
own under the same TOML.  No `mcp/` files touched.

### §2.6 Six unit tests in `test_structured.py` (spec called for 4)

The spec's §5.1 listed four tests:
- `test_run_structured_inference_returns_parsed`
- `test_run_structured_inference_returns_none_when_invalid`
- `test_run_structured_inference_threads_kwargs`
- `test_on_inference_callback_fires`

Two additional tests were added:
- `test_omits_params_when_json_mode_is_none` — covers the branch
  where no `params` dict is sent (TZ uses the variant's configured
  `json_mode`).  Easy to miss in M3 if the retry loop accidentally
  forces `json_mode` even when the caller did not request one.
- `test_raises_when_function_returns_chat_response` — covers the
  `isinstance(raw, JsonInferenceResponse)` guard.  Without this
  test, the runtime check is untested by the offline suite.

No carry-forward implication.

## §3 Open questions raised or resolved

### Resolved

- **Q1** — `JsonInferenceResponse` shape verified at
  `.venv/.../tensorzero/types.py:209-238`,
  `tensorzero/__init__.py:243-244`.  See
  `02-open-questions.md` for the full resolution.
- **Q3** — `output_schema: Path` resolves against
  `Path.cwd()` (relative) or verbatim (absolute).  Documented in
  `docs/structured-outputs.md`.  Code in
  `src/t0_agents/core/config.py:206-230`.
- **Q11** — `dryrun: Optional[bool]` is a top-level kwarg on
  `AsyncTensorZeroGateway.inference(...)`
  (`tensorzero/tensorzero.pyi:1048`).  `run_structured_inference()`
  forwards verbatim.  Test:
  `tests/test_core/test_structured.py::test_threads_every_kwarg_into_inference_call`.

### Raised (new during M1)

None.  M2 and M3 will discover their own.

## §4 Carry-forward notes

### For M2 — `Agent.run` JSON dispatch

- The dispatch decision is `config.function_kind == "json"`.  The
  per-call kwargs (`output_schema`, `json_mode`) take precedence over
  `AgentConfig` per the contract §3.2; M1 did not implement that
  precedence — `Agent.run` itself owns it.
- `AgentResult.function_kind` (not `response_kind`) is the field to
  set.  Default is `"chat"`.
- `AgentResult.response_source` stays `"text"` for JSON runs — no
  thought-block path exists.  Contract §3.5 records this.
- `Agent.run` for JSON does not loop — it calls
  `run_structured_inference()` once and wraps the result in
  `AgentResult`.  Memory hooks (`load`, `save`) still fire; guardrail
  hooks `before_run` / `after_run` still fire; tool-level hooks
  (`before_tool` / `after_tool`) do not because there is no tool
  call.
- The contract §3.6 forbids `sub_agents` with `function_kind ==
  "json"`; M1 enforces this in `AgentConfig.__post_init__`.  M2 does
  not need to re-check.

### For M3 — permissive parsing + retry

- `StructuredOutput.parse_strategy` already carries the four-value
  vocabulary (`"strict"`, `"permissive"`, `"retry"`, `"none"`).  M1
  emits only `"strict"` and `"none"`.  M3 adds the in-Python
  permissive layer (sets `"permissive"`) and the retry loop (sets
  `"retry"`).
- `StructuredOutput.retry_attempts: int = 0` is wired and tested.
  M3 will start populating it.
- The retry loop lives in `Agent.run` (per contract §4.4), not in
  `run_structured_inference()` — the helper stays single-shot.
- The `on_inference` callback fires per inference; M3's retry loop
  must invoke it once per attempt.
- Q8 (episode_id continuity across retries) is still open — M3 will
  resolve it.
- The contract documents §3.4: `AgentResult.inference_ids` includes
  every inference id produced during retries, in order.  Make sure
  the retry orchestration appends in the right order.

### For M4 — safety + quirks

- The runtime guard `json_mode == "tool"` + non-empty tool list
  belongs in `Agent.__init__` (contract §5.1, last bullet).  M1
  could not implement this because `AgentConfig.__post_init__` has no
  access to the tool_provider's `list_tools()`.  M4 owns this.
- `Agent.stream(...)` with `function_kind == "json"` must raise
  `ConfigurationError` at the start of the async generator before
  yielding any event (contract §3.3, §5.3).

### For M5 — testing helpers + polish

- `MockGateway` is chat-only in v0.9.0; M5 should fold the JSON
  branch into it so `tests/test_core/test_structured.py` can drop
  the local `FakeJsonT0` / `_fake_json_response` fakes.  Pattern to
  mirror: `tests/_fake_t0.py:FakeChatResponse + FakeT0`.
- The shape-contract test
  (`tests/test_testing/test_mock_shape.py`) should grow a JSON
  branch — extend `_INFERENCE_KWARGS` if M3 added any new kwargs to
  `Agent.run` JSON dispatch.
- Consolidate the per-milestone `## [0.10.0-M<N>]` CHANGELOG
  sub-sections into a single `## [0.10.0] — Unreleased` block.

## §5 Verification

Commands run during M1 close-out (all from the repo root):

- `uv run ruff check src/` — **passed.**  All source files clean.
  (`tests/` has eight pre-existing ruff failures in unrelated files,
  none introduced by this milestone; per the CLAUDE.md convention the
  required gate is `src/` only.)
- `uv run mypy src/` — **passed.**  "Success: no issues found in 54
  source files."
- `uv run pytest -m "not integration"` — **passed.**  549 passed, 2
  skipped, 50 deselected (integration).  The 17 new offline tests
  introduced by M1 are included in the 549.
- `uv run pytest tests/test_core/test_structured.py
  tests/test_core/test_config_structured.py -v` — **passed.**  17/17
  green.
- Smoke check: `python -c "from t0_agents import StructuredOutput,
  AgentConfig, AgentResult; ..."` — **passed.**  All defaults
  byte-identical to v0.9.0.
- `git diff src/t0_agents/mcp/` — **empty.**  `mcp/` is byte-identical
  per contract §2.1.
- `git diff src/t0_agents/core/agent.py` — **empty.**  No `Agent.run`
  or `Agent.stream` changes (those land in M2 / M4).

### Integration tests

`uv run pytest -m integration` was not run during the milestone — the
user has not yet confirmed `docker compose up -d` is up with
`GROQ_API_KEY` configured.  The two new tests under
`tests/test_core/test_structured_integration.py` are wired and ready;
the user will run them after standing up the gateway and report the
result in a follow-up.
