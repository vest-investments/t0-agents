# Milestone 3 — Dev Log

> Implementation log for `M3 — Permissive parsing and retry`.  Format
> follows the template in `01-implementation-plan.md` §5: §1 Summary,
> §2 Deviations from spec, §3 Open questions raised or resolved,
> §4 Carry-forward notes, §5 Verification.

---

## §1 Summary

### Files added (4)

- `src/t0_agents/core/structured.py` — gained
  `PermissiveStrategy` literal alias,
  `_extract_first_balanced` (string-literal-aware balanced-brace
  scanner), `_extract_fenced_code` (Markdown fenced-block extractor),
  `_ordered_pairs_for` (earliest-opener disambiguator), and the
  public `permissive_parse(raw)`.  File grew from 187 → 349 lines.
  Also gained the `prior_messages: list[Mapping[str, Any]] | None =
  None` kwarg on `run_structured_inference` so the retry orchestration
  can thread the failed turn + corrective hint back into the
  conversation.
- `tests/test_core/test_permissive_parse.py` — 184 lines, 22 unit
  tests covering the strict / permissive / none branches and the
  deterministic sweep that replaces the spec's hypothesis property
  test.
- `tests/test_core/test_agent_json_retry.py` — 286 lines, 9 unit
  tests for the retry orchestration (strict / permissive / retry /
  none transitions, per-call override, episode-id preservation,
  permissive opt-out, negative-budget config error, retry-hint
  substring in history).
- `tests/test_core/test_agent_json_retry_integration.py` — 89 lines,
  1 integration test (marked `@pytest.mark.integration`) that runs an
  end-to-end recovery against a real Groq provider with
  `json_mode="off"`.
- `examples/17_structured/recovery_demo.py` — 185 lines, a
  self-contained local-fake demo of all four `parse_strategy`
  outcomes.
- `examples/17_structured/recovery_real_provider.py` — 86 lines, the
  docker-gated real-provider analogue.

### Files modified (5)

- `src/t0_agents/core/config.py` — added two `AgentConfig` fields
  (`permissive_json_parsing: bool = True`,
  `max_json_retries: int = 2`) and a `__post_init__` validation that
  raises `ConfigurationError` on `max_json_retries < 0`.
- `src/t0_agents/core/types.py` — added two `AgentResult` fields
  (`parse_strategy: ParseStrategy = "strict"`,
  `json_parse_attempts: int = 0`); imports `ParseStrategy` from
  `core.structured`.
- `src/t0_agents/core/agent.py` — added the `_RETRY_HINT_BASE`
  module-level constant, the schema-aware `_build_retry_hint(schema)`
  helper, the `max_json_retries: int | None = None` kwarg on
  `Agent.run`, and rewrote `Agent._run_json`'s single-shot
  delegation into a bounded retry loop that tracks `attempts`,
  `prior_msgs`, `inference_ids`, and `captured_episode` across
  attempts.
- `examples/17_structured/README.md` — gained a "Robust JSON recovery"
  section with the three-layer diagram and pointers to the two new
  scripts; the "Run" block lists all four examples; "What to observe"
  documents the demo and real-provider outputs.
- `docs/structured-outputs.md` — gained a "Robust JSON parsing"
  section (three-layer diagram, config/observability tables, two
  worked examples, opt-out callout, schema-aware retry hint note,
  `permissive_parse` recipe).  The milestone summary at the top now
  covers M1–M3 instead of M1–M2.
- `docs/agent.md` — documents the two new `AgentConfig` fields, the
  new `Agent.run` kwarg, the new `AgentResult` fields, and the updated
  `turns` / `inference_ids` semantics for JSON runs.
- `docs/internal/structured-outputs/02-open-questions.md` — marked
  Q4, Q5, Q6, Q8, Q9 as resolved with the M3 decisions; Q2 remains
  open (out of scope per Q12).
- `CHANGELOG.md` — prepended a new `## [0.10.0-M3] — Unreleased`
  section above the M2 section, with Added / Changed / Compatibility
  blocks.
- `src/t0_agents/__init__.py` — re-exports `permissive_parse` from
  the package root; `__all__` updated accordingly.

### Public API delta (4 new symbols)

- `t0_agents.permissive_parse` (and `t0_agents.core.structured.permissive_parse`)
- `t0_agents.core.config.AgentConfig.permissive_json_parsing`
- `t0_agents.core.config.AgentConfig.max_json_retries`
- `t0_agents.core.types.AgentResult.parse_strategy`
- `t0_agents.core.types.AgentResult.json_parse_attempts`
- `t0_agents.core.agent.Agent.run(max_json_retries=...)`
- `t0_agents.core.structured.run_structured_inference(prior_messages=...)`

(The two underscored-private helpers `_RETRY_HINT_BASE` and
`_build_retry_hint(...)` are deliberately internal and not part of the
public API.  They are imported by the retry unit tests for the
substring assertion in `test_retry_hint_appears_in_history`.)

### Tests added

| File                                              | Count |
|---------------------------------------------------|-------|
| `tests/test_core/test_permissive_parse.py`        | 22 unit |
| `tests/test_core/test_agent_json_retry.py`        | 9 unit |
| `tests/test_core/test_agent_json_retry_integration.py` | 1 integration |
| **Total (M3)**                                    | **31 new tests** |

Combined with M1's 19 + M2's 13, the structured-outputs surface now
ships ~63 tests across `tests/test_core/`.

---

## §2 Deviations from the milestone spec

### a) Schema-aware retry hint (vs the spec's hard-coded English)

The M3 spec §4.5 proposed a single fixed corrective sentence.  During
M3 planning the user opted for a richer hint that, when an
`output_schema` is in play, also includes the schema's top-level
`title` (when present) and `required` field names (when present).

**What changed:** the orchestration calls
`_build_retry_hint(effective_schema)` instead of inlining the constant
string.  The helper is additive — it returns the spec's base sentence
when called with `schema=None`.

**Why:** richer signal for weak models; cheap to implement; deterministic
and easy to test.  The base sentence is exposed as `_RETRY_HINT_BASE`
so tests can pattern-match on a substring rather than locking the exact
wording.

**Impact on subsequent milestones:** none — the helper is private and
behaviour-compatible.  If M4 or a future milestone adds an i18n /
per-call hint hook, it can plug into the same call site without
changing the orchestration shape.

### b) Deterministic sweep instead of hypothesis (vs spec §5.2)

The spec proposed a hypothesis-based property test for
`permissive_parse`.  `hypothesis` is not in this project's dev
dependencies and adding it for one milestone felt disproportionate.

**What changed:** `tests/test_core/test_permissive_parse.py` ships a
deterministic sweep (`TestDeterministicSweep::test_sweep_objects_with_prose`)
that iterates a curated cartesian product of `(prefix, obj, suffix)` —
7 prefixes × 6 objects × 6 suffixes = 252 cases — and asserts recovery.
Top-level arrays get a parametrised companion test.

**Why:** the user picked option (c) "Deterministic sweep instead" in
planning; no new dev dependency; reproducible by index rather than
shrunk counterexample.

**Impact on subsequent milestones:** none.  If M5's testing helpers
benefit from a real property test, `hypothesis` can be added then
without retroactively changing M3.

### c) Examples directory is `17_structured/`, not the spec's `16_structured/`

The spec was written before the actual examples directory was
created.  M1 chose `16_parallel/` for the parallel-tools example and
`17_structured/` for the structured-outputs example; M2 added
`structured_agent.py` to `17_structured/`.

**What changed:** the two M3 examples (`recovery_demo.py`,
`recovery_real_provider.py`) live in `examples/17_structured/`
alongside their M1/M2 siblings, not in a new `16_structured/`.

**Why:** keeping all structured-outputs examples in one directory is
better pedagogy than splitting them by milestone number.

**Impact on subsequent milestones:** M5's dynamic-schema NER example
should also live in `examples/17_structured/` for the same reason.

### d) `permissive_parse` opener ordering fix

The spec's natural-language description ("Try `{...}` first, then
`[...]`") would have produced an off-by-one for inputs like
`'[{"k": 1}, {"k": 2}]'` (top-level array containing objects): the
scanner would extract the inner `{...}` and stop.  The deterministic
sweep caught this on the first run.

**What changed:** added `_ordered_pairs_for(text)` which returns the
two pairs ordered by whichever opener appears earliest in `text`.
When both openers are present, the earliest one wins; when only one is
present, the other is still tried as a fall-back (covers
`"prose [unbalanced {valid: 1}"`-style inputs).

**Why:** the natural intuition — "extract whatever JSON-shaped span
appears first" — turned out to need explicit positional disambiguation.
The fix is local to `permissive_parse` and adds one short helper.

**Impact on subsequent milestones:** none.  Documented in this log
and in `02-open-questions.md` Q5's resolution so M4/M5 readers do not
re-discover the bug.

### e) Post-review: iterate balanced spans and fenced blocks

PR #24 surfaced two `gemini-code-assist[bot]` comments (medium
priority) flagging that the original `_extract_first_balanced` and
`_extract_fenced_code` would silently lose recovery cases.  Both are
real failure modes for chain-of-thought / apology-prone models.

**What changed:** promoted both internal helpers from "return the first
hit" to "yield every hit, in order":

- `_extract_first_balanced(text, opener, closer) -> str | None` is
  replaced by `_iter_balanced_spans(text, opener, closer) -> Iterator[str]`.
  `permissive_parse` iterates each yielded span and tries `json.loads`
  on it; this also catches the case where the first balanced span is
  syntactically balanced but invalid JSON (e.g. an apology containing
  `{key: value}` without quoted keys, followed by valid JSON in the
  same response).
- `_extract_fenced_code(text) -> str | None` is replaced by
  `_iter_fenced_code(text) -> Iterator[str]`.  Chain-of-thought models
  commonly emit a prose ```` ``` ```` reasoning block before the
  ```` ```json ```` answer block; the iterator yields every block
  in order so `permissive_parse` retries past the unparseable ones.

**Why:** both changes are strictly additive — every input the original
`_extract_first_*` helpers recovered is still recovered by their
iterating successors, plus the previously-lost cases above.  Two new
unit tests (`test_skips_unbalanced_opener_in_prose` and
`test_recovers_second_fenced_block_when_first_unparseable`) pin the
two motivating scenarios.

**Impact on subsequent milestones:** none.  The two helpers remain
private (underscore-prefixed); only `permissive_parse` is public, and
its semantics — "recover if any layer can; never raise" — are
unchanged.

---

## §3 Open questions raised or resolved

Resolved during M3 (with citations to `02-open-questions.md`):

- **Q4** (retry hint phrasing) — resolved schema-aware-rich; see
  deviation (a) above and `02-open-questions.md` Q4 resolution.
- **Q5** (top-level arrays) — resolved yes, with the
  `_ordered_pairs_for` disambiguator; see deviation (d) above and
  `02-open-questions.md` Q5 resolution.
- **Q6** (wall-clock budget) — resolved attempt-count-only for v1;
  see `02-open-questions.md` Q6 resolution.  Callers needing a
  wall-clock ceiling can wrap `Agent.run(...)` in
  `asyncio.wait_for(..., timeout=…)`.
- **Q8** (episode_id continuity) — resolved exactly as hypothesised;
  `Agent._run_json` captures the first response's `episode_id` and
  threads it into every subsequent attempt.  Test coverage:
  `test_retry_preserves_episode_id`.  See `02-open-questions.md` Q8
  resolution.
- **Q9** (memory persistence with retries) — resolved exactly as
  hypothesised; final `output.raw` + full ordered `inference_ids` +
  `turns = 1 + json_parse_attempts`.  See `02-open-questions.md` Q9
  resolution.

Still open (deferred):

- **Q2** (schema validation post-TZ) — remains open.  Per **Q12**'s
  earlier decision, schema validation in Python is out of scope for
  the roadmap, so M3 does not address it.  If the contract ever
  changes (e.g. the project takes on a `jsonschema` dependency), Q2's
  hypothesis can be revisited.
- **Q10** (provider auto-detection for safety warnings) — expected to
  land in M4.

No new questions were raised during M3.

---

## §4 Carry-forward notes

### For M4 — Safety & quirks

- **Streaming retry semantics.**  TZ does not emit `parsed` for
  streaming JSON (guide §7.1, §11.3), so the M3 retry loop is
  non-applicable to `Agent.stream(...)`.  M4 must decide whether to
  raise `ConfigurationError` when streaming is requested with a
  non-zero `max_json_retries`, or to silently ignore it and document
  that streaming is "TZ-pure" by construction.  The contract today
  (§3.3) implies the former — confirm during M4 planning.
- **Reusing `permissive_parse` on the streaming side.**  M4 could
  buffer all `JsonChunk.raw` deltas, concatenate, and run
  `permissive_parse` on the assembled string before the terminal
  event.  This adds value without changing the public API; it is
  separate from the retry decision above.
- **`_build_retry_hint(schema)` is a private helper.**  If M4 (or a
  future i18n hook) wants to expose the retry-hint phrasing for
  per-call override, the cleanest path is a new kwarg on `Agent.run`
  that takes a `Callable[[Mapping | None], str]`; the orchestrator
  swaps `_build_retry_hint(effective_schema)` for the callable's
  output.  No structural refactor needed.
- **The `memory_strategy="messages"` runtime guard in `_run_json`
  stays.**  M3 plumbed `prior_messages` through the helper but
  reserved it for the retry loop.  Lifting the guard would require
  turning `MemoryRecord`s into `prior_messages` shape, which is its
  own design exercise; not in M3's scope.

### For M5 — Testing & polish

- **`MockGateway` consolidation.**  Both
  `tests/test_core/test_structured.py` (M1) and
  `tests/test_core/test_agent_json_run.py` (M2) shipped local
  `FakeJsonT0` + `_fake_json_response` helpers; M3's
  `test_agent_json_retry.py` adds a third copy with one tweak (does
  not auto-derive `parsed` from `raw`).  M5's `MockGateway` JSON
  branch should support the *multi-call scripted* shape M3 uses
  (queue of `(raw, parsed)` pairs popped per `inference()` call) so
  the three local fakes can be deleted in one pass.
- **Shape-contract test (`tests/test_testing/test_mock_shape.py`).**
  Confirm the eventual JSON branch of `MockGateway` covers
  `episode_id`, `inference_id`, `output.raw`, `output.parsed`, and
  `usage` — the retry orchestration reads all four.

### General

- **`permissive_parse` is now part of the public API.**  Any future
  internal change to its return shape is a breaking change; callers
  may treat the function as stable.

No other carry-forward notes.

---

## §5 Verification

Commands run and their outcomes (machine where M3 was implemented:
Linux, Python 3.12, `uv` installed):

| Command                                                  | Outcome |
|----------------------------------------------------------|---------|
| `uv run ruff check src/`                                  | passed — `All checks passed!` |
| `uv run mypy src/`                                        | passed — `Success: no issues found in 54 source files` |
| `uv run pytest -m "not integration"`                      | passed — 596 tests pass, 2 skipped, 52 deselected (2.61s, post-review) |
| `uv run pytest tests/test_core/test_permissive_parse.py`  | passed — 22 tests in 0.09 s |
| `uv run pytest tests/test_core/test_agent_json_retry.py`  | passed — 9 tests in 0.10 s |
| `uv run python examples/17_structured/recovery_demo.py`   | passed — four scenarios print four distinct `parse_strategy` outcomes |
| `git diff src/t0_agents/mcp/`                             | empty (contract §2.1 invariant holds) |
| `git diff tests/test_mcp/ tests/test_stream/ tests/test_serve/ tests/test_compose/ tests/test_guardrails/ tests/test_memory/ tests/test_cli/ tests/test_testing/` | empty (chat-byte-identity invariant holds) |

Not run during M3 implementation (user-gated):

- `uv run pytest -m integration` (requires
  `docker compose -f docker-compose.yml --env-file .env up -d` and
  `GROQ_API_KEY` populated).  The new
  `test_agent_json_retry_integration.py` is included in the
  integration suite — the user can verify it after spinning up the
  gateway.
- `uv run python examples/17_structured/recovery_real_provider.py`
  (same docker prerequisite).

The chat-byte-identity gate is the most important regression-prevention
check; it held on every iteration of M3 because the offline suite
includes 32 tests under `test_stream/` and 27 under `test_mcp/` that
pin the chat surface.

---

*End of Milestone 3 dev log.*
