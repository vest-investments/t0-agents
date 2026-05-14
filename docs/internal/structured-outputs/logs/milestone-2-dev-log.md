# Milestone 2 — Implementation Log

> Author: Claude Code (Opus 4.7), 2026-05-14.
> Reviewer: pending (gustavo@mivest.io).

## §1 Summary — what was done

### Files added

- `tests/test_core/test_agent_json_run.py` (375 lines, 12 unit tests across
  five test classes — `TestJsonDispatch`, `TestPerCallOverrides`,
  `TestGuardrails`, `TestMemory`, `TestEpisodePropagation`,
  `TestMessagesMemoryStrategy`).
- `tests/test_core/test_agent_json_run_integration.py` (68 lines, one
  `@pytest.mark.integration` test — `test_agent_run_json_end_to_end`).
- `examples/17_structured/structured_agent.py` (93 lines, runnable
  `Agent`-flavour demo against the M1-shipped `extract_customer_info`
  function).
- `docs/internal/structured-outputs/logs/milestone-2-dev-log.md` (this
  file).

### Files modified

- `src/t0_agents/core/agent.py` — added module-level
  `_make_json_cost_tracker` next to `_make_cost_tracker`; added imports
  for `Mapping`, `JsonInferenceResponse`, `StructuredOutput`,
  `run_structured_inference`; extended `Agent.run` signature with
  `output_schema` and `json_mode` kwargs; added a 16-line dispatch
  guard at the top of `Agent.run`; expanded the `run` docstring;
  appended `_run_json` (~155 lines) between `Agent.stream` and the
  private-helpers section.  Net `+270/-3`.
- `examples/17_structured/README.md` — rewritten to document both
  `main.py` (M1 helper) and `structured_agent.py` (M2 `Agent`);
  expanded the "What to observe" section to cover both scripts.
- `docs/structured-outputs.md` — added a "Structured `Agent`" section
  with quickstart code, per-call override example, dispatch diagram,
  and a memory + guardrails behaviour table; updated the milestone
  block at the top of the page to mark M2 as shipped; updated the
  "What's coming" section accordingly.
- `docs/agent.md` — updated the `function_kind` row in the
  `AgentConfig` table (removed "Wiring lands in M2"); added the new
  `output_schema` / `json_mode` kwargs to the `Agent.run` signature in
  the API reference; expanded the `run` member description; added
  `function_kind` and `parsed` rows to the `AgentResult` table; tweaked
  the existing rows (`response`, `inference_ids`, `turns`,
  `response_source`) to mention their JSON-run behaviour.
- `CHANGELOG.md` — prepended a new
  `## [0.10.0-M2] — Unreleased` sub-section above the existing M1
  entry.

### Public API delta

- `Agent.run` signature gains two optional kwargs (`output_schema`,
  `json_mode`); chat callers are byte-identical because both default
  to `None` and only the JSON dispatch path consumes them.
- No new top-level imports added to `t0_agents.__init__` — `Agent` /
  `AgentConfig` / `AgentResult` were already re-exported by M1, and
  `run_structured_inference` continues to live under
  `t0_agents.core.structured`.

### Tests added

- 12 unit tests in `tests/test_core/test_agent_json_run.py`.  The M2
  spec called for 9; the file ships 12 because the natural test layout
  split a few cases:
  - `TestPerCallOverrides` has 5 tests instead of the spec's 4 — added
    `test_config_schema_used_when_per_call_omits_override` for the
    fallback path and `test_per_call_json_mode_kwarg_forbidden_on_chat_agent`
    as a sibling to the `output_schema` rejection test (the spec
    mentions both kwargs collectively).
  - `TestMessagesMemoryStrategy` is a new class for the runtime guard
    against `memory_strategy="messages"` (see §2.4 below).
- 1 integration test in
  `tests/test_core/test_agent_json_run_integration.py`.
- Total: 13 new tests (12 offline + 1 integration).  Offline suite
  count moved from 549 → 561.

### Docs added or modified

- New: 1 (this dev-log).
- Modified: 4 (`docs/structured-outputs.md`, `docs/agent.md`,
  `examples/17_structured/README.md`, `CHANGELOG.md`).

## §2 Deviations from the milestone spec

### §2.1 Integration test uses `extract_customer_info`, not `extract_email`

The spec (§5.2) calls for an integration test against an `extract_email`
JSON function.  After clarification, the user opted to reuse the
existing `extract_customer_info` function from M1's
`config/tensorzero.toml` rather than adding a second TOML function.
The schema's `{"name": str|null, "email": str|null}` shape covers the
M2 assertion ("`result.parsed` carries an email") without extra wiring,
and CI's per-run Groq token budget stays unchanged.  No impact on M3+.

### §2.2 Example folder path

The spec (§6, §8) names the example
`examples/16_structured/structured_agent.py`.  M1's dev-log §2.1
already corrected `16_structured` → `17_structured` because
`examples/16_parallel/` exists.  This milestone follows M1's correction
— file lands at `examples/17_structured/structured_agent.py`.  No
impact on M3+ (forward references in `01-implementation-plan.md` were
fixed in M1).

### §2.3 `AgentResult.function_kind` (not `response_kind`)

The spec (§1, §5.1) uses `response_kind` in assertion examples.  M1's
dev-log §2.2 amended the field name to `function_kind` because
`AgentResult.response_source: Literal["text", "thought"]` already
exists and `response_kind` would have been confusable.  This milestone
sets `result.function_kind = "json"` (already declared by M1's
dataclass).  No impact on M3+.

### §2.4 Runtime guard for `memory_strategy="messages"` on JSON functions

The M2 spec (§4.1 step 1) explicitly enumerates `prepended_message`
and `system_template` as the supported strategies for JSON runs and is
silent on `"messages"`.  `run_structured_inference` has no
`prior_messages` kwarg (TZ's JSON functions are single-shot), so a
loaded conversation history could not be forwarded to the gateway.
`_run_json` rejects the strategy at the top of the method —
immediately after `_resolve_memory_strategy()` returns `"messages"`,
before any `memory.load` round-trip — and raises
`ConfigurationError` with a message pointing at `"prepended_message"`
or `"system_template"`.  The strategy-level check is uniform across
episode states: a misconfigured agent fails the same way whether the
caller provides no `episode_id`, an `episode_id` with empty history,
or an `episode_id` with non-empty history.

Initial implementation keyed the check off `"prior_messages" in
extra_kwargs` (after `_prepare_memory_injection`), which silently
passed when the history was empty or `episode_id` was `None`.
Switched to the strategy-level check after gemini-code-assist flagged
the inconsistency in PR review.  Three tests in
`TestMessagesMemoryStrategy` lock the new behaviour in
(non-empty-history, empty-history, no-episode-id).

**Impact on M3:** M3's retry loop reuses `_run_json` (per contract §4.4
the retry orchestration lives inside `Agent.run`'s JSON path); the
guard is hit before any inference attempt and therefore does not
interact with retry.  If a real workload needs threaded history with
JSON, M3 can either (a) extend `run_structured_inference` to accept
`prior_messages` and remove the guard, or (b) document the constraint
formally in `00-integration-contract.md` §3.5.

### §2.5 Cost-tracker duplicated, not consolidated

The spec (§4.1) mentions the `on_inference` cost-tracker hook but does
not specify a refactor.  This milestone adds
`_make_json_cost_tracker` next to `_make_cost_tracker`.  Both are
single-purpose closures with identical bodies, but `ChatInferenceResponse`
and `JsonInferenceResponse` are not related by inheritance — pushing a
parameterised tracker through both call sites would have required
either a structural `Union` or `Any` typing.  Documented in the
function's docstring; consolidation is an M3 candidate once the retry
path adds a second JSON inference per run.

**Impact on M3:** none load-bearing — M3 will likely call the same
tracker once per retry attempt.

### §2.6 No `docs/agent.md` per-version feature roll-call

The spec (§7.2) calls for a "v0.3.0 / v0.4.0 / etc. feature roll-call"
entry.  No such section exists in `docs/agent.md` — the page never
adopted a per-version feature ledger and adding one would have been
new scaffolding rather than an update.  The `function_kind` row in the
`AgentConfig` table now says "(since v0.10.0-M2)" and the page-level
"See also" forward-link in `docs/structured-outputs.md` mirrors the
same version anchor, which together carry the same information.

### §2.7 12 unit tests rather than 9

The spec (§5.1) lists 9 tests by name.  All 9 ship; three additional
tests cover natural variants (fallback to config schema when the
per-call kwarg is omitted, the `json_mode` analogue of the chat-kwarg
rejection test, and the runtime guard for `memory_strategy="messages"`).
No impact on M3+ — adds coverage without changing the spec's
behavioural contract.

### §2.8 M1 compose fix: `clickhouse:lts` (not `clickhouse/clickhouse-server:lts`)

`examples/17_structured/docker-compose.yml` shipped in M1 referenced
`clickhouse/clickhouse-server:lts`, but Docker Hub has deprecated the
`lts` tag in that repository (the user hit
`manifest unknown: manifest unknown` on `docker compose up`).  The
repo-root `docker-compose.yml` uses Docker's official `clickhouse:lts`
image, which is still published.  Updated the example compose to match.
This is technically an M1 carry-over, not a M2 task, but it was
blocking M2's integration-test verification — fixing it inline avoided
a roundtrip.  No impact on M3+.

## §3 Open questions raised or resolved

No new open questions were raised during M2.

No previously-open questions were resolved by M2 — the four still-open
items in `02-open-questions.md` (Q2, Q4, Q5, Q6, Q8, Q9, Q10) all
target M3 or M4 by design.  Q13 ("system_arguments threading through
JSON") is implicitly confirmed by
`TestPerCallOverrides::test_per_call_output_schema_override`'s sibling
assertion that `system_arguments` flows through the helper unchanged,
but the question was already marked decided in M1.

## §4 Carry-forward notes

### For M3 — permissive parsing + retry-on-invalid-json

- The retry orchestration lives in `Agent.run`'s JSON path
  (contract §4.4), not in `run_structured_inference`.  Concretely,
  M3 extends `_run_json` (added by M2) — not the helper — with a
  retry loop around the `run_structured_inference` call.
- `AgentResult.inference_ids` is currently a single-entry list for
  JSON runs.  M3 must append every retry's inference id to this
  list, in order, per contract §3.4.
- `AgentResult.turns` is currently always `1` for JSON runs.  M3
  must increment it on retries (per contract §3.4: `turns = 1 +
  json_parse_attempts`).
- Episode-id continuity across retries (Q8 in
  `02-open-questions.md`) — M3 should preserve the first attempt's
  `episode_id` on subsequent attempts so the retry stays in the
  same TZ episode.  `run_structured_inference` already accepts an
  `episode_id` kwarg.
- The `_make_json_cost_tracker` closure increments `ctx.turn` per
  inference; M3 will benefit because that means cumulative cost
  tracking already works across retries without additional wiring.
- Memory save (M2's step 6) currently writes `agent_response =
  result.response` — i.e. the raw JSON of the **final** attempt.
  This matches contract §3.5; M3 should preserve this — do not
  switch to a list of all attempts.
- Resolved-`output_schema` / `json_mode` are computed once at the
  top of `_run_json`.  Reuse the same values across retries; do
  not re-resolve.
- The runtime guard for `memory_strategy="messages"` fires before
  the inference loop, so M3 inherits the constraint for free.
  Revisit the guard only if M3 extends
  `run_structured_inference` to accept `prior_messages`.

### For M4 — provider safety, streaming guard, quirks

- `Agent.stream` still has its original v0.9.0 signature.  M4
  needs to add (a) the JSON-stream guard (raise
  `ConfigurationError` at the top of the async generator when
  `config.function_kind == "json"`) and (b) the
  `output_schema`/`json_mode` kwargs to match `Agent.run`'s
  signature, even if the kwargs only make sense to pass through
  to a future streaming-JSON helper.
- Runtime check for `json_mode == "tool"` + non-empty tool list in
  `Agent.__init__` (contract §5.1 last bullet) — still outstanding.
  M1 deferred it because `AgentConfig` has no access to the tool
  provider; M4 should add it in `Agent.__init__`.
- Anthropic extended-thinking provider warning (Q10) — still open.
  M4 logs a `logger.warning` on `Agent.__init__` when
  `json_mode not in {"strict", "off"}` and the configured variant
  reports an Anthropic extended-thinking model.  See TZ guide
  snapshot §11.

### For M5 — testing helpers, examples, polish

- `MockGateway` is chat-only in v0.9.0 + M1 + M2.  The local
  `FakeJsonT0` / `_fake_json_response` helpers in
  `tests/test_core/test_agent_json_run.py` mirror M1's identical
  helpers in `tests/test_core/test_structured.py`.  M5 should
  fold both into `MockGateway` (its JSON branch) and update both
  test files to drop the local fakes.
- The shape-contract test
  (`tests/test_testing/test_mock_shape.py`) still hard-codes the
  chat `_INFERENCE_KWARGS` surface.  M5 extends it to a JSON
  branch — this was already flagged in M1 dev-log §4 and remains
  open after M2 (we did not touch the file).
- Consolidate the per-milestone CHANGELOG sub-sections
  (`## [0.10.0-M1]`, `## [0.10.0-M2]`, and the upcoming
  `## [0.10.0-M3]`, `## [0.10.0-M4]`) into a single
  `## [0.10.0]` block when M5 closes the roadmap.

## §5 Verification

Commands run at M2 close-out (all timestamps 2026-05-14, on the
implementer's local checkout — pre-PR):

- `uv run ruff check src/` — **passed.**
- `uv run ruff check tests/test_core/test_agent_json_run.py tests/test_core/test_agent_json_run_integration.py` — **passed.**
  (The shared `tests/` directory has 7 pre-existing ruff errors in
  unrelated files; M2 does not touch them — out of scope.)
- `uv run mypy src/` — **passed.**  No issues across 54 source files
  under strict mode.
- `uv run mypy tests/test_core/test_agent_json_run.py tests/test_core/test_agent_json_run_integration.py examples/17_structured/structured_agent.py` — **passed.**
- `uv run pytest -m "not integration"` — **passed.**  561 passed,
  2 skipped, 51 deselected (vs M1's baseline of 549 passed, 2
  skipped, 50 deselected — +12 unit tests, +1 deselected integration
  test).
- `uv run pytest tests/test_core/test_agent_json_run.py -v` —
  **passed.**  12/12 green in 0.10s.
- `git diff src/t0_agents/mcp/` — **empty.**  Byte-identical per
  contract §2.1.
- `git diff src/t0_agents/core/structured.py src/t0_agents/core/config.py src/t0_agents/core/types.py src/t0_agents/core/exceptions.py src/t0_agents/__init__.py` — **empty.**  M1 territory untouched.
- `git diff --stat src/t0_agents/` — `core/agent.py | 273 +++- 1
  file changed, 270 insertions(+), 3 deletions(-)`.

Integration tests (require `docker compose up -d` from the repo root —
the root compose mounts `config/tensorzero.toml`, which declares both
`agent` (chat) and `extract_customer_info` (json), so the whole
integration suite runs against a single gateway):

- `uv run pytest -m integration` — **passed (user verification on
  2026-05-14).**  All 51 selected integration tests green, including
  the new `test_agent_json_run_integration::test_agent_run_json_end_to_end`,
  the pre-existing M1 `test_structured_integration` pair, and the
  pre-existing v0.9.0 suite (`test_mcp`, `test_compose`,
  `test_agent_integration`, `test_guardrails`, `test_memory`,
  `test_serve`, `test_cli`).  First attempt against the example's
  compose file (`examples/17_structured/docker-compose.yml`) failed
  the non-M2 tests because that config only declares
  `extract_customer_info`; switching to the repo-root compose fixed
  it — see §2.8.
- `uv run python examples/17_structured/structured_agent.py` —
  **pending** (the user has not reported this back; not blocking the
  milestone since the integration test exercises the same code path
  against the same function).
