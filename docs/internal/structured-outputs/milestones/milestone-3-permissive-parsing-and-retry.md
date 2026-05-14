# Milestone 3 — Permissive Parsing & Retry-on-Invalid-JSON

> The value-add layer that justifies `t0-agents` building on top of
> raw TensorZero.  Adds in-Python recovery for prose-wrapped JSON
> (zero provider cost) and an inference-retry loop with a corrective
> hint (provider cost, bounded by `max_json_retries`).
>
> **This is the milestone the user called out as "fundamental e
> importante" — it's what makes t0-agents work cleanly even with
> weaker models that occasionally violate `json_mode = "strict"`.**

## §1 Goal

Make JSON runs survive imperfect model output.  After this milestone,
the following all return a usable `result.parsed`:

```python
# Model returned: '{"email": "a@b.com"}'
# Layer: TZ strict parse → result.parse_strategy == "strict"

# Model returned: 'Here is your json: {"email": "a@b.com"}'
# Layer: in-Python permissive parse → result.parse_strategy == "permissive"

# Model returned: 'I cannot help with that' (first), then '{"email": "a@b.com"}' (after retry)
# Layer: retry with hint → result.parse_strategy == "retry",
#                          result.json_parse_attempts == 1
```

And the failure path is observable, not silent:

```python
# Model returned invalid JSON every time
# Layer: none of the above succeeded
# result.parsed is None
# result.parse_strategy == "none"
# result.json_parse_attempts == config.max_json_retries
# (caller inspects and decides; no exception raised by default)
```

## §2 Scope

### In scope

- New helper `permissive_parse(raw: str) -> tuple[Any | None, Literal["strict","permissive","none"]]`
  in `core/structured.py`.
- New `AgentConfig` fields: `permissive_json_parsing: bool = True`,
  `max_json_retries: int = 2`.
- New `AgentResult` fields: `json_parse_attempts: int = 0`,
  `parse_strategy: Literal["strict","permissive","retry","none"] = "strict"`.
- New per-call kwarg on `Agent.run`: `max_json_retries: int | None = None`.
- Retry orchestration inside `_run_json` (`core/agent.py`) — uses the
  M1 helper repeatedly with a hint message appended between attempts.
- Tests: 10+ unit tests, 1 integration test, 1 property-based test
  exercising prose-wrapped extraction.
- Example: `examples/16_structured/recovery_demo.py`.
- Docs: "Robust JSON parsing" section in `docs/structured-outputs.md`.

### Out of scope

- JSON-Schema validation in Python (Q12, decided).  Permissive parser
  returns whatever JSON it can extract; schema-conformance is the
  caller's concern.
- Per-call hook to customise the retry hint message (Q4 — deferred;
  hard-coded short hint in M3).
- Wall-clock retry budget (Q6 — attempt count only in M3).
- `Agent.stream(...)` retry behaviour (M4 raises `ConfigurationError`
  for the JSON stream path).

### Non-goals

- Auto-retrying on TensorZero transport errors (5xx, timeouts).  Those
  are TZ's variant-level concern; this milestone only retries when
  the output is not parseable JSON.

## §3 Constraints

- `mcp/` is byte-identical to v0.9.0.
- Chat-path callers see byte-identical behaviour: new fields default to
  values that match v0.9.0 output exactly (`parse_strategy="strict"`
  default is intentional — for chat runs the field exists but is not
  meaningful; users can ignore it).
- Permissive parsing **does not run** when TZ already returned
  `parsed is not None` — no over-parsing.
- Retries **do not happen** when TZ returned valid JSON on the first
  try, even if some other criterion failed (e.g. schema validation —
  but we don't do schema validation, so this is moot in v1).
- The retry hint message is hard-coded in English; if i18n becomes a
  need, a follow-up adds a hook.
- `max_json_retries = 0` disables the retry layer entirely; only the
  permissive parser runs (when enabled).
- `permissive_json_parsing = False` and `max_json_retries = 0` together
  is the "TZ-pure" mode: no value-add layer.
- mypy strict, ruff clean, line length 100.

## §4 Module additions

### §4.1 `core/structured.py` (modified)

Add `permissive_parse`:

```python
import json
import re
from typing import Any, Literal

PermissiveStrategy = Literal["strict", "permissive", "none"]


def permissive_parse(
    raw: str,
) -> tuple[Any | None, PermissiveStrategy]:
    """Try increasingly relaxed JSON extraction strategies.

    Returns ``(value, "strict")`` when ``raw`` is itself valid JSON.
    Returns ``(value, "permissive")`` when JSON could be extracted from
    surrounding prose (via balanced-brace scan or fenced code block).
    Returns ``(None, "none")`` if no strategy succeeded.

    Empty string ``raw`` returns ``(None, "none")``.
    """
    if not raw:
        return None, "none"
    # Layer 1: strict parse
    try:
        return json.loads(raw), "strict"
    except (json.JSONDecodeError, ValueError):
        pass
    # Layer 2: balanced-brace scan
    value = _extract_first_balanced(raw, openers="{[", closers="}]")
    if value is not None:
        return value, "permissive"
    # Layer 3: fenced code block
    fenced = _extract_fenced_code(raw)
    if fenced is not None:
        try:
            return json.loads(fenced), "permissive"
        except (json.JSONDecodeError, ValueError):
            pass
    return None, "none"


def _extract_first_balanced(
    raw: str, *, openers: str, closers: str,
) -> Any | None:
    """Find the first balanced {...} or [...] and try to parse it.

    Returns the parsed value on success; ``None`` on failure.  Uses
    a simple depth counter that tolerates nested objects/arrays.
    Quoted strings are respected so braces inside string literals
    do not affect depth.
    """
    ...

def _extract_fenced_code(raw: str) -> str | None:
    """Extract content from the first ```json ... ``` or ``` ... ```
    fenced block.  Returns the inner string or ``None``."""
    ...
```

The two private helpers handle the heavy lifting.  Both are pure-
function, side-effect-free, and individually unit-tested.

### §4.2 `core/structured.py` — `StructuredOutput` update

`StructuredOutput.parse_strategy` widens to include `"retry"`
(declared in M1, used in M3).  `retry_attempts` is set by the caller
(`Agent._run_json`) rather than the helper.

### §4.3 `core/config.py` (modified)

`AgentConfig` gains two fields:

```python
@dataclass
class AgentConfig:
    # ... existing fields, plus M1 fields ...
    permissive_json_parsing: bool = True
    max_json_retries: int = 2
```

`__post_init__` validations:

- `max_json_retries < 0` ⇒ `ConfigurationError`.

(No new validation for `permissive_json_parsing` — it's a bool.)

### §4.4 `core/types.py` (modified)

`AgentResult` gains two fields:

```python
@dataclass
class AgentResult:
    # ... existing fields, plus M1 fields ...
    json_parse_attempts: int = 0
    parse_strategy: Literal["strict", "permissive", "retry", "none"] = "strict"
```

### §4.5 `core/agent.py` — `_run_json` retry orchestration (modified)

The orchestration sketch:

```python
async def _run_json(self, user_message, *, ..., output_schema, json_mode, max_json_retries):
    # 1. memory load + before_run guardrails  (same as M2)
    ...

    # 2. resolve effective config
    permissive_enabled = self._config.permissive_json_parsing
    retry_budget = (
        max_json_retries
        if max_json_retries is not None
        else self._config.max_json_retries
    )
    effective_schema = output_schema or self._config.output_schema
    effective_mode = json_mode or self._config.json_mode

    # 3. inference + retry loop
    inference_ids: list[str] = []
    attempts = 0
    structured: StructuredOutput | None = None
    last_raw: str = ""
    history_messages: list[Message] = []  # for retry hints

    while True:
        structured = await run_structured_inference(
            self._t0,
            self._config.function_name,
            user_message,
            output_schema=effective_schema,
            json_mode=effective_mode,
            system_arguments=...,
            episode_id=episode_id,
            tags=...,
            namespace=...,
            on_inference=on_inference,
            # prior_messages=history_messages,   # see §4.6 note
        )
        inference_ids.append(structured.inference_id)
        episode_id = structured.episode_id  # propagate to subsequent retries
        last_raw = structured.raw

        # If TZ already parsed → done.
        if structured.parsed is not None:
            final = structured  # parse_strategy stays "strict"
            break

        # TZ failed.  Try permissive parser.
        if permissive_enabled:
            parsed, layer = permissive_parse(structured.raw)
            if parsed is not None:
                final = dataclasses.replace(
                    structured, parsed=parsed, parse_strategy=layer,
                )
                break

        # Permissive failed too.  Retry?
        if attempts >= retry_budget:
            final = dataclasses.replace(
                structured, parse_strategy="none",
            )
            break

        attempts += 1
        # Append a corrective hint to the conversation for the next try.
        history_messages.append(_assistant_turn(structured.raw))
        history_messages.append(_user_turn(RETRY_HINT))

    # 4. build AgentResult
    result = AgentResult(
        response=last_raw,
        parsed=final.parsed,
        response_kind="json",
        episode_id=final.episode_id,
        inference_ids=inference_ids,
        turns=1 + attempts,
        json_parse_attempts=attempts,
        parse_strategy=("retry" if attempts > 0 and final.parsed is not None
                        else final.parse_strategy),
        guardrail_log=...,
    )

    # 5. memory save + after_run guardrails  (same as M2)
    ...
    return result


RETRY_HINT = (
    "Your previous response was not valid JSON matching the schema. "
    "Return ONLY a valid JSON object that matches the schema, with no "
    "commentary or prose."
)
```

### §4.6 Note on retry hints and TZ message history

The helper `run_structured_inference` does not currently accept a
`prior_messages` argument — it's single-shot, with `user_message` as
the only input.  For the retry path, we need to thread prior turns
so the model sees its own bad output and the correction request.

Two options:

1. **Add `prior_messages` to `run_structured_inference`.**  Mirrors
   the v0.9.0 `run_agent(prior_messages=...)` kwarg.  Cleanest; the
   helper stays single-shot but accepts conversation state.  **Pick
   this.**
2. Encode the retry as system_arguments / template variables.  Couples
   the retry mechanism to the template, which is brittle.  Reject.

The dev-log records the choice.  If option 1 surfaces an SDK incompat
(some TZ JSON-function endpoints may not accept `prior_messages`),
the dev-log records the workaround.

## §5 Tests

### §5.1 Unit tests for `permissive_parse` (`tests/test_core/test_permissive_parse.py`)

- `test_strict_pass`: `'{"a": 1}'` → `({"a": 1}, "strict")`.
- `test_prose_prefix`: `'Here is your json: {"a": 1}'` →
  `({"a": 1}, "permissive")`.
- `test_prose_suffix`: `'{"a": 1} and that is the result.'` →
  `({"a": 1}, "permissive")`.
- `test_fenced_code_block`: `'```json\n{"a": 1}\n```'` →
  `({"a": 1}, "permissive")`.
- `test_fenced_code_block_no_lang`: `'```\n{"a": 1}\n```'` →
  `({"a": 1}, "permissive")`.
- `test_top_level_array`: `'[1, 2, 3]'` → `([1,2,3], "strict")`.
- `test_array_with_prose`: `'Here it is: [1, 2, 3] done.'` →
  `([1,2,3], "permissive")`.
- `test_nested_braces_in_string`: `'{"text": "}{nested}{"}'` parses
  to that exact value (depth counter respects string literals).
- `test_invalid_returns_none`: `'totally not json'` → `(None, "none")`.
- `test_empty_string`: `''` → `(None, "none")`.
- `test_unbalanced_returns_none`: `'{not balanced'` → `(None, "none")`.

### §5.2 Property-based test (hypothesis)

`tests/test_core/test_permissive_parse_property.py`:

```python
from hypothesis import given, strategies as st

@given(
    prefix=st.text(alphabet=st.characters(blacklist_characters="{[]}")),
    obj=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
    suffix=st.text(alphabet=st.characters(blacklist_characters="{[]}")),
)
def test_permissive_recovers_embedded_json(prefix, obj, suffix):
    raw = f"{prefix}{json.dumps(obj)}{suffix}"
    parsed, strategy = permissive_parse(raw)
    assert parsed == obj
    assert strategy in {"strict", "permissive"}
```

(Note: `hypothesis` may already be a transitive dep; if not, this test
is marked `pytest.importorskip("hypothesis")` or moved to an opt-in
extra.  Decide during implementation; record in dev-log.)

### §5.3 Unit tests for the retry orchestration (`tests/test_core/test_agent_json_retry.py`)

- `test_no_retry_when_first_attempt_strict`: `MockGateway` returns
  valid JSON on the first call.  Assert `attempts == 0`,
  `parse_strategy == "strict"`.
- `test_no_retry_when_first_attempt_permissive`: `MockGateway`
  returns prose-wrapped JSON; TZ's `parsed` is None.  Assert
  `attempts == 0`, `parse_strategy == "permissive"`,
  `result.parsed == ...`.
- `test_retry_once_succeeds`: `MockGateway` scripts `(invalid → valid)`
  responses.  Assert `attempts == 1`, `parse_strategy == "retry"`,
  `len(inference_ids) == 2`.
- `test_retry_budget_exhausted`: `MockGateway` scripts
  `(invalid → invalid → invalid)` with `max_json_retries=2`.  Assert
  `attempts == 2`, `parse_strategy == "none"`,
  `result.parsed is None`, `len(inference_ids) == 3`.
- `test_per_call_retry_override`: agent has `max_json_retries=5` but
  per-call kwarg passes `max_json_retries=0` — assert no retry happens.
- `test_retry_preserves_episode_id`: scripted retries; assert every
  inference shares the same `episode_id`.
- `test_permissive_disabled_skips_layer`: agent with
  `permissive_json_parsing=False` + scripted prose-wrapped response
  + `max_json_retries=0`.  Assert `parsed is None`,
  `parse_strategy == "none"` (the permissive parser did not run).
- `test_negative_max_json_retries_is_config_error`: building
  `AgentConfig(... max_json_retries=-1)` raises `ConfigurationError`.
- `test_retry_hint_appears_in_history`: capture the `prior_messages`
  passed on the second `run_structured_inference` call; assert it
  contains the assistant's prior bad output and a user-turn with
  the `RETRY_HINT` text.

### §5.4 Integration test (`@pytest.mark.integration`)

`tests/test_core/test_agent_json_retry_integration.py`:

- Configure a function with **`json_mode = "off"`** (so the provider
  has no schema enforcement) and a deliberately under-specified
  prompt template that often produces prose-wrapped JSON.
- Run the agent against this function with
  `permissive_json_parsing=True`, `max_json_retries=2`.
- Assert that across N runs (N=5), the success rate is materially
  higher than running with both layers off.

This test is **noisy** by nature (depends on the provider).  Mark it
`@pytest.mark.integration` and `@pytest.mark.flaky` (or similar) so CI
treats it as informative, not gating.  Record the observation in the
dev-log §5.

## §6 Example

`examples/16_structured/recovery_demo.py`:

- Single script that demonstrates all four `parse_strategy` outcomes
  using a `MockGateway` (so it runs without docker, instantly).
- Shows the `result.parse_strategy` and `result.json_parse_attempts`
  fields.
- README explains the three layers (TZ → permissive → retry) with a
  flow diagram.

A companion runnable script
`examples/16_structured/recovery_real_provider.py` exercises the
same agent against a real provider (Groq through TZ).  Optional;
requires `docker compose up` + `GROQ_API_KEY`.

## §7 Documentation

`docs/structured-outputs.md` gains a "Robust JSON parsing" section:

- The three-layer diagram (TZ strict → in-Python permissive → retry).
- Configuration table: `permissive_json_parsing`, `max_json_retries`,
  per-call overrides.
- Observability table: `parse_strategy` values, `json_parse_attempts`,
  `inference_ids` length.
- Worked example: weak-model output → permissive recovery → final
  parsed.
- "Opt out for TZ-pure behaviour" callout:
  `AgentConfig(..., permissive_json_parsing=False, max_json_retries=0)`.
- "Best-practice" callout: leave the defaults on unless you need byte-
  perfect fidelity to raw TZ semantics.

## §8 Checklist

### Pre-flight
- [ ] Read `logs/milestone-1-dev-log.md` and
      `logs/milestone-2-dev-log.md` in full — every deviation from
      the spec is potentially load-bearing for M3.
- [ ] Re-read `00-integration-contract.md` §3.4 — the four new fields
      and their semantics.
- [ ] Re-read the guide section §11.14 ("`parsed` can be `null` even
      on a successful HTTP request") and §11.15 (undocumented
      validation behaviour).

### Code
- [ ] Add `permissive_parse` + private helpers to `core/structured.py`.
- [ ] Decide and document `prior_messages` plumbing through
      `run_structured_inference` (§4.6).
- [ ] Add `permissive_json_parsing`, `max_json_retries` to
      `AgentConfig`; validate `max_json_retries >= 0`.
- [ ] Add `json_parse_attempts`, `parse_strategy` to `AgentResult`.
- [ ] Implement retry orchestration in `Agent._run_json`.
- [ ] Add `max_json_retries` kwarg to `Agent.run`.
- [ ] Verify `mcp/` is byte-identical to v0.9.0.

### Tests
- [ ] `tests/test_core/test_permissive_parse.py` — 11 unit tests.
- [ ] `tests/test_core/test_permissive_parse_property.py` — 1+
      hypothesis test (or skip + note in dev-log if hypothesis is
      not available).
- [ ] `tests/test_core/test_agent_json_retry.py` — 9 unit tests.
- [ ] `tests/test_core/test_agent_json_retry_integration.py` — 1
      integration test, flagged noisy.
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `uv run pytest -m integration` green (with docker up).

### Example
- [ ] `examples/16_structured/recovery_demo.py` runnable instantly
      (uses `MockGateway`).
- [ ] `examples/16_structured/recovery_real_provider.py` runnable
      with docker compose up.
- [ ] README updated with the recovery diagram.

### Docs
- [ ] `docs/structured-outputs.md` gains "Robust JSON parsing" section.

### Style gates
- [ ] `uv run ruff check src/`.
- [ ] `uv run mypy src/`.

### Implementation log
- [ ] Write `docs/internal/structured-outputs/logs/milestone-3-dev-log.md`.
      Record:
      - Q2 (schema validation in TZ) — whatever was observed.
      - Q4 (retry hint phrasing) — final wording committed.
      - Q5 (top-level arrays in permissive parse) — yes per §4.1.
      - Q6 (wall-clock budget) — open or decided.
      - Q8 (`episode_id` preservation across retries) — decision.
      - Q9 (memory persistence of multi-attempt run) — decision.

### CHANGELOG
- [ ] Append `## [0.10.0-M3] — Unreleased` under M2's entry.

## §9 Acceptance criteria

1. `permissive_parse('{"a": 1}')` → `({"a": 1}, "strict")`.
2. `permissive_parse('Here it is: {"a": 1}')` →
   `({"a": 1}, "permissive")`.
3. `permissive_parse('totally not json')` → `(None, "none")`.
4. A JSON agent with `max_json_retries=2` and a `MockGateway` script
   `(invalid, invalid, valid)` returns `result.parsed = ...`,
   `result.json_parse_attempts = 2`, `result.parse_strategy = "retry"`,
   `len(result.inference_ids) == 3`.
5. A JSON agent with `max_json_retries=0` and a `MockGateway` script
   `(invalid_but_extractable_prose,)` returns `result.parsed = ...`
   via permissive parse — no retries.
6. Setting `permissive_json_parsing=False, max_json_retries=0` yields
   byte-identical behaviour to M2 (no recovery layer).
7. The chat path remains byte-identical to v0.9.0.
8. `mcp/` source is byte-identical to v0.9.0.
9. Every existing test still passes; new tests pass.
10. `examples/16_structured/recovery_demo.py` runs in < 1 s on a
    fresh clone without docker.
11. `docs/structured-outputs.md` has the "Robust JSON parsing"
    section with the three-layer diagram.
12. `CHANGELOG.md` has the `## [0.10.0-M3] — Unreleased` entry.
13. `logs/milestone-3-dev-log.md` exists with all five sections.

## §10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Balanced-brace scanner is buggy in edge cases (escaped quotes, nested arrays inside object values, unicode escapes) | Property-based test with `hypothesis`; explicit unit test for nested strings; consider falling back to `json.JSONDecoder.raw_decode` (which can decode the first valid JSON in a buffer) as a simpler implementation. |
| `hypothesis` is not in the dev dependencies | Check `pyproject.toml`; if absent, either add it to the `[dev]` group or skip the property test with a `pytest.importorskip` and record the choice in the dev-log. |
| `prior_messages` does not flow through `run_structured_inference` cleanly because the TZ SDK rejects them on JSON function calls | Verify against the SDK first; if blocked, encode the retry as a sequence of `input.messages` directly in the helper (which is what `prior_messages` ultimately does in the chat path). |
| The retry hint changes the model's behaviour in unexpected ways (e.g. the model now apologises in JSON) | This is a UX concern, not a correctness one — the hint is documented and the caller can disable the layer (`max_json_retries=0`). |
| The noisy integration test in §5.4 fails CI intermittently | Mark it `@pytest.mark.flaky` (or use `pytest-rerunfailures`); document in the dev-log that it is informative-only. |
| Memory persistence — saving `output.raw` from the last (possibly invalid) attempt is misleading | Document the `agent_response` semantics in `docs/memory.md` (cross-link) and in the dev-log; the caller can check `parse_strategy == "none"` before trusting the record. |

## §11 Estimated surface

- New files: 2 (`examples/16_structured/recovery_demo.py`,
  `examples/16_structured/recovery_real_provider.py`).
- Modified files: 4 (`core/structured.py`, `core/config.py`,
  `core/types.py`, `core/agent.py`).
- New tests: ~22 (unit + property + integration).
- Modified docs: 1 (`structured-outputs.md` "Robust JSON parsing").
- Lines of code added: ~150 in `core/structured.py` (permissive parser
  + helpers), ~80 in `core/agent.py` (retry orchestration).
