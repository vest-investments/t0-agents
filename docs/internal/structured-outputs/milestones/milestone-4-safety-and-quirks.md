# Milestone 4 — Provider Safety, Streaming Guard, Quirks

> Adds the configuration-error surfaces for incompatible combinations
> and documents the provider quirks `t0-agents` cannot detect at
> runtime.  Ships an opt-in `require_parsed()` guardrail for callers
> who want a hard abort when JSON parsing exhausts every layer.

## §1 Goal

Make obviously-wrong configurations fail fast with actionable messages,
warn about silently-degrading ones, and provide an opt-in guardrail
for callers who want strict failure semantics.

After this milestone:

- `AgentConfig(function_kind="json", json_mode="tool")` + a
  `tool_provider` advertising tools → `ConfigurationError` at
  `Agent.__init__` with a message pointing to TZ guide §11.1.
- `agent.stream("...")` on a JSON-flavoured agent → `ConfigurationError`
  before any event is yielded, with a message pointing to
  `agent.run(...)`.
- `AgentConfig(function_kind="json", json_mode="on")` with a runtime
  warning *iff* the gateway can be heuristically identified as routing
  to Anthropic extended-thinking models (best-effort; documented as
  may-miss).
- `from t0_agents.guardrails import require_parsed; AgentConfig(...,
  guardrails=[require_parsed()])` → run with that guardrail
  aborts with `GuardrailBlockedError` when `result.parsed is None`.

## §2 Scope

### In scope

- `ConfigurationError` raised in `Agent.__init__` when:
  - `config.function_kind == "json"` AND `config.json_mode == "tool"`
    AND `tool_provider.list_tools()` returns a non-empty list.
- `ConfigurationError` raised in `Agent.stream(...)` when
  `config.function_kind == "json"` — raised **before** yielding any
  event from the async generator.
- Best-effort `log.warning` at `Agent.__init__` (or on the first run,
  whichever is more practical) when `json_mode in {"on", "tool"}`
  for users on Anthropic extended-thinking routes.  The warning is
  generic ("verify provider compatibility") because the SDK does not
  expose provider routing.
- Documentation of the Bedrock/Vertex + Anthropic + `strict`
  degradation (guide §11.5).
- New built-in guardrail `t0_agents.guardrails.require_parsed()`
  factory that returns a `Guardrail` whose `after_run` raises
  `GuardrailBlockedError` when the run's result has
  `parsed is None`.
- Tests for every new error path.
- Documentation: "Safety & provider quirks" section in
  `docs/structured-outputs.md`; cross-link from `docs/guardrails.md`.

### Out of scope

- Provider auto-detection (Q10 — punted; we log a generic warning).
- Translating the silent-degradation case (Bedrock + strict) into a
  runtime error.  We cannot detect it; only document it.
- New guardrails beyond `require_parsed()`.

### Non-goals

- Schema validation in `require_parsed()`.  It only checks the
  `parsed is None` case.  Schema conformance is the caller's
  responsibility (guide §11.15 / Q2).

## §3 Constraints

- `mcp/` is byte-identical to v0.9.0.
- Chat-path callers see byte-identical behaviour: the new checks gate
  exclusively on `function_kind == "json"`.
- The streaming guard does not affect chat streams — only JSON-flavoured
  configs.
- `require_parsed()` is opt-in; it does not become part of any default
  policy set.
- The Anthropic warning fires at most **once per Agent instance** —
  not once per call — to avoid log spam.

## §4 Module additions

### §4.1 `core/agent.py` (modified)

#### §4.1.1 `Agent.__init__` checks

Add the tool-mode conflict check (uses `await
tool_provider.list_tools()` lazily — see §4.1.2 for the bootstrap
problem).

```python
def __init__(self, config, *, t0, tool_provider):
    # ... existing validations ...
    if (
        config.function_kind == "json"
        and config.json_mode == "tool"
    ):
        # Eagerly check that the tool provider has no tools (TZ guide §11.1).
        # tool_provider.list_tools() is async, so we either:
        # (a) defer this check to the first `run`, or
        # (b) require the caller to provide an empty tool_provider.
        # Decision: (a), with a clear error message at first call.
        self._needs_tool_mode_check = True
    else:
        self._needs_tool_mode_check = False
    # ...
```

In `_run_json` (M2), at the start:

```python
if self._needs_tool_mode_check:
    tools = await self._tool_provider.list_tools()
    if tools:
        raise ConfigurationError(
            "AgentConfig(json_mode='tool') cannot be combined with a "
            "tool_provider that advertises tools — see TensorZero guide "
            "§11.1.  Either change json_mode to 'strict'/'on'/'off' or "
            "use an empty tool_provider."
        )
    self._needs_tool_mode_check = False  # cache
```

#### §4.1.2 Anthropic warning

```python
def __init__(self, config, *, t0, tool_provider):
    # ...
    self._warned_anthropic_combination = False
    # ...

def _maybe_warn_anthropic_quirk(self):
    if self._warned_anthropic_combination:
        return
    if self._config.json_mode in {"on", "tool"}:
        logger.warning(
            "AgentConfig(json_mode=%r) — Anthropic extended-thinking "
            "models reject this combination (TZ guide §11.4).  Verify "
            "your provider routing.",
            self._config.json_mode,
        )
        self._warned_anthropic_combination = True
```

Called at the top of `_run_json`.

#### §4.1.3 Streaming guard

```python
async def stream(self, user_message, *, ...):
    if self._config.function_kind == "json":
        raise ConfigurationError(
            "Agent.stream() is not supported for function_kind='json'.  "
            "TensorZero does not produce `output.parsed` mid-stream "
            "(guide §7.1, §11.3); use Agent.run() instead."
        )
    # ... existing body ...
```

### §4.2 `guardrails/builtins.py` (modified)

Add `require_parsed()`:

```python
def require_parsed(*, name: str = "require_parsed") -> Guardrail:
    """Hard-abort when a JSON run completes with `parsed is None`.

    Opt-in: add to AgentConfig(guardrails=[require_parsed()]) when
    your application cannot tolerate an unparseable JSON output.

    For chat (text) runs, this guardrail is a no-op — `parsed` is
    always `None` for them, so an unconditional fail would be wrong.
    The guardrail checks `result.response_kind == "json"` before
    evaluating.
    """

    class _RequireParsed:
        @property
        def name(self) -> str: return name

        async def after_run(self, ctx, result):
            if result is None:
                return
            if result.response_kind != "json":
                return
            if result.parsed is None:
                raise GuardrailBlockedError(
                    f"{name}: JSON run completed with parsed=None "
                    f"(parse_strategy={result.parse_strategy}, "
                    f"attempts={result.json_parse_attempts})"
                )

    return cast(Guardrail, _RequireParsed())
```

Export from `t0_agents.guardrails.__init__`.

### §4.3 No other module changes

`mcp/`, `memory/`, `serve/`, `cli/`, `stream/`, `compose/`, `testing/`
are untouched.

## §5 Tests

### §5.1 Unit tests (`tests/test_core/test_agent_json_safety.py`)

- `test_tool_mode_with_tools_errors`: build an agent with
  `json_mode="tool"` and a `MockToolProvider({"add": ...})`.  Call
  `agent.run("...")` — expect `ConfigurationError`.
- `test_tool_mode_with_empty_tools_works`: same setup but
  `MockToolProvider({})` — no error; run succeeds.
- `test_stream_on_json_agent_errors`: build a JSON agent; iterate
  `agent.stream("...")` — expect `ConfigurationError` on the first
  `__anext__`.
- `test_stream_on_chat_agent_byte_identical`: build a chat agent;
  stream works as before.
- `test_anthropic_warning_fires_once`: capture logs; build an agent
  with `json_mode="on"`, call `run` twice — assert exactly one
  warning emitted.
- `test_anthropic_warning_not_emitted_for_strict`: `json_mode="strict"`
  — no warning.

### §5.2 Unit tests (`tests/test_guardrails/test_require_parsed.py`)

- `test_require_parsed_passes_when_parsed`: a `MockGateway` returns
  valid JSON; guardrail does not abort.
- `test_require_parsed_aborts_when_none`: `MockGateway` returns
  invalid JSON, `permissive_json_parsing=False`, `max_json_retries=0`;
  guardrail raises `GuardrailBlockedError`.
- `test_require_parsed_noop_on_chat_run`: a chat agent (no parsed)
  with `require_parsed()` — assert no exception.

### §5.3 Integration test
(`tests/test_core/test_agent_json_safety_integration.py`)

`@pytest.mark.integration` — verifies the stream guard against a real
gateway: build a JSON agent and an async generator from `agent.stream`,
confirm `ConfigurationError` is raised at first `__anext__`.

## §6 Example

No new top-level example.  The recovery demo from M3 grows a section
showing `require_parsed()` aborting when retries are exhausted.

## §7 Documentation

### §7.1 `docs/structured-outputs.md` (extended)

Add the "Safety & provider quirks" section:

- Table of `ConfigurationError` cases (with one-line resolution
  each).
- `json_mode` × provider compatibility table (copy of guide §13.1
  with our enforcement layer overlaid: which combinations are
  config errors vs runtime warnings vs silent degradation we cannot
  detect).
- "Why you cannot stream JSON functions" subsection.
- "Opt-in hard failure" subsection introducing `require_parsed()`.

### §7.2 `docs/guardrails.md` (modified)

Add `require_parsed()` to the built-ins list with a one-paragraph
explanation and cross-link to `docs/structured-outputs.md`.

## §8 Checklist

### Pre-flight
- [ ] Read `logs/milestone-1-dev-log.md`,
      `logs/milestone-2-dev-log.md`,
      `logs/milestone-3-dev-log.md` in full.
- [ ] Re-read contract §3.3 (stream behaviour) and §5 (error model).

### Code
- [ ] Add `_needs_tool_mode_check` and `_warned_anthropic_combination`
      state to `Agent.__init__`.
- [ ] Add the tool-mode check at the top of `_run_json`.
- [ ] Add the Anthropic warning logic.
- [ ] Add the JSON stream guard in `Agent.stream`.
- [ ] Implement `require_parsed()` in `guardrails/builtins.py`.
- [ ] Export `require_parsed` from `t0_agents.guardrails.__init__`.
- [ ] Verify `mcp/` is byte-identical to v0.9.0.

### Tests
- [ ] `tests/test_core/test_agent_json_safety.py` — 6 unit tests.
- [ ] `tests/test_guardrails/test_require_parsed.py` — 3 unit tests.
- [ ] `tests/test_core/test_agent_json_safety_integration.py` — 1
      integration test.
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `uv run pytest -m integration` green (with docker up).

### Docs
- [ ] `docs/structured-outputs.md` "Safety & provider quirks" section.
- [ ] `docs/guardrails.md` `require_parsed()` entry.

### Style gates
- [ ] `uv run ruff check src/`.
- [ ] `uv run mypy src/`.

### Implementation log
- [ ] Write `docs/internal/structured-outputs/logs/milestone-4-dev-log.md`.

### CHANGELOG
- [ ] Append `## [0.10.0-M4] — Unreleased` under M3's entry.

### Open questions to resolve
- [ ] Q10 (Anthropic provider detection) — confirm decision: generic
      warning, no detection.

## §9 Acceptance criteria

1. `AgentConfig(function_kind="json", json_mode="tool")` combined
   with a non-empty tool provider raises `ConfigurationError`
   when `Agent.run(...)` is called.
2. `Agent.stream(...)` on a JSON-flavoured agent raises
   `ConfigurationError` before any event is yielded.
3. `Agent.stream(...)` on a chat agent is byte-identical to v0.9.0.
4. `Agent.__init__` with `json_mode in {"on","tool"}` logs exactly
   one warning per instance (verified via caplog).
5. `require_parsed()` aborts when `result.response_kind == "json"`
   and `result.parsed is None`.
6. `require_parsed()` is a no-op for chat runs.
7. The chat path remains byte-identical to v0.9.0.
8. `mcp/` source is byte-identical to v0.9.0.
9. Every existing test still passes; new tests pass.
10. `docs/structured-outputs.md` has the "Safety & provider quirks"
    section.
11. `docs/guardrails.md` lists `require_parsed()` in the built-ins.
12. `CHANGELOG.md` has the `## [0.10.0-M4] — Unreleased` entry.
13. `logs/milestone-4-dev-log.md` exists.

## §10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| The tool-mode check runs on every `_run_json` call until the first verification, which adds a `list_tools` round-trip | Cache the result on first call (`self._needs_tool_mode_check = False` after the check). |
| `list_tools()` on a real FastMCP `Client` requires the Client be entered (`async with mcp_client:`); the agent does not own the lifecycle | Document that the check happens lazily during the first `run`.  If the Client is not entered, the existing `_run_json` code path already fails with a clear FastMCP error; we just need to ensure our check runs **after** the implicit context check. |
| Anthropic warning is noisy in environments where the user knows the routing is fine | Once-per-instance; the user can also silence the t0_agents logger. |
| `require_parsed()`'s `after_run` could interact badly with other guardrails' `after_run` ordering | Existing guardrail loop fires them in registration order; document the interaction. |
| The streaming guard might be too strict — what if a future TZ version supports `parsed` mid-stream? | Document the guard as "current TZ semantics".  When TZ changes, this milestone's M4 work is revisited via a follow-up. |

## §11 Estimated surface

- New files: 0 (everything is additions to existing modules).
- Modified files: 3 (`core/agent.py`, `guardrails/builtins.py`,
  `guardrails/__init__.py`).
- New tests: ~10.
- Modified docs: 2 (`structured-outputs.md`, `guardrails.md`).
- Lines of code added: ~80–120 across the modified files.
