# Milestone 2 — `Agent.run` JSON Dispatch (Non-Streaming)

> Wires the `Agent` class to use the M1 helper when the config declares
> a JSON function.  Adds per-call schema/mode overrides.  Defines the
> memory and guardrail behaviour for the JSON path.  No retry or
> permissive parsing yet — those land in M3.

## §1 Goal

Make `Agent.run(...)` work transparently for both chat and JSON
functions.  After this milestone a developer can:

```python
config = AgentConfig(
    name="customer-extractor",
    function_name="extract_customer_info",
    function_kind="json",
    json_mode="strict",
)
agent = Agent(config, t0=gateway, tool_provider=mcp_client)

result = await agent.run("Hi, I'm Sarah at sarah@example.com")
assert result.response_kind == "json"
assert result.parsed == {"name": "Sarah", "email": "sarah@example.com"}
assert result.response == '{"name":"Sarah","email":"sarah@example.com"}'
```

And override per call:

```python
result = await agent.run(
    "...",
    output_schema={"type": "object", "properties": {"email": {"type": "string"}}, ...},
    json_mode="tool",  # override config's "strict"
)
```

## §2 Scope

### In scope

- `Agent.run(...)` dispatches to `run_structured_inference()` when
  `config.function_kind == "json"`.
- New per-call kwargs on `Agent.run`: `output_schema`, `json_mode`.
- `AgentResult` for a JSON run is populated per contract §3.4:
  `response = output.raw`, `parsed = output.parsed`,
  `response_kind = "json"`, `inference_ids = [first_inference_id]`,
  `turns = 1`.
- Memory persistence for JSON runs (`agent_response = output.raw`).
- Run-level guardrails (`before_run`, `after_run`) fire normally for
  the JSON path.
- Configuration-error surface: `output_schema` / `json_mode` per-call
  kwargs raise `ConfigurationError` when `function_kind == "chat"`.
- Documentation: complete the "Structured Agent" section of
  `docs/structured-outputs.md`.
- Example: `examples/16_structured/structured_agent.py`.

### Out of scope

- Permissive parsing layer (M3).
- Retry loop (M3).
- `Agent.stream(...)` JSON-path behaviour (M4 — `ConfigurationError`).
- `MockGateway` JSON shapes (M5).
- New assertion helpers (M5).

### Non-goals

- Tool-level guardrail hooks for JSON runs.  By contract (§3.4), a
  JSON function does not run the tool loop, so `before_tool` /
  `after_tool` never fire.  Documented; not "fixed".
- Sub-agent composition for JSON functions (forbidden by §3.6;
  already errors at config construction in M1).
- Output-schema validation in t0-agents (Q12, decided out of scope).

## §3 Constraints

- `mcp/` is byte-identical to v0.9.0.
- Chat-path callers of `Agent.run` see byte-identical behaviour: the
  dispatch check is the first thing the method does, and the chat
  branch is unchanged.
- New kwargs default to `None` and are only consulted when
  `function_kind == "json"`.
- Memory backends (`InMemoryMemory`, `SqliteMemory`, `TensorZeroMemory`)
  receive the same `MemoryRecord` shape they receive today — only the
  `agent_response` content differs (now a JSON string for JSON runs).
- mypy strict, ruff clean, line length 100.

## §4 Module additions

### §4.1 `src/t0_agents/core/agent.py` (modified)

`Agent.run(...)` gains three optional kwargs and a dispatch branch.
The full new signature:

```python
async def run(
    self,
    user_message: str,
    *,
    episode_id: str | None = None,
    tags: dict[str, str] | None = None,
    system_arguments: dict[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    json_mode: Literal["off", "on", "strict", "tool"] | None = None,
) -> AgentResult: ...
```

The dispatch:

```python
async def run(self, user_message, *, ..., output_schema=None, json_mode=None):
    if self._config.function_kind == "json":
        return await self._run_json(
            user_message,
            episode_id=episode_id,
            tags=tags,
            system_arguments=system_arguments,
            output_schema=output_schema,
            json_mode=json_mode,
        )
    # Existing chat path is byte-identical to v0.9.0.
    if output_schema is not None or json_mode is not None:
        raise ConfigurationError(
            "output_schema/json_mode kwargs are only valid when "
            "AgentConfig.function_kind == 'json'"
        )
    return await self._run_chat(user_message, ...)  # existing body
```

(The existing body of `Agent.run` is renamed to `_run_chat` or kept
inline — implementer's choice based on readability.  Whichever path
is taken, the dev-log records it.)

`_run_json` does:

1. Load memory if `config.memory` and `episode_id` are set — same
   contract as chat (`prepended_message` strategy applies to the
   user message; `system_template` strategy threads through
   `system_arguments`).
2. Fire `before_run` guardrails (parity with `_run_chat`).
3. Resolve effective `output_schema` and `json_mode`:
   - Per-call override > config.
   - Static `config.output_schema` (Mapping or pre-loaded Path) is the
     fallback.
4. Call `run_structured_inference(...)` with the resolved args.
5. Build an `AgentResult`:
   ```python
   AgentResult(
       response=structured.raw,
       parsed=structured.parsed,
       response_kind="json",
       episode_id=structured.episode_id,
       inference_ids=[structured.inference_id],
       turns=1,
       guardrail_log=guard_log,
   )
   ```
6. Save memory if applicable.
7. Fire `after_run` guardrails (parity with chat path; `result` is the
   `AgentResult`).
8. Return.

The `on_inference` cost-tracker (used by `max_cost` guardrail in M3
of the agentic-feature roadmap) is hooked the same way as the chat
path so cumulative cost works for JSON runs too.

### §4.2 `src/t0_agents/core/exceptions.py` (modified, minor)

No new exception classes.  `ConfigurationError` already exists and is
the right type for the per-call kwarg validation.

### §4.3 No changes elsewhere

- `mcp/` untouched.
- `stream/` untouched (M4 adds the stream guard).
- `memory/`, `guardrails/`, `compose/`, `serve/`, `cli/`, `testing/`
  untouched.

## §5 Tests

### §5.1 Unit tests (`tests/test_core/test_agent_json_run.py`)

- `test_run_dispatches_to_json_when_function_kind_is_json`: a
  `MockGateway` script returning a `JsonInferenceResponse`-shaped
  response.  Assert `result.response_kind == "json"`,
  `result.parsed == {...}`, `result.response == raw_string`,
  `result.turns == 1`.
- `test_chat_path_is_byte_identical`: build an agent with the same
  `MockGateway` setup as an existing chat test in
  `tests/test_core/test_agent.py`.  Assert the new fields default to
  `parsed=None`, `response_kind="text"` and every other field is
  byte-identical to v0.9.0.
- `test_per_call_output_schema_override`: `MockGateway` captures
  kwargs; assert the per-call `output_schema=` reaches the gateway
  (overriding the config).
- `test_per_call_json_mode_override`: same as above for `json_mode`.
- `test_per_call_kwargs_forbidden_on_chat_agent`: build a
  chat-flavoured agent and call `agent.run("...",
  output_schema={...})` — expect `ConfigurationError`.
- `test_before_run_after_run_fire_on_json`: register two custom
  guardrails (one `before_run`, one `after_run`); both are called and
  `result.guardrail_log` has two `pass` entries.
- `test_before_tool_after_tool_do_not_fire_on_json`: register a
  custom guardrail with `before_tool` / `after_tool`; assert neither
  is called.
- `test_memory_save_persists_raw_json`: `InMemoryMemory` is set up;
  after a JSON run, assert the saved record has
  `agent_response == structured.raw` and `inference_ids ==
  [first_inference_id]`.
- `test_episode_id_propagation`: pass `episode_id="abc"`; assert the
  gateway received it and the result echoes it back.

### §5.2 Integration test
(`tests/test_core/test_agent_json_run_integration.py`)

- Marked `@pytest.mark.integration`.
- Build a real `AsyncTensorZeroGateway`.  Build an agent with
  `function_kind="json"`, `function_name="extract_email"`,
  `json_mode="strict"`.  Call `agent.run("...hello@example.com...")`.
- Assert `result.parsed == {"email": "hello@example.com"}`.

## §6 Example

`examples/16_structured/structured_agent.py`:

- Builds an `AgentConfig(function_kind="json", json_mode="strict",
  function_name="extract_customer_info")`.
- Calls `agent.run("...")` twice in a row, varying the input.
- Prints `result.parsed` for each.
- README in the same folder explains:
  - The config field `function_kind="json"`.
  - The required `json_mode` field.
  - Memory continuity (each call gets its own JSON parse).
  - Provider matrix (point at `docs/structured-outputs.md`).

## §7 Documentation

### §7.1 `docs/structured-outputs.md` (extended)

Add the "Structured `Agent`" section:

- Quickstart with `AgentConfig(function_kind="json", ...)`.
- Per-call override examples.
- Memory + guardrails behaviour table for JSON runs.
- Diagram: how `Agent.run` decides to dispatch
  (chat → `run_agent`, json → `run_structured_inference`).

### §7.2 `docs/agent.md` (modified)

Expand the v0.3.0 / v0.4.0 / etc. feature roll-call to include
"v0.10.0: Structured outputs and JSON functions".  Cross-link to
`docs/structured-outputs.md`.

## §8 Checklist

### Pre-flight
- [ ] Read every prior dev-log under
      `docs/internal/structured-outputs/logs/` (only `milestone-1`
      should exist).
- [ ] Re-read `00-integration-contract.md` §3.2, §3.5 — Agent contract
      changes and memory contract.

### Code
- [ ] Modify `core/agent.py`: add new kwargs, dispatch branch,
      `_run_json` helper.
- [ ] Validate per-call kwargs are rejected on chat agents.
- [ ] Re-run all existing chat tests to confirm byte-identity.
- [ ] Verify `mcp/` source files are byte-identical to v0.9.0.

### Tests
- [ ] `tests/test_core/test_agent_json_run.py` — 9 unit tests above.
- [ ] `tests/test_core/test_agent_json_run_integration.py` — 1
      integration test.
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `uv run pytest -m integration` green (with docker up).

### Example
- [ ] `examples/16_structured/structured_agent.py` runnable.
- [ ] README updated.

### Docs
- [ ] `docs/structured-outputs.md` gains "Structured `Agent`" section.
- [ ] `docs/agent.md` gains feature roll-call entry + cross-link.

### Style gates
- [ ] `uv run ruff check src/`.
- [ ] `uv run mypy src/`.

### Implementation log
- [ ] Write `docs/internal/structured-outputs/logs/milestone-2-dev-log.md`.

### CHANGELOG
- [ ] Append `## [0.10.0-M2] — Unreleased` under M1's entry in
      `CHANGELOG.md`.

## §9 Acceptance criteria

1. `Agent.run(...)` with a JSON-flavoured config returns a populated
   `AgentResult` with `response_kind="json"` and a non-None `parsed`
   (when the model output is valid JSON).
2. Per-call `output_schema=` and `json_mode=` kwargs override the
   config.
3. Setting per-call kwargs on a chat agent raises
   `ConfigurationError` with an actionable message.
4. Every chat-path test in the repo passes unchanged.
5. `mcp/` source is byte-identical to v0.9.0.
6. The integration test in `tests/test_core/test_agent_json_run_integration.py`
   passes with docker compose up.
7. `examples/16_structured/structured_agent.py` runs successfully.
8. `docs/structured-outputs.md` has the "Structured `Agent`" section.
9. `CHANGELOG.md` has the `## [0.10.0-M2] — Unreleased` entry.
10. `docs/internal/structured-outputs/logs/milestone-2-dev-log.md`
    exists with the five mandatory sections populated.

## §10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `_run_json` duplicates memory/guardrail scaffolding from `_run_chat`, drifting over time | Factor common scaffolding (memory load, guardrail before/after_run, memory save) into a private helper or context manager.  The dev-log records the refactoring decision. |
| `output_schema` mapping must be JSON-serialisable; arbitrary `Mapping` values may break | Validate at call time that the schema can be JSON-dumped; surface a clear error. |
| The shape-contract test (`tests/test_testing/test_mock_shape.py`) hard-codes `ChatInferenceResponse`-only kwargs; introducing `output_schema` could break it | Update the contract test in M5 (the testing milestone); flag the broken assertion in the M2 dev-log §4 (carry-forward). |
| Memory `save()` may surprise `TensorZeroMemory` (the no-op writer) — but TZ already records the inference, so this should be transparent | Verify with the integration test against `TensorZeroMemory`; document. |

## §11 Estimated surface

- New files: 1 (`examples/16_structured/structured_agent.py`).
- Modified files: 2 (`core/agent.py`, possibly small edits to
  `core/exceptions.py`).
- Modified docs: 2 (`structured-outputs.md`, `agent.md`).
- New tests: ~10 (unit + integration).
- Lines of code added: ~120–180 in `core/agent.py`.
