# Structured Outputs — Integration Contract

> **Status:** authoritative. Every milestone spec under `milestones/`
> must trace back to this document.  When the contract changes, the
> author of the change records a pointer in their milestone dev-log
> at `logs/milestone-<N>-dev-log.md`.

## §1 Purpose

Add first-class support for TensorZero **JSON functions** and
**structured outputs** to `t0-agents` while preserving every invariant
established by the agentic-feature roadmap (phases A–H).

The motivation is direct: TensorZero distinguishes `chat` vs `json`
function types at the gateway level — different inference shape
(`JsonInferenceResponse` vs `ChatInferenceResponse`), different config
contract (`json_mode` required for `json`, forbidden for `chat`),
different response payload (`output.raw` + `output.parsed` vs
`content` + `tool_calls`).  A `tensorzero-first` library cannot
plausibly omit half of TZ's functional surface.

In addition, two pragmatic gaps are filled (deliberate, narrow value-add
layer documented in §3.4): in-Python permissive JSON parsing and a
schema-validation-aware retry loop, because real-world LLMs emit prose-
wrapped or truncated JSON even under `json_mode = "strict"`.

## §2 Invariants

The following invariants from the agentic-feature contract continue
unchanged for this work.  Every milestone must respect them.

### §2.1 `mcp/` stays standalone (frozen)

The `src/t0_agents/mcp/` submodule **does not** gain JSON-mode awareness.
A developer who `pip install t0-agents` and uses
`from t0_agents.mcp import run_agent` must observe byte-identical
behaviour to v0.9.0.  No new imports, no signature changes, no new
optional kwargs to `run_agent`, `convert_tools`,
`serialize_tool_result`, `build_tool_result`, `execute_tool_calls`, or
`has_tool_calls`.

> Source: memory `feedback_mcp_standalone`.

### §2.2 TensorZero-first

Every JSON-mode feature is a wrapper over an existing TZ primitive
(`output_schema`, `json_mode`, `output.parsed`, `output.raw`,
`response_format`, `tensorzero::params.chat_completion.json_mode`).
We do not:

- Re-implement JSON Schema validation.
- Build a parallel "structured output" abstraction.
- Replace TZ retries for transport-level failures (5xx, timeouts —
  those remain at the gateway-variant level).

We do (under the **narrow-gap exception** of memory
`feedback_tensorzero_first`):

- Add in-Python permissive JSON extraction *when* TZ returned
  `parsed = null`.  TZ does not attempt this — it is genuinely
  application-level recovery, not duplication.
- Add a loop that re-runs the inference with a corrective hint after
  invalid-JSON outputs.  TZ's variant-level retry handles transport
  errors, not "the model returned text but it was not valid JSON
  per the schema".

> Source: memory `feedback_tensorzero_first`.

### §2.3 Async-first

All new code is async, mypy-strict typed, ruff-compliant (rules
`E, F, I, N, UP, RUF`, line length 100), Python ≥ 3.11.

### §2.4 Byte-identical chat path

A `function_kind == "chat"` agent observes byte-identical behaviour
to v0.9.0.  Every existing test under `tests/test_mcp/`,
`tests/test_core/`, `tests/test_stream/`, `tests/test_serve/`,
`tests/test_compose/`, `tests/test_guardrails/`, `tests/test_memory/`,
`tests/test_cli/`, and `tests/test_testing/` must stay green without
modification, except when a test is intentionally updated to cover
new behaviour (and that update is documented in the dev-log of the
milestone that ships it).

### §2.5 Declarative function kind

Because TensorZero v2026.4.0's Python SDK does **not** expose a
`get_function_config(name)` API (memory:
`project_agentic_feature`, Phase B resolved-question matrix), runtime
introspection of whether a function is `chat` or `json` is not
possible.  The caller declares it via
`AgentConfig.function_kind: Literal["chat", "json"] = "chat"`.

The default is `"chat"` so every existing `AgentConfig(...)` keeps
working unchanged.

### §2.6 Elicitation-only HITL

JSON-mode features do not introduce a separate human-in-the-loop
mechanism.  Elicitation (FastMCP) remains the only HITL primitive.

> Source: memory `feedback_tensorzero_first` (Problem 6 in
> `problemas.md`).

## §3 Agent contract changes

### §3.1 `AgentConfig` new fields

Five new optional fields are added.  All have defaults that preserve
v0.9.0 behaviour when omitted.

| Field | Type | Default | Milestone | Semantics |
|-------|------|---------|-----------|-----------|
| `function_kind` | `Literal["chat", "json"]` | `"chat"` | M1 | Declares the TZ function type.  Drives dispatch in `Agent.run`. |
| `output_schema` | `Mapping[str, Any] \| Path \| None` | `None` | M1 | Optional static schema.  `Mapping` is sent inline; `Path` is read at construction time.  Forbidden when `function_kind == "chat"`. |
| `json_mode` | `Literal["off", "on", "strict", "tool"] \| None` | `None` | M1 | TZ json_mode override at the variant level.  **Required** when `function_kind == "json"`; **forbidden** when `function_kind == "chat"` (TZ guide §11.2). |
| `permissive_json_parsing` | `bool` | `True` | M3 | Enables in-Python recovery layer when TZ returns `parsed == None`.  Set to `False` for TZ-pure behaviour. |
| `max_json_retries` | `int` | `2` | M3 | Maximum extra inferences when the response still cannot be parsed.  `0` disables the retry layer.  Negative values raise `ConfigurationError`. |

`AgentConfig.__post_init__` enforces:

- `function_kind == "json"` requires `json_mode is not None`.
- `function_kind == "chat"` forbids `json_mode` and `output_schema`.
- `function_kind == "json"` with `sub_agents` is a `ConfigurationError`
  (loop incompatibility — see §3.6).
- `function_kind == "json"` with `json_mode == "tool"` and a
  `tool_provider` whose `list_tools()` is non-empty is a
  `ConfigurationError` raised in `Agent.__init__` (the config alone
  cannot check this — see §5.2; the actual enforcement lives in
  `Agent.__init__`).

### §3.2 `Agent.run(...)` signature additions

Three new kwargs, all defaulting to `None` to preserve byte-identity:

```python
async def run(
    self,
    user_message: str,
    *,
    episode_id: str | None = None,
    tags: dict[str, str] | None = None,
    system_arguments: dict[str, Any] | None = None,
    # NEW (M2 / M3):
    output_schema: Mapping[str, Any] | None = None,
    json_mode: Literal["off", "on", "strict", "tool"] | None = None,
    max_json_retries: int | None = None,
) -> AgentResult: ...
```

Per-call overrides take precedence over `AgentConfig` fields.
`None` means "use config".  Setting any of these on a
`function_kind == "chat"` agent raises `ConfigurationError`.

### §3.3 `Agent.stream(...)` behaviour

When `function_kind == "json"`, `Agent.stream(...)` raises
`ConfigurationError` with an actionable message pointing to
`Agent.run(...)`.  Rationale: TZ streams emit `raw` deltas but never
`parsed` (guide §7.1, §11.3) — surfacing this as
`AsyncIterator[AgentEvent]` would yield events whose `parsed` field is
permanently `None`, which is a worse contract than refusing to
stream.

The chat path of `Agent.stream` remains byte-identical to v0.9.0.

### §3.4 `AgentResult` new fields

Four new fields, all with defaults that produce byte-identical output
for chat callers:

| Field | Type | Default | Milestone | Semantics |
|-------|------|---------|-----------|-----------|
| `function_kind` | `Literal["chat", "json"]` | `"chat"` | M1 | `"json"` only when the underlying function is `json`.  Echoes `AgentConfig.function_kind` for downstream consumption. |
| `parsed` | `Any \| None` | `None` | M1 | Populated when `function_kind == "json"` and parse succeeded (via TZ, permissive, or retry). |
| `parse_strategy` | `Literal["strict", "permissive", "retry", "none"]` | `"strict"` | M3 | Reports which layer produced `parsed`.  `"strict"`: TZ parsed directly.  `"permissive"`: in-Python recovery succeeded on the first response.  `"retry"`: a re-emitted inference produced parseable output.  `"none"`: every layer failed; `parsed is None`. |
| `json_parse_attempts` | `int` | `0` | M3 | Count of extra inferences emitted after the first one (so `0` means "first response parsed" — via TZ or permissive layer).  Bounded by `max_json_retries`. |

> **Amendment (M1, 2026-05-14):** the M1 field is named `function_kind`
> rather than the planning-time `response_kind`.  `AgentResult` already
> carries `response_source: Literal["text", "thought"]` (which content
> block produced the response text); a near-duplicate `response_kind`
> field would have been confusing.  `function_kind` echoes
> `AgentConfig.function_kind` the same way `episode_id` echoes the
> TZ-assigned id.  No semantic change.

Existing `AgentResult` fields (`response`, `episode_id`,
`inference_ids`, `turns`, `guardrail_log`) are unchanged.  For
`function_kind == "json"`:

- `response` = `output.raw` (the JSON string returned by the model,
  or the last raw string if retries exhausted).
- `inference_ids` includes every inference id produced during retries,
  in order.
- `turns` = `1 + json_parse_attempts` (one initial inference plus
  retries).
- `guardrail_log` reflects `before_run` / `after_run` activations
  only; tool-level hooks never fire because the JSON path has no
  tool loop.
- `response_source` stays `"text"` for JSON runs — the JSON path has
  no `Thought` content blocks, so the field is degenerate but kept
  for byte-identity with chat callers.

### §3.5 Memory contract for JSON runs

`MemoryProvider.save()` is called with a `MemoryRecord` whose
`agent_response` field contains `output.raw` (the raw JSON string).
This is consistent with v0.9.0 behaviour for chat runs (where
`agent_response` is the joined terminating text).  The choice
guarantees that JSON memory entries are forward-compatible with the
existing `prepended_message` and `system_template` rendering paths.

`MemoryProvider.load()` and `load_history()` are unchanged.  For an
agent whose prior turns ran on a JSON function, the loaded record's
`agent_response` is treated as opaque text — exactly as it would be
for any other model output.

### §3.6 Sub-agent composition for JSON runs

`AgentConfig(function_kind="json", sub_agents=[...])` raises
`ConfigurationError`.  Rationale: sub-agents register themselves as
MCP tools on the orchestrator's FastMCP server, and a JSON function
with `json_mode = "tool"` cannot have free tools (guide §11.1); with
`json_mode = "strict"`/`"on"`/`"off"` the loop is single-shot
(no tools), so sub-agents are unreachable.

To compose a chat orchestrator that produces structured output, the
caller chains two agents manually (chat → json).  Sugar for this
pattern was scoped out of this roadmap (deferred to a follow-up).

## §4 Module additions

### §4.1 `src/t0_agents/core/structured.py` (new, M1)

Top-level new module under `core/`.  Exports:

- `run_structured_inference(t0, function_name, user_message, *, ...) -> StructuredOutput` —
  single-shot JSON inference.  Sibling of `mcp/loop.run_agent`.
- `StructuredOutput` (frozen dataclass) — `raw: str`, `parsed: Any | None`,
  `episode_id: str`, `inference_id: str`, `usage: Usage | None`,
  `parse_strategy: Literal["strict","permissive","retry","none"]`,
  `retry_attempts: int`.
- `permissive_parse(raw: str) -> tuple[Any | None, Literal["strict","permissive","none"]]`
  (M3) — pure-Python multi-strategy JSON extractor.

The module **must not** import from `mcp/`, `memory/`, `guardrails/`,
`serve/`, `stream/`, `compose/`, `testing/`, or `cli/` — it lives
alongside `core/agent.py`, `core/config.py`, `core/types.py` with the
same dependency hygiene.

### §4.2 `src/t0_agents/core/types.py` (modified, M1 + M3)

`AgentResult` gains the four new fields described in §3.4.  All
defaults preserve byte-identity for chat callers.

### §4.3 `src/t0_agents/core/config.py` (modified, M1 + M3)

`AgentConfig` gains the five new fields described in §3.1, plus the
five new validation cases in `__post_init__`.

### §4.4 `src/t0_agents/core/agent.py` (modified, M2 + M3 + M4)

`Agent.run` and `Agent.stream` gain the dispatch logic described in
§3.2, §3.3.  The retry orchestration (M3) lives here, not in
`run_structured_inference` — the inference helper is single-shot;
the loop above it is what knows the retry budget and the hint
message.

### §4.5 `src/t0_agents/guardrails/builtins.py` (modified, M4)

New built-in `require_parsed()` guardrail factory.  Aborts a JSON run
with `GuardrailBlockedError` when `result.parsed is None` after all
retries.  Opt-in; not added to the default policy set.

### §4.6 `src/t0_agents/testing/` (modified, M5)

`MockGateway` learns to produce duck-typed `JsonInferenceResponse`
shapes (`output.raw`, `output.parsed`).  New assertion helpers added
under `t0_agents.testing.asserts`.

### §4.7 Public re-exports (`__init__.py`, M1 + M3)

`from t0_agents import StructuredOutput` (M1).
No new top-level symbols for the retry layer — its observability is
surfaced through `AgentResult` fields, not new classes.

## §5 Error model

### §5.1 Configuration errors

Raised at `AgentConfig.__post_init__` or `Agent.__init__`:

- `function_kind == "json"` with `json_mode is None`.
- `function_kind == "chat"` with `output_schema` or `json_mode` set.
- `function_kind == "json"` with non-empty `sub_agents`.
- `function_kind == "json"` with `json_mode == "tool"` and a
  `tool_provider` advertising tools.
- `function_kind == "chat"` with any per-call structured-output kwarg
  set on `Agent.run`.
- `max_json_retries < 0`.

### §5.2 Runtime non-errors (`parsed is None`)

`output.parsed is None` after exhausting permissive parsing and
retries is **not** an exception.  The caller inspects
`result.parsed` and `result.parse_strategy == "none"`.  If a hard
abort is desired, the caller adds the `require_parsed()` guardrail
(M4).

### §5.3 Stream + JSON

`Agent.stream(...)` with `function_kind == "json"` raises
`ConfigurationError` at the start of the async generator, before
yielding any event.

### §5.4 Transport-level errors

Provider 5xx, timeouts, and other transport-level failures continue
to propagate through TensorZero's variant-level retry machinery,
unchanged.  The JSON retry loop is **only** for invalid-JSON outputs;
it does not trigger on `TensorZeroError`.

## §6 Backwards-compatibility guarantee

A caller running v0.9.0 code against this roadmap's final release
observes byte-identical behaviour without changes, provided their
`AgentConfig` keeps `function_kind` at its default of `"chat"`.
This includes:

- `Agent.run`, `Agent.stream` return values.
- All `AgentEvent` types and shapes.
- All `AgentResult` field values for chat runs (new fields default to
  their byte-identity defaults).
- All memory, guardrail, sub-agent, serve, CLI, and testing surfaces.

The `mcp/` submodule is bit-for-bit identical (no source changes).

## §7 Open items at contract time

These questions are recorded in `02-open-questions.md` and resolved
during the milestones that need them.  None of them gates the
contract above.

- **Q1.** Does `tensorzero.inference()` return distinct Python classes
  for `chat` vs `json` functions, or a single union type?  Verify
  against `.venv/.../tensorzero/tensorzero.pyi`.
- **Q2.** Does TensorZero run JSON-Schema validation on top of
  `JSON.parse`, or only the parse step?  Affects what "valid JSON"
  means for `parsed`.  Guide §11.15 flags this as undocumented.
- **Q3.** How is `output_schema: Path` resolved — relative to cwd, to
  the tensorzero.toml location, to the calling module?  Choose one
  and document.
- **Q4.** Retry hint phrasing — short and corrective vs. verbose with
  schema text.  Decide during M3.
- **Q5.** Should `permissive_parse` accept a list output (JSON Schema
  top-level array)?  Default: yes, balance both `{...}` and `[...]`.
- **Q6.** Should the retry loop bound by wall-clock time in addition
  to attempt count?  Default: attempt count only; revisit if it
  surfaces during M3 implementation.

Resolutions land in `02-open-questions.md` with `Status: resolved`
and a citation (file:line) when source-validated.
