# Milestone 1 — JSON-Inference Foundations

> Builds the substrate every later milestone depends on: a single-shot
> structured-inference helper, the new types, and the M1 fields on
> `AgentConfig` and `AgentResult`.  No changes to `Agent.run` /
> `Agent.stream` behaviour yet — those land in M2.

## §1 Goal

Introduce the building blocks required to invoke a TensorZero JSON
function from `t0-agents` without yet wiring them into the `Agent`
class.  After this milestone, a developer can:

```python
from t0_agents import StructuredOutput
from t0_agents.core.structured import run_structured_inference

result: StructuredOutput = await run_structured_inference(
    t0=gateway,
    function_name="extract_customer_info",
    user_message="Hi, I'm Sarah Johnson at sarah.j@example.com",
    json_mode="strict",
)
print(result.parsed)  # {"name": "Sarah Johnson", "email": "sarah.j@example.com"}
```

And configure (but not yet use, until M2) a JSON-flavoured agent:

```python
config = AgentConfig(
    name="extractor",
    function_name="extract_customer_info",
    function_kind="json",
    json_mode="strict",
)
```

## §2 Scope

### In scope

- New module `src/t0_agents/core/structured.py` with
  `run_structured_inference()` and the `StructuredOutput` dataclass.
- New `AgentConfig` fields: `function_kind`, `output_schema`,
  `json_mode`.  Validation in `__post_init__`.
- New `AgentResult` fields: `response_kind`, `parsed`.
- Public re-export: `from t0_agents import StructuredOutput`.
- One single-shot example at `examples/16_structured/extract_customer_info.py`.
- Stub `docs/structured-outputs.md` with the "Single-shot JSON inference"
  section and the `json_mode` provider matrix.
- One-paragraph "Chat vs JSON functions" primer added to `docs/concepts.md`.
- Tests for the helper (unit + integration) and the new config fields.

### Out of scope

- Changes to `Agent.run` or `Agent.stream` (M2).
- Permissive parsing or retry layer (M3).
- Provider safety checks, `Agent.stream` guard, `require_parsed` (M4).
- `MockGateway` JSON support and `assert_parsed_*` helpers (M5).
- Synthesis sugar (`Agent.run_with_synthesis`) — deferred to a follow-up.

### Non-goals

- Modifying `mcp/` (contract §2.1 — frozen).
- Adding a Pydantic dependency or `pydantic_output_model` config field
  (Q7, decided out of scope).
- Building schema-validation in Python (Q12, decided out of scope).

## §3 Constraints

- `core/structured.py` **must not** import from `mcp/`, `memory/`,
  `guardrails/`, `serve/`, `stream/`, `compose/`, `testing/`, or `cli/`.
- New `AgentConfig` fields must default to values that leave existing
  v0.9.0 callers byte-identical.  `function_kind: Literal["chat",
  "json"] = "chat"`, `output_schema = None`, `json_mode = None`.
- New `AgentResult` fields must default to byte-identity values for
  chat callers: `response_kind = "text"`, `parsed = None`.
- mypy strict, ruff clean (`E, F, I, N, UP, RUF`), line length 100,
  Python ≥ 3.11.
- TensorZero SDK version: v2026.4.0 (whatever is in `pyproject.toml`).
  Do not bump.

## §4 Module additions

### §4.1 `src/t0_agents/core/structured.py` (new)

```python
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import JsonInferenceResponse  # subject to Q1 verification

logger = logging.getLogger(__name__)

ParseStrategy = Literal["strict", "permissive", "retry", "none"]


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    """Result of a single TensorZero JSON-function inference."""

    raw: str
    parsed: Any | None
    episode_id: str
    inference_id: str
    usage: Any | None  # TZ Usage type — TBD per Q1
    parse_strategy: ParseStrategy = "strict"
    retry_attempts: int = 0  # always 0 in M1; populated in M3


async def run_structured_inference(
    t0: AsyncTensorZeroGateway,
    function_name: str,
    user_message: str,
    *,
    output_schema: Mapping[str, Any] | None = None,
    json_mode: Literal["off", "on", "strict", "tool"] | None = None,
    system_arguments: dict[str, Any] | None = None,
    episode_id: str | None = None,
    tags: dict[str, str] | None = None,
    namespace: str | None = None,
    dryrun: bool = False,
    on_inference: Callable[[JsonInferenceResponse], Awaitable[None]] | None = None,
) -> StructuredOutput:
    """Single-shot JSON function inference.

    Sibling of :func:`t0_agents.mcp.loop.run_agent`.  Does NOT loop —
    JSON functions are single-shot by contract.

    Parameters mirror :func:`run_agent`'s where they make sense; the
    JSON-specific ones (``output_schema``, ``json_mode``) override the
    function's static configuration on a per-call basis (TZ guide §4).
    """
    ...
```

Implementation notes:

- The signature **deliberately excludes** `parallel_tools`, `tool_choice`,
  `tool_timeout`, `prior_messages`, `max_turns` — none of them are
  meaningful for a single-shot JSON inference.
- The call into TensorZero passes `output_schema` (mapping) and
  `params.chat_completion.json_mode` per the guide §5.5.
- `on_inference` parity: same callback shape as `run_agent`'s, used
  later by the cost guardrail.  In M1 it is purely a hook.

### §4.2 `src/t0_agents/core/types.py` (modified)

`AgentResult` gains two new fields with byte-identity defaults:

```python
@dataclass
class AgentResult:
    # ... existing fields unchanged ...
    response_kind: Literal["text", "json"] = "text"
    parsed: Any | None = None
```

### §4.3 `src/t0_agents/core/config.py` (modified)

`AgentConfig` gains three new fields:

```python
@dataclass
class AgentConfig:
    # ... existing fields unchanged ...
    function_kind: Literal["chat", "json"] = "chat"
    output_schema: Mapping[str, Any] | Path | None = None
    json_mode: Literal["off", "on", "strict", "tool"] | None = None
```

`__post_init__` enforces:

- `function_kind == "json"` ⇒ `json_mode is not None`.
- `function_kind == "chat"` ⇒ `json_mode is None` and
  `output_schema is None`.
- `function_kind == "json"` with non-empty `sub_agents` ⇒
  `ConfigurationError`.

`output_schema: Path` is resolved at construction time:

- Absolute paths are loaded verbatim.
- Relative paths are resolved against `cwd` (per Q3 resolution).
- File is read as JSON; result becomes a `Mapping[str, Any]` stored
  in the config.

### §4.4 `src/t0_agents/__init__.py` (modified)

```python
from t0_agents.core.structured import StructuredOutput

__all__ = [..., "StructuredOutput", ...]
```

## §5 Tests

### §5.1 Unit tests (`tests/test_core/test_structured.py`)

- `test_run_structured_inference_returns_parsed`: `MockGateway` returns
  a `JsonInferenceResponse`-shaped object with `output.raw =
  '{"a": 1}'` and `output.parsed = {"a": 1}`.  Assert
  `result.parsed == {"a": 1}`, `result.raw == '{"a": 1}'`,
  `result.parse_strategy == "strict"`.
- `test_run_structured_inference_returns_none_when_invalid`:
  `MockGateway` returns `output.raw = "not json"`, `output.parsed =
  None`.  Assert `result.parsed is None`, `result.parse_strategy ==
  "none"`.  (M1 does NOT run permissive parsing — that lands in M3.
  This test pins the M1 behaviour as "trust TZ's parse only".)
- `test_run_structured_inference_threads_kwargs`: capture the
  kwargs passed to `gateway.inference()` and assert
  `output_schema`, `json_mode`, `system_arguments`, `tags`,
  `namespace`, `episode_id`, `dryrun` all reach the gateway in the
  shape required by the SDK.
- `test_on_inference_callback_fires`: a counter callback is incremented
  exactly once per call.

### §5.2 Unit tests (`tests/test_core/test_config_structured.py`)

- `test_function_kind_json_requires_json_mode`: building
  `AgentConfig(function_kind="json", json_mode=None)` raises
  `ConfigurationError`.
- `test_function_kind_chat_forbids_json_mode`: building
  `AgentConfig(function_kind="chat", json_mode="strict")` raises
  `ConfigurationError`.
- `test_function_kind_chat_forbids_output_schema`: building
  `AgentConfig(function_kind="chat", output_schema={})` raises
  `ConfigurationError`.
- `test_function_kind_json_forbids_sub_agents`: building
  `AgentConfig(function_kind="json", json_mode="strict",
  sub_agents=[some_agent])` raises `ConfigurationError`.
- `test_output_schema_loads_from_path`: a temp file containing a JSON
  schema is loaded and stored as a `Mapping`.
- `test_default_function_kind_is_chat`: omitting `function_kind`
  yields the v0.9.0 behaviour byte-identically.

### §5.3 Integration test (marked `@pytest.mark.integration`)

`tests/test_core/test_structured_integration.py`: runs against
`docker compose up -d` with a real TensorZero gateway and a function
configured as:

```toml
[functions.extract_email]
type = "json"
output_schema = "extract_email_schema.json"

[functions.extract_email.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
system_template = "extract_email_system.minijinja"
json_mode = "strict"
```

Test: call `run_structured_inference(function_name="extract_email",
user_message="Hi, contact me at hello@example.com")`.  Assert
`result.parsed == {"email": "hello@example.com"}`.

A second integration test exercises a dynamic schema override (override
adds a `domain` field) per guide §4.

## §6 Example

`examples/16_structured/extract_customer_info.py`:

- Tiny FastMCP-less script (no tools needed for a JSON function).
- Builds an `AsyncTensorZeroGateway`, calls
  `run_structured_inference(...)`, prints `result.parsed`.
- README explains the configuration in `examples/16_structured/config/tensorzero.toml`
  with `extract_customer_info` set up as a `type = "json"` function with
  the schema from guide §3.2.
- `docker-compose.yml` provided so `docker compose up -d` + `python
  extract_customer_info.py` is a one-shot runnable.

## §7 Documentation

### §7.1 `docs/structured-outputs.md` (new, stub form)

Sections in M1:

- "What are JSON functions?" — one-paragraph primer.
- "Single-shot JSON inference" — shows `run_structured_inference`.
- "Configuring a JSON function" — points to TensorZero docs for the
  TOML side; explains the t0-agents side
  (`AgentConfig(function_kind="json", json_mode=...)`).
- "`json_mode` provider matrix" — copy of guide §13.1 table.
- "What's coming" — placeholder for M2 (Agent dispatch), M3
  (Robust parsing), M4 (Safety/quirks).

### §7.2 `docs/concepts.md` (modified)

Add a short "Chat vs JSON functions" paragraph alongside the existing
agent-loop primer.

### §7.3 `docs/agent.md` (modified)

Add a one-line note that `function_kind` exists; defer detail to M2's
update.

## §8 Checklist

### Code
- [ ] Create `src/t0_agents/core/structured.py` with
      `StructuredOutput`, `run_structured_inference`.
- [ ] Update `src/t0_agents/core/types.py`: add `response_kind`,
      `parsed` to `AgentResult`.
- [ ] Update `src/t0_agents/core/config.py`: add `function_kind`,
      `output_schema`, `json_mode`; add `__post_init__` validations.
- [ ] Update `src/t0_agents/__init__.py`: re-export `StructuredOutput`.
- [ ] Verify `mcp/` source files are byte-identical to v0.9.0.

### Tests
- [ ] `tests/test_core/test_structured.py` — 4 unit tests above.
- [ ] `tests/test_core/test_config_structured.py` — 6 unit tests above.
- [ ] `tests/test_core/test_structured_integration.py` — 2 integration
      tests (`@pytest.mark.integration`).
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `uv run pytest -m integration` green (with docker up).

### Example
- [ ] `examples/16_structured/` directory.
- [ ] `examples/16_structured/extract_customer_info.py` runnable.
- [ ] `examples/16_structured/config/tensorzero.toml`.
- [ ] `examples/16_structured/README.md`.
- [ ] `examples/16_structured/docker-compose.yml`.

### Docs
- [ ] `docs/structured-outputs.md` stub form.
- [ ] `docs/concepts.md` primer paragraph.
- [ ] `docs/agent.md` one-line forward-reference.

### Style gates
- [ ] `uv run ruff check src/`.
- [ ] `uv run mypy src/`.

### Implementation log
- [ ] Write `docs/internal/structured-outputs/logs/milestone-1-dev-log.md`
      with sections §1 Summary, §2 Deviations, §3 Open questions,
      §4 Carry-forward, §5 Verification.

### CHANGELOG
- [ ] Append `## [0.10.0-M1] — Unreleased` under v0.9.0 entry in
      `CHANGELOG.md`.

### Open questions to resolve
- [ ] Q1 (response type — `JsonInferenceResponse`).
- [ ] Q3 (path resolution policy).
- [ ] Q11 (`dryrun` parity).

## §9 Acceptance criteria

1. `from t0_agents import StructuredOutput` works without errors.
2. `from t0_agents.core.structured import run_structured_inference` works.
3. `examples/16_structured/extract_customer_info.py` runs to completion
   against `docker compose up -d` and prints
   `{"name": "...", "email": "..."}`.
4. Every test in `tests/test_core/test_structured.py` and
   `tests/test_core/test_config_structured.py` passes.
5. Every existing test in `tests/test_mcp/`, `tests/test_core/`,
   `tests/test_stream/`, `tests/test_compose/`, `tests/test_guardrails/`,
   `tests/test_memory/`, `tests/test_serve/`, `tests/test_cli/`,
   `tests/test_testing/` still passes — byte-identical chat behaviour.
6. `mcp/` source files are byte-identical to v0.9.0.
7. `docs/structured-outputs.md` exists with the M1 sections.
8. `CHANGELOG.md` has a new `## [0.10.0-M1] — Unreleased` entry.
9. `docs/internal/structured-outputs/logs/milestone-1-dev-log.md`
   exists with the five mandatory sections populated.

## §10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Q1 turns out wrong (the SDK uses a different class name or shape) | Implementer reads `.venv/.../tensorzero/tensorzero.pyi` first thing.  If the actual class is different, update the `StructuredOutput` source mapping and write the resolution in the dev-log. |
| `output_schema: Path` resolution is contentious (relative-to-what?) | Q3 picks cwd; document the recommendation to use absolute paths.  If the user pushes back, the dev-log records the alternative and M2 adjusts. |
| `dryrun` semantics surprise — TZ may not return a meaningful response in dryrun mode for a JSON function | Test in M1 with `dryrun=True` and assert behaviour matches TZ docs; if surprising, document explicitly. |
| `Mapping[str, Any]` vs `dict[str, Any]` typing friction | Use `Mapping` in signatures (read-only), `dict` in storage.  Source-validate against TZ SDK's expected types. |

## §11 Estimated surface

- New files: 2 (`core/structured.py`, `examples/16_structured/*`).
- Modified files: 3 (`core/config.py`, `core/types.py`,
  `t0_agents/__init__.py`).
- Modified docs: 3 (`structured-outputs.md` new, `concepts.md` +
  `agent.md` minor edits).
- New tests: ~12 (unit + integration).
- Lines of code added: ~250–350 in `core/structured.py` + ~80 in
  config validations + minor surface in `types.py`.
