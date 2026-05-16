# Milestone 3 — `OutputMode[T]` Polymorphism

> **Roadmap:** v1.0 four-abstraction refactor.
> **Deps:** M1 (`Run`), M2 (`Middleware` — specifically `JsonRetryMiddleware`).
> **Dev-log:** `docs/internal/v1-refactor/logs/milestone-3-dev-log.md` (always-write at close).

---

## §1 — Goal

**Remove** `AgentConfig.function_kind` and the 4 JSON-only kwargs (`json_mode`, `output_schema`, `permissive_json_parsing`, `max_json_retries`).  Replace them with a single canonical field: `AgentConfig.output: OutputMode = ChatOutput()`.

`AgentResult` becomes `AgentResult[T]` generic with `value: T`.  All v0.12 result-side fields (`response`, `parsed`, `parse_strategy`, `json_parse_attempts`, `function_kind`) remain accessible as `@property` aliases — those are read-side and don't need migration.

Five built-in output modes ship: `ChatOutput`, `JsonOutput`, `StructuredOutput[M]` (Pydantic), `ToolCallOutput` (router pattern), `StreamingOutput` (placeholder for future TZ streaming JSON support).

Third-party developers can plug in custom `OutputMode` implementations (YAML extraction, Instructor-style retry, SQLAlchemy row materialisation, etc.) without forking.

**Migration cost (Tier 2 — contract §2.4):** the 5 legacy kwargs are removed directly, no deprecation path.  M3 ships the migration of the canonical JSON example (`examples/17_structured/`) as part of the milestone deliverable.  Downstream PaaS callers migrate one line per `AgentConfig` construction.

---

## §2 — Scope

### In scope
- `core/output.py` — `OutputMode` protocol, types (`InferenceRequest`, `Attempt`, `ParseFailure`, `Recovery`).
- `core/outputs/{chat,json,structured,tool_call,streaming}.py` — 5 built-ins.
- `core/types.py` refactor: `AgentResult[T]` generic with `value: T` + `OutputDiagnostics`; properties for v0.12 fields (`response`, `parsed`, `parse_strategy`, `json_parse_attempts`, `function_kind`).
- `core/config.py` — **remove** the 5 legacy kwargs (`function_kind`, `json_mode`, `output_schema`, `permissive_json_parsing`, `max_json_retries`); **add** `output: OutputMode = ChatOutput()`.  `__post_init__` simplifies (no migration of legacy kwargs; validation delegated to `OutputMode`).
- `core/agent.py` — dispatch via `output.expects_tool_loop` (no more `if function_kind == "json"`).
- `core/structured.py` — migrate `permissive_parse`'s retry-hint logic into `JsonOutput.recover()`.  Keep `permissive_parse` public.  Keep `run_structured_inference` low-level helper unchanged.
- `core/middleware/builtins.py` — `JsonRetryMiddleware` activates (delegates to `output.recover()`).
- `__init__.py` — export `OutputMode`, 5 built-ins, `OutputDiagnostics`, helper types.
- **Migrate `examples/17_structured/`** to the new `output=JsonOutput(...)` shape (and a sibling `output=StructuredOutput(...)` example showcasing Pydantic).

### Out of scope
- Streaming JSON: `StreamingOutput(base=JsonOutput(...))` raises `ConfigurationError` "not yet supported" (Q11 deferral).
- Custom recovery policies as builtins (e.g., Instructor-style).  Third-party can plug in.
- Cross-mode interpolation (e.g., agent that switches from chat to JSON mid-run).

### Non-goals
- Compatibility break: `AgentConfig(function_kind="json", json_mode="strict", ...)` continues to work byte-identical.
- Forcing users to learn the new API in v1.0 — the migration guide (M5) lays out the path.

---

## §3 — Constraints

Inherited from `00-integration-contract.md`:
- §2.1 `mcp/` standalone — OutputMode lives entirely in `core/`.
- §2.2 TensorZero-first — `JsonOutput.shape_inference` sets `json_mode` and `output_schema` on the TZ inference call; never reimplements TZ knobs.
- §2.3 Chat byte-identity — chat-mode examples run byte-identical to v0.12 (modulo deprecation warnings from M1 shims).
- §2.4 Tier 0 read-side compat — `AgentResult.response`, `.parsed`, `.parse_strategy`, `.json_parse_attempts`, `.function_kind` remain readable via `@property`.
- §2.4 Tier 2 — `AgentConfig` write-side legacy kwargs (`function_kind`, `json_mode`, `output_schema`, `permissive_json_parsing`, `max_json_retries`) are **removed** in v1.0.  Constructing `AgentConfig(function_kind="json", ...)` raises `TypeError`.  The migration is documented and the canonical example is updated as part of this milestone.

Milestone-specific:
- `AgentResult[T]` generic; default `T = str` for backwards compat in untyped code.
- `OutputMode.expects_tool_loop` capability flag drives dispatch (no `isinstance` checks).
- `OutputMode.supports_streaming` capability flag — `Agent.stream()` raises early `ConfigurationError` if `False`.
- Recovery types: `Recovery.retry(hint)`, `Recovery.accept(value)`, `Recovery.fail()`.

---

## §4 — Module additions

### New: `src/t0_agents/core/output.py`

```python
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from tensorzero.types import ChatInferenceResponse, JsonInferenceResponse
    from t0_agents.core.types import AgentResult

T_co = TypeVar("T_co", covariant=True)

@dataclass(slots=True)
class InferenceRequest:
    function_name: str
    input: dict[str, Any]            # {"system": ..., "messages": [...]}
    additional_tools: list[Any] | None
    tool_choice: Any | None
    tags: dict[str, str] | None
    namespace: str | None
    dryrun: bool
    # JSON-specific kwargs (none for chat):
    output_schema: Mapping[str, Any] | None = None
    params: dict[str, Any] = None

@dataclass(slots=True)
class Attempt:
    inference_id: str
    raw_text: str | None
    parsed: Any | None
    usage: Any | None

@dataclass(slots=True)
class ParseFailure:
    raw_text: str
    inference_id: str
    reason: str               # "json.JSONDecodeError" / "ValidationError" / etc.
    detail: str

@dataclass(slots=True)
class Recovery:
    kind: Literal["retry", "accept", "fail"]
    hint: str | None = None   # corrective hint for retry
    value: Any = None         # for accept

    @classmethod
    def retry(cls, hint: str) -> "Recovery":
        return cls(kind="retry", hint=hint)
    @classmethod
    def accept(cls, value: Any) -> "Recovery":
        return cls(kind="accept", value=value)
    @classmethod
    def fail(cls) -> "Recovery":
        return cls(kind="fail")

ParseStrategy = Literal["strict", "permissive", "retry", "none"]

@dataclass(slots=True)
class OutputDiagnostics:
    parse_strategy: ParseStrategy = "strict"
    attempts: int = 0
    raw_text: str | None = None

@runtime_checkable
class OutputMode(Protocol[T_co]):
    expects_tool_loop: bool
    supports_streaming: bool

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest: ...
    async def materialize(self, response: Any, attempts: list[Attempt]) -> "AgentResult[T_co]": ...
    async def recover(self, failure: ParseFailure, attempts: list[Attempt]) -> Recovery: ...
```

### New: `src/t0_agents/core/outputs/chat.py`

```python
class ChatOutput:
    expects_tool_loop = True
    supports_streaming = True
    name = "chat"

    def shape_inference(self, req): return req     # no-op
    async def materialize(self, response, attempts):
        # join Text blocks; pull thoughts; build AgentResult[str]
        ...
    async def recover(self, failure, attempts):
        return Recovery.fail()                      # chat does not retry parse failures
```

### New: `src/t0_agents/core/outputs/json.py`

```python
@dataclass(slots=True)
class JsonOutput:
    schema: Mapping[str, Any] | None = None
    json_mode: Literal["off", "on", "strict", "tool"] = "strict"
    permissive: bool = True
    max_retries: int = 2

    expects_tool_loop: bool = False
    supports_streaming: bool = False
    name: str = "json"

    def shape_inference(self, req):
        req.params = req.params or {}
        req.params.setdefault("chat_completion", {})["json_mode"] = self.json_mode
        if self.schema is not None:
            req.output_schema = self.schema
        return req

    async def materialize(self, response, attempts):
        # 1. Try response.output.parsed (TZ-native).
        # 2. If None and self.permissive → permissive_parse(raw_text).
        # 3. Build AgentResult[dict] with OutputDiagnostics.
        ...

    async def recover(self, failure, attempts):
        if len(attempts) >= self.max_retries:
            return Recovery.fail()
        return Recovery.retry(
            f"The previous response could not be parsed as JSON. "
            f"Reason: {failure.reason}.  Please re-emit a valid JSON object."
        )
```

### New: `src/t0_agents/core/outputs/structured.py`

```python
M = TypeVar("M", bound="BaseModel")

@dataclass(slots=True)
class StructuredOutput(Generic[M]):
    model: type[M]
    json_mode: Literal["off", "on", "strict", "tool"] = "strict"
    max_retries: int = 2

    expects_tool_loop: bool = False
    supports_streaming: bool = False

    @property
    def name(self) -> str: return f"structured/{self.model.__name__}"

    def shape_inference(self, req):
        # Derive JSON schema from Pydantic model; thread as JsonOutput would.
        schema = self.model.model_json_schema()
        req.output_schema = schema
        req.params = req.params or {}
        req.params.setdefault("chat_completion", {})["json_mode"] = self.json_mode
        return req

    async def materialize(self, response, attempts):
        # 1. Parse via JsonOutput path.
        # 2. self.model.model_validate(parsed) → M.
        # 3. Return AgentResult[M].
        ...

    async def recover(self, failure, attempts):
        if len(attempts) >= self.max_retries:
            return Recovery.fail()
        # Field-path hint from ValidationError.
        return Recovery.retry(
            f"Validation failed: {failure.detail}.  Re-emit conforming to schema."
        )
```

### New: `src/t0_agents/core/outputs/tool_call.py`

```python
@dataclass(slots=True)
class ToolCallOutput:
    tool_choice: Any = "required"  # tensorzero.ToolChoice

    expects_tool_loop: bool = False  # we WANT the call, not its result
    supports_streaming: bool = False
    name: str = "tool_call"

    def shape_inference(self, req):
        req.tool_choice = self.tool_choice
        return req

    async def materialize(self, response, attempts):
        # Extract ToolCall blocks from response.content; do not execute.
        # AgentResult[list[ToolCall]] with .value = tool_calls.
        ...
    async def recover(self, failure, attempts):
        return Recovery.fail()
```

### New: `src/t0_agents/core/outputs/streaming.py`

```python
@dataclass(slots=True)
class StreamingOutput:
    base: "OutputMode[Any]" = field(default_factory=ChatOutput)
    expects_tool_loop: bool = True   # inherited from base
    supports_streaming: bool = True
    name: str = "streaming"

    def __post_init__(self):
        if not getattr(self.base, "supports_streaming", False):
            raise ConfigurationError(
                f"StreamingOutput(base={self.base.name}) is not yet supported. "
                "Only base=ChatOutput() works in v1.0; streaming JSON awaits TZ upstream."
            )
        self.expects_tool_loop = self.base.expects_tool_loop

    def shape_inference(self, req): return self.base.shape_inference(req)
    async def materialize(self, response, attempts): return await self.base.materialize(response, attempts)
    async def recover(self, failure, attempts): return await self.base.recover(failure, attempts)
```

### Modified: `src/t0_agents/core/types.py`

```python
T = TypeVar("T")

@dataclass
class AgentResult(Generic[T]):
    value: T                                # NEW: the typed result
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
    guardrail_log: list[GuardrailLogEntry] = field(default_factory=list)
    diagnostics: OutputDiagnostics | None = None  # NEW

    # ---- v0.12 compat properties ----
    @property
    def response(self) -> str:
        if isinstance(self.value, str):
            return self.value
        return self.diagnostics.raw_text if self.diagnostics else ""

    @property
    def parsed(self) -> Any | None:
        if isinstance(self.value, str):
            return None
        return self.value

    @property
    def parse_strategy(self) -> ParseStrategy:
        return self.diagnostics.parse_strategy if self.diagnostics else "strict"

    @property
    def json_parse_attempts(self) -> int:
        return self.diagnostics.attempts if self.diagnostics else 0

    @property
    def function_kind(self) -> Literal["chat", "json"]:
        return "chat" if isinstance(self.value, str) else "json"
```

### Modified: `src/t0_agents/core/config.py`

The 5 legacy kwargs are **deleted** from the dataclass.  Constructing with any of them raises `TypeError` (dataclass default behaviour).

```python
@dataclass(slots=True)
class AgentConfig:
    # ... existing identity / loop / TZ / memory / guardrails / sub_agents / middleware fields ...

    # CANONICAL (replaces the 5 legacy kwargs entirely):
    output: "OutputMode" = field(default_factory=ChatOutput)

    def __post_init__(self):
        # ... existing validations ...
        # No legacy migration — the 5 legacy kwargs no longer exist on the dataclass.
        # `self.output` is always an OutputMode (default ChatOutput()) — no None case.

        # Optional eager validation of incompatible combos (delegated to OutputMode):
        if not self.output.expects_tool_loop and self.sub_agents:
            raise ConfigurationError(
                f"AgentConfig.sub_agents is incompatible with output={self.output.name} "
                f"(single-shot mode — sub-agent tools would be unreachable). "
                f"Use ChatOutput() if you need sub-agents."
            )
```

**Migration table (documented in M5 migration guide):**

| v0.12 `AgentConfig(...)` | v1.0 equivalent |
|---|---|
| `function_kind="chat"` (default) | omit (default `output=ChatOutput()`) |
| `function_kind="json", json_mode="strict", output_schema=S` | `output=JsonOutput(schema=S, mode="strict")` |
| `function_kind="json", json_mode="strict", output_schema=S, permissive_json_parsing=False` | `output=JsonOutput(schema=S, mode="strict", permissive=False)` |
| `function_kind="json", json_mode="strict", output_schema=S, max_json_retries=5` | `output=JsonOutput(schema=S, mode="strict", max_retries=5)` |
| `output_schema=Path("schema.json")` | load JSON yourself and pass dict, OR use `StructuredOutput(MyPydanticModel)` |

### Modified: `src/t0_agents/core/agent.py`

Dispatch via `output.expects_tool_loop`:

```python
def _make_terminal(self) -> Callable[[RunContext], Awaitable[AgentResult]]:
    async def terminal(ctx: RunContext) -> AgentResult:
        if ctx.output.expects_tool_loop:
            # Chat path: run_agent loop
            ...
        else:
            # Single-shot path: shape inference + materialize + recover loop
            ...
    return terminal
```

`Agent.stream()` consults `output.supports_streaming`:

```python
async def stream(self, msg, **kw):
    output = kw.get("output") or self.config.output
    if not output.supports_streaming:
        raise ConfigurationError(
            f"OutputMode {output.name} does not support streaming. "
            "Use ChatOutput or StreamingOutput(base=ChatOutput())."
        )
    async for ev in self.invoke(msg, **kw).events():
        yield ev
```

### Modified: `src/t0_agents/core/structured.py`

The retry-hint and permissive-parse logic that lived inline migrate to `JsonOutput.recover()` and `JsonOutput.materialize()`.

`permissive_parse` stays as a public helper (third-party uses it).

`run_structured_inference` is unchanged at the signature level (low-level helper).

### Modified: `src/t0_agents/core/middleware/builtins.py`

`JsonRetryMiddleware` activates:

```python
class JsonRetryMiddleware:
    name = "json_retry"
    async def __call__(self, ctx, next_):
        if ctx.output.expects_tool_loop:
            return await next_(ctx)  # chat path; no retry orchestration here
        # JSON path: retry orchestration delegated to OutputMode.
        return await next_(ctx)   # the terminal itself runs the retry loop
        # (placement decided by Q13 — innermost, terminal-owned)
```

(Q13 resolution: the retry loop is owned by the terminal, not by middleware.  `JsonRetryMiddleware` becomes a thin observability wrapper that records retry counts on `ctx.run` for tracing.)

### Modified: `src/t0_agents/__init__.py`

Add exports: `OutputMode`, `ChatOutput`, `JsonOutput`, `StructuredOutput`, `ToolCallOutput`, `StreamingOutput`, `OutputDiagnostics`, `Recovery`, `ParseFailure`, `Attempt`, `InferenceRequest`.

---

## §5 — Tests

### `tests/test_output/test_chat_output.py`
- `ChatOutput.expects_tool_loop is True`, `supports_streaming is True`.
- `shape_inference` no-op.
- `materialize` produces `AgentResult[str]` byte-identical to v0.12 chat agent.

### `tests/test_output/test_json_output.py`
- `JsonOutput(schema=...)` injects `json_mode` and `output_schema` correctly into request.
- `materialize` reads `response.output.parsed` first; falls back to `permissive_parse` when `permissive=True`; reports `parse_strategy` in diagnostics.
- `recover` produces retry hint up to `max_retries`; then `Recovery.fail()`.

### `tests/test_output/test_structured_output.py`
- `StructuredOutput[Customer](Customer)` derives schema from model.
- `materialize` validates via `Customer.model_validate(parsed)`.
- `ValidationError` triggers `recover` with field-path hint.
- `AgentResult[Customer].value` is a `Customer` instance.

### `tests/test_output/test_tool_call_output.py`
- `ToolCallOutput(tool_choice="required")` sets `tool_choice` correctly.
- `materialize` returns `AgentResult[list[ToolCall]]` with `value` containing the calls (NOT executed).
- `expects_tool_loop is False` → terminal does NOT run tool execution.

### `tests/test_output/test_streaming_output.py`
- `StreamingOutput(base=ChatOutput())` works.
- `StreamingOutput(base=JsonOutput(...))` raises `ConfigurationError` at construction.
- `StreamingOutput` delegates `materialize` / `recover` to base.

### `tests/test_output/test_agent_result_compat.py`
- `AgentResult[str](value="hi", ...)`: `.response == "hi"`, `.parsed is None`, `.function_kind == "chat"`.
- `AgentResult[dict](value={"a": 1}, diagnostics=OutputDiagnostics(parse_strategy="permissive", attempts=1, raw_text='{"a": 1}'))`: `.response == '{"a": 1}'`, `.parsed == {"a": 1}`, `.parse_strategy == "permissive"`, `.json_parse_attempts == 1`, `.function_kind == "json"`.

### `tests/test_output/test_legacy_kwargs_removed.py`
- `AgentConfig(function_kind="json")` raises `TypeError` (kwarg removed).
- `AgentConfig(json_mode="strict")` raises `TypeError`.
- `AgentConfig(output_schema={...})` raises `TypeError`.
- `AgentConfig(permissive_json_parsing=False)` raises `TypeError`.
- `AgentConfig(max_json_retries=5)` raises `TypeError`.
- Migrated `examples/17_structured/` produces functionally equivalent output to v0.12 (same parsed dict structure, same `episode_id`-shaped logging).

### `tests/test_output/test_stream_rejects_json.py`
- `Agent(AgentConfig(..., output=JsonOutput(...))).stream("hi")` raises `ConfigurationError` early (before any inference).

---

## §6 — Example

M3 ships the **migrated** `examples/17_structured/`.  The original v0.12 example used `AgentConfig(function_kind="json", json_mode="strict", output_schema=S, ...)`.  The migrated version uses `AgentConfig(output=JsonOutput(schema=S, mode="strict", ...))`.  Same functionality, new shape.

Additionally, M3 adds a sibling demo file (`examples/17_structured/structured_pydantic.py`) showing the new `StructuredOutput[Customer]` capability that didn't exist in v0.12.

Full M5 example (`examples/21_output_modes/`) consolidates Pydantic + custom YAML demos.

---

## §7 — Documentation

- `docs/agent.md` — note that `output=` kwarg is now canonical; `function_kind=` is legacy.
- `docs/api-reference.md` — list new symbols.
- `docs/structured-outputs.md` — update with note that JSON mode is now driven by `OutputMode`; v0.12 syntax continues to work.
- Internal: update `02-open-questions.md` Q11, Q12, Q13.

---

## §8 — Checklist (Definition of Done)

### Pre-flight
- [ ] Read `00-integration-contract.md`.
- [ ] Read `01-implementation-plan.md`.
- [ ] Read `02-open-questions.md` (Q11, Q12, Q13).
- [ ] Read **M1 dev-log** + **M2 dev-log** (mandatory).
- [ ] Read project memory.
- [ ] Inspect SDK source: `.venv/lib/python*/site-packages/tensorzero/types.py` for `ChatInferenceResponse`/`JsonInferenceResponse` shape.

### Code
- [ ] `core/output.py` implements protocol + types.
- [ ] `core/outputs/{chat,json,structured,tool_call,streaming}.py` implemented.
- [ ] `core/types.py` — `AgentResult[T]` generic; properties for v0.12 fields.
- [ ] `core/config.py` — `output` field; legacy → `OutputMode` migration in `__post_init__`.
- [ ] `core/agent.py` — dispatch via `output.expects_tool_loop`; `stream()` rejects non-streaming modes.
- [ ] `core/structured.py` — retry/permissive logic migrated; public helper preserved.
- [ ] `core/middleware/builtins.py` — `JsonRetryMiddleware` activated.
- [ ] `__init__.py` exports updated.

### Tests
- [ ] All 8 test files in `tests/test_output/` pass.
- [ ] Legacy `tests/test_structured/` (if exists) byte-identical.

### Example
- [ ] `examples/17_structured/structured_agent.py` migrated to `output=JsonOutput(...)` shape; runs end-to-end against `docker compose up`.
- [ ] Sibling demo `examples/17_structured/structured_pydantic.py` showcasing `StructuredOutput[Customer]` added.

### Docs
- [ ] `docs/agent.md` updated.
- [ ] `docs/api-reference.md` updated.
- [ ] `docs/structured-outputs.md` updated.

### Style gates
- [ ] `uv run ruff check src/` clean.
- [ ] `uv run mypy src/` strict pass with generics.

### Implementation log
- [ ] `docs/internal/v1-refactor/logs/milestone-3-dev-log.md` written.

### CHANGELOG
- [ ] `## [1.0.0-M3] — Unreleased` entry.

### Open questions to resolve
- [ ] Q11 (StreamingOutput activation) — deferred officially.
- [ ] Q12 (AgentResult[T] type narrowing helpers) — decided.
- [ ] Q13 (JsonRetryMiddleware placement) — decided (innermost / terminal-owned).

### Invariant checks (gate)
- [ ] `git diff src/t0_agents/mcp/` → empty.
- [ ] 17 chat-mode examples (`examples/00..16, 18`) run byte-identical (stderr may include `DeprecationWarning` from M1 shims).
- [ ] Migrated `examples/17_structured/` runs end-to-end with `output=JsonOutput(...)`.
- [ ] All pre-existing tests pass, **except** any test that explicitly exercises legacy JSON kwargs — those are updated to use `output=` and noted in dev-log §2.

---

## §9 — Acceptance criteria

- `from t0_agents import OutputMode, ChatOutput, JsonOutput, StructuredOutput, ToolCallOutput, StreamingOutput, OutputDiagnostics, Recovery` works.
- `AgentConfig(name="x", function_name="extract", output=StructuredOutput(Customer))` constructs without error.
- `result.value` is typed `Customer` (mypy verified).
- `AgentConfig(function_kind="json", ...)` raises `TypeError` (legacy kwargs removed).
- `AgentResult.parsed`, `.parse_strategy`, `.json_parse_attempts`, `.function_kind` still readable via `@property` — read-side compat preserved.
- `agent.invoke(...)` on a JSON-mode agent works; `agent.invoke(...).events()` raises `ConfigurationError` (streaming JSON not supported until TZ ships it).
- Migrated `examples/17_structured/` produces functionally equivalent JSON extraction.

---

## §10 — Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `AgentResult[T]` generic breaks downstream mypy | Default `Agent` (untyped) = `Agent[str]`; document explicitly. |
| Properties on `AgentResult` change semantics subtly (mutability, lazy eval) | Tests compare attribute reads byte-by-byte with v0.12. |
| `JsonOutput.recover()` retry orchestration in terminal vs middleware ambiguity | Q13 resolution: terminal-owned.  Document in `00-integration-contract.md` §3.3. |
| `StreamingOutput(base=JsonOutput)` user surprise | Clear `ConfigurationError` message at construction time. |
| Pydantic optional dependency for `StructuredOutput` | `StructuredOutput` imports `pydantic` lazily; `ImportError` surfaces clearly when missing. |

---

## §11 — Estimated surface

- `core/output.py`: ~250 LoC.
- `core/outputs/chat.py`: ~80 LoC.
- `core/outputs/json.py`: ~200 LoC.
- `core/outputs/structured.py`: ~150 LoC.
- `core/outputs/tool_call.py`: ~80 LoC.
- `core/outputs/streaming.py`: ~80 LoC.
- `core/types.py` modifications: ~+100 LoC (generic + properties).
- `core/config.py` modifications: ~+50 LoC.
- `core/agent.py` modifications: ~+50 LoC.
- `core/structured.py` modifications: ~−100 LoC (logic migrated out).
- `tests/test_output/`: ~700 LoC across 8 files.
- Docs delta: ~50 lines.

**Total: ~+1000 LoC src + ~700 LoC tests.  Estimated 5 days full-time.**
