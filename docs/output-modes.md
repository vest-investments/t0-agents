# Output modes

`OutputMode[T]` is the polymorphic surface for how an agent's output
is shaped, materialised, and recovered from parse failures.  v0.12
hardcoded a chat/JSON dichotomy inside `core/agent.py` with 5 fields
on `AgentConfig` and ~80 lines of cross-field validation; v1.0
collapses that into one protocol that any callsite can plug into.

Five built-in modes ship: `ChatOutput` (the default, tool-loop chat),
`JsonOutput` (single-shot JSON with retry), `StructuredOutput[M]`
(Pydantic-validated extraction), `ToolCallOutput` (router pattern),
`StreamingOutput` (placeholder for v1.1 streaming JSON).  Third
parties can ship their own — YAML extraction, Instructor-style
retry, SQLAlchemy row materialisation — without forking the library.

> Companion reading: [`docs/run.md`](run.md) (the handle that wraps
> the materialised result), [`docs/middleware.md`](middleware.md)
> (`JsonRetryMiddleware` drives the recovery handshake),
> [`docs/structured-outputs.md`](structured-outputs.md) (legacy
> reference for the `run_structured_inference()` helper and the
> `json_mode` provider matrix),
> [`docs/api-reference.md` — Output modes](api-reference.md#output-modes).

---

## The `OutputMode[T]` protocol

```python
from typing import Any, Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class OutputMode(Protocol[T_co]):
    name: str
    expects_tool_loop: bool
    supports_streaming: bool

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest: ...

    async def materialize(
        self,
        response: Any,
        attempts: list[Attempt],
    ) -> AgentResult[T_co]: ...

    async def recover(
        self,
        failure: ParseFailure,
        attempts: list[Attempt],
    ) -> Recovery: ...
```

Concept by concept:

- **`name: str`** — short identifier surfaced in error messages and
  `ConfigurationError` text.  Built-ins use `"chat"`, `"json"`,
  `"structured"`, `"tool_call"`, `"streaming"`.
- **`expects_tool_loop: bool`** — `True` for chat modes that drive
  the tool-call loop (`agent.invoke` requires a `tool_provider` when
  this is `True`); `False` for single-shot modes that make one
  inference call.
- **`supports_streaming: bool`** — `True` when the mode can yield
  `AgentEvent`s mid-run.  Drives the gate on `run.events()`: when
  `False`, `events()` raises `ConfigurationError`.
- **`shape_inference(req)`** — mutates `req` (an `InferenceRequest`)
  to add the TZ-side kwargs the mode needs (`json_mode`,
  `output_schema`, `tool_choice`, ...).
- **`materialize(response, attempts)`** — converts the raw
  `t0.inference` response into an `AgentResult[T]`, including the
  diagnostics payload (`parse_strategy`, `attempts`, `raw_text`,
  `failure_detail`).
- **`recover(failure, attempts)`** — called by `JsonRetryMiddleware`
  on parse failure.  Returns `Recovery.retry(hint)`,
  `Recovery.accept(value)`, or `Recovery.fail()`.

Single-shot modes (`expects_tool_loop=False`, `supports_streaming=False`)
fire one inference and materialise once; on failure the
`JsonRetryMiddleware` consults `recover` to decide whether to loop
with a corrective hint, accept a coerced fallback, or surface the
failure.  Chat modes use `shape_inference` for tool-call setup and
generally return `Recovery.fail()` from `recover` — there's no retry
loop for chat.

---

## The 5 built-in modes

| Mode | `T` | `expects_tool_loop` | `supports_streaming` | Construction |
|---|---|---|---|---|
| `ChatOutput()` | `str` | `True` | `True` | default — no kwargs |
| `JsonOutput(schema=, json_mode=, permissive=, max_retries=)` | `dict[str, Any]` | `False` | `False` | all kwargs optional |
| `StructuredOutput[M](model)` | `M` (Pydantic) | `False` | `False` | `model` required; `[pydantic]` extra |
| `ToolCallOutput(tool_choice=)` | `list[ToolCall]` | `True` | `False` | `tool_choice` optional |
| `StreamingOutput(base=ChatOutput())` | delegates | delegates | `True` | only `base=ChatOutput()` works in v1.0 |

### `ChatOutput` — the default

The tool-loop chat path that every v0.12 chat agent used.  `T=str`.
Materialises `AgentResult.value: str` from the final assistant
response.  `recover` always returns `Recovery.fail()` — chat doesn't
retry on parse "failures" because there is no parse step.

```python
from t0_agents import Agent, AgentConfig, ChatOutput

config = AgentConfig(
    name="assistant",
    function_name="agent",
    output=ChatOutput(),   # the default; can be omitted
)
agent = Agent(config, t0=gateway, tool_provider=mcp_client)
result = await agent.invoke("Hello").result()
print(result.value)        # str — the LLM's response
print(result.response)     # same str — v0.12 alias
```

### `JsonOutput` — single-shot JSON with retry

```python
from t0_agents import Agent, AgentConfig, JsonOutput

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
    "required": ["name", "email"],
}
config = AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=JsonOutput(
        schema=schema,         # or None for the function's static schema
        json_mode="strict",    # "off" | "on" | "strict" | "tool"
        permissive=True,       # accept dirty JSON via permissive_parse
        max_retries=2,         # retries with corrective hint
    ),
)
agent = Agent(config, t0=gateway)   # no tool_provider needed
result = await agent.invoke("Sarah <sarah@example.com>").result()
print(result.value)        # dict — {"name": "Sarah", "email": "sarah@example.com"}
print(result.parsed)       # same dict — v0.12 alias
print(result.parse_strategy)   # "strict" | "permissive" | "retry" | "none"
print(result.json_parse_attempts)
```

Setting `permissive=True` enables the `permissive_parse()` helper
that handles common formatting drift (fence blocks, trailing commas,
unquoted keys).  `max_retries=N` lets the middleware re-emit
inference up to N times with a corrective hint extracted from the
parse failure.

### `StructuredOutput[M]` — Pydantic-validated

```python
from pydantic import BaseModel
from t0_agents import Agent, AgentConfig, StructuredOutput

class Customer(BaseModel):
    name: str
    email: str
    company: str | None = None

config = AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=StructuredOutput(Customer),
)
agent = Agent(config, t0=gateway)
result = await agent.invoke("Sarah from ACME <sarah@acme.com>").result()
print(result.value)        # Customer(name='Sarah', email='sarah@acme.com', company='ACME')
print(type(result.value))  # <class 'Customer'>  — mypy verifies this via overload
```

Per-call overloads on `Agent.invoke` narrow `Run[T]` automatically:
`agent.invoke(msg, output=StructuredOutput(Customer))` returns
`Run[Customer]`; `await run.result()` returns `AgentResult[Customer]`
with `result.value: Customer` statically typed — this is why no
`Agent[T]` class-level generic exists in v1.0.

`StructuredOutput` lazily imports pydantic at construction time and
derives the JSON schema via `Model.model_json_schema()`.  Install the
extra: `pip install 't0-agents[pydantic]'`.  Construction without
pydantic installed surfaces as `ConfigurationError` with an install
hint.

On validation failure the `recover` body builds a field-aware
corrective hint summarising the `ValidationError` (which fields, what
went wrong) so the LLM can fix specifically what was broken — this
is what `OutputDiagnostics.failure_detail` carries (see
[`OutputDiagnostics` below](#outputdiagnostics)).

### `ToolCallOutput` — the router pattern

```python
from t0_agents import Agent, AgentConfig, ToolCallOutput

config = AgentConfig(
    name="router",
    function_name="route_query",
    output=ToolCallOutput(tool_choice="required"),  # or specific name
)
agent = Agent(config, t0=gateway, tool_provider=mcp_client)
result = await agent.invoke("Book a flight to Lisbon next Tuesday").result()
for call in result.value:    # list[ToolCall]
    print(call.name, call.arguments)
```

Extracts tool calls from a single chat response **without executing
them**.  Useful when you want the LLM to pick a route but the actual
dispatch happens elsewhere (e.g., persisted as a job, sent through
an approval queue).  When the response has no tool calls,
`result.value` is `[]` and `result.parse_strategy` is `"none"`.

### `StreamingOutput` — placeholder for streaming JSON

```python
from t0_agents import StreamingOutput, ChatOutput, JsonOutput

# Works — delegates to ChatOutput.
config = AgentConfig(..., output=StreamingOutput(base=ChatOutput()))

# Raises ConfigurationError at construction — see Q11.
config = AgentConfig(..., output=StreamingOutput(base=JsonOutput(...)))
```

`StreamingOutput(base=ChatOutput())` is functionally equivalent to
`ChatOutput()`; the wrapper is intentional documentation that the
caller opted into streaming explicitly.
`StreamingOutput(base=JsonOutput(...))` is deferred (Q11) until
TensorZero ships mid-stream `parsed` upstream.

---

## Per-call override on `Agent.invoke`

`AgentConfig(output=...)` sets the default mode for every invocation
on that agent.  Per call, you can override:

```python
config = AgentConfig(name="multi", function_name="agent", output=ChatOutput())
agent = Agent(config, t0=gateway, tool_provider=mcp)

chat = await agent.invoke("Hi").result()           # Run[Any] (ChatOutput)
data = await agent.invoke(                            # Run[dict]
    "Extract customer info",
    output=JsonOutput(schema=customer_schema),
).result()
typed = await agent.invoke(                           # Run[Customer]
    "Extract customer info",
    output=StructuredOutput(Customer),
).result()
```

The per-call value wins over `config.output`.  Mypy narrows `Run[T]`
via overload pair: when `output=` is supplied with a
`OutputMode[T]`, the return type is `Run[T]`; without `output=`,
it's `Run[Any]`.

This per-call narrowing is the v1.0 answer to "how do I get
`result.value: Customer` mypy-verified?" — overload narrowing on
`Agent.invoke` is the canonical path, rather than a class-level
`Agent[T]` generic.

---

## The retry-and-recovery handshake

`JsonRetryMiddleware` drives the recovery loop for single-shot modes
(`expects_tool_loop=False`).  The handshake:

```
                 ┌─────────────────────────────────────────────┐
   next_(ctx) ─→ │ terminal (run_structured_inference)         │
                 │   ↓                                         │
                 │  raw_response                               │
                 │   ↓                                         │
                 │ ctx.output.materialize(response, attempts) ─┼─→ result
                 └─────────────────────────────────────────────┘
                            │
                            ▼
   if result.diagnostics.parse_strategy != "none": return result
   else:
       failure = ParseFailure(raw_text=..., inference_id=..., reason=...)
       recovery = await ctx.output.recover(failure, attempts)
       match recovery:
           case Recovery.retry(hint):
               # Push prior turn pair + corrective hint, loop.
               ctx.prior_messages.append(... assistant turn ...)
               ctx.user_message = hint
               continue
           case Recovery.accept(value):
               # Synthesise AgentResult with parse_strategy="permissive".
               return AgentResult(value=value, ...)
           case Recovery.fail():
               # Surface the failed result as-is.
               return result
```

`Recovery` is a frozen dataclass with three constructors:

- `Recovery.retry(hint: str)` — re-emit inference with `hint` as the
  next user turn.
- `Recovery.accept(value: Any)` — bypass parsing, take this value as
  the materialised result with `parse_strategy="permissive"`.
- `Recovery.fail()` — surface the failure to the caller (`result`
  flows through with `value=None`, `parse_strategy="none"`,
  `parsed=None`).

Built-in `JsonOutput.recover` builds a corrective hint that pastes
the raw text + a "the previous response wasn't valid JSON; return
ONLY a JSON object matching the schema" instruction.
`StructuredOutput.recover` extends this with field-aware
`ValidationError` summarisation.

Custom modes implement `recover` in any way they like — see
[Custom output mode walkthrough](#custom-output-mode-walkthrough)
below.

### `OutputDiagnostics`

Every materialised `AgentResult` carries a `diagnostics` payload:

```python
@dataclass(frozen=True)
class OutputDiagnostics:
    parse_strategy: ParseStrategy   # "strict" | "permissive" | "retry" | "none"
    attempts: list[Attempt]
    raw_text: str | None
    failure_detail: str | None      # non-recovery parse-failure context
```

The first three fields back the v0.12 read-side properties
(`result.parse_strategy`, `result.json_parse_attempts`,
`result.raw_text` if the mode exposes one).

The fourth field — `failure_detail` — carries non-recovery context
between `materialize` and `recover`.  When a mode's `materialize`
reports `parse_strategy="none"`, it can populate
`failure_detail` with a structured summary the recovery hint can
incorporate.  `StructuredOutput` uses this to ferry the Pydantic
`ValidationError` summary from `materialize` to `recover` without
conflating it with the model's raw output (`raw_text`).

---

## The `[pydantic]` extra

`StructuredOutput` requires pydantic >= 2.0.  The library installs
clean without it; pydantic is pulled by the `[pydantic]` extra:

```bash
pip install 't0-agents[pydantic]'
# or
uv add t0-agents --extra pydantic
```

`StructuredOutput.__init__` does a lazy `import pydantic`; an
`ImportError` surfaces as `ConfigurationError` with an install hint
that points at this extra.

The `[serve]` and `[cli]` extras also pull pydantic transitively (for
FastAPI), so users of either may already have pydantic available
without explicitly adding `[pydantic]`.

---

## Custom output mode walkthrough

Third-party output modes implement the protocol the same way the
built-ins do.  A worked example: a 50-line `YamlOutput` that
materialises YAML responses instead of JSON.

```python
from dataclasses import dataclass
from typing import Any
import yaml      # pip install pyyaml

from t0_agents import AgentResult
from t0_agents.core.output import (
    Attempt, InferenceRequest, OutputDiagnostics, ParseFailure, Recovery,
)


@dataclass
class YamlOutput:
    """Single-shot YAML extraction.  Mirror of JsonOutput but for YAML."""

    name: str = "yaml"
    expects_tool_loop: bool = False
    supports_streaming: bool = False
    max_retries: int = 1

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        # YAML is plain text; nothing to inject into request.
        return req

    async def materialize(
        self,
        response: Any,
        attempts: list[Attempt],
    ) -> AgentResult[dict[str, Any]]:
        raw = response.content[-1].text if response.content else ""
        try:
            parsed = yaml.safe_load(raw)
            return AgentResult(
                value=parsed,
                episode_id=response.episode_id,
                inference_ids=[response.inference_id],
                turns=1,
                diagnostics=OutputDiagnostics(
                    parse_strategy="strict",
                    attempts=attempts,
                    raw_text=raw,
                    failure_detail=None,
                ),
            )
        except yaml.YAMLError as e:
            return AgentResult(
                value=None,
                episode_id=response.episode_id,
                inference_ids=[response.inference_id],
                turns=1,
                diagnostics=OutputDiagnostics(
                    parse_strategy="none",
                    attempts=attempts,
                    raw_text=raw,
                    failure_detail=str(e),
                ),
            )

    async def recover(
        self,
        failure: ParseFailure,
        attempts: list[Attempt],
    ) -> Recovery:
        if len(attempts) >= self.max_retries + 1:
            return Recovery.fail()
        hint = (
            f"The previous response wasn't valid YAML "
            f"(parser said: {failure.detail}).\n\n"
            f"Return ONLY a YAML object — no commentary, no fences."
        )
        return Recovery.retry(hint)


# Use it:
config = AgentConfig(name="yaml_ext", function_name="agent", output=YamlOutput())
agent = Agent(config, t0=gateway)
result = await agent.invoke("Give me ACME's contact info").result()
print(result.value)   # dict from parsed YAML
```

That's the full surface — any third-party mode that exposes the
three methods + the two flags works with the rest of v1.0
(middleware chain, `Agent.invoke`, `Run.events`, the
`AgentResult[T]` generics).

Other realistic third-party patterns the protocol supports:

- **Instructor-style retry** — wrap a Pydantic model with a custom
  validator and a richer recovery hint that mentions field-specific
  examples; ship as a separate library.
- **SQLAlchemy row materialiser** — `materialize` constructs an ORM
  instance from the parsed JSON; `recover` adds DB-constraint
  context to the hint.
- **Multi-language output** — `materialize` parses YAML or TOML or
  XML depending on a constructor kwarg.

---

## Wire-format preservation (HTTP/WS)

When you serve an agent over HTTP or WebSocket via
`t0_agents.serve.create_app(..., streaming=True)`, the
`DoneEvent.result` payload serialises through the v0.12 read-side
keys — `response`, `parsed`, `parse_strategy`, `json_parse_attempts`,
`function_kind` — rather than the new canonical `value` /
`diagnostics`.  This preserves the v0.12 wire format so downstream
clients keep working byte-identical despite `AgentResult` becoming
generic.

The projection lives at
[`src/t0_agents/serve/streaming_routes.py`](../src/t0_agents/serve/streaming_routes.py)
in `_event_to_json`.  The projection keeps downstream
`DoneEvent` consumers byte-identical with v0.12 while letting
`AgentResult` become generic on the Python side.

If you build a third-party server that serialises `AgentResult`, you
have two choices:

- **Stay v0.12-shaped** (recommended for backward compat) — project
  through the read-side properties: `{"response": result.response,
  "parsed": result.parsed, ...}`.
- **Use the new shape** — serialise `result.value` directly.  Cleaner
  for new clients that know they're talking to a v1.0 backend.

---

## Related

- [`docs/run.md`](run.md) — the handle around the materialised result.
- [`docs/middleware.md`](middleware.md) — `JsonRetryMiddleware`
  drives the recovery handshake.
- [`docs/structured-outputs.md`](structured-outputs.md) — legacy
  reference for the `run_structured_inference()` helper, the recovery
  ladder, and the `json_mode` provider matrix (Pydantic / Groq /
  OpenAI / etc.).
- [`docs/migration-v0.12-to-v1.0.md`](migration-v0.12-to-v1.0.md) — the
  5 Tier 2 kwargs that became `output=JsonOutput(...)`.
- [`docs/api-reference.md` — Output modes](api-reference.md#output-modes).
- [`docs/glow/getting-started.md` — Structured outputs by return
  type](glow/getting-started.md#structured-outputs-by-return-type) —
  glow infers the `OutputMode` from the function's return annotation;
  the dispatch table lives there.
