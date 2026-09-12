# Structured Outputs

TensorZero distinguishes two function types — `chat` and `json`. This page documents how t0-agents supports the `json` flavour: the single-shot `run_structured_inference()` helper, the declarative `Agent` dispatch with single-shot output modes, the three-layer recovery ladder that absorbs imperfect model output, the safety surface that catches incompatible configurations, the opt-in `require_parsed()` guardrail, the testing primitives, and the `json_mode` provider matrix.

The chat-path surface is byte-identical to v0.9.0; everything below is additive for callers who opt into JSON output.

> **v1.0 migration.**  v1.0 replaces the v0.12 `AgentConfig(function_kind="json", json_mode="strict", output_schema=S, permissive_json_parsing=True, max_json_retries=2)` cluster with a single canonical field:
>
> ```python
> from t0_agents import AgentConfig, JsonOutput
> config = AgentConfig(
>     name="extractor",
>     function_name="extract_customer_info",
>     output=JsonOutput(
>         schema=S,  # or None to use the function's static schema
>         json_mode="strict",
>         permissive=True,
>         max_retries=2,
>     ),
> )
> ```
>
> The 5 legacy v0.12 kwargs raise `TypeError` if passed.  Read-side `AgentResult.{parsed, parse_strategy, json_parse_attempts, function_kind, response}` properties continue to work (preserved via `@property` aliases over the new `AgentResult[T].value` field).  Per-call overrides live on the canonical surface: `agent.invoke(msg, output=JsonOutput(schema=..., json_mode=..., max_retries=...))`.
>
> **Canonical surface.**  [`docs/output-modes.md`](output-modes.md) is the v1.0 narrative for the full `OutputMode[T]` polymorphism (chat / JSON / Pydantic / tool-call / streaming + custom modes).  This page remains the reference for the legacy `run_structured_inference()` helper, the recovery ladder, and the `json_mode` provider matrix.  For the migration walkthrough see [`docs/migration-v0.12-to-v1.0.md`](migration-v0.12-to-v1.0.md).

## What are JSON functions?

A TensorZero `json` function is single-shot: the gateway sends the user message to a model, asks the model to emit a JSON object that conforms to a JSON Schema, and returns both the raw string and the parsed mapping. There is no tool loop, no multi-turn conversation, no termination criterion. The shape of the response is:

- `output.raw`  — the JSON string the model produced.
- `output.parsed` — `dict | None`; `None` when the model output failed to parse.

t0-agents flattens that shape onto a single `StructuredOutput` dataclass for ergonomic consumption — see [Single-shot JSON inference](#single-shot-json-inference) below.

## Configuring a JSON function

The TensorZero side of the configuration is unchanged from the gateway docs — declare a function with `type = "json"` and a variant that picks a `json_mode`:

```toml
[functions.extract_customer_info]
type = "json"
output_schema = "functions/extract_customer_info/output_schema.json"

[functions.extract_customer_info.variants.gpt_oss_20]
type = "chat_completion"
model = "gpt_oss_20"
templates.system.path = "functions/extract_customer_info/system_template.minijinja"
json_mode = "strict"
```

The schema file is plain JSON Schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name":  { "type": ["string", "null"] },
    "email": { "type": ["string", "null"] }
  },
  "required": ["name", "email"],
  "additionalProperties": false
}
```

See `examples/agent/03_output_modes/` for a runnable copy.

## Single-shot JSON inference

```python
from tensorzero import AsyncTensorZeroGateway
from t0_agents.core.structured import run_structured_inference

async with await AsyncTensorZeroGateway.build_http(
    gateway_url="http://localhost:3000"
) as t0:
    result = await run_structured_inference(
        t0,
        function_name="extract_customer_info",
        user_message="Hi, I'm Sarah Johnson at sarah.j@example.com",
        json_mode="strict",
    )

# result.parsed       — {"name": "Sarah Johnson", "email": "sarah.j@example.com"}
# result.raw          — '{"name":"Sarah Johnson","email":"sarah.j@example.com"}'
# result.parse_strategy — "strict" (the gateway parsed it directly)
# result.episode_id   — the TZ-assigned episode UUID
# result.inference_id — the inference UUID
# result.usage        — tensorzero.Usage (input_tokens, output_tokens, cost)
```

The kwargs accepted by `run_structured_inference()` mirror the JSON-relevant subset of `run_agent`'s — `episode_id`, `tags`, `namespace`, `dryrun`, `system_arguments`, plus the JSON-specific `output_schema` (dynamic override) and `json_mode` (variant override). It deliberately omits `parallel_tools`, `tool_choice`, `tool_timeout`, and `prior_messages` because none of them are meaningful for a single-shot JSON call.

The full signature:

```python
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
) -> StructuredOutput
```

### Dynamic schema override

Pass `output_schema={...}` as an inline mapping to override the function's static schema for a single call. TensorZero's behaviour is documented in §4 of [the gateway's structured-outputs guide](https://www.tensorzero.com/docs/gateway/generate-structured-outputs).

### Path-based schema

`JsonOutput.schema` accepts an inline `dict` mapping.  When you want to keep the schema in its own JSON file, read it from disk once at construction time and pass the parsed mapping in:

```python
import json
from pathlib import Path

schema = json.loads(Path("functions/extract_customer_info/output_schema.json").read_text())
config = AgentConfig(
    name="extractor",
    function_name="extract_customer_info",
    output=JsonOutput(schema=schema, json_mode="strict"),
)
```

> **Recommendation:** prefer absolute paths in production code.  Relative paths feel natural in tests and one-off scripts but can surprise downstream callers when the working directory drifts.

### Schema validation in Python

`parsed` may be valid JSON without being schema-conforming (e.g. missing a required field).  t0-agents does **not** run JSON-Schema validation in Python — TensorZero already does that on its side when the provider supports it, and adding `jsonschema` as a dependency was deliberately ruled out (see the integration contract).  If you need strict schema conformance, validate with [Pydantic](https://docs.pydantic.dev/) on the consumer side:

```python
from pydantic import BaseModel, ValidationError

class Address(BaseModel):
    street: str
    city: str
    country: str

class Customer(BaseModel):
    name: str | None
    email: str | None
    address: Address | None = None

result = await run_structured_inference(...)
if result.parsed is None:
    # Every recovery layer failed.  Decide whether to retry, fall back
    # to a partial value, or surface the failure to the caller.
    raise RuntimeError(f"Could not extract customer info: {result.response!r}")

try:
    customer = Customer.model_validate(result.parsed)
except ValidationError as exc:
    # parsed was structurally valid JSON but the schema constraints
    # (types, nesting, required fields) were not met.  This is the
    # "valid JSON but wrong shape" case TensorZero's strict mode is
    # supposed to prevent — falling through here usually means the
    # provider doesn't support native strict mode (e.g. Bedrock).
    raise ValueError(f"Schema mismatch: {exc.errors()}") from exc
```

This pattern keeps the contract local to your application: t0-agents guarantees `result.parsed` is either valid JSON (or `None`); your model declaration enforces the rest.  See [Testing your JSON agent](#testing-your-json-agent) below for a `jsonschema`-based alternative when Pydantic isn't already in your dependency graph.

## Structured `Agent`

The high-level `Agent` wrapper dispatches to `run_structured_inference()` whenever `AgentConfig.output` is a single-shot mode (`JsonOutput` or `StructuredOutput`) — no manual branching required.  Build the same `AgentConfig` you would for a chat agent, swap `output=ChatOutput()` (the default) for `output=JsonOutput(...)`, and call `agent.invoke(...)` exactly as before.

```python
from tensorzero import AsyncTensorZeroGateway
from t0_agents import Agent, AgentConfig, JsonOutput

config = AgentConfig(
    name="customer-extractor",
    function_name="extract_customer_info",
    output=JsonOutput(json_mode="strict"),
)

async with await AsyncTensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    agent = Agent(config, t0=t0)
    result = await agent.invoke("Hi, I'm Sarah at sarah@example.com")

# result.function_kind  — "json"
# result.parsed         — {"name": "Sarah", "email": "sarah@example.com"}
# result.response       — '{"name":"Sarah","email":"sarah@example.com"}'
# result.turns          — 1   (JSON is single-shot)
# result.inference_ids  — [<one uuid>]
```

Single-shot output modes never dispatch MCP tools, so the `Agent` constructor accepts no `tool_provider` here — pass just the configuration and the TensorZero gateway.  (Chat-mode agents — i.e. `output=ChatOutput()`, the default — still require a `tool_provider`; see [`docs/agent.md`](agent.md).)  Passing an explicit `tool_provider` to a JSON agent is still accepted — useful if you mix output modes across a single fleet — but it is no longer required.  For a runnable copy see `examples/agent/03_output_modes/structured_agent.py`.

### Per-call overrides

Pass a per-call `output=JsonOutput(...)` to override schema, mode, retries, or permissive parsing for a single invocation. Per-call values win over the `AgentConfig` fields:

```python
result = await agent.invoke(
    "...",
    output=JsonOutput(
        schema={"type": "object", "properties": {"email": {"type": "string"}}, ...},
        json_mode="tool",  # overrides config's "strict"
    ),
)
```

A chat-mode agent paired with a per-call single-shot `output=` (or vice versa) is rejected at `Agent.invoke()` with `ConfigurationError`.

### How `Agent.invoke` dispatches

```
agent.invoke(message, ...) → Run[T]
        │
        ▼
  output.expects_tool_loop?
        │
   ┌────┴────┐
   ▼         ▼
  False     True
   │         │
   │         └─► run_agent (chat loop)
   │
   └─► run_structured_inference (single-shot JSON inference)
```

### Memory and guardrails for JSON runs

JSON functions are single-shot — there is no tool loop, no parallel tool dispatch, no sub-agent orchestration (the last is rejected at `AgentConfig.__post_init__`). The scaffolding fires accordingly:

| Hook | Fires on JSON run? | Notes |
|------|--------------------|-------|
| `Guardrail.before_run` | Yes | Same `GuardrailContext`, same fatal/non-fatal contract as chat. |
| `Guardrail.before_tool` | **No** | JSON functions never call tools. |
| `Guardrail.after_tool` | **No** | JSON functions never call tools. |
| `Guardrail.after_run` | Yes | Same contract as chat — receives the final `AgentResult` (or `None` on fatal abort). |
| `MemoryProvider.load` | Yes when `episode_id` is set | `prepended_message` (default) prepends a summary; `system_template` threads into `input.system`. |
| `MemoryProvider.save` | Yes | `MemoryRecord.agent_response` is the raw JSON string (i.e. `result.response`). |

The `memory_strategy="messages"` mode is **not yet supported** for JSON functions — `Agent.invoke` raises `ConfigurationError` at runtime whenever a JSON-flavoured agent is configured with that strategy, regardless of whether a prior history exists. Use `"prepended_message"` (the default) or `"system_template"` instead.

## Robust JSON parsing

Real models sometimes ignore the JSON-mode contract: a Groq model wraps its reply in prose, a fenced ` ```json ` block, or an apology before the JSON.  TensorZero's strict parser returns `parsed == None` in those cases, even though the JSON is right there in `output.raw`.  t0-agents adds two in-Python layers that recover the JSON without losing the byte-for-byte TensorZero-pure path.

```
        TensorZero strict parse        ← layer 1: parse_strategy="strict"
                  │ parsed is None
                  ▼
        permissive_parse(output.raw)   ← layer 2: parse_strategy="permissive"
                  │ parsed is None
                  ▼
        retry with corrective hint     ← layer 3: parse_strategy="retry"
        (up to max_json_retries extra attempts)
                  │ every attempt failed
                  ▼
        result.parsed is None,         ← terminal: parse_strategy="none"
        caller inspects strategy
```

The layers are configured per-agent on `JsonOutput` (which lives on `AgentConfig.output`):

| Field                       | Default | What it controls                                                                 |
|-----------------------------|---------|----------------------------------------------------------------------------------|
| `JsonOutput(permissive=...)` | `True`  | Enables the in-Python recovery layer when TensorZero returned `parsed == None`. |
| `JsonOutput(max_retries=...)` | `2`     | Maximum number of *extra* inferences after the first.  `0` disables retries.    |
| `agent.invoke(msg, output=JsonOutput(max_retries=N))` | from config | Per-call override of the retry budget for one invocation. |

Each call reports which layer produced the result on the new `AgentResult` fields:

| Field                          | Meaning                                                                                     |
|--------------------------------|---------------------------------------------------------------------------------------------|
| `result.parse_strategy`        | `"strict"`, `"permissive"`, `"retry"`, or `"none"` — which layer produced `parsed`.        |
| `result.json_parse_attempts`   | Count of *extra* inferences after the first.  `0` on the happy path, `JsonOutput.max_retries` at most. |
| `len(result.inference_ids)`    | `1 + json_parse_attempts` — every attempt's inference id is recorded, in order.            |
| `result.turns`                 | Same as `len(result.inference_ids)` for JSON runs.                                          |

### Worked example: permissive recovery

The model wraps its JSON in a fenced code block:

```python
result = await agent.invoke("Tell me about Sarah Johnson, sarah.j@example.com.")
# Model returned: 'Sure! Here it is:\n```json\n{"name": "Sarah", "email": "sarah.j@example.com"}\n```'
# TensorZero says: output.parsed == None
# In-Python permissive layer extracts the JSON span and parses it.

result.parsed         # {"name": "Sarah", "email": "sarah.j@example.com"}
result.parse_strategy # "permissive"
result.json_parse_attempts  # 0  (no retry was needed)
len(result.inference_ids)   # 1
```

### Worked example: retry

The model produces invalid output twice, then succeeds on the third attempt:

```python
config = AgentConfig(
    name="extractor",
    function_name="extract_customer_info",
    output=JsonOutput(
        json_mode="strict",
        max_retries=2,         # the default
        permissive=True,       # the default
    ),
)
result = await agent.invoke("Tell me about a famous engineer.")
# Attempt 0: "Sorry, I cannot answer that."  → parsed=None, no recoverable JSON.
# Attempt 1: another non-JSON apology         → still no JSON.
# Attempt 2: '{"name": "...", "email": "..."}' → parsed.

result.parse_strategy        # "retry"
result.json_parse_attempts   # 2  (two extra inferences after the first)
result.turns                 # 3
len(result.inference_ids)    # 3
```

### Opt out for TensorZero-pure mode

To get byte-identical TensorZero behaviour — no in-Python recovery, no retries — set both knobs to off:

```python
config = AgentConfig(
    ...,
    output=JsonOutput(
        json_mode="strict",
        permissive=False,
        max_retries=0,
    ),
)
```

In that mode an invalid TZ response surfaces as `result.parsed is None` and `result.parse_strategy == "none"` on the first attempt — no in-Python fallback, no retry.

### The corrective hint

Every retry sends a corrective `user` message after the failed assistant turn.  The hint starts with a fixed base sentence (`"Your previous response was not valid JSON. Return ONLY a valid JSON object..."`) and, when an `output_schema` is in play, appends:

- The schema's top-level `title` when present (`"The schema is named 'Customer'."`).
- The top-level `required` field names when present (`"Required top-level fields: 'name', 'email'."`).

The hint is hard-coded English today; if you need a different wording, the cleanest path is to disable retries (`JsonOutput(max_retries=0)`) and build the loop in your application code.  An i18n / per-agent override hook is on the deferred-work list.

### Calling `permissive_parse` directly

The permissive parser is exported at the package root and is a pure-function utility — no TensorZero gateway required.  Use it when you have already obtained a raw model string from elsewhere and want to apply the same recovery ladder:

```python
from t0_agents import permissive_parse

value, strategy = permissive_parse('Sure! ```json\n{"a": 1}\n```')
# value    == {"a": 1}
# strategy == "permissive"
```

The function never raises — every JSON decode error along the recovery ladder is swallowed and the next layer is tried.

## Testing your JSON agent

`t0_agents.testing.MockGateway` covers the JSON path — the same gateway primitive that scripts chat responses (`script` / `canned`) also scripts JSON responses (`script_json` / `canned_json`).  No docker, no provider keys, no flaky LLM behaviour; unit tests for JSON agents run in milliseconds.

### Scripting a JSON response

```python
from typing import cast
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.testing import MockGateway

gateway = MockGateway.script_json([
    {"raw": '{"name": "Ada", "email": "ada@example.com"}'},
    # parsed is auto-derived from raw when the key is absent.  Pass
    # parsed=None explicitly to script a retry scenario where TZ
    # returned the JSON string but parsed=null.
])

agent = Agent(
    AgentConfig(
        name="extract",
        function_name="extract_customer_info",
        output=JsonOutput(json_mode="strict"),
    ),
    t0=cast(AsyncTensorZeroGateway, gateway),
)

result = await agent.invoke("Hi, I'm Ada at ada@example.com")
```

Dict entries accept the keys `raw`, `parsed`, `episode_id`, `inference_id`, `usage`, `variant_name`, and `finish_reason` — all optional except `raw`.  Pre-built `JsonInferenceResponse` instances bypass the dict path entirely (useful when a test needs to pin a specific `Usage` shape or re-use the same response across attempts).

### Scripting the retry ladder

Each list entry corresponds to one inference call; the retry orchestration in `Agent._run_json` pops them in order.  Pin `parsed: None` to script "TZ returned a JSON string but couldn't parse it" — the permissive layer kicks in upstream:

```python
gateway = MockGateway.script_json([
    {"raw": "totally not json", "parsed": None},      # attempt 0 — fails both layers
    {"raw": '{"a": 2}', "parsed": {"a": 2}},           # attempt 1 — TZ parses cleanly
])

result = await agent.invoke("hello")
# result.parse_strategy      == "retry"
# result.json_parse_attempts == 1
# len(result.inference_ids)  == 2
```

### JSON-aware assertion helpers

Five new helpers complement the chat-flavoured `assert_tool_called` / `assert_response_contains` family:

| Helper | What it pins |
|--------|--------------|
| `assert_parsed_equals(result, expected)` | `result.parsed == expected`; failure message includes `parse_strategy`. |
| `assert_parsed_matches_schema(result, schema)` | Validates `result.parsed` against a JSON Schema using `jsonschema`.  See note below. |
| `assert_invalid_json(result)` | `result.parsed is None` AND `result.parse_strategy == "none"`. |
| `assert_parse_strategy(result, strategy)` | `result.parse_strategy == strategy` (one of `"strict"`, `"permissive"`, `"retry"`, `"none"`). |
| `assert_retry_attempts(result, n)` | `result.json_parse_attempts == n`. |

```python
from t0_agents.testing import (
    assert_parse_strategy,
    assert_parsed_equals,
    assert_retry_attempts,
)

assert_parsed_equals(result, {"a": 2})
assert_parse_strategy(result, "retry")
assert_retry_attempts(result, 1)
```

`assert_parsed_matches_schema` is the one helper that requires an optional dependency: `jsonschema`.  It's imported lazily inside the helper, so the rest of the testing surface stays usable without it.  When called without `jsonschema` installed the helper raises `ImportError` with the install hint `pip install jsonschema`.  If your project already depends on Pydantic, the [Pydantic recipe](#schema-validation-in-python) above achieves the same result with one fewer dependency.

## Safety & provider quirks

JSON functions sit at the intersection of three contracts (TensorZero, the model provider, and your application) and several combinations are either explicitly forbidden by the gateway, silently degrade on certain providers, or are conceptually impossible to support today.  This section documents what t0-agents catches at runtime, what it warns about, and what it cannot detect.

### Configuration errors

These combinations raise `ConfigurationError`.  Most are caught at `AgentConfig` construction; two are caught lazily on the first `Agent.invoke(...)` because the check requires async I/O.

| Combination | Resolution |
|-------------|------------|
| `output=ChatOutput()` (default) paired with a per-call `output=JsonOutput(...)` (or vice versa) | Match the modes — either keep the chat agent and stop forcing single-shot JSON per call, or build a JSON agent on `AgentConfig(output=JsonOutput(...))` to begin with. |
| `output=JsonOutput(...)` (or `StructuredOutput(...)`) with `sub_agents` non-empty | Build a chat orchestrator and call the JSON agent as a downstream step. |
| `output=JsonOutput(...)` with `memory_strategy="messages"` | Use `"prepended_message"` (default) or `"system_template"` — JSON functions are single-shot and do not consume `prior_messages`. |
| `JsonOutput(max_retries=N)` with `N < 0` | Use `0` to disable retry, positive values to allow it. |
| `JsonOutput(json_mode="tool")` with a tool provider that advertises tools | Switch to `"strict"`/`"on"`/`"off"`, or pass an empty tool provider (TZ guide §11.1 — `"tool"` mode cannot share a request with other tools). Verified on first `Agent.invoke(...)` because `list_tools()` is async. |
| `run.events()` on a JSON-flavoured agent | Use `await agent.invoke(...)`.  TensorZero does not emit `output.parsed` mid-stream (TZ guide §7.1, §11.3); the guard fires at first `__anext__` before any event is yielded. |

### Why you cannot stream JSON functions

TensorZero's streaming response for a JSON function only carries `raw` deltas — there is no `parsed` field per chunk and the gateway does not emit one on the terminal chunk either (TZ guide §7.1: *"The TensorZero Gateway does **not** provide a `parsed` field for streaming JSON inferences."*).  Surfacing this through `Run.events(...)` would yield a stream of `AgentEvent`s whose `parsed` is permanently `None`, which is a strictly worse contract than refusing.

If you have a workload that genuinely needs incremental JSON, the recommended pattern is to call `Agent.invoke(...)` and present the JSON object on the consumer side once it lands.  Reusing the [`permissive_parse`](#calling-permissive_parse-directly) layer over reassembled `raw` deltas is on the deferred-work list; the current refusal is by design.

### Anthropic extended-thinking warning

TensorZero guide §11.4 documents that Anthropic's extended-thinking models reject every `json_mode` value except `"strict"` (direct Anthropic) and `"off"`.  The Python SDK does not expose provider routing, so t0-agents cannot enforce this at runtime — but it surfaces a generic `WARNING` on the `t0_agents.core.agent` logger the first time `Agent.invoke(...)` fires on an agent whose driving `OutputMode` carries `json_mode in {"on", "tool"}`:

```
OutputMode(json_mode='on') — Anthropic extended-thinking models reject this
combination (TensorZero guide §11.4).  Verify your provider routing.
```

The warning fires at most **once per Agent instance**, so a long-lived agent does not generate log spam.  Modes `"strict"` and `"off"` are safe across every provider and do not emit a warning.  Resolution options:

- Pin a non-Anthropic-extended-thinking variant in your `tensorzero.toml`.
- Switch the agent's `JsonOutput(json_mode=...)` to `"strict"` or `"off"`.
- Suppress the warning at the logger level (`logging.getLogger("t0_agents.core.agent").setLevel(logging.ERROR)`) if you have verified your routing.

### Provider quirks t0-agents cannot detect

A handful of provider-specific behaviours are documented in the TZ guide but cannot be enforced from the Python SDK.  These will not raise or warn; consult the table when picking a `json_mode`.

| Provider / route | Behaviour | Reference |
|------------------|-----------|-----------|
| Anthropic via AWS Bedrock or GCP Vertex | `json_mode="strict"` **silently degrades** to prompt-based JSON mode. | TZ guide §11.5 |
| Gemini (Vertex / AI Studio) | Supports a subset of JSON Schema only; certain schemas are rejected by the model. | TZ guide §11.6 |
| OpenAI-compat endpoint | `response_format` is a partial no-op; use `tensorzero::params.chat_completion.json_mode` instead. | TZ guide §11.7 |
| DeepSeek (Chat Completions) | `response_format` accepts only `text` and `json_object`, so `json_mode="strict"` **warns and downgrades**.  `json_object` carries a prompt precondition — see below. | [DeepSeek docs](https://api-docs.deepseek.com/guides/json_mode) |

See [the gateway guide §8](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#handle-model-provider-limitations) for the authoritative provider matrix.

### DeepSeek: `json_object` has a prompt precondition

DeepSeek's Chat Completions endpoint has no schema-enforced mode at all — `response_format` accepts only `text` and `json_object` — so `json_mode="strict"` can only warn and downgrade.  `"on"` is not a workaround there; it is the only mode the endpoint offers.

Two consequences never surface as parse failures, so the [recovery ladder](#robust-json-parsing) cannot see them:

- **The prompt must contain the word "json" and an example of the shape.**  Without it, *"the model may generate an unending stream of whitespace until the generation reaches the token limit"* — a hung request that burns `max_tokens`, not a parse error.  A docstring that already asks for a JSON object and shows the keys satisfies this; a terse one does not.
- **The API may return empty content**, and content is cut off mid-JSON when `finish_reason == "length"`.  Size `max_tokens` against the schema's worst case.

Schema enforcement does exist on DeepSeek, on two routes the `deepseek` provider does not take: the Responses API (`response_format={"type": "json_schema", ...}`) and per-function `"strict": true` tool calls against the Beta base URL.

Source: [JSON output](https://api-docs.deepseek.com/guides/json_mode) and [create chat completion](https://api-docs.deepseek.com/api/create-chat-completion).

### Opt-in hard failure: `require_parsed()`

By default, an unrecoverable JSON output surfaces as a runtime non-error: `result.parsed is None` and `result.parse_strategy == "none"`.  For applications where this is unacceptable — interactive workflows that must show the user structured data, batch jobs that should fail loudly rather than persist garbage — add the `require_parsed()` guardrail:

```python
from t0_agents import Agent, AgentConfig, JsonOutput
from t0_agents.guardrails import require_parsed

config = AgentConfig(
    name="extractor",
    function_name="extract_customer_info",
    output=JsonOutput(json_mode="strict"),
    guardrails=[require_parsed()],
)
```

When the run completes with `parsed is None` (every layer failed), the guardrail's `after_run` hook raises `GuardrailBlockedError` and the exception propagates out of `Agent.invoke(...)`.  Chat runs are a strict no-op: `parsed` is structurally absent on the chat path, so the guardrail short-circuits and never aborts a text run.  See [`docs/guardrails.md`](guardrails.md#require_parsed) for the full reference.

## `json_mode` provider matrix

| Mode       | What it does                                                                       | Best for                                                       |
| ---------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `off`      | No special JSON handling (**not recommended** per TZ docs)                         | Debugging                                                      |
| `on`       | Provider JSON mode if available                                                    | Providers that have a non-strict JSON-only mode, e.g. DeepSeek |
| `strict`   | Strict JSON / Structured Outputs (OpenAI uses Structured Outputs; direct Anthropic uses structured outputs) | OpenAI, direct Anthropic                                       |
| `tool`     | TensorZero converts your schema to a tool call under the hood and unwraps the response back to JSON | Providers without native strict, e.g. AWS Bedrock              |

Forbidden combinations:

- `json_mode` on a `chat` function (any value) — configuration error.
- `json_mode = "tool"` + any other tool in the same request — not allowed.
- `json_mode` other than `"strict"` (direct Anthropic) or `"off"` on Anthropic extended-thinking models.

See [the gateway guide §8](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#handle-model-provider-limitations) for the authoritative provider matrix.

## Deferred work

Open follow-up candidates:

- Streaming JSON with the permissive layer buffering reassembled `raw` deltas (today: `Run.events(...)` raises on JSON agents).
- A `require_schema(model_or_schema)` built-in guardrail that runs Pydantic / jsonschema validation in the `after_run` hook.
- An i18n / per-call hook for the corrective retry hint phrasing.

## See also

- [Agent](agent.md) — declarative front door; carries the `output=JsonOutput(...)` / `output=StructuredOutput(...)` dispatch and the JSON-aware `AgentResult` properties (`function_kind`, `parsed`, `parse_strategy`, `json_parse_attempts`).
- [Concepts](concepts.md) — overview of TensorZero's chat-vs-json distinction.
- [Guardrails](guardrails.md#require_parsed) — `require_parsed()` built-in for callers that want a hard abort on `parsed is None`.
- [Memory](memory.md) — `MemoryRecord.agent_response` carries the raw JSON string for JSON runs; the inference id list captures every retry attempt.
- [Streaming](streaming.md) — chat-only.  `Run.events(...)` on a JSON agent raises `ConfigurationError`; see [Why you cannot stream JSON functions](#why-you-cannot-stream-json-functions) above.
- [Testing](testing.md#testing-json-agents) — `MockGateway.script_json` / `canned_json` and the five JSON-aware assert helpers.
- [`examples/agent/03_output_modes/`](../examples/agent/03_output_modes/) — five runnable demos: single-shot helper, `Agent` dispatch, local and real-provider recovery ladder, dynamic-schema NER.
- [TensorZero structured-outputs guide](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) — authoritative source for `output_schema`, `json_mode`, `response_format`, and provider quirks.
