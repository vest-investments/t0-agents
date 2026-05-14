# Structured Outputs

TensorZero distinguishes two function types — `chat` and `json`. This page documents how t0-agents supports the `json` flavour and what each upcoming milestone adds to the surface.

> **Milestones 1 and 2 (M1, M2):** M1 shipped the single-shot `run_structured_inference()` helper, the new `AgentConfig` fields (`function_kind`, `output_schema`, `json_mode`), and the new `AgentResult` fields (`function_kind`, `parsed`).  M2 wires `Agent.run(...)` to dispatch JSON functions through the helper transparently and surfaces per-call `output_schema` / `json_mode` kwargs.  Permissive parsing and the retry loop land in M3; provider-safety warnings and the `Agent.stream(...)` guard land in M4; testing helpers and the dynamic-schema NER example land in M5.

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

See `examples/17_structured/` for a runnable copy.

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

`AgentConfig.output_schema` (used in M2 once `Agent.run` dispatch lands) also accepts a `pathlib.Path`. The file is read as JSON at construction time:

- Absolute paths are loaded verbatim.
- Relative paths are resolved against the current working directory.

> **Recommendation:** prefer absolute paths in production code.  Relative paths feel natural in tests and one-off scripts but can surprise downstream callers when the working directory drifts.

### Schema validation in Python

`parsed` may be valid JSON without being schema-conforming (e.g. missing a required field).  t0-agents does **not** run JSON-Schema validation in Python — TensorZero already does that on its side when the provider supports it, and adding `jsonschema` as a dependency was deliberately ruled out (see the integration contract).  If you need strict schema conformance, validate with [Pydantic](https://docs.pydantic.dev/) on the consumer side:

```python
from pydantic import BaseModel

class Customer(BaseModel):
    name: str | None
    email: str | None

result = await run_structured_inference(...)
customer = Customer.model_validate(result.parsed)
```

## Structured `Agent`

The high-level `Agent` wrapper recognises `function_kind="json"` and dispatches to `run_structured_inference()` automatically — no manual branching required. Build the same `AgentConfig` you would for a chat agent, set the two new fields, and call `agent.run(...)` exactly as before.

```python
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway
from t0_agents import Agent, AgentConfig

config = AgentConfig(
    name="customer-extractor",
    function_name="extract_customer_info",
    function_kind="json",
    json_mode="strict",
)

async with await AsyncTensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0, Client(FastMCP("noop")) as mcp_client:
    agent = Agent(config, t0=t0, tool_provider=mcp_client)
    result = await agent.run("Hi, I'm Sarah at sarah@example.com")

# result.function_kind  — "json"
# result.parsed         — {"name": "Sarah", "email": "sarah@example.com"}
# result.response       — '{"name":"Sarah","email":"sarah@example.com"}'
# result.turns          — 1   (JSON is single-shot)
# result.inference_ids  — [<one uuid>]
```

JSON functions never call MCP tools, but `Agent` still requires a `tool_provider` for its constructor — pass any `ToolProvider` (e.g. an empty in-process FastMCP `Client`, as in the snippet above). For a runnable copy see `examples/17_structured/structured_agent.py`.

### Per-call overrides

Both `output_schema` and `json_mode` can be overridden for a single call. Per-call values win over the `AgentConfig` fields:

```python
result = await agent.run(
    "...",
    output_schema={"type": "object", "properties": {"email": {"type": "string"}}, ...},
    json_mode="tool",  # overrides config's "strict"
)
```

Passing either kwarg on a chat-flavoured agent (`function_kind == "chat"`) raises `ConfigurationError` — the kwargs are specifically scoped to JSON functions.

### How `Agent.run` dispatches

```
agent.run(message, ...)
        │
        ▼
  function_kind == "json"?
        │
   ┌────┴────┐
   ▼         ▼
  yes        no
   │         │
   │         ├─► output_schema/json_mode set?  ─yes─► ConfigurationError
   │         │
   │         └─► run_agent (chat loop, byte-identical to v0.9.0)
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

The `memory_strategy="messages"` mode is **not yet supported** for JSON functions — `Agent.run` raises `ConfigurationError` at runtime whenever a JSON-flavoured agent is configured with that strategy, regardless of whether a prior history exists. Use `"prepended_message"` (the default) or `"system_template"` instead.

## `json_mode` provider matrix

| Mode       | What it does                                                                       | Best for                                                       |
| ---------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `off`      | No special JSON handling (**not recommended** per TZ docs)                         | Debugging                                                      |
| `on`       | Provider JSON mode if available                                                    | Providers that have a non-strict JSON-only mode                |
| `strict`   | Strict JSON / Structured Outputs (OpenAI uses Structured Outputs; direct Anthropic uses structured outputs) | OpenAI, direct Anthropic                                       |
| `tool`     | TensorZero converts your schema to a tool call under the hood and unwraps the response back to JSON | Providers without native strict, e.g. AWS Bedrock              |

Forbidden combinations:

- `json_mode` on a `chat` function (any value) — configuration error.
- `json_mode = "tool"` + any other tool in the same request — not allowed.
- `json_mode` other than `"strict"` (direct Anthropic) or `"off"` on Anthropic extended-thinking models.

See [the gateway guide §8](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#handle-model-provider-limitations) for the authoritative provider matrix.

## What's coming

The structured-outputs roadmap ships in five milestones; M1 and M2 are closed.

- **M3** — adds `permissive_json_parsing` (in-Python recovery when the gateway returns `parsed == None`) and `max_json_retries` (re-emit the inference with a corrective hint).  `parse_strategy` will report which layer produced the parse.
- **M4** — adds the `Agent.stream(...) + function_kind == "json"` guard (TZ does not emit `parsed` for streaming JSON), the opt-in `require_parsed()` guardrail, and provider-safety warnings.
- **M5** — adds `MockGateway` support for JSON shapes, the `assert_parsed_*` testing helpers, and the dynamic-schema NER example.

## See also

- [Agent](agent.md) — declarative front door; gains `function_kind == "json"` dispatch (since v0.10.0-M2).
- [Concepts](concepts.md) — overview of TensorZero's chat-vs-json distinction.
- [TensorZero structured-outputs guide](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) — authoritative source for `output_schema`, `json_mode`, `response_format`, and provider quirks.
