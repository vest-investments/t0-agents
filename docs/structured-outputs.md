# Structured Outputs

TensorZero distinguishes two function types — `chat` and `json`. This page documents how t0-agents supports the `json` flavour and what each upcoming milestone adds to the surface.

> **Milestone 1 (M1):** ships the single-shot `run_structured_inference()` helper, the new `AgentConfig` fields (`function_kind`, `output_schema`, `json_mode`), and the new `AgentResult` fields (`function_kind`, `parsed`).  `Agent.run(...)` dispatch lands in M2; permissive parsing and the retry loop land in M3; provider-safety warnings and the `Agent.stream(...)` guard land in M4; testing helpers and the dynamic-schema NER example land in M5.

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

The structured-outputs roadmap ships in five milestones; M1 closes only the first.

- **M2** — `Agent.run(...)` dispatches `function_kind == "json"` agents to `run_structured_inference()` transparently and surfaces per-call `output_schema` / `json_mode` kwargs.
- **M3** — adds `permissive_json_parsing` (in-Python recovery when the gateway returns `parsed == None`) and `max_json_retries` (re-emit the inference with a corrective hint).  `parse_strategy` will report which layer produced the parse.
- **M4** — adds the `Agent.stream(...) + function_kind == "json"` guard (TZ does not emit `parsed` for streaming JSON), the opt-in `require_parsed()` guardrail, and provider-safety warnings.
- **M5** — adds `MockGateway` support for JSON shapes, the `assert_parsed_*` testing helpers, and the dynamic-schema NER example.

## See also

- [Agent](agent.md) — declarative front door; will gain `function_kind == "json"` dispatch in M2.
- [Concepts](concepts.md) — overview of TensorZero's chat-vs-json distinction.
- [TensorZero structured-outputs guide](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) — authoritative source for `output_schema`, `json_mode`, `response_format`, and provider quirks.
