# TensorZero — JSON Mode & Structured Outputs: Comprehensive Implementation Guide

> A documentation-grounded reference for engineers implementing structured-output workflows on TensorZero. Every claim, snippet, parameter, and limitation in this guide is traceable to the project documentation files cited at the end of each section. Where documentation is silent, this guide says so explicitly rather than guessing.

---

## Table of Contents

1. [Mental model: `chat` functions vs. `json` functions](#1-mental-model-chat-functions-vs-json-functions)
2. [The four `json_mode` strategies](#2-the-four-json_mode-strategies)
3. [Configuring a static schema (the recommended path)](#3-configuring-a-static-schema-the-recommended-path)
4. [Dynamic schemas at inference time](#4-dynamic-schemas-at-inference-time)
5. [Setting / overriding `json_mode` at inference time](#5-setting--overriding-json_mode-at-inference-time)
6. [Response shapes: native API vs OpenAI-compatible](#6-response-shapes-native-api-vs-openai-compatible)
7. [Streaming JSON outputs](#7-streaming-json-outputs)
8. [Provider-specific limitations](#8-provider-specific-limitations)
9. [Batch inference with JSON functions](#9-batch-inference-with-json-functions)
10. [Datasets & datapoints for JSON functions](#10-datasets--datapoints-for-json-functions)
11. [Pitfalls, "gotchas", and undocumented edges](#11-pitfalls-gotchas-and-undocumented-edges)
12. [Two end-to-end recipes](#12-two-end-to-end-recipes)
13. [Quick reference cheat sheet](#13-quick-reference-cheat-sheet)
14. [Sources used](#14-sources-used)

---

## 1. Mental model: `chat` functions vs. `json` functions

TensorZero Functions come in two flavors:

- **`chat`** — the default choice for most LLM chat completion use cases. It returns unstructured text responses.
- **`json`** — a specialized function type when your goal is generating structured outputs that must conform to a JSON Schema.

**Rule of thumb from the docs:** use JSON functions if you have a **single, well-defined output schema**. If you need more flexibility (e.g., letting the model pick between multiple tools, or whether to pick a tool at all), then Chat Functions with **tool use** might be a better fit.

The skeleton of a function configuration:

```toml
[functions.my_function_name]
type = "..." # "chat" or "json"
# ... other fields depend on the function type ...
```

A variant is a particular implementation of a function. The simplest variant type is **`chat_completion`**, which is the typical chat completion format used by OpenAI and many other LLM providers.

```toml
[functions.my_function_name.variants.my_variant_name]
type = "..." # e.g. "chat_completion"
model = "..." # e.g. "openai::gpt-5" or "my_gpt_5"
# ... other fields (e.g. prompt templates, inference parameters) ...
```

> Sources: `gateway-generate-structured-outputs.md` > "How to generate structured outputs"; `gateway-configure-functions-and-variants.md`.

---

## 2. The four `json_mode` strategies

`json_mode` is a **variant-level** field that defines the strategy for generating JSON outputs.

- **Type:** string
- **Required:** **yes for `json` functions, forbidden for `chat` functions.**

The supported modes (verbatim from the configuration reference):

- **`off`** — Make a chat completion request without any special JSON handling (**not recommended**).
- **`on`** — Make a chat completion request with JSON mode (if supported by the provider).
- **`strict`** — Make a chat completion request with strict JSON mode (if supported by the provider). For example, the TensorZero Gateway uses Structured Outputs for OpenAI.
- **`tool`** — Make a special-purpose tool use request under the hood, and convert the tool call into a JSON response.

Configured in the variant section:

```toml
[functions.draft-email.variants.prompt-v1]
# ...
json_mode = "strict"
# ...
```

**Practical guidance (also verbatim):** Use `"strict"` mode for providers that support it (e.g., OpenAI) or `"tool"` for others.

### How `tool` mode actually works under the hood

> For providers without native support [for strict schema enforcement], you can still generate structured outputs with `json_mode = "tool"`. TensorZero converts your output schema into a tool call, then transforms the tool response back into JSON output.

**Critical pitfall:**

> When using `json_mode = "tool"`, you cannot use other tools in the same inference request.

> Sources: `gateway-configuration-reference.md` > `json_mode` (under variants); `gateway-generate-structured-outputs.md` > "Lack of native support (e.g. AWS Bedrock)".

---

## 3. Configuring a static schema (the recommended path)

The docs explicitly recommend specifying a **fixed schema in the configuration whenever possible**, only resorting to dynamic schemas if the use case demands it.

### 3.1 Function-level field: `output_schema`

Under `type = "json"`:

> **`output_schema`**
> - **Type:** string (path)
> - **Required:** no (default: `{}`, the empty JSON schema that accepts any valid JSON output)
>
> Defines the path to the output schema file, which should contain a JSON Schema for the output of the function. The path is relative to the configuration file. This schema is used for validating the output of the function.

```toml
[functions.extract-customer-info]
# ...
type = "json"
output_schema = "./functions/extract-customer-info/output_schema.json"
# ...
```

> If you don't specify an `output_schema`, the gateway will default to accepting any valid JSON output.

### 3.2 Minimal end-to-end example (verbatim from the guide)

**`tensorzero.toml`:**

```toml
[functions.extract_data]
type = "json"
output_schema = "output_schema.json"  # optional

[functions.extract_data.variants.baseline]
type = "chat_completion"
model = "openai::gpt-5-mini"
system_template = "system_template.minijinja"
json_mode = "strict"
```

**`output_schema.json`:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": ["string", "null"],
      "description": "The customer's full name"
    },
    "email": {
      "type": ["string", "null"],
      "description": "The customer's email address"
    }
  },
  "required": ["name", "email"],
  "additionalProperties": false
}
```

**`system_template.minijinja`:**

```txt
You are a helpful AI assistant that extracts customer information from messages.

Extract the customer's name and email address if present. Use null for any fields that are not found.

Your output should be a JSON object with the following schema:

{
  "name": string or null,
  "email": string or null
}

---

Examples:

User: Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com
Assistant: {"name": "Sarah Johnson", "email": "sarah.j@example.com"}

User: My email is contact@company.com
Assistant: {"name": null, "email": "contact@company.com"}

User: This is John Doe reaching out
Assistant: {"name": "John Doe", "email": null}
```

> Tip from the docs: Including examples in your prompt helps the model understand the expected output format and improves accuracy.

### 3.3 Calling the function (OpenAI-compatible endpoint)

**Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="tensorzero::function_name::extract_data",
    messages=[
        {
            "role": "user",
            "content": "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
        }
    ],
)
```

Sample response (the `content` is the JSON string):

```python
ChatCompletion(
    id='019a78dd-8e77-7c21-ab70-5dc4b897f7d2',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='{"name":"Sarah Johnson","email":"sarah.j@example.com"}',
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=None
            )
        )
    ],
    created=1762964379,
    model='tensorzero::function_name::extract_data::variant_name::baseline',
    object='chat.completion',
    service_tier=None,
    system_fingerprint='',
    usage=CompletionUsage(
        completion_tokens=90,
        prompt_tokens=252,
        total_tokens=342,
        completion_tokens_details=None,
        prompt_tokens_details=None
    ),
    episode_id='019a78dd-8e77-7c21-ab70-5ddb585eb35e'
)
```

**HTTP:**

```bash
curl -X POST "http://localhost:3000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::extract_data",
    "messages": [
      {
        "role": "user",
        "content": "Hi, I'\''m Sarah Johnson and you can reach me at sarah.j@example.com"
      }
    ]
  }'
```

### 3.4 Calling via the **native** TensorZero API

The native API uses `client.inference(...)` or `POST /inference`, and the JSON response has the shape `output.raw` + `output.parsed` (see §6 for details).

**Python (native client):**

```python
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="extract_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [
                {
                    "role": "user",
                    "content": "...blah blah blah hello@tensorzero.com blah blah blah..."
                }
            ]
        }
        # optional: stream=True,
    )
```

**HTTP:**

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "extract_email",
    "input": {
      "system": "You are an AI assistant...",
      "messages": [
        {
          "role": "user",
          "content": "...blah blah blah hello@tensorzero.com blah blah blah..."
        }
      ]
    }
    // optional: "stream": true
  }'
```

> Sources: `gateway-generate-structured-outputs.md` > "Generate structured outputs with a static schema"; `gateway-configuration-reference.md` > `output_schema` (under `type: "json"`); `gateway-api-reference.md` > "JSON Function" example.

---

## 4. Dynamic schemas at inference time

> While we recommend specifying a fixed schema in the configuration whenever possible, you can provide the output schema dynamically at inference time if your use case demands it.

You can override the static schema two different ways depending on which endpoint you use.

### 4.1 Native API: the `output_schema` request field

From `gateway-api-reference.md` > native `/inference` request:

> **`output_schema`**
> - **Type:** object (valid JSON Schema)
> - **Required:** no
>
> If set, this schema will override the `output_schema` defined in the function configuration for a JSON function. This dynamic output schema is used for validating the output of the function, and sent to providers which support structured outputs.

### 4.2 OpenAI-compatible API: the `response_format` field

From `gateway-api-reference.md` > OpenAI-compatible request:

> **`response_format`**
> - **Type:** either a string or an object
> - **Required:** no (default: `null`)
>
> Options here are `"text"`, `"json_object"`, and `"{"type": "json_schema", "schema": ...}"`, where the schema field contains a valid JSON schema. This field is not actually respected except for the `"json_schema"` variant, in which the `schema` field can be used to dynamically set the output schema for a `json` function.

**Pitfall to memorize:** only the `"json_schema"` variant is actually respected. Passing `"text"` or `"json_object"` will not produce the effect you might expect from raw OpenAI semantics — those values are silently inactive in TensorZero.

### 4.3 Example: dynamic schema via OpenAI-compatible Python SDK

Base configuration (the static schema is overridden at inference time):

```toml
# ...
[functions.extract_email]
type = "json"
output_schema = "output_schema.json"
# ...
```

```json
// output_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "email": {
      "type": "string"
    }
  },
  "required": ["email"]
}
```

Override at request time to also extract `domain`:

```python
from openai import AsyncOpenAI

dynamic_output_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "email": { "type": "string" },
    "domain": { "type": "string" }
  },
  "required": ["email", "domain"]
}

async with AsyncOpenAI(
    base_url="http://localhost:3000/openai/v1"
) as client:
    result = await client.chat.completions.create(
        model="tensorzero::function_name::extract_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [
                {
                    "role": "user",
                    "content": "...blah blah blah hello@tensorzero.com blah blah blah..."
                }
            ]
        }
        # Override the output schema using the `response_format` field
        response_format={"type": "json_schema", "schema": dynamic_output_schema}
        # optional: stream=True,
    )
```

Equivalent HTTP:

```bash
curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::extract_email",
    "input": {
      "system": "You are an AI assistant...",
      "messages": [
        {
          "role": "user",
          "content": "...blah blah blah hello@tensorzero.com blah blah blah..."
        }
      ]
    },
    "response_format": {
      "type": "json_schema",
      "schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
          "email": { "type": "string" },
          "domain": { "type": "string" }
        },
        "required": ["email", "domain"]
      }
    }
  }'
```

Sample response (OpenAI-compatible regular mode):

```json
{
  "id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "model": "extract_email_variant",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "content": "{\"email\": \"hello@tensorzero.com\", \"domain\": \"tensorzero.com\"}"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 100,
    "total_tokens": 200,
    "tensorzero_cost": 0.0003
  }
}
```

> Sources: `gateway-generate-structured-outputs.md` > "Generate structured outputs with a dynamic schema"; `gateway-api-reference.md` > native `output_schema` field; `gateway-api-reference.md` > OpenAI-compatible `response_format` field; `gateway-api-reference.md` > "JSON Function with Dynamic Output Schema" example.

---

## 5. Setting / overriding `json_mode` at inference time

You can set `json_mode` for a particular request using `params`. This value **takes precedence over any default behaviors or `json_mode` in the configuration.**

The accepted values at inference time are: `"strict"`, `"tool"`, `"on"`, `"off"`.

### 5.1 Python (OpenAI-compatible)

```python
response = client.chat.completions.create(
    # ...
    extra_body={
        "tensorzero::params": {
            "chat_completion": {
                "json_mode": "strict",  # or: "tool", "on", "off"
            }
        }
    }
    # ...
)
```

### 5.2 Node (OpenAI-compatible)

```ts
const response = await client.chat.completions.create({
  // ...
  "tensorzero::params": {
    chat_completion: {
      json_mode: "strict", // or: "tool", "on", "off"
    },
  },
  // ...
});
```

### 5.3 HTTP

```bash
curl -X POST "http://localhost:3000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    // ...
    "tensorzero::params": {
      "chat_completion": {
        "json_mode": "strict", // or: "tool", "on", "off"
      }
    }
    // ...
  }'
```

> Tip from the docs: Dynamic inference parameters like `json_mode` apply to specific variant types. Unless you're using an advanced variant type, the variant type will be `chat_completion`.

### 5.4 Precedence rule for OpenAI-compatible requests

From the OpenAI-compatible reference (the **`tensorzero::params`** section):

> When using the OpenAI-compatible endpoint, values specified in `tensorzero::params` take precedence over parameters provided directly in the request body (e.g., top-level `temperature`, `max_tokens`) or inferred from other fields (e.g., `json_mode` inferred from `response_format`).

So if you simultaneously pass a `response_format` and a `tensorzero::params.chat_completion.json_mode`, the `tensorzero::params` value wins.

### 5.5 Full list of parameters you can override in `tensorzero::params.chat_completion`

- `frequency_penalty` (float)
- `json_mode` (object)
- `max_tokens` (integer)
- `presence_penalty` (float)
- `reasoning_effort` (string)
- `seed` (integer)
- `service_tier` (string)
- `stop_sequences` (list of strings)
- `temperature` (float)
- `thinking_budget_tokens` (integer)
- `top_p` (float)
- `verbosity` (string)

> Sources: `gateway-generate-structured-outputs.md` > "Set `json_mode` at inference time"; `gateway-api-reference.md` > `tensorzero::params`.

---

## 6. Response shapes: native API vs OpenAI-compatible

This is one of the most important distinctions to internalize when implementing a client.

### 6.1 Native `/inference` response for a JSON function

The response (non-streaming) is a JSON object with these fields:

- **`inference_id`** (UUID): The ID assigned to the inference.
- **`episode_id`** (UUID): The ID of the episode associated with the inference.
- **`output`** (object): contains
  - **`raw`**: The raw response from the model provider (which might be invalid JSON).
  - **`parsed`**: The parsed response from the model provider (`null` if invalid JSON).
- **`variant_name`** (string): The name of the variant used for the inference.
- **`usage`** (object, optional): `input_tokens`, `output_tokens`, `cost` (dollars; may be `null`).
- **`raw_response`** (array, optional, only when `include_raw_response` is `true`): An array of raw provider-specific response data from all model inferences. Each entry contains `model_inference_id`, `provider_type`, `api_type` (one of `"chat_completions"`, `"responses"`, or `"embeddings"`), and `data` (raw response string).
- **`raw_usage`** (array, optional, only when `include_raw_usage` is `true`): Raw provider-specific usage data.

Concrete example:

```json
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "output": {
    "raw": "{\"email\": \"hello@tensorzero.com\"}",
    "parsed": {
      "email": "hello@tensorzero.com"
    }
  }
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100,
    "cost": 0.0003
  }
}
```

> The `parsed` field is `null` if the model produced invalid JSON. Always check for `null` before consuming it.

### 6.2 OpenAI-compatible `/openai/v1/chat/completions` response for a JSON function

The JSON is returned as a **string** in `choices[0].message.content`. The client must parse it.

```json
{
  "id": "019a78de-97d4-79d3-8b61-bcab4c697281",
  "episode_id": "019a78de-97d4-79d3-8b61-bcb10a8c02f4",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "content": "{\"name\":\"Sarah Johnson\",\"email\":\"sarah.j@example.com\"}",
        "tool_calls": null,
        "role": "assistant"
      }
    }
  ],
  "created": 1762964446,
  "model": "tensorzero::function_name::extract_data::variant_name::baseline",
  "system_fingerprint": "",
  "service_tier": null,
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 252,
    "completion_tokens": 26,
    "total_tokens": 278
  }
}
```

The `model` field reports both the function name and the chosen variant: `tensorzero::function_name::<fn>::variant_name::<variant>`.

### 6.3 Comparison table

| Aspect                          | Native API (`/inference`)                                | OpenAI-compatible (`/openai/v1/chat/completions`)                 |
| ------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| Where the JSON lives            | `output.raw` (string) and `output.parsed` (object/null)  | `choices[0].message.content` (JSON-encoded string)                |
| Parsing done for you            | Yes (`parsed`)                                           | No — client must parse                                            |
| Invalid JSON signal             | `parsed === null`                                        | Same string in `content`; client parse will fail                  |
| ID field                        | `inference_id`                                           | `id`                                                              |
| Episode ID                      | `episode_id`                                             | `episode_id`                                                      |
| Usage shape                     | `input_tokens`/`output_tokens`/`cost`                    | `prompt_tokens`/`completion_tokens`/`total_tokens`/`tensorzero_cost` |
| Dynamic schema field            | `output_schema` (request top-level)                      | `response_format` with `type: "json_schema"`                      |
| Unknown content blocks          | Preserved                                                | **Dropped + warning logged** (see §11)                            |

> Sources: `gateway-api-reference.md` > "JSON Function" Regular response; `gateway-generate-structured-outputs.md` sample responses; `gateway-api-reference.md` > OpenAI-compatible Response.

---

## 7. Streaming JSON outputs

You can stream by passing `stream=True` (Python) / `"stream": true` (HTTP).

### 7.1 Native streaming response

> In streaming mode, the response is an SSE stream of JSON messages, followed by a final `[DONE]` message.

Each chunk has these fields:

- `episode_id` (UUID)
- `inference_id` (UUID)
- `raw` (string) — The raw response delta from the model provider.
- `variant_name` (string)
- `usage` (object, optional)
- `raw_usage` (array, optional, only when `include_raw_usage` is `true`)

**Critical:**

> The TensorZero Gateway does **not** provide a `parsed` field for streaming JSON inferences. If your application depends on a well-formed JSON response, we recommend using regular (non-streaming) inference.

Example chunk:

```json
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "raw": "{\"email\":",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100,
    "cost": 0.0003
  }
}
```

### 7.2 OpenAI-compatible streaming response (JSON function)

Each chunk follows the OpenAI streaming shape; deltas of the JSON string land in `choices[0].message.content`:

```json
{
  "id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "model": "extract_email_variant",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "content": "{\"email\":"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 100,
    "total_tokens": 200,
    "tensorzero_cost": 0.0003
  }
}
```

You can optionally enable usage reporting in streaming mode:

```json
{
  "stream_options": {
    "include_usage": true
  }
}
```

> Sources: `gateway-api-reference.md` > "JSON Function" Streaming (native and OpenAI-compatible).

---

## 8. Provider-specific limitations

The docs are explicit that not all providers support every `json_mode`. The matrix below summarizes what is and isn't covered in the documentation.

### 8.1 OpenAI

- **`strict`**: TensorZero uses OpenAI's **Structured Outputs** feature, providing guaranteed schema compliance for supported OpenAI models.
- Strict is also leveraged for tool parameters when `tools.<tool>.strict = true`:
  > Only a few providers support strict JSON generation. For example, the TensorZero Gateway uses Structured Outputs for OpenAI. If the provider does not support strict mode, the TensorZero Gateway ignores this field.

### 8.2 Anthropic

> For the direct Anthropic provider, `json_mode = "strict"` automatically uses Anthropic's structured outputs feature for guaranteed schema compliance.
>
> AWS Bedrock and GCP Vertex AI do not support Anthropic's structured outputs, so `json_mode = "strict"` falls back to prompt-based JSON mode. Use `json_mode = "tool"` for more reliable schema compliance on these providers.

**Extended thinking models:**

> For Anthropic's extended thinking models, only `json_mode = "strict"` (direct Anthropic) or `json_mode = "off"` are compatible. Other modes use prefill or forced tool use, which conflict with thinking.

So with Anthropic extended-thinking models:

| Provider routing                    | Allowed `json_mode` values for thinking models                    |
| ----------------------------------- | ----------------------------------------------------------------- |
| Direct Anthropic                    | `"strict"` or `"off"`                                             |
| AWS Bedrock / GCP Vertex (Anthropic)| The docs only state that `strict` falls back to prompt-based; combined with the extended-thinking constraint, **the docs do not explicitly enumerate what's allowed for thinking models on those routes.** Verify before relying on a specific mode there. |

### 8.3 Gemini (GCP Vertex AI, Google AI Studio)

> GCP Vertex AI Gemini and Google AI Studio support structured outputs, but only support a **subset of the JSON Schema specification**. TensorZero automatically handles some known limitations, but certain output schemas will still be rejected by the model provider. Refer to the [Google documentation](https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#json_schema_support) for details on supported JSON Schema features.

### 8.4 AWS Bedrock (and other providers without native strict support)

> Some model providers (e.g. OpenAI, Google) support strictly enforcing output schemas natively, but others (e.g. AWS Bedrock) do not.
>
> For providers without native support, you can still generate structured outputs with `json_mode = "tool"`. TensorZero converts your output schema into a tool call, then transforms the tool response back into JSON output.

You can set `json_mode = "tool"` in your configuration file **or** at inference time.

> Sources: `gateway-generate-structured-outputs.md` > "Handle model provider limitations"; `gateway-configuration-reference.md` > `strict` (under tools).

---

## 9. Batch inference with JSON functions

Batch inference accepts a list of inputs and (for JSON functions) optionally a per-element list of schemas.

### 9.1 `output_schemas`

From `gateway-api-reference.md`:

> **`output_schemas`**
> - **Type:** list of optional objects (valid JSON Schema)
> - **Required:** no
>
> A list of JSON schemas that will be used to validate the output of the function for each inference in the batch. Each element in the list corresponds to a single inference in the batch. These can be `null` for elements that need to use the `output_schema` defined in the function configuration. This schema is used for validating the output of the function, and sent to providers which support structured outputs.

### 9.2 `params` in batch mode

Each batch parameter under `params.chat_completion.<param>` must be **a list the same length as the batch size**, with `null` permitted for entries that should use the configuration value:

- `frequency_penalty`
- `json_mode`
- `max_tokens`
- `presence_penalty`
- `reasoning_effort`
- `seed`
- `service_tier`
- `stop_sequences`
- `temperature`
- `thinking_budget_tokens`
- `top_p`
- `verbosity`

> Note that the parameters will apply to every variant of the specified type.

### 9.3 Polling batch results

A completed batch returns objects that "exactly match the response body in the inference endpoint" — i.e., the per-inference JSON Function response shape from §6.1 applies inside the `inferences` array. While pending: `{"status": "pending"}`. On failure: `{"status": "failed"}`. You can also poll a single inference at `GET /batch_inference/:batch_id/inference/:inference_id`.

> Sources: `gateway-api-reference.md` > batch inference `output_schemas`, batch inference `params`, and batch polling examples.

---

## 10. Datasets & datapoints for JSON functions

When you build a dataset (e.g., for SFT, DICL, evaluations, or GEPA), JSON datapoints use these fields:

For `json` datapoints:

- `function_name` (string, required)
- `input` (object, required, identical to an inference's `input`)
- `output` (object, optional, an object that matches the `output_schema` of the function)
- `output_schema` (object, optional, **a dynamic JSON schema that overrides the output schema of the function**)
- `episode_id` (UUID, optional)
- `tags` (map of string to string, optional)
- `name` (string, optional)

The `POST /v1/datasets/{dataset_name}/datapoints` endpoint (or `create_datapoints` on the client) returns an object with `ids`, a list of IDs (strings, UUIDv7) of the newly created datapoints. You can also create datapoints from existing inferences via `POST /v1/datasets/{dataset_name}/from_inferences`.

**Data model in storage (ClickHouse):**

For the `JsonInference` table, each row stores:

- `id` (UUID)
- `function_name`, `variant_name` (strings)
- `episode_id` (UUID)
- `input` (String/JSON) — identical to the `input` field of the `/inference` request body
- `output` (String/JSON) — **an object with `parsed` and `raw` fields**
- `output_schema` (String/JSON) — the schema the output must conform to
- `inference_params` (String/JSON) — e.g. `{"chat_completion": {"temperature": 0.5}}`
- `processing_time_ms`, `timestamp`, `tags`, `extra_body`, `auxiliary_content`, ...

The `JsonInferenceDatapoint` table mirrors this for dataset entries (with `dataset_name`, `is_deleted`, `staled_at`, `source_inference_id`, `is_custom`, `name`, `snapshot_hash`).

> Sources: `gateway-api-reference.md` > "Create datapoints" (json datapoint fields); `gateway-data-model.md` > `JsonInference` and `JsonInferenceDatapoint`.

---

## 11. Pitfalls, "gotchas", and undocumented edges

This is the most important section for an implementer. Every item below comes directly from a documented statement; nothing is inferred without flagging.

### 11.1 `json_mode = "tool"` blocks other tool calls

> When using `json_mode = "tool"`, you cannot use other tools in the same inference request.

This is hard-blocking: if you need a JSON function that also needs to call tools, you cannot use `tool` mode. Use `strict` (where supported) or pick a `chat` function with tools instead.

### 11.2 `json_mode` is **required for json**, **forbidden for chat**

From the configuration reference, under `json_mode`:

> **Required:** **yes for `json` functions, forbidden for `chat` functions**

Setting `json_mode` on a `chat` variant is a configuration error. Conversely, omitting it on a `json` variant is also an error.

### 11.3 No `parsed` field in streaming

> The TensorZero Gateway does not provide a `parsed` field for streaming JSON inferences. If your application depends on a well-formed JSON response, we recommend using regular (non-streaming) inference.

Reassemble `raw` deltas yourself and parse only once the stream ends — or just don't stream JSON functions.

### 11.4 Anthropic extended-thinking compatibility

> For Anthropic's extended thinking models, only `json_mode = "strict"` (direct Anthropic) or `json_mode = "off"` are compatible. Other modes use prefill or forced tool use, which conflict with thinking.

If you select `"on"` or `"tool"` on an Anthropic extended-thinking model, the docs flag this as incompatible.

### 11.5 Anthropic structured outputs aren't supported through Bedrock or Vertex

> AWS Bedrock and GCP Vertex AI do not support Anthropic's structured outputs, so `json_mode = "strict"` falls back to prompt-based JSON mode. Use `json_mode = "tool"` for more reliable schema compliance on these providers.

In other words: setting `strict` will silently degrade to prompt-only JSON on those routes; reach for `tool` if you actually need reliability.

### 11.6 Gemini supports only a subset of JSON Schema

> GCP Vertex AI Gemini and Google AI Studio support structured outputs, but only support a subset of the JSON Schema specification. TensorZero automatically handles some known limitations, but certain output schemas will still be rejected by the model provider.

Some schemas you can use with OpenAI strict mode will be rejected by Gemini.

### 11.7 `response_format` is partially a no-op

> Options here are `"text"`, `"json_object"`, and `"{"type": "json_schema", "schema": ...}"`, where the schema field contains a valid JSON schema. **This field is not actually respected except for the `"json_schema"` variant**, in which the `schema` field can be used to dynamically set the output schema for a `json` function.

Do **not** rely on `response_format: "json_object"` to enable JSON mode on the OpenAI-compatible endpoint — it's documented as inactive. Instead use `tensorzero::params.chat_completion.json_mode` or set `json_mode` on the variant configuration.

### 11.8 `tensorzero::params` overrides OpenAI-style fields

> When using the OpenAI-compatible endpoint, values specified in `tensorzero::params` take precedence over parameters provided directly in the request body (e.g., top-level `temperature`, `max_tokens`) or inferred from other fields (e.g., `json_mode` inferred from `response_format`).

A mix of top-level fields and `tensorzero::params` will not error, but `tensorzero::params` always wins. Pick one place to set each parameter.

### 11.9 Inference-time `json_mode` applies to a variant type, not a variant

> Dynamic inference parameters like `json_mode` apply to specific variant types. Unless you're using an advanced variant type, the variant type will be `chat_completion`.

So `tensorzero::params.chat_completion.json_mode` applies to **every chat-completion variant** of the function. If you mix `chat_completion` with `experimental_best_of_n` etc., understand which type is being targeted.

### 11.10 Empty default schema means "any valid JSON"

> default: `{}`, the empty JSON schema that accepts any valid JSON output

If you omit `output_schema`, TensorZero will not constrain the output beyond "must be valid JSON". You won't get structured-output enforcement at the provider level either, since there's no schema to send. For real validation, define a schema.

### 11.11 Native vs OpenAI-compatible response shape mismatch

If you write code expecting `output.parsed`, that works on `/inference` but **not** on `/openai/v1/chat/completions`. Conversely, OpenAI SDKs expect `choices[].message.content`. Pick one and be consistent across your codebase.

### 11.12 OpenAI-compatible endpoint drops unknown content blocks

> The OpenAI-compatible inference endpoint can't handle unknown content blocks in the response. If the model provider returns an unknown content block, the gateway will drop the content block from the response and log a warning.
>
> If you need to access unknown content blocks, use the native TensorZero API.

This is general (not JSON-specific), but matters because thinking/reasoning content blocks can be silently lost on the OpenAI-compatible endpoint.

### 11.13 The OpenAI-compatible endpoint has limited credentials handling

> Most notably, this endpoint doesn't support dynamic credentials, so they must be specified with a different method.

And from `gateway-api-reference.md`:

> The gateway will use the credentials specified in the `tensorzero.toml` file. In most cases, these credentials will be environment variables available to the TensorZero gateway — *not* your OpenAI client. API keys sent from the OpenAI client will be ignored.

### 11.14 `parsed` can be `null` even on a successful HTTP request

> `parsed`: The parsed response from the model provider (`null` if invalid JSON).

A 200 OK with `output.parsed === null` means the model produced something that didn't parse as JSON. Code defensively for this.

### 11.15 Output validation behavior on schema mismatch is not explicitly documented

The configuration reference says `output_schema` is "used for validating the output of the function." However, the docs do **not** explicitly describe what happens when the model produces structurally valid JSON that nevertheless fails the schema (e.g., missing a required field). This is not covered in the documentation I have access to — verify against the latest TensorZero docs before relying on a specific failure mode.

> Sources: cited inline above.

---

## 12. Two end-to-end recipes

### 12.1 Recipe A — Customer information extractor with strict OpenAI structured outputs

**Goal:** extract `name` and `email` from a freeform user message, guaranteed to comply with a schema.

**Directory layout:**

```
config/
├── tensorzero.toml
└── functions/
    └── extract_customer_info/
        ├── output_schema.json
        └── baseline/
            └── system_template.minijinja
```

**`config/tensorzero.toml`:**

```toml
[functions.extract_customer_info]
type = "json"
output_schema = "./functions/extract_customer_info/output_schema.json"

[functions.extract_customer_info.variants.baseline]
type = "chat_completion"
model = "openai::gpt-5-mini"
system_template = "./functions/extract_customer_info/baseline/system_template.minijinja"
json_mode = "strict"
```

**`output_schema.json`:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name":  { "type": ["string", "null"], "description": "The customer's full name" },
    "email": { "type": ["string", "null"], "description": "The customer's email address" }
  },
  "required": ["name", "email"],
  "additionalProperties": false
}
```

**`system_template.minijinja`:**

```
You are a helpful AI assistant that extracts customer information from messages.

Extract the customer's name and email address if present. Use null for any fields that are not found.

Your output should be a JSON object with the following schema:

{
  "name": string or null,
  "email": string or null
}
```

**Client (native API):**

```python
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="extract_customer_info",
        input={
            "messages": [
                {"role": "user", "content": "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com"}
            ]
        }
    )
    # result.output.parsed -> {'name': 'Sarah Johnson', 'email': 'sarah.j@example.com'}
    # result.output.raw    -> '{"name":"Sarah Johnson","email":"sarah.j@example.com"}'
```

**Client (OpenAI-compatible):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="unused")

resp = client.chat.completions.create(
    model="tensorzero::function_name::extract_customer_info",
    messages=[
        {"role": "user", "content": "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com"}
    ],
)
# resp.choices[0].message.content is a JSON string: '{"name":"Sarah Johnson","email":"sarah.j@example.com"}'
```

### 12.2 Recipe B — Named-Entity-Recognition with a JSON function (verbatim style from the optimization guide)

This pattern is used as the canonical example for SFT / DICL / GEPA workflows in `optimization.md`.

**Configuration:**

```toml
[functions.extract_entities]
type = "json"
output_schema = "functions/extract_entities/output_schema.json"

[functions.extract_entities.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
templates.system.path = "functions/extract_entities/initial_prompt/system_template.minijinja"
json_mode = "strict"
```

> Note: this example uses the **newer** `templates.system.path` form rather than the legacy `system_template`. Both forms appear across the documentation. The migration table from `gateway-create-a-prompt-template.md`:
>
> | Legacy Configuration | Updated Configuration      |
> | -------------------- | -------------------------- |
> | `system_template`    | `templates.system.path`    |
> | `system_schema`      | `schemas.system.path`      |
> | `user_template`      | `templates.user.path`      |
> | `user_schema`        | `schemas.user.path`        |
> | `assistant_template` | `templates.assistant.path` |
> | `assistant_schema`   | `schemas.assistant.path`   |

**System template (`system_template.minijinja`):**

```
You are an assistant that is performing a named entity recognition task.
Your job is to extract entities from a given text.

The entities you are extracting are:

- people
- organizations
- locations
- miscellaneous other entities

Please return the entities in the following JSON format:

{
"person": ["person1", "person2", ...],
"organization": ["organization1", "organization2", ...],
"location": ["location1", "location2", ...],
"miscellaneous": ["miscellaneous1", "miscellaneous2", ...]
}
```

**Dynamic schema override at request time** (illustrating §4):

```python
dynamic_output_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "person":        {"type": "array", "items": {"type": "string"}},
    "organization":  {"type": "array", "items": {"type": "string"}},
    "location":      {"type": "array", "items": {"type": "string"}},
    "miscellaneous": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["person", "organization", "location", "miscellaneous"],
  "additionalProperties": false
}

resp = client.chat.completions.create(
    model="tensorzero::function_name::extract_entities",
    messages=[{"role": "user", "content": "Third was Ford with 35,563 registrations , or 11.7 percent ."}],
    response_format={"type": "json_schema", "schema": dynamic_output_schema},
)
```

**Switch to `tool` mode at request time when calling a non-strict provider**:

```python
resp = client.chat.completions.create(
    model="tensorzero::function_name::extract_entities",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "tensorzero::params": {
            "chat_completion": {
                "json_mode": "tool",
            }
        }
    },
)
```

> Sources: `optimization.md` (NER configuration); `gateway-create-a-prompt-template.md` (legacy vs new template/schema syntax); `gateway-generate-structured-outputs.md`.

---

## 13. Quick reference cheat sheet

### 13.1 `json_mode` values

| Mode       | What it does                                                                                                   | Best for                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `off`      | No special JSON handling (**not recommended** per docs)                                                        | Debugging; not recommended in production                       |
| `on`       | Provider JSON mode if available                                                                                | Providers that have a non-strict JSON-only mode                |
| `strict`   | Strict JSON / Structured Outputs (OpenAI uses Structured Outputs; direct Anthropic uses structured outputs)    | OpenAI, direct Anthropic                                       |
| `tool`     | TensorZero converts your schema to a tool call under the hood and unwraps the response back to JSON            | Providers without native strict, e.g. AWS Bedrock              |

Forbidden combos:

- `json_mode` on a `chat` function (any value) — configuration error.
- `json_mode = "tool"` + any other tool in the same request — not allowed.
- `json_mode` other than `"strict"` (direct Anthropic) or `"off"` on Anthropic extended-thinking models.

### 13.2 Where to set things

| What                                | Configuration (preferred)                          | Inference-time override (native API)              | Inference-time override (OpenAI-compat)                                  |
| ----------------------------------- | -------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------ |
| Output schema                       | `[functions.x] output_schema = "path.json"`        | request body `output_schema` (object)             | request body `response_format = {"type":"json_schema","schema":...}`     |
| `json_mode`                         | `[functions.x.variants.y] json_mode = "strict"`    | `params.chat_completion.json_mode`                | `extra_body["tensorzero::params"].chat_completion.json_mode`             |
| Other chat-completion params        | variant fields (`temperature`, `max_tokens`, etc.) | `params.chat_completion.<param>`                  | `extra_body["tensorzero::params"].chat_completion.<param>`               |

### 13.3 Response field map

| Concept                      | Native `/inference`                | OpenAI-compatible                                   |
| ---------------------------- | ---------------------------------- | --------------------------------------------------- |
| Inference ID                 | `inference_id`                     | `id`                                                |
| Episode ID                   | `episode_id`                       | `episode_id`                                        |
| Variant chosen               | `variant_name`                     | `model` (encoded as `…::variant_name::<v>`)         |
| Raw JSON string              | `output.raw`                       | `choices[0].message.content`                        |
| Parsed JSON object           | `output.parsed` (`null` if invalid)| Not provided — parse `content` yourself             |
| Input tokens                 | `usage.input_tokens`               | `usage.prompt_tokens`                               |
| Output tokens                | `usage.output_tokens`              | `usage.completion_tokens`                           |
| Total tokens                 | (not provided as a sum)            | `usage.total_tokens`                                |
| Cost (USD)                   | `usage.cost`                       | `usage.tensorzero_cost`                             |
| Raw provider response        | `raw_response` (when requested)    | (not documented as available)                       |

### 13.4 Decision flow for choosing a setup

1. Do you have a single, well-defined output shape? → use a `json` function. Otherwise consider a `chat` function with tool use.
2. Pick `json_mode` by provider:
   - OpenAI → `strict`.
   - Anthropic direct → `strict` (use `off` instead if the model is an extended-thinking model and `strict` is somehow unavailable).
   - Anthropic via Bedrock/Vertex → `tool` (for reliability).
   - Gemini → `strict` (mind the JSON Schema subset limitation).
   - AWS Bedrock (non-Anthropic) → `tool`.
3. Static schema in config when possible. Dynamic only if the shape varies per request.
4. Never stream a JSON function unless you can tolerate having to reassemble and parse `raw` deltas yourself.

---

## 14. Sources used

This guide reflects only the content of the following project files. Every quoted passage is verbatim from them; every example is reproduced from them. Where documentation is silent on a question, this is flagged in §11.

- `gateway-generate-structured-outputs.md` — primary guide. Original URL: `https://www.tensorzero.com/docs/gateway/generate-structured-outputs.md`.
- `gateway-configuration-reference.md` — `json_mode`, `output_schema`, `strict`, template/schema fields, variant types. Sections used:
  - `[functions.function_name]` (including `type = "json"` accordion and `output_schema`)
  - `[functions.function_name.variants.variant_name]` `chat_completion` accordion (`json_mode`, templates, params)
  - `[tools.tool_name]` (`strict` field for tool parameters)
- `gateway-api-reference.md` — native `/inference` request and response shapes; batch inference (`output_schemas`, batch `params`); JSON Function and JSON Function with Dynamic Output Schema examples; OpenAI-compatible `response_format`, `tensorzero::params`, and response constraints. Original URLs: `https://www.tensorzero.com/docs/gateway/api-reference/inference.md` and `https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md`.
- `gateway-data-model.md` — `JsonInference` and `JsonInferenceDatapoint` table schemas (storage shape for `parsed`/`raw` outputs and `output_schema`).
- `gateway-configure-functions-and-variants.md` — `chat` vs `json` overview; variant skeleton.
- `gateway-create-a-prompt-template.md` — legacy vs new template/schema syntax migration table; `tensorzero::template` content block usage.
- `optimization.md` — canonical `extract_entities` JSON function configuration reused across SFT, DICL, and GEPA guides.
- `evaluations.md` — LLM-judge variants using `json_mode = "strict"` as illustrative examples.

If you find any discrepancy between this guide and the live TensorZero documentation, treat the live docs as authoritative — this guide is a snapshot of the project files at the time of writing.
