---

## Phase 0 Guide: Infrastructure — TensorZero Side

### P0.1 — T0 Gateway starts with Docker Compose

**Required file structure:**

```
your-project/
├── config/
│   └── tensorzero.toml
├── docker-compose.yml
└── .env   (or export variables)
```

**Minimal `tensorzero.toml` file** (for a chat function without tools, sufficient for P0.2):

```toml
# config/tensorzero.toml

[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
```

> The shorthand notation `openai::gpt-4o-mini` avoids having to define `[models.*]` and `[models.*.providers.*]` sections. Providers that support shorthand are: `anthropic`, `deepseek`, `fireworks`, `google_ai_studio_gemini`, `gcp_vertex_gemini`, `gcp_vertex_anthropic`, `hyperbolic`, `groq`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.
>
> (source: `gateway-configuration-reference.md` > variant `model` field, URL: https://www.tensorzero.com/docs/gateway/configuration-reference.md)

**Reference Docker Compose** (with Postgres for observability):

```yaml
# docker-compose.yml
# Simplified for development. Do not use in production.

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:3000/health",
        ]
      start_period: 10s
      start_interval: 1s
      timeout: 1s
    depends_on:
      postgres:
        condition: service_healthy
      gateway-run-postgres-migrations:
        condition: service_completed_successfully

  postgres:
    image: tensorzero/postgres:17
    command: ["postgres", "-c", "cron.database_name=tensorzero"]
    environment:
      POSTGRES_DB: tensorzero
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d tensorzero"]
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  gateway-run-postgres-migrations:
    image: tensorzero/gateway
    environment:
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
    depends_on:
      postgres:
        condition: service_healthy
    command: ["--run-postgres-migrations"]

volumes:
  postgres-data:
```

(source: `quickstart.md` > "Deploying TensorZero", URL: https://www.tensorzero.com/docs/quickstart.md)

**Important detail:** There is a separate `gateway-run-postgres-migrations` service that runs Postgres migrations **before** the gateway starts. The gateway has `depends_on` with `condition: service_completed_successfully` pointing to this service. If you omit this, the gateway may fail to start.

**Important detail:** If you do **not** want observability (neither Postgres nor ClickHouse), you can run the gateway in minimal mode with `--default-config` and without `TENSORZERO_POSTGRES_URL`:

```bash
docker run \
  -e OPENAI_API_KEY \
  -p 3000:3000 \
  tensorzero/gateway \
  --default-config
```

(source: `deployment.md` > "Deploy the TensorZero Gateway", URL: https://www.tensorzero.com/docs/deployment/tensorzero-gateway.md)

But for the full quickstart, **using Postgres is recommended**.

**Verification:** After `docker compose up`, the gateway's healthcheck validates `http://localhost:3000/health`. You can also use `http://localhost:3000/status`.

(source: `deployment.md` > "Add a status or health check")

---

### P0.2 — Basic inference without tools against T0

TensorZero exposes **two inference APIs**:

1. **Native API**: `POST /inference` — uses the Python client `tensorzero` (`AsyncTensorZeroGateway`)
2. **OpenAI-compatible API**: `POST /openai/v1/chat/completions` — uses the OpenAI SDK pointing to T0

**Option A: OpenAI-compatible API (simpler to start with)**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

response = client.chat.completions.create(
    model="tensorzero::function_name::generate_haiku",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about TensorZero.",
        }
    ],
)

print(response)
```

The `model` field uses the format `tensorzero::function_name::<function_name>` to call functions defined in the config, or `tensorzero::model_name::<provider>::<model>` to call a model directly.

(source: `quickstart.md` > "Our First TensorZero API Call" and `gateway-call-any-llm.md`, URL: https://www.tensorzero.com/docs/quickstart.md)

**Option B: TensorZero native API**

```python
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about TensorZero."
                }
            ]
        }
    )
```

(source: `gateway-api-reference.md` > "API Reference: Inference" > "Chat Function" example, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

**Native response** (not OpenAI-compatible):

```json
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "gpt_4o_mini",
  "content": [
    {
      "type": "text",
      "text": "..."
    }
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100,
    "cost": 0.0003
  }
}
```

**Important detail — `dryrun`:** The `/inference` endpoint supports a `dryrun` field (boolean). If `true`, the inference is executed but **not stored in the database**. Useful for testing if you don't want to contaminate your observability data.

(source: `gateway-api-reference.md` > `dryrun` field, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

**Important detail — `variant_name`:** For tests you can pin a specific variant with `variant_name="gpt_4o_mini"`, although in production it is recommended to let the gateway select it.

---

### P0.3, P0.4, P0.5 — FastMCP

_(You fill in this part with the FastMCP documentation: `@mcp.tool`, `Client(mcp_instance)`, `list_tools()`, `call_tool()`, `CallToolResult`)_

---

### P0.6 — pytest configured with `asyncio_mode = "auto"`

This is standard `pytest-asyncio` configuration. In your `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

This allows all your `async def test_*` functions to run automatically without needing the `@pytest.mark.asyncio` decorator on each test.

---

### Additional details relevant to future integration (T0 + tools context)

Although Phase 0 verifies that both systems work **separately**, it is useful to know how T0 handles tools from now on, because in later phases you will need to connect them.

**How T0 defines tools in config (static):**

```toml
[functions.weather_bot]
type = "chat"
tools = ["get_temperature"]
tool_choice = "auto"  # options: "none", "auto", "required", { specific = "tool_name" }

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "get_temperature.json"   # path relative to the config file
strict = false  # optional, default false
```

The `get_temperature.json` file contains a JSON Schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for"
    },
    "units": {
      "type": "string",
      "enum": ["fahrenheit", "celsius"]
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

(source: `gateway-api-reference.md` > "Chat Function with Tool Use" and `gateway-configuration-reference.md` > `[tools.tool_name]`, URL: https://www.tensorzero.com/docs/gateway/configuration-reference.md)

**How T0 defines tools dynamically (at runtime — `additional_tools`):**

In the inference request you can pass tools without them being in the config:

```json
{
  "function_name": "my_function",
  "input": { "messages": [...] },
  "additional_tools": [
    {
      "name": "get_temperature",
      "description": "Get the current temperature in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        },
        "required": ["location"]
      },
      "strict": false
    }
  ]
}
```

(source: `gateway-api-reference.md` > `additional_tools` field, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

**How T0 responds with tool calls:**

```json
{
  "content": [
    {
      "type": "tool_call",
      "arguments": {"location": "Tokyo", "units": "celsius"},
      "id": "123456789",
      "name": "get_temperature",
      "raw_arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}",
      "raw_name": "get_temperature"
    }
  ]
}
```

**How to send tool_results back to T0** (multi-turn):

```json
{
  "function_name": "weather_bot",
  "episode_id": "<episode_id from the previous response>",
  "input": {
    "messages": [
      {"role": "user", "content": "What is the weather like in Tokyo?"},
      {
        "role": "assistant",
        "content": [
          {
            "type": "tool_call",
            "id": "123456789",
            "name": "get_temperature",
            "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "tool_result",
            "id": "123456789",
            "name": "get_temperature",
            "result": "22"
          }
        ]
      }
    ]
  }
}
```

(source: `gateway-api-reference.md` > `input.messages` content blocks, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

**Key details not to overlook:**

1. The `episode_id` links multiple inferences in a conversation. If you don't pass it, T0 generates a new one. For multi-turn (sending tool results), **you must reuse the `episode_id`** from the previous response.

2. `tool_result` messages use `role: "user"`, not a special role. The `id` must match the `id` of the assistant's `tool_call`.

3. `tool_call` entries in assistant messages use `arguments` as a JSON string (not as a parsed object), unlike the gateway response where you have both `arguments` (parsed) and `raw_arguments` (string).

4. T0 validates tool calls against the defined JSON Schema. If the LLM's response is invalid, `arguments` will be `null` and `name` will be `null` in the response, but `raw_arguments` and `raw_name` will contain what the model generated.

---

### Main sources

| File | Section | URL |
|---|---|---|
| `quickstart.md` | Full quickstart | https://www.tensorzero.com/docs/quickstart.md |
| `deployment.md` | Deploy the TensorZero Gateway | https://www.tensorzero.com/docs/deployment/tensorzero-gateway.md |
| `gateway-configure-functions-and-variants.md` | Functions & variants | https://www.tensorzero.com/docs/gateway/configure-functions-and-variants.md |
| `gateway-configuration-reference.md` | Config Reference (tools, functions, variants) | https://www.tensorzero.com/docs/gateway/configuration-reference.md |
| `gateway-api-reference.md` | API Reference: Inference | https://www.tensorzero.com/docs/gateway/api-reference/inference.md |
| `gateway-call-any-llm.md` | Call any LLM | https://www.tensorzero.com/docs/gateway/call-any-llm.md |


## Continuation: Supplementary Details for Phase 0

### TensorZero-specific parameters in the OpenAI-compatible API

When using the OpenAI SDK against T0, there are parameters passed via `extra_body` with the `tensorzero::` prefix. These are the ones relevant for Phase 0 and testing:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

response = client.chat.completions.create(
    model="tensorzero::function_name::generate_haiku",
    messages=[
        {"role": "user", "content": "Write a haiku about testing."}
    ],
    extra_body={
        # Does not persist the inference in the database (useful for tests)
        "tensorzero::dryrun": True,
        # Pins the variant (useful for deterministic tests)
        "tensorzero::variant_name": "gpt_4o_mini",
        # Associates with an existing episode (for multi-turn)
        "tensorzero::episode_id": "11111111-1111-1111-1111-111111111111",
    },
)
```

**Important detail:** API keys sent from the OpenAI client are **ignored** by the gateway. Provider credentials are configured as environment variables of the gateway (e.g. `OPENAI_API_KEY`), not of the client.

(source: `gateway-api-reference.md` > "API Reference: Inference (OpenAI-Compatible)" > Request section, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md)

---

### Format of the `model` field in the OpenAI-compatible API

It is important to understand the three variants of the `model` field:

| To call... | Format |
|---|---|
| A function defined in `tensorzero.toml` as `[functions.my_function]` | `tensorzero::function_name::my_function` |
| A model defined in `tensorzero.toml` as `[models.my_model]` | `tensorzero::model_name::my_model` |
| A provider's model directly (without defining it in config) | `tensorzero::model_name::{provider_type}::{model_name}` |

Concrete example:

```python
# Calls the "generate_haiku" function defined in config
model="tensorzero::function_name::generate_haiku"

# Calls OpenAI gpt-4o-mini directly without a function
model="tensorzero::model_name::openai::gpt-4o-mini"

# Calls Anthropic Claude directly
model="tensorzero::model_name::anthropic::claude-sonnet-4-5"
```

(source: `gateway-api-reference.md` > "API Reference: Inference (OpenAI-Compatible)" > `model` field, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md)

---

### Differences between the native API (`/inference`) and the OpenAI-compatible API (`/openai/v1/chat/completions`)

For your integration you will need to decide which one to use. Key documented differences:

**Native API (`POST /inference`):**
- Uses the Python client `tensorzero` (`AsyncTensorZeroGateway`)
- Fields: `function_name`, `input`, `dryrun`, `episode_id`, `additional_tools`, `allowed_tools`, `tool_choice`, `variant_name`, `stream`, `tags`, `params`, `credentials`
- The response has `inference_id`, `episode_id`, `variant_name`, `content` (list of content blocks), `usage`
- Supports dynamic credentials (`credentials` field)
- Supports `unknown` content blocks in the response

**OpenAI-compatible API (`POST /openai/v1/chat/completions`):**
- Uses the standard OpenAI SDK
- T0-specific parameters go in `extra_body` with the `tensorzero::` prefix
- The response follows the OpenAI format: `id`, `choices[].message.content`, `choices[].message.tool_calls`, `usage`
- **Does not support** dynamic credentials
- **Cannot** handle `unknown` content blocks (drops them with a warning)

(source: `gateway-api-reference.md` > "API Reference: Inference (OpenAI-Compatible)" and "API Reference: Inference", URLs: https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md, https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

---

### Tool use via the OpenAI-compatible API

When using the OpenAI SDK to pass tools dynamically:

```python
response = client.chat.completions.create(
    model="tensorzero::function_name::my_agent",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for"
                        }
                    },
                    "required": ["location"]
                },
                "strict": False
            }
        }
    ],
    tool_choice="auto",
)
```

The response with tool calls in OpenAI format:

```json
{
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "content": null,
        "tool_calls": [
          {
            "id": "123456789",
            "type": "function",
            "function": {
              "name": "get_temperature",
              "arguments": "{\"location\": \"Tokyo\"}"
            }
          }
        ],
        "role": "assistant"
      }
    }
  ]
}
```

And to send the tool result back (multi-turn with OpenAI SDK):

```python
response = client.chat.completions.create(
    model="tensorzero::function_name::my_agent",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "123456789",
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "arguments": "{\"location\": \"Tokyo\"}"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "123456789",
            "content": "22"
        }
    ],
    extra_body={
        "tensorzero::episode_id": "<episode_id from the previous response>"
    },
)
```

**Important detail for the OpenAI-compatible API:** The role for tool results is `tool` (not `user` as in the native API), and it uses `tool_call_id` (not `id` + `name` as in the native API). This is because it follows the OpenAI convention.

(source: `gateway-api-reference.md` > "API Reference: Inference (OpenAI-Compatible)" > `messages` field, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md)

---

### Debug mode in configuration

For development, you can enable debug mode which logs more detailed inputs and outputs:

```toml
# config/tensorzero.toml
[gateway]
debug = true
```

> Normally, TensorZero does not include inputs and outputs in logs or errors to avoid leaking sensitive data. When `debug = true`, the gateway will log more verbose errors.

(source: `gateway-configuration-reference.md` > `debug` field, URL: https://www.tensorzero.com/docs/gateway/configuration-reference.md)

---

### Verification checklist for Phase 0

Here is an executable summary to validate each item:

**P0.1 — Gateway starts:**
```bash
export OPENAI_API_KEY="sk-..."
docker compose up
# Verify: curl http://localhost:3000/health → 200
```

**P0.2 — Basic inference works:**
```bash
curl http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::function_name::generate_haiku",
    "messages": [
      {"role": "user", "content": "Write a haiku about testing."}
    ]
  }'
# Verify: response has "choices" with non-empty content
```

Or with Python:
```python
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(
    gateway_url="http://localhost:3000"
) as client:
    result = await client.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {"role": "user", "content": "Write a haiku about testing."}
            ]
        },
        dryrun=True,  # does not persist in DB
    )
    assert result.content  # has content blocks
    assert result.inference_id  # has an ID
```

**P0.3, P0.4, P0.5 — FastMCP** _(you fill in)_

**P0.6 — pytest:**
```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```bash
pytest -v  # all async tests run without an extra decorator
```

---

### Documented gotchas and common errors

1. **Port 5432 occupied:** If you already have a local Postgres running, docker-compose will fail to bind the port. Change the port mapping or stop your local Postgres.

2. **Missing credentials → gateway won't start:** If the config references a provider (e.g. `openai::gpt-4o-mini`) but the corresponding environment variable (`OPENAI_API_KEY`) is not defined, the gateway fails at startup, not at runtime.
   (source: `deployment.md` > "Set up model provider credentials", URL: https://www.tensorzero.com/docs/deployment/tensorzero-gateway.md)

3. **Docker image:** The image is `tensorzero/gateway` (no version tag mentioned in the quickstart, implicitly uses latest). For Postgres the image is `tensorzero/postgres:17` — note that this is a custom TensorZero image, not the official Postgres image. This is because it includes the `pg_cron` extension.

4. **Config file path inside the container:** The volume mounts `./config:/app/config:ro` and the command uses `--config-file /app/config/tensorzero.toml`. Paths to schemas and templates inside the TOML (such as `parameters = "get_temperature.json"`) are **relative to the configuration file**, i.e., relative to `/app/config/`.
   (source: `gateway-configuration-reference.md` > `parameters` field in tools, URL: https://www.tensorzero.com/docs/gateway/configuration-reference.md)

5. **`extra_hosts: host.docker.internal:host-gateway`:** This allows the gateway inside Docker to connect to services running on your host (e.g. if you run a local MCP server). It is already included in the reference docker-compose.

---

### Complete sources

| File | Section | Original URL |
|---|---|---|
| `quickstart.md` | Quickstart | https://www.tensorzero.com/docs/quickstart.md |
| `deployment.md` | Deploy the TensorZero Gateway | https://www.tensorzero.com/docs/deployment/tensorzero-gateway.md |
| `gateway-configure-functions-and-variants.md` | Functions & variants | https://www.tensorzero.com/docs/gateway/configure-functions-and-variants.md |
| `gateway-configuration-reference.md` | Full Config Reference | https://www.tensorzero.com/docs/gateway/configuration-reference.md |
| `gateway-api-reference.md` | API Reference: Inference | https://www.tensorzero.com/docs/gateway/api-reference/inference.md |
| `gateway-api-reference.md` | API Reference: Inference (OpenAI-Compatible) | https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible.md |
| `gateway-call-any-llm.md` | Call any LLM | https://www.tensorzero.com/docs/gateway/call-any-llm.md |
