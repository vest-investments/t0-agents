# 17 - Structured Outputs (M1 single-shot helper)

Single-shot JSON extraction with `run_structured_inference()` — the
sibling of `run_agent` for TensorZero functions whose `type = "json"`.
No MCP server, no tool loop: the gateway parses the model output
against the declared JSON Schema and returns both `output.raw` (the
JSON string) and `output.parsed` (the JSON-parsed mapping).

## What it demonstrates

1. Configuring an `extract_customer_info` JSON function in
   `tensorzero.toml` with a static `output_schema` and
   `json_mode = "strict"`.
2. Calling the function via `run_structured_inference()` with one
   user message.
3. Inspecting the `StructuredOutput` fields — `parsed`, `raw`,
   `parse_strategy`, `usage`, `episode_id`, `inference_id`.

The roadmap's later milestones build on this same helper:

- **M2** wires `Agent.run(...)` to dispatch JSON functions through the
  helper transparently.
- **M3** adds permissive parsing and a retry loop for invalid JSON.
- **M4** adds provider-safety warnings and the
  `Agent.stream(...) + json` guard.
- **M5** ships testing helpers and the dynamic-schema NER example.

## Prerequisites

- Python 3.11+
- Dependencies installed: `uv sync`
- `GROQ_API_KEY` set in `.env` at the project root.

## Start the gateway

```bash
docker compose -f examples/17_structured/docker-compose.yml \
    --env-file .env up -d
```

## Run

```bash
uv run python examples/17_structured/main.py
```

## Stop the gateway

```bash
docker compose -f examples/17_structured/docker-compose.yml \
    --env-file .env down
```

## What to observe

- `Parsed:` shows a dict like `{"name": "Sarah Johnson",
  "email": "sarah.j@example.com"}`.
- `Raw:` is the JSON string the model produced.
- `Parse strategy: strict` — TensorZero parsed the model output
  directly.  M3 will add `permissive` (in-Python recovery) and
  `retry` (re-emit with a corrective hint).
- `Episode ID` / `Inference ID` are the IDs ClickHouse recorded; you
  can query them from the dashboard or via `tensorzero` tooling.

## Notes on `json_mode`

The variant declares `json_mode = "strict"`, which routes through
OpenAI's Structured Outputs (or direct Anthropic's structured outputs)
when the provider supports it.  For providers without native strict
support — e.g. AWS Bedrock non-Anthropic — use `json_mode = "tool"`
instead.  See `docs/structured-outputs.md` for the provider matrix.
