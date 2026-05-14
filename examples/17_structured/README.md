# 17 - Structured Outputs

Two complementary entry points to the same JSON function:

- **`main.py`** (M1) — single-shot extraction with
  `run_structured_inference()`, the low-level sibling of `run_agent`
  for TensorZero functions whose `type = "json"`.
- **`structured_agent.py`** (M2) — the declarative `Agent` wrapper
  dispatching to the same helper transparently when
  `AgentConfig.function_kind == "json"`.

Both target the `extract_customer_info` function declared in
`config/tensorzero.toml` with a static `output_schema` and
`json_mode = "strict"`.  No MCP tools are called — JSON functions are
single-shot by contract.

## What `main.py` demonstrates

1. Configuring the JSON function in `tensorzero.toml` with a static
   schema and `json_mode = "strict"`.
2. Calling `run_structured_inference()` directly with one user message.
3. Inspecting the `StructuredOutput` fields — `parsed`, `raw`,
   `parse_strategy`, `usage`, `episode_id`, `inference_id`.

## What `structured_agent.py` demonstrates

1. Building an `AgentConfig(function_kind="json", json_mode="strict",
   …)` against the same TensorZero function.
2. Calling `agent.run(message)` twice in a row — `Agent` dispatches to
   `run_structured_inference()` because of `function_kind`.
3. Inspecting the resulting `AgentResult` — `function_kind == "json"`,
   `parsed` carrying the JSON object, `response` set to the raw JSON
   string, `turns == 1`, and one entry in `inference_ids`.

See [`docs/structured-outputs.md`](../../docs/structured-outputs.md)
for the full provider matrix, per-call override semantics, and the
roadmap's later milestones (M3 permissive parsing + retry; M4
streaming guard and provider warnings; M5 testing helpers).

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
# M1 — low-level helper
uv run python examples/17_structured/main.py

# M2 — Agent wrapper
uv run python examples/17_structured/structured_agent.py
```

## Stop the gateway

```bash
docker compose -f examples/17_structured/docker-compose.yml \
    --env-file .env down
```

## What to observe

For `main.py`:

- `Parsed:` shows a dict like `{"name": "Sarah Johnson",
  "email": "sarah.j@example.com"}`.
- `Raw:` is the JSON string the model produced.
- `Parse strategy: strict` — TensorZero parsed the model output
  directly.  M3 will add `permissive` (in-Python recovery) and
  `retry` (re-emit with a corrective hint).
- `Episode ID` / `Inference ID` are the IDs ClickHouse recorded; you
  can query them from the dashboard or via `tensorzero` tooling.

For `structured_agent.py`:

- Two consecutive runs print `function_kind: json` and a populated
  `parsed:` dict.
- `response:` shows the same raw JSON string the helper would have
  returned — they target the same function and variant.
- `turns: 1` confirms the JSON path is single-shot (no tool loop).
- Each run prints a fresh `episode_id` because no `episode_id` is
  threaded into `agent.run(...)`.

## Notes on `json_mode`

The variant declares `json_mode = "strict"`, which routes through
OpenAI's Structured Outputs (or direct Anthropic's structured outputs)
when the provider supports it.  For providers without native strict
support — e.g. AWS Bedrock non-Anthropic — use `json_mode = "tool"`
instead.  See `docs/structured-outputs.md` for the provider matrix.
