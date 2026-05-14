# 17 - Structured Outputs

Four complementary entry points covering the structured-outputs surface:

- **`main.py`** (M1) — single-shot extraction with
  `run_structured_inference()`, the low-level sibling of `run_agent`
  for TensorZero functions whose `type = "json"`.
- **`structured_agent.py`** (M2) — the declarative `Agent` wrapper
  dispatching to the same helper transparently when
  `AgentConfig.function_kind == "json"`.
- **`recovery_demo.py`** (M3, no docker) — local in-process fake
  gateway that scripts all four `parse_strategy` outcomes (`strict`,
  `permissive`, `retry`, `none`) so you can see the recovery ladder
  end-to-end in under a second.
- **`recovery_real_provider.py`** (M3, docker) — same wiring as
  `structured_agent.py` but with `json_mode="off"` per call so the
  real model is free to emit prose-wrapped JSON; the in-Python
  permissive layer and the retry loop absorb the result.

The first two and `recovery_real_provider.py` target the
`extract_customer_info` function declared in `config/tensorzero.toml`
with a static `output_schema` and `json_mode = "strict"`.  No MCP
tools are called — JSON functions are single-shot by contract.

## Robust JSON recovery (M3)

When TensorZero's strict parser returns `parsed == None`, the agent
walks a three-layer recovery ladder before declaring the run a failure:

```
        TensorZero strict parse        ← layer 1: ``parse_strategy="strict"``
                  │ parsed is None
                  ▼
        permissive_parse(output.raw)   ← layer 2: ``parse_strategy="permissive"``
                  │ parsed is None
                  ▼
        retry with corrective hint     ← layer 3: ``parse_strategy="retry"``
        (up to ``max_json_retries`` extra attempts)
                  │ every attempt failed
                  ▼
        result.parsed is None,         ← terminal: ``parse_strategy="none"``
        caller inspects strategy
```

The permissive layer (`t0_agents.permissive_parse`) and the retry
layer are both opt-out-able — pass
`AgentConfig(..., permissive_json_parsing=False, max_json_retries=0)`
for byte-perfect TensorZero-pure behaviour.

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

## What `recovery_demo.py` demonstrates

1. A local `ScriptedGateway` in-process fake (no docker required) that
   pops one canned `JsonInferenceResponse` per `inference()` call.
2. Four scenarios scripted to land on each of the four `parse_strategy`
   outcomes in turn.
3. A side-by-side table of `parse_strategy`, `json_parse_attempts`,
   `len(inference_ids)`, and `parsed` so the four behaviours are
   visible at a glance.

## What `recovery_real_provider.py` demonstrates

1. The same `Agent` setup as `structured_agent.py` but with
   `permissive_json_parsing=True, max_json_retries=2` (the defaults).
2. A deliberately under-specified prompt and a per-call
   `json_mode="off"` override, so the model decides on its output
   shape instead of being forced into strict JSON.
3. The resulting `AgentResult.parse_strategy` and
   `json_parse_attempts` reveal whether TensorZero parsed cleanly,
   whether the in-Python permissive layer rescued the response, or
   whether a retry was needed.  Re-run a few times — model behaviour
   is non-deterministic.

See [`docs/structured-outputs.md`](../../docs/structured-outputs.md)
for the full provider matrix, per-call override semantics, and the
roadmap's later milestones (M4 streaming guard and provider warnings;
M5 testing helpers and consolidation).

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
# M1 — low-level helper (needs docker)
uv run python examples/17_structured/main.py

# M2 — Agent wrapper (needs docker)
uv run python examples/17_structured/structured_agent.py

# M3 — local recovery ladder demo (no docker, < 1s)
uv run python examples/17_structured/recovery_demo.py

# M3 — recovery against a real provider (needs docker)
uv run python examples/17_structured/recovery_real_provider.py
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

For `recovery_demo.py`:

- The output table lists one row per scenario.
- The `strict` row reports `parse_strategy=strict`, `attempts=0`,
  `inference_ids=1` — the happy path.
- The `permissive` row reports `parse_strategy=permissive`,
  `attempts=0`, `inference_ids=1` — TZ failed, but the in-Python
  layer recovered the JSON without retrying.
- The `retry` row reports `parse_strategy=retry`, `attempts=1`,
  `inference_ids=2` — first inference invalid, second succeeded.
- The `none` row reports `parse_strategy=none`, `attempts=2`,
  `inference_ids=3`, `parsed=None` — every layer and every attempt
  failed.

For `recovery_real_provider.py`:

- `parse_strategy` will vary across runs — Groq's models sometimes
  emit clean JSON (`strict`), sometimes prose-wrapped JSON
  (`permissive`), occasionally needing a retry (`retry`).
- `json_parse_attempts` is the count of *extra* inferences after the
  first; `turns == 1 + json_parse_attempts`.
- `inference_ids` accumulates every attempt's id in order, including
  the ones that produced invalid output — useful when grepping the
  TensorZero dashboard for the conversation.

## Notes on `json_mode`

The variant declares `json_mode = "strict"`, which routes through
OpenAI's Structured Outputs (or direct Anthropic's structured outputs)
when the provider supports it.  For providers without native strict
support — e.g. AWS Bedrock non-Anthropic — use `json_mode = "tool"`
instead.  See `docs/structured-outputs.md` for the provider matrix.
