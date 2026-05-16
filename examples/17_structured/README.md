# 17 - Structured Outputs

Five complementary entry points covering the structured-outputs surface:

- **`main.py`** — single-shot extraction with
  `run_structured_inference()`, the low-level sibling of `run_agent`
  for TensorZero functions whose `type = "json"`.
- **`structured_agent.py`** — the declarative `Agent` wrapper
  dispatching to the same helper transparently when
  `AgentConfig.function_kind == "json"`.
- **`recovery_demo.py`** *(no docker)* — local in-process fake
  gateway that scripts all four `parse_strategy` outcomes (`strict`,
  `permissive`, `retry`, `none`) so you can see the recovery ladder
  end-to-end in under a second.
- **`recovery_real_provider.py`** *(docker)* — same wiring as
  `structured_agent.py` but with `json_mode="off"` per call so the
  real model is free to emit prose-wrapped JSON; the in-Python
  permissive layer and the retry loop absorb the result.
- **`ner_dynamic_schema.py`** *(docker)* — NER agent exercising
  per-call `output_schema` overrides and per-call `json_mode` swaps
  against the `extract_entities` function (people, locations,
  organizations, topics — five categories with the per-call override).

The first two, `recovery_real_provider.py`, and `ner_dynamic_schema.py`
target functions declared in `config/tensorzero.toml`
(`extract_customer_info` for the first three, `extract_entities` for
the fifth) with a static `output_schema` and `json_mode = "strict"`
at the variant level.  No MCP tools are called — JSON functions are
single-shot by contract.

## Robust JSON recovery

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
2. Calling `agent.invoke(message)` twice in a row — `Agent` dispatches to
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

## What `ner_dynamic_schema.py` demonstrates

1. **Default config** (scenario 1) — `function_kind="json"`,
   `json_mode="strict"`, the gateway uses the static
   `functions/extract_entities/output_schema.json` (four categories:
   people, locations, organizations, topics).
2. **Per-call schema override** (scenario 2) — passes
   `output_schema=` at call time with a five-category schema (adds
   `products`); TensorZero validates against the per-call schema
   instead of the variant-level one (TZ guide §4).
3. **Per-call `json_mode="tool"` swap** (scenario 3) — flips the
   gateway-side strict mode to tool-call mode at call time.  Legal
   because the agent's MCP tool provider is empty (the safety guard's
   positive path).
4. **Recovery demo** (scenario 4) — `json_mode="off"` per call so the
   provider can emit prose-wrapped JSON; the permissive parsing layer
   recovers the result.

See [`docs/structured-outputs.md`](../../docs/structured-outputs.md)
for the full provider matrix, per-call override semantics, the
testing helpers (`MockGateway.script_json` and the `assert_parsed_*`
family), and the Pydantic recipe.

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
# Low-level helper (needs docker)
uv run python examples/17_structured/main.py

# Agent wrapper (needs docker)
uv run python examples/17_structured/structured_agent.py

# Local recovery ladder demo (no docker, < 1s)
uv run python examples/17_structured/recovery_demo.py

# Recovery against a real provider (needs docker)
uv run python examples/17_structured/recovery_real_provider.py

# Dynamic schema override + json_mode swap (needs docker)
uv run python examples/17_structured/ner_dynamic_schema.py
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
  directly.  See `recovery_demo.py` for the `permissive` (in-Python
  recovery), `retry` (re-emit with a corrective hint), and `none`
  outcomes.
- `Episode ID` / `Inference ID` are the IDs ClickHouse recorded; you
  can query them from the dashboard or via `tensorzero` tooling.

For `structured_agent.py`:

- Two consecutive runs print `function_kind: json` and a populated
  `parsed:` dict.
- `response:` shows the same raw JSON string the helper would have
  returned — they target the same function and variant.
- `turns: 1` confirms the JSON path is single-shot (no tool loop).
- Each run prints a fresh `episode_id` because no `episode_id` is
  threaded into `agent.invoke(...)`.

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

For `ner_dynamic_schema.py`:

- Scenario 1 prints a four-key `parsed` (`people`, `locations`,
  `organizations`, `topics`).
- Scenario 2 prints a five-key `parsed` (the four above plus
  `products`).  The script `assert`s on `"products" in parsed` to
  prove the per-call override threaded through the gateway.
- Scenario 3 prints the same shape as scenario 1; the value of
  `parse_strategy` should be `"strict"`.  If you adapt the example to
  attach a non-empty tool provider, the per-call `json_mode="tool"`
  swap raises `ConfigurationError` per the safety guard.
- Scenario 4 typically reports `parse_strategy="permissive"` because
  Groq's model wraps the JSON in prose under `json_mode="off"`.  Re-
  run a few times — the value is non-deterministic.

## Notes on `json_mode`

The variant declares `json_mode = "strict"`, which routes through
OpenAI's Structured Outputs (or direct Anthropic's structured outputs)
when the provider supports it.  For providers without native strict
support — e.g. AWS Bedrock non-Anthropic — use `json_mode = "tool"`
instead.  See `docs/structured-outputs.md` for the provider matrix.
