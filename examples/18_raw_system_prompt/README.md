# Example 18: Raw System Prompt

Runnable demo of the **`system="..."`** kwarg added in 0.13.0 — the
runtime escape hatch that forwards a raw string to TensorZero's
`input.system` field.  Same agent, three different runtime-supplied
personalities (haiku, pirate, lawyer).

Companion to [`examples/03_system_prompt/`](../03_system_prompt/),
which demonstrates the **recommended** pattern: system prompts in
`tensorzero.toml` via `system_template`, with template variables
flowing through `system_arguments`.

## What this demonstrates

Prompt-as-a-service style flows where the prompt body itself is
*request data* — for example, a product that lets each end-user bring
their own system prompt and runs inference on top of t0-agents.  In
that shape there's nothing to template against, and TensorZero accepts
`input.system` as a plain string as long as the function does not
declare a system schema.

## When to use this vs `system_arguments`

| Use `system_arguments={...}` (default) when… | Use `system="<str>"` (this example) when… |
|---|---|
| The prompt structure lives in `tensorzero.toml` | The prompt body is *request data* (e.g. user-supplied) |
| You want A/B-testable variants by template_id | You don't have a fixed template to A/B test |
| You want structured rows in ClickHouse for fine-tuning | The prompt *is* the user input — structure isn't useful |
| The TensorZero function declares `schemas.system.path` | The function has **no** system schema (HTTP 400 otherwise) |

See [`docs/api-reference.md` § Runtime system prompt](../../docs/api-reference.md#runtime-system-prompt-system-vs-system_arguments)
for the full decision matrix and validation rules.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A `GROQ_API_KEY` environment variable (get one at <https://console.groq.com>)

## How to run

```bash
# 1. Export your API key
export GROQ_API_KEY="gsk_..."

# 2. Start the gateway and ClickHouse
docker compose -f examples/18_raw_system_prompt/docker-compose.yml \
    --env-file .env up -d

# 3. Run the example
uv run python examples/18_raw_system_prompt/main.py

# 4. Tear down when done
docker compose -f examples/18_raw_system_prompt/docker-compose.yml down
```

## What to observe

- Three blocks of output, one per personality.
- The haiku response is three short lines, the pirate response is
  pirate slang, the lawyer response is a terse two-sentence answer —
  all from the **same** TensorZero function, only the runtime
  `system="..."` argument differs.
- The `echo` tool advertised by the MCP server is never called: the
  runtime system prompt steers the model to plain-text replies.

## Also available on

The same `system="..."` kwarg is exposed on every public inference
surface — pick whichever fits the call site:

- `t0_agents.run_agent(..., system="...")` — low-level loop
- `Agent.run(user_message, system="...")` — what this example uses
- `Agent.stream(user_message, system="...")` — streaming variant
- `t0_agents.core.structured.run_structured_inference(..., system="...")` — JSON functions
- `StreamingAgentLoop.stream(..., system="...")` — streaming runner

Validation rules are identical across all five: mutually exclusive
with `system_arguments`, and incompatible with
`memory_strategy="system_template"` on the `Agent`.
