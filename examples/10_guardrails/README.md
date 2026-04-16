# `10_guardrails` — Trusted Agent (M3)

Declarative guardrails composed on an `Agent`:

- **`require_approval(tools=["send_email", "delete_task"])`** — intercepts
  at the client before the tool dispatches. Uses the same
  `elicitation_handler` signature as FastMCP, so one handler powers both
  the client-side approval gate and any server-side `ctx.elicit()` calls.
- **`max_cost(per_episode=0.50)`** — post-inference cost accumulation;
  aborts the run via `GuardrailBlockedError` when `ctx.cost_so_far`
  crosses the cap.
- **`max_tool_calls(per_run=8)`** — counter incremented pre-dispatch;
  race-safe under `parallel_tools=True`.
- **`block_tools(["drop_database"])`** — hard deny-list; the call
  short-circuits with `"ERROR: tool 'drop_database' is blocked by policy"`
  so the LLM can try something else.

## Where each guardrail fires

| Surface | Guardrail | Hook | Outcome |
|---|---|---|---|
| Client before dispatch | `require_approval` | `before_tool` | Soft block (`"ERROR: user declined ..."`) |
| Client before dispatch | `block_tools` | `before_tool` | Soft block (`"ERROR: tool 'drop_database' is blocked by policy"`) |
| Client after inference | `max_cost` | `before_tool` | **Hard** abort (`GuardrailBlockedError`) |
| Client before dispatch | `max_tool_calls` | `before_tool` | **Hard** abort (`GuardrailBlockedError`) |

Every hook invocation is mirrored on `AgentResult.guardrail_log`.

## Running the example

```bash
docker compose up -d                        # boot gateway + ClickHouse
export GROQ_API_KEY=gsk_...                 # (or use .env)
uv run python examples/10_guardrails/main.py
```

Expected interactive session (approximate):

1. `read_calendar` runs without prompting.
2. `send_email(to=alice@example.com, ...)` prompts for approval — type
   `y` to continue, `n` to surface an `ERROR: user declined` result to the
   LLM.
3. If the LLM tries `drop_database` anywhere, you will see a log entry
   like `before_tool rejection block_tools([...])` and the LLM will read
   the policy violation.
4. If the conversation exceeds eight tool calls, the run aborts with
   `GuardrailBlockedError: Tool-call cap 8 hit ...`.

## Caveat: `max_cost` needs provider pricing

`max_cost` reads `response.usage.cost` after every inference. TensorZero
only populates this when pricing is declared in `tensorzero.toml`. The
default config in this repo (`config/tensorzero.toml`) uses Groq-hosted
`gpt_oss_20` without an explicit pricing block, so `Usage.cost = None`
and the cap never fires. Declare pricing (see the TensorZero config
reference for `pricing_per_million_tokens_input`/`_output`) or use
`max_tool_calls` as a deterministic fallback cap.

## Emitting feedback for guardrail outcomes

The framework does **not** auto-emit `t0.feedback(...)` for guardrail
events — every metric must be pre-declared in `tensorzero.toml`
(T0 Q5 resolution). To capture approval rejections as a learnable signal,
add the following block to your config and extend `main.py` with a
post-run feedback call:

```toml
[metrics.action_approved]
type = "boolean"
level = "inference"
optimize = "max"
```

```python
if any(entry.kind == "rejection" for entry in result.guardrail_log):
    await t0.feedback(
        metric_name="action_approved",
        inference_id=result.inference_ids[-1],
        value=False,
    )
```

See [`docs/guardrails.md`](../../docs/guardrails.md) for the full
public guide, including the "Writing your own guardrail" recipes
(regex validator, PII scrubber, schedule-bound execution gate).
