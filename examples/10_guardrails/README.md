# `10_guardrails` — Built-in guardrails on one Agent

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
events — TensorZero requires every metric to be pre-declared in
`tensorzero.toml`, so the library surfaces the audit trail on
`AgentResult.guardrail_log` and leaves emission to the caller. To capture
approval rejections as a learnable signal, add the following block to
your config and extend `main.py` with a post-run feedback call:

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

## Custom guardrail — `custom_guardrail.py`

A second runnable script illustrates the Advanced tier from the public
doc: writing your own `Guardrail` with a domain-specific
`T0AgentsFatalError` subclass, stateful accumulation on
`GuardrailContext.extra`, and a summary emitted from `after_run` (which
fires even on fatal abort).

```bash
uv run python examples/10_guardrails/custom_guardrail.py
```

The scenario asks the agent to send an email to a recipient outside the
allowlist. The custom `enforce_email_allowlist` guardrail inspects
`tool_call.arguments["to"]`, extracts the domain, records the attempt on
`ctx.extra["email_attempts"]`, and raises `UnauthorizedRecipientError`
(a `T0AgentsFatalError`). Because `after_run` fires in the `finally`
branch of `Agent.invoke`, the audit summary is logged regardless of whether
the run completed.

## See also

See [`docs/guardrails.md`](../../docs/guardrails.md) for the full public
guide — protocol reference, lifecycle diagram, observability hooks, and
production notes.
