# Guardrails

Declarative safety layer on top of an `Agent`. Compose one or more
`Guardrail` instances to gate sensitive tool calls, cap per-run cost,
abort at a tool-call ceiling, or hard-deny specific tool names — all
without inventing a parallel Approval system.

## Installation

Guardrails ship in the base package — no extras required:

```bash
pip install t0-agents
```

## Quick start

```python
from t0_agents import Agent, AgentConfig
from t0_agents.guardrails import (
    require_approval,
    max_cost,
    max_tool_calls,
    block_tools,
)

agent = Agent(
    AgentConfig(
        name="executive_assistant",
        function_name="agent",
        guardrails=[
            require_approval(
                tools=["send_email", "make_purchase"],
                handler=cli_elicitation_handler,
            ),
            max_cost(per_episode=0.50),        # USD
            max_tool_calls(per_run=15),
            block_tools(["drop_database"]),
        ],
    ),
    t0=t0,
    tool_provider=client,
)

result = await agent.run("Draft a reply to Alice and send it.")
```

## What is a `Guardrail`?

`Guardrail` is a `@runtime_checkable` Protocol defined in
`t0_agents.core.types`. Any object implementing these four async hooks
and a `name` attribute satisfies it:

```python
class Guardrail(Protocol):
    name: str

    async def before_run(self, ctx: GuardrailContext) -> None: ...
    async def before_tool(
        self, ctx: GuardrailContext, tool_call: ToolCall,
    ) -> str | None: ...
    async def after_tool(
        self, ctx: GuardrailContext, tool_call: ToolCall, result_str: str,
    ) -> None: ...
    async def after_run(
        self, ctx: GuardrailContext, result: AgentResult | None,
    ) -> None: ...
```

### Return semantics

- `before_tool` returning `None` → pass through; the inner tool runs.
- `before_tool` returning `str` → soft-block; the string is surfaced to
  the LLM as the tool's error result (single `"ERROR: "` prefix added by
  the adapter). The run continues; the LLM may try another tool.
- Raising `GuardrailBlockedError` (a `T0AgentsFatalError`) from any hook
  → hard abort; the exception propagates past the agent loop and out of
  `Agent.run()`.

## Built-in guardrails

### `require_approval`

Intercept specific tool names and ask the user via the same callable
signature as FastMCP's `elicitation_handler`. You can reuse the exact
handler you already wired on `Client(elicitation_handler=...)`.

```python
from fastmcp.client.elicitation import ElicitResult

async def cli_handler(message, response_type, params, context):
    if response_type is None:
        yes = input(f"{message} [y/n]: ").strip().lower() in ("y", "yes")
        return ElicitResult(
            action="accept" if yes else "decline", content={},
        )
    ...

guard = require_approval(tools=["send_email"], handler=cli_handler)
```

When `handler is None`, `require_approval` soft-blocks with
`"ERROR: approval required for <tool> but no handler is configured"` and
logs a single WARNING per run.

### `max_cost`

Cumulative cost ceiling for a single `Agent.run()`. Reads
`response.usage.cost` from each `ChatInferenceResponse` via the
framework's `on_inference` callback. A missing cost (provider-side)
counts as `0.0`.

```python
max_cost(per_episode=0.50, currency="USD")
```

Cost is **post-inference**, so the guardrail is not a predictive budget —
it catches overruns at the next tool-call boundary.

> **Cost reporting depends on TOML pricing.** TensorZero only populates
> `Usage.cost` when the model's pricing is declared in `tensorzero.toml`
> (via `[models.<name>.providers.<p>]`'s `pricing_per_million_tokens_*`
> fields, or the provider's own pricing registry). Models without
> explicit pricing — for example Groq-hosted `gpt_oss_20` in this
> project's default config — return `Usage.cost = None`, which the
> tracker treats as `0.0`. If you need a deterministic cap regardless of
> provider pricing, use `max_tool_calls(per_run=...)` as a fallback.

### `max_tool_calls`

Cap the number of tool invocations per run. Counter increments at
dispatch time (race-safe under `parallel_tools=True`).

```python
max_tool_calls(per_run=15)
```

### `block_tools`

Deny-list specific tool names. Soft-blocks (the LLM reads the policy
error and may pivot). Does NOT filter the output of
`Client.list_tools()` — enforcement is at call time.

```python
block_tools(["drop_database", "wire_transfer"])
```

## Writing your own guardrail

The Protocol's four hooks are all optional — implement only what you
need. A no-op implementation of the other three hooks satisfies
`runtime_checkable`.

### Example 1 — regex input validator

```python
import re
from t0_agents.core.types import Guardrail
from t0_agents.guardrails.base import GuardrailContext

def block_arguments_matching(pattern: str, *, reason: str) -> Guardrail:
    rx = re.compile(pattern)

    class _BlockArgs:
        name = f"block_arguments_matching({pattern!r})"

        async def before_run(self, ctx): ...
        async def after_tool(self, ctx, tool_call, result_str): ...
        async def after_run(self, ctx, result): ...

        async def before_tool(self, ctx, tool_call):
            if rx.search(repr(tool_call.arguments)):
                return f"ERROR: blocked by policy ({reason})"
            return None

    return _BlockArgs()
```

### Example 2 — PII scrubber stub (audit-only)

```python
def audit_pii() -> Guardrail:
    class _Audit:
        name = "audit_pii"

        async def before_run(self, ctx): ...
        async def before_tool(self, ctx, tool_call): ...
        async def after_run(self, ctx, result): ...

        async def after_tool(self, ctx, tool_call, result_str):
            # Bring your own classifier here.
            if "ssn" in result_str.lower():
                ctx.extra.setdefault("pii_hits", []).append(tool_call.name)

    return _Audit()
```

### Example 3 — schedule-bound execution gate

```python
from datetime import datetime, time

def business_hours_only() -> Guardrail:
    start, end = time(9, 0), time(18, 0)

    class _Hours:
        name = "business_hours_only"

        async def before_run(self, ctx):
            now = datetime.now().time()
            if not (start <= now <= end):
                from t0_agents.core.exceptions import GuardrailBlockedError
                raise GuardrailBlockedError(
                    "agent runs are restricted to 09:00–18:00"
                )

        async def before_tool(self, ctx, tool_call): ...
        async def after_tool(self, ctx, tool_call, result_str): ...
        async def after_run(self, ctx, result): ...

    return _Hours()
```

## The guardrail log

Every hook activation appends a `GuardrailLogEntry` to
`AgentResult.guardrail_log`:

```python
@dataclass
class GuardrailLogEntry:
    guardrail_name: str
    hook: Literal["before_run", "before_tool", "after_tool", "after_run"]
    kind: Literal["pass", "rejection", "block", "modification"]
    tool_name: str | None = None
    detail: str = ""
```

The log is populated even when the run aborts with a fatal error. When
`GuardrailBlockedError` propagates, `after_run` is still invoked (via
`try`/`finally`) with `result=None` so your audit guardrails always get
a chance to record the abort.

## Concurrency

`GuardrailContext` is **per-run**, not per-agent. Two sequential
`Agent.run()` calls on the same `Agent` instance each get a fresh
context — counters start at zero, `cost_so_far` starts at zero, and
`extra` starts empty.

Under `parallel_tools=True`, multiple tool dispatches share one
`GuardrailContext`. The built-in counters mutate inside `before_tool`
(pre-dispatch), which runs to its first `await` before other coroutines
can interleave — so increments are atomic in practice. If your custom
guardrail needs more, put an `asyncio.Lock` on `ctx.extra`.

## Emitting feedback for guardrail outcomes

**The framework does not auto-emit `t0.feedback(...)` for guardrail
events.** TensorZero requires every metric to be pre-declared in
`tensorzero.toml`, so forcing an `[metrics.action_approved]` block on
every deployer would be over-reach. Inspect the guardrail log and emit
feedback manually:

```toml
# tensorzero.toml
[metrics.action_approved]
type = "boolean"
level = "inference"
optimize = "max"
```

```python
result = await agent.run("Send the email.")

if any(entry.kind == "rejection" for entry in result.guardrail_log):
    await t0.feedback(
        metric_name="action_approved",
        inference_id=result.inference_ids[-1],
        value=False,
    )
```

## Error model

- `GuardrailBlockedError` — hard abort. Inherits from
  `T0AgentsFatalError`, which the agent loop's
  `_execute_single_tool_call` re-raises past its belt-and-braces
  `except Exception` trap so the exception surfaces verbatim.
- Non-fatal exceptions raised from guardrail hooks (bugs in your own
  guardrail) are caught by `GuardedToolProvider`, logged at ERROR with
  the stack trace, and treated as pass for `before_tool` / noop for the
  other hooks. Your tool call still runs — a broken guardrail cannot
  corrupt the agent loop.

## See also

- `examples/10_guardrails/` — all four built-ins composed on one agent.
- `docs/elicitation.md` — background on the FastMCP elicitation
  signature that `require_approval` rides.
- `docs/agent.md` — `AgentConfig` reference, including the `guardrails`
  field and per-run lifecycle.
