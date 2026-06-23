# Middleware

Middleware is the v1.0 extension surface for cross-cutting concerns
around an agent invocation: memory hydration, guardrail gating, cost
tracking, sub-agent orchestration, JSON-retry recovery, observability
spans.  In v0.12 each of those lived inline in ``core/agent.py``
(~1400 LoC of hand-rolled `_execute_*_legacy` orchestration).  v1.0
collapses them into a Starlette-style chain — small composable
coroutines that wrap a single terminal.

`AgentConfig(memory=, guardrails=, sub_agents=, ...)` materialises the
v0.12-equivalent chain automatically — every existing caller keeps
working byte-identical.  Power users compose custom chains via
`AgentConfig(middleware=[...])` to plug in PII redactors, rate
limiters, observability hooks, and recovery policies that were
impossible in v0.12 without forking the library.

> Companion reading: [`docs/run.md`](run.md) (the invocation handle),
> [`docs/output-modes.md`](output-modes.md) (the canonical home for
> `OutputMode` + the JSON retry handshake),
> [`docs/api-reference.md` — Middleware](api-reference.md#middleware-1)
> (exhaustive symbol table).

---

## The `Middleware` protocol

```python
from typing import Awaitable, Callable, Protocol, runtime_checkable

from t0_agents import AgentResult, RunContext


Next = Callable[[RunContext], Awaitable[AgentResult]]


@runtime_checkable
class Middleware(Protocol):
    name: str

    async def __call__(
        self,
        ctx: RunContext,
        next_: Next,
    ) -> AgentResult: ...
```

A middleware is an object with a `name` string (used for observability
spans and error messages) and an async callable that takes the
current `RunContext` plus a `Next` callable.  The body decides
whether and when to call `next_(ctx)`:

- Call `next_(ctx)` exactly once → setup before, teardown after — the
  canonical pattern.
- Skip `next_(ctx)` → short-circuit; return a synthesised
  `AgentResult` without invoking the terminal (useful for caches or
  hard-deny policy).
- Call `next_(ctx)` twice → raises `T0AgentsError` by default
  (the `_NextGuard` double-call detector).  Retry-style middlewares
  opt in to this pattern via `allow_retry = True` class attribute
  (see [`JsonRetryMiddleware`](#jsonretrymiddleware)).

`middleware[0]` is the **outermost** layer — entered first on the way
in, exited last on the way out.  The terminal sits at the innermost
position.

---

## `RunContext` — the mutable per-Run state

```python
from dataclasses import dataclass, field


@dataclass
class RunContext:
    # 1. Identity / request
    config: AgentConfig
    user_message: str              # mutable — middlewares can rewrite
    episode_id: str | None
    tags: dict[str, str] | None    # mutable
    system: str | None
    system_arguments: dict[str, Any] | None
    output: OutputMode[Any] | None

    # 2. Gateway + tools
    t0: AsyncTensorZeroGateway
    tool_provider: ToolProvider | None  # mutable via replace_tool_provider

    # 3. Per-run scratch
    on_inference: OnInferenceCallback | None = None
    prior_messages: list[Mapping[str, Any]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # 4. Outputs
    result: AgentResult[Any] | None = None
    run: Run[Any] | None = None

    def chain_on_inference(self, cb: OnInferenceCallback) -> None: ...
    def replace_tool_provider(self, tp: ToolProvider) -> None: ...
```

Key fields by behaviour:

- **`user_message: str`** — mutable.  A PII redactor or a
  memory-prepended-message strategy replaces it before the terminal
  runs.  The immutable original is always reachable via
  `ctx.run.input.user_message`.
- **`tags: dict[str, str] | None`** — mutable.  A tenant-scope
  middleware can inject `{"tenant_id": value}` here; downstream
  middlewares and the terminal see the augmented dict.
- **`tool_provider`** — replaced by `ctx.replace_tool_provider(tp)`,
  which stacks the prior provider on
  `ctx.extras["_tool_provider_stack"]` for audit.  Both
  `GuardrailsMiddleware` and `SubAgentMiddleware` wrap the
  tool_provider; the stack reveals the wrapping order.
- **`on_inference`** — composed via `ctx.chain_on_inference(cb)`.
  Multiple middlewares can observe each inference without
  overwriting each other; callbacks fire in registration order.
- **`extras`** — free-form per-middleware scratch.  Convention:
  prefix keys with the middleware name (e.g.
  `"_tool_provider_stack"`, `"_output_attempts"`).

---

## The default chain

`AgentConfig.__post_init__` materialises the default chain from the
existing kwargs:

```
Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal
```

- **`Memory`** present iff `memory=` is set.
- **`SubAgents`** present iff `sub_agents=` is set.
- **`Guardrails`** present iff `guardrails=` is set.
- **`CostTracking`** always present.
- **`JsonRetry`** always present (no-op for `ChatOutput`).

```python
from t0_agents import AgentConfig

config = AgentConfig(
    name="assistant",
    function_name="agent",
    memory=InMemoryMemory(),
    guardrails=[max_cost(per_episode=0.10)],
)
# Chain becomes:
#   Memory → Guardrails → CostTracking → JsonRetry → terminal
print([m.name for m in config.middleware])
# ['memory', 'guardrails', 'cost_tracking', 'json_retry']
```

### Why this ordering?

The order is **load-bearing**.  It preserves the v0.12 invariant
*"guardrails see sub-agent delegations as tool calls"*:

`SubAgentMiddleware` wraps `ctx.tool_provider` with an
`_OrchestrationToolProvider` that dispatches sub-agent invocations
inline.  `GuardrailsMiddleware` wraps the result with a
`GuardedToolProvider` that fires `before_tool` / `after_tool` hooks.
Because each `replace_tool_provider` stacks on the previous one and
the wrap that runs **last** ends up **outermost**, running
`SubAgents` before `Guardrails` makes `GuardedToolProvider` the
outer wrap — so every tool call (including sub-agent dispatches)
fires guardrails first.

Re-ordering them is a silent break of the guardrails contract:
the v0.12 invariant that guardrails fire on sub-agent delegations
only holds when `SubAgentMiddleware` runs *before* `GuardrailsMiddleware`
so the latter ends up the outer wrap around the orchestration tool
provider.

---

## The 6 built-in middlewares

| Class | Purpose | When fires |
|---|---|---|
| `MemoryMiddleware` | Hydrate prior context; persist a record at the end | `before_run` (load + inject), `finally` (save) |
| `SubAgentMiddleware` | Wrap `tool_provider` with `_OrchestrationToolProvider`; thread `parent_episode_id` ContextVar | wrap on entry, restore on exit |
| `GuardrailsMiddleware` | Fire `before_run` / `after_run`; wrap `tool_provider` with `GuardedToolProvider`; install cost-tracker for `max_cost` | full lifecycle |
| `CostTrackingMiddleware` | Accumulate token + cost totals onto `Run.cost()` / `Run.usage()` | per-inference via `on_inference` |
| `JsonRetryMiddleware` | Drive parse-failure retries for single-shot output modes | wraps terminal; loops on `Recovery.retry(hint)` |
| `ObservabilityMiddleware` | Span-wrap the run via any `tracer.start_as_current_span(name)` | wraps terminal |

### `MemoryMiddleware`

Dispatches on `AgentConfig.memory_strategy`:

- `"prepended_message"` (default) — prepends a rendered summary of
  prior records to `ctx.user_message`.
- `"system_template"` — surfaces the summary under
  `ctx.system_arguments["memory_context"]`.  Caller-explicit
  `system_arguments` win on key collisions.
- `"messages"` — populates `ctx.prior_messages` via
  `memory.load_history(episode_id)`.  Caller-explicit
  `prior_messages` (e.g. from `Run.fork`) take precedence and skip
  the load entirely.

Save runs in `finally` so the record persists even on exception paths
when the run produced a result; failures of load or save are logged
but never propagate (memory is best-effort).

### `SubAgentMiddleware`

Activates only when `AgentConfig.sub_agents` is non-empty.  Builds an
`_OrchestrationToolProvider` that intercepts `call_tool(...)` for the
configured sub-agent tool names and dispatches them inline as
`sub_agent.invoke(...).result()` — no FastMCP transport round-trip,
no extra latency.  Refreshes the `parent_episode_id` ContextVar so
sub-Runs tag their inferences with the parent's episode id, enabling
ClickHouse hierarchy reconstruction.

### `GuardrailsMiddleware`

Wraps the **entire run** in a single `try: ... finally:` so
`after_run` fires once around the lifecycle — including the
fatal-abort-in-`before_run` edge case (covered by
`tests/test_guardrails/test_agent_with_guardrails.py::TestAfterRunWithFatalAbort`).
Registers a cost-tracker callback via `ctx.chain_on_inference` so the
`max_cost` guardrail's per-inference accumulator keeps working.

### `CostTrackingMiddleware`

Always-on.  Reads `response.usage.{input_tokens, output_tokens, cost}`
on every inference via `ctx.chain_on_inference` and updates a
`RunUsage` accumulator on the live `Run`.  After the run completes,
`run.cost()` returns the aggregated cost (float) and `run.usage()`
returns the `RunUsage` (or `None` if no inference fired).

When the caller overrides the chain with `middleware=[]`, the tracker
doesn't run — `run.cost()` returns `0.0` and `run.usage()` returns
`None`.  This is intentional: explicit "no middleware"
means **no middleware** including cost tracking.

For the footprint-vs-billing distinction, the cache-token fields,
`cost_is_priced`, and the per-conversation accumulator, see
[Token usage](usage.md).

### `JsonRetryMiddleware`

No-op pass-through for chat agents (`output.expects_tool_loop is True`).
For single-shot agents (`JsonOutput`, `StructuredOutput`, `ToolCallOutput`):

1. Initialise `attempts: list[Attempt] = []` on `ctx.extras["_output_attempts"]`.
2. Call `next_(ctx)` — the terminal materialises a result via
   `ctx.output.materialize(response, attempts)`.
3. If `result.diagnostics.parse_strategy != "none"` → success; return.
4. Otherwise build a `ParseFailure`, append an `Attempt`, and call
   `ctx.output.recover(failure, attempts)`:
   - `Recovery.fail()` → return the failed result as-is so the v0.12
     `parsed=None, parse_strategy="none"` shape flows through.
   - `Recovery.accept(value)` → synthesise an `AgentResult` carrying
     the coerced value with `parse_strategy="permissive"`.
   - `Recovery.retry(hint)` → append the prior user/assistant turn
     pair to `ctx.prior_messages`, set `ctx.user_message` to the
     corrective hint, and loop.

This middleware needs to call `next_(ctx)` more than once per
invocation, so it sets the class attribute `allow_retry = True` —
`build_pipeline` skips the `_NextGuard` for this layer.  See
[Custom middleware authoring → `allow_retry`](#custom-middleware-authoring) below.

### `ObservabilityMiddleware`

Opt-in only.  Wraps `next_(ctx)` in
`tracer.start_as_current_span(f"agent.invoke/{config.name}")`.
Accepts any tracer with that method:

- OpenTelemetry: `from opentelemetry import trace; tracer = trace.get_tracer("my-service")`.
- Datadog: `from ddtrace import tracer`.
- Custom: any duck-typed object with `start_as_current_span(name)`.

`tracer=None` (the default) falls back to `contextlib.nullcontext()`
so the middleware is always safe to include in a chain.

---

## The `middleware=[]` opt-out

Passing `middleware=[]` is the explicit "no middleware" override —
distinct from `middleware=None` (use the auto-materialised default
chain):

```python
# Auto-materialise default chain (Memory + CostTracking + JsonRetry).
config = AgentConfig(name="...", function_name="...", memory=mem)

# Explicit empty chain — NO middleware runs.
config = AgentConfig(name="...", function_name="...", middleware=[])
```

**Important trade-off**: with `middleware=[]`,
`CostTrackingMiddleware` doesn't run, so `Run.cost()` returns `0.0`
and `Run.usage()` returns `None`.  Guardrails configured on
`AgentConfig(guardrails=[...])` are also skipped because the
materialiser that turns them into a `GuardrailsMiddleware` doesn't
run.  Use `middleware=[]` only when you know what you're opting out
of.

A common pattern for power users who want **partial** override:
sandwich the default chain.

```python
from t0_agents import AgentConfig, ObservabilityMiddleware

config = AgentConfig(
    name="traced",
    function_name="agent",
    memory=InMemoryMemory(),
    guardrails=[max_cost(per_episode=0.10)],
    middleware=[
        ObservabilityMiddleware(tracer=my_tracer),  # outermost
        *AgentConfig.default_middleware_for(
            memory=InMemoryMemory(),
            guardrails=[max_cost(per_episode=0.10)],
            sub_agents=None,
        ),
    ],
)
```

`AgentConfig.default_middleware_for(...)` is the single source of
truth for the default chain shape — call it explicitly to compose
around it without re-deriving the ordering by hand.

---

## Custom middleware authoring

A middleware is any object with a `name` string and an `async
__call__(ctx, next_)`.  No inheritance is required — duck typing via
the `runtime_checkable` Protocol does the rest.

### Worked example: PII redactor

A 30-line middleware that scrubs email addresses out of
`ctx.user_message` before the terminal runs:

```python
import re
from typing import Any
from t0_agents import RunContext, AgentResult


class PiiRedactor:
    """Apply (regex, replacement) pairs to ctx.user_message."""

    name = "pii_redactor"

    def __init__(self, patterns: list[tuple[re.Pattern[str], str]]) -> None:
        self._patterns = patterns

    async def __call__(self, ctx: RunContext, next_) -> AgentResult[Any]:
        redacted = ctx.user_message
        for pattern, replacement in self._patterns:
            redacted = pattern.sub(replacement, redacted)
        ctx.user_message = redacted
        return await next_(ctx)


# Use it:
EMAIL = re.compile(r"[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d{1,3}[- ]?\d{3,4}[- ]?\d{3,4}")

config = AgentConfig(
    name="redacted",
    function_name="agent",
    middleware=[
        PiiRedactor(patterns=[(EMAIL, "[EMAIL]"), (PHONE, "[PHONE]")]),
        *AgentConfig.default_middleware_for(),
    ],
)
```

The immutable original is on `ctx.run.input.user_message`, so the
memory record stores the un-redacted form and audit isn't lost.

### `allow_retry = True` — re-invoking the inner chain

The `_NextGuard` wrapping each `next_` raises `T0AgentsError` on the
second call.  If your middleware genuinely needs to re-invoke the
inner chain (retry pattern, recovery, replay), opt out of the guard:

```python
class RetryOnce:
    """Retry once if the inner chain raises a transient error."""

    name = "retry_once"
    allow_retry = True   # class attribute — skips _NextGuard

    async def __call__(self, ctx, next_):
        try:
            return await next_(ctx)
        except TransientError:
            return await next_(ctx)
```

Without `allow_retry = True`, the second call would raise a clear
"middleware called next_() more than once" error — the guard exists
because the second invocation enters the inner chain with potentially
mutated state, producing surprising results.  Use the opt-in
deliberately; the most common consumers are retry/recovery loops
that own the mutation explicitly.

`JsonRetryMiddleware` uses this opt-in; see the
[`JsonRetryMiddleware`](#jsonretrymiddleware) section for the full
retry handshake.

---

## Composing middleware order — practical patterns

The chain order is the contract.  Some patterns and their reasoning:

- **Outermost = observation, innermost = orchestration.**
  `ObservabilityMiddleware` makes sense at position 0 — its span
  encloses every other middleware's work.  `JsonRetryMiddleware` sits
  innermost so memory and guardrails see only the final successful
  inference (a failed retry is not a "real" outcome the outer layers
  should observe).

- **Tool-provider wrappers come after observers, before terminal.**
  `SubAgentMiddleware` and `GuardrailsMiddleware` both stack
  `replace_tool_provider`; the **later** middleware ends up the
  **outer** wrap because each wrap layers on top of the current
  provider.  Run `SubAgents` first to get `Guarded(_Orchestration(client))`.

- **Cost tracking after policy.**  `CostTrackingMiddleware` runs
  after `GuardrailsMiddleware` so its inference observer is composed
  on top of the guardrail's cost-budget tracker (both subscribe via
  `ctx.chain_on_inference`).  Subscription order in chain order
  preserves the guardrail's per-budget semantics.

If you replace the default chain entirely, preserve these invariants
manually or call `AgentConfig.default_middleware_for(...)` and
sandwich your additions around it.

---

## What middleware can't do

A few intentional non-goals to set expectations:

- **Modify the gateway.**  `ctx.t0` is read-only; replacing it
  mid-run would invalidate every downstream middleware's
  assumption.  Tag-based routing belongs in `AgentConfig.tags=` /
  `agent.invoke(tags=...)`.
- **Bypass `mcp/`.**  The terminal calls `run_agent(...)` (or its
  streaming / single-shot peer).  `mcp/` is an invariant — middleware
  can't reach inside it.  If you want per-tool behaviour, wrap
  `ctx.tool_provider`.
- **Survive across runs.**  Each `Agent.invoke(...)` creates a fresh
  `RunContext`; middleware instance state (e.g. a
  `BulkheadMiddleware` semaphore) is fine, but `ctx`-level mutation
  is per-run.  Cross-run state must be on the middleware instance.

These constraints keep the chain predictable — the boundaries to the
gateway, to `mcp/`, and across runs stay clear.

---

## Related

- [`docs/run.md`](run.md) — the `Run` handle that the chain produces;
  `Run.cost()` / `Run.usage()` read live data populated here.
- [`docs/usage.md`](usage.md) — footprint vs. cumulative billing, the
  cache fields, `cost_is_priced`, and the serve `usage` block, all
  populated by `CostTrackingMiddleware`.
- [`docs/output-modes.md`](output-modes.md) — `OutputMode[T]` and the
  `JsonRetryMiddleware` retry handshake.
- [`docs/multi-agent.md`](multi-agent.md) — `SubAgentMiddleware` and
  the `<name>_ask` sub-agent dispatch path.
- [`docs/guardrails.md`](guardrails.md) — the policy layer that runs
  via `GuardrailsMiddleware`.
- [`docs/memory.md`](memory.md) — the persistence layer that runs via
  `MemoryMiddleware`.
- [`docs/api-reference.md` — Middleware](api-reference.md#middleware-1)
  — exhaustive symbol table.
- [`docs/glow/escape-hatch.md`](glow/escape-hatch.md) — per-call
  middleware override from a `@app.agent` wrapper via
  `wrapper._agent.invoke(..., middleware=[...])`.
