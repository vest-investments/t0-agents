# Guardrails

Memory decides what the agent knows. Guardrails decide what it may do. This chapter covers the declarative policy layer that sits between the LLM's tool-call request and the actual dispatch — approval gates for sensitive tools, hard caps on cost and tool-call counts, deny-lists for dangerous names, and a Protocol for writing your own domain rules.

Declarative policy layer for an `Agent`. Compose one or more `Guardrail`
instances to gate sensitive tool calls, cap cost and tool-call budgets,
hard-deny specific tool names, or encode your own domain rules — all
without inventing a parallel approval system. Every activation is
recorded on `AgentResult.guardrail_log` so you can audit runs and emit
manual TensorZero feedback.

The subsystem ships four built-ins:

| Factory | Outcome on breach | Surface |
|---|---|---|
| `require_approval(tools=..., handler=...)` | Soft-block (error string to the LLM) | Before dispatch |
| `max_cost(per_episode=...)` | **Hard abort** (`GuardrailBlockedError`) | Before dispatch, driven by `on_inference` |
| `max_tool_calls(per_run=...)` | **Hard abort** (`GuardrailBlockedError`) | Before dispatch |
| `block_tools(names=...)` | Soft-block (error string to the LLM) | Before dispatch |

Bring your own by implementing the structural `Guardrail` Protocol —
four optional async hooks and a `name` attribute. No inheritance required.

---

## Installation

Guardrails ship in the base package. No extras are needed:

```bash
pip install t0-agents
# or:
uv add t0-agents
```

The subsystem depends only on `t0_agents.core` and on FastMCP's
`ElicitResult` (already pulled in by the library). The MCP gateway
([Getting Started](getting-started.md)) must be running for any live
example — guardrails wrap the existing agent loop, they do not replace it.

---

## Quick start

```python
import asyncio
from typing import Any

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, GuardrailBlockedError
from t0_agents.guardrails import (
    block_tools,
    max_cost,
    max_tool_calls,
    require_approval,
)


async def cli_handler(message, response_type, params, context):
    """FastMCP-shaped elicitation handler reused by require_approval."""
    if response_type is None:
        yes = input(f"{message} [y/n]: ").strip().lower() in ("y", "yes")
        return ElicitResult(
            action="accept" if yes else "decline", content={},
        )
    # (structured-input branch omitted for brevity — see examples/10_guardrails/)
    return ElicitResult(action="decline")


async def main() -> None:
    gateway = await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )
    client: Client[Any] = Client(mcp, elicitation_handler=cli_handler)
    config = AgentConfig(
        name="executive_assistant",
        function_name="agent",
        tool_timeout=None,  # humans can take their time
        guardrails=[
            require_approval(
                tools=["send_email", "make_purchase"],
                handler=cli_handler,
            ),
            max_cost(per_episode=0.50),       # USD
            max_tool_calls(per_run=15),
            block_tools(["drop_database"]),
        ],
    )

    async with gateway as t0, client:
        agent = Agent(config, t0=t0, tool_provider=client)
        try:
            result = await agent.run("Draft a reply to Alice and send it.")
        except GuardrailBlockedError as exc:
            print(f"Aborted by guardrail: {exc}")
            return

    print(result.response)
    for entry in result.guardrail_log:
        print(f"{entry.hook:11} {entry.kind:10} {entry.guardrail_name}")


asyncio.run(main())
```

The full runnable example — including the server, the structured-input
branch of the handler, and the expected interactive session — lives at
[`examples/10_guardrails/`](../examples/10_guardrails/).

---

## Concepts

### What is a `Guardrail`?

`Guardrail` is a `@runtime_checkable` Protocol defined in
`t0_agents.core.types`. Any object that exposes a `name` string and the
four async hooks below satisfies it — `isinstance(obj, Guardrail)` returns
`True` without inheritance:

```python
from typing import Protocol, runtime_checkable
from t0_agents import AgentResult


@runtime_checkable
class Guardrail(Protocol):
    name: str

    async def before_run(self, ctx) -> None: ...
    async def before_tool(self, ctx, tool_call) -> str | None: ...
    async def after_tool(self, ctx, tool_call, result_str: str) -> None: ...
    async def after_run(self, ctx, result: AgentResult | None) -> None: ...
```

`ctx` is a `GuardrailContext` constructed fresh for every `Agent.run()`
call. `tool_call` carries `name` (`str`) and `arguments` (`dict[str, Any]`).
`result_str` is the already-serialised tool result — identical to the
string the LLM will read next turn.

### Return semantics

`before_tool` is the primary decision point:

| Returns | Meaning |
|---|---|
| `None` | Pass through; the inner `call_tool` runs. |
| `str` | **Soft-block**. The string becomes the tool's error result. A single `"ERROR: "` prefix is added by the adapter — do not double it. The run continues; the LLM may try another tool. |
| Raises `GuardrailBlockedError` | **Hard abort**. The exception propagates past the agent loop and out of `Agent.run()`. Use for budget caps and policy violations that must stop execution. |

The other three hooks are observational (`before_run`, `after_tool`,
`after_run`) — they return `None` and any exception they raise is treated
as non-fatal (logged, then the run continues). A custom hook that needs to
abort should raise a subclass of `T0AgentsFatalError`, which bypasses the
non-fatal trap.

### Lifecycle

`Agent.run()` wraps the caller's `tool_provider` in a
`GuardedToolProvider` when `AgentConfig.guardrails` is non-empty. The
wrapper threads a shared `GuardrailContext` through every hook:

```
Agent.run(message)
    │
    ├─ construct GuardrailContext(agent_name, function_name, user_message)
    ├─ for g in guardrails: await g.before_run(ctx)
    │
    ├─ enter turn loop (run_agent with on_inference=cost_tracker)
    │    ├─ response = await t0.inference(...)
    │    ├─ cost_tracker: ctx.cost_so_far += response.usage.cost or 0.0
    │    │                ctx.turn += 1
    │    ├─ for tool_call in response.tool_calls:
    │    │       GuardedToolProvider.call_tool(name, arguments)
    │    │           ├─ for g: await g.before_tool(ctx, tool_call)
    │    │           │         └─ None → pass | str → short-circuit | raise → abort
    │    │           ├─ (pass)       result = await inner.call_tool(...)
    │    │           ├─ (soft-block) result = synthetic CallToolResult(is_error=True)
    │    │           └─ for g: await g.after_tool(ctx, tool_call, serialized_result)
    │    └─ loop until the LLM stops emitting tool calls or max_turns hits
    │
    └─ finally:
           for g in guardrails: await g.after_run(ctx, result or None)
           return AgentResult(..., guardrail_log=collected_entries)
```

Every hook activation — pass, rejection, or error — appends a
`GuardrailLogEntry` to the shared log. `after_run` fires even when the
run aborted before producing an `AgentResult`, so audit guardrails always
see the abort and can act on `result is None`.

### Why a handler kwarg on `require_approval`?

FastMCP's `Client` stores its elicitation handler in
`_session_kwargs["elicitation_callback"]` with no public getter. A
client-side guardrail cannot read the `Client`'s handler back out, so
`require_approval` takes its own `handler` reference at construction.

The recommended pattern is **one callable, two surfaces**: define your
elicitation handler once, wire it on both
`Client(elicitation_handler=my_handler)` and
`require_approval(handler=my_handler)`. The server uses it when a tool
calls `ctx.elicit()`; the guardrail uses it when a gated tool is about to
dispatch. Users see the same UX for both paths.

For HTTP and WebSocket deployments where the human lives in a different
process, pass `ElicitationForwarder`'s `handler` attribute instead — see
[Production considerations](#production-considerations).

---

## Built-in guardrails

### `require_approval`

Intercepts the tools you list and asks the user via an elicitation-shaped
callable. Gated tools pause at `before_tool`; non-gated tools pass through
untouched.

```python
from fastmcp.client.elicitation import ElicitResult

async def cli_handler(message, response_type, params, context):
    if response_type is None:
        yes = input(f"{message} [y/n]: ").strip().lower() in ("y", "yes")
        return ElicitResult(
            action="accept" if yes else "decline", content={},
        )
    return ElicitResult(action="decline")

guard = require_approval(
    tools=["send_email", "make_purchase"],
    handler=cli_handler,
    prompt_template="Approve calling {tool_name} with {arguments}?",
)
```

**Behaviour**

- Non-gated tool → pass.
- Handler returns `ElicitResult(action="accept", ...)` → pass.
- Handler returns `action="decline"` or `"cancel"` → soft-block
  `"ERROR: user declined to run '<tool>'"`.
- Handler returns a raw dict (not an `ElicitResult`) → normalised to
  `accept` with the dict as `content`. Matches FastMCP's callback wrapper.

**Edge cases**

- `handler=None` → soft-block `"ERROR: approval required for '<tool>'
  but no handler is configured"`, plus one `WARNING` log per run.
- Handler raises any exception → soft-block
  `"ERROR: approval handler failed for '<tool>': <exc>"`. Never aborts
  the run; a broken handler degrades to a policy decline.
- `prompt_template` supports `{tool_name}` and `{arguments}` placeholders.

### `max_cost`

Cumulative cost ceiling for a single `Agent.run()`. The framework reads
`response.usage.cost` from each `ChatInferenceResponse` via an
`on_inference` hook that fires *before* tools dispatch, so the cap can
abort on the very next tool call after a budget-breaking inference.

```python
max_cost(per_episode=0.50, currency="USD")
```

**Behaviour**

- Checks `ctx.cost_so_far < per_episode` at `before_tool`.
- Breach → raises `GuardrailBlockedError("Cost limit … exceeded …")`.
- Cost is **post-inference**, so this is a corrective cap rather than a
  predictive budget.

**Edge cases**

- `per_episode <= 0` → raises `ValueError` at construction.
- `response.usage.cost is None` (provider without declared pricing) → the
  tracker treats it as `0.0`. See the next box.
- `currency` is cosmetic only; no FX conversion is performed.

> **Cost reporting depends on TensorZero pricing.** TensorZero only
> populates `Usage.cost` when the model's pricing is declared in
> `tensorzero.toml` (via `pricing_per_million_tokens_input` /
> `_output` on the provider) or when the provider ships with built-in
> rates. Models without pricing return `Usage.cost = None`, which the
> tracker treats as `0.0` — the cap will never fire. For a deterministic
> budget regardless of provider pricing, pair `max_cost` with
> `max_tool_calls(per_run=…)`.

### `max_tool_calls`

Caps the number of tool invocations per run. The counter increments at
dispatch time (inside `before_tool`), not after completion.

```python
max_tool_calls(per_run=15)
```

**Behaviour**

- Increments `ctx.tool_calls_made` before any `await` point. Under
  `parallel_tools=True` the asyncio runtime runs each coroutine to its
  first await without yielding, so the `+=` is atomic in practice.
- Breach → raises `GuardrailBlockedError("Tool-call cap … hit …")`.

**Edge cases**

- `per_run=0` is a valid "no tool calls allowed" configuration (read-only
  agent) — any dispatch attempt aborts the run.
- `per_run<0` → raises `ValueError` at construction.

### `block_tools`

Deny-lists specific tool names. The block fires at call time, so a name
that the LLM never attempts passes through with zero overhead.

```python
block_tools(["drop_database", "wire_transfer"])
```

**Behaviour**

- Soft-block: returns `"ERROR: tool '<name>' is blocked by policy"`.
  The LLM reads the policy violation and may pick a different tool.
- Every blocked attempt is recorded with `kind="rejection"` on the
  guardrail log.

**Edge cases**

- `block_tools` does **not** filter `Client.list_tools()` output. The LLM
  still sees the tool name and may attempt to call it — enforcement is at
  `before_tool` only. This asymmetry is deliberate: it keeps the
  wrapper's `list_tools` byte-identical to the inner provider and catches
  LLMs that hallucinate tool names the server does not advertise.

### `require_parsed`

Opt-in counterpart to the M3 permissive-parsing layer for JSON / structured-
output agents. Hard-aborts a JSON run that completes with `result.parsed is
None` — i.e. TensorZero's strict parser, the in-Python permissive layer, and
every retry attempt all failed. Chat runs are a strict no-op because
`parsed` is structurally absent on the chat path.

```python
from t0_agents.guardrails import require_parsed

AgentConfig(
    function_kind="json",
    json_mode="strict",
    guardrails=[require_parsed()],
)
```

**Behaviour**

- Fires in `after_run`; raises `GuardrailBlockedError` when
  `result.function_kind == "json"` and `result.parsed is None`.
- The error message carries the run's `parse_strategy` and
  `json_parse_attempts` for debugging the layer that "should have" recovered.
- For chat runs the hook returns `None` immediately — adding the guardrail
  to a `function_kind="chat"` config is safe (no false positives).

**When to use it**

- Interactive workflows that must surface a structured object to the user —
  silently returning `parsed=None` is worse than an explicit failure.
- Batch jobs whose downstream consumer cannot handle a null payload — pair
  with the M3 `max_json_retries` knob to define exactly how hard the agent
  tries before giving up.
- Schema-validation pipelines where "parseable JSON" is a strictly weaker
  guarantee than "schema-conforming JSON" — combine with a Pydantic
  validator on `result.parsed` for the conformance check (TZ guide §11.15
  documents that schema conformance is not validated server-side in every
  configuration).

**Edge cases**

- `result is None` (fatal abort teardown) short-circuits; `require_parsed`
  does not synthesise a second exception on top of an existing one.
- The guardrail's `after_run` raises a `T0AgentsFatalError` subclass; per
  the [Return semantics](#return-semantics) section, the fatal trap
  propagates out of `Agent.run(...)` and `Agent.stream(...)`'s lifecycle.
- See [`docs/structured-outputs.md`](structured-outputs.md#opt-in-hard-failure-require_parsed)
  for the structured-outputs angle (recovery layers, when to opt in vs out).

---

## API reference

### Factories (`from t0_agents.guardrails`)

| Symbol | Signature | Returns |
|---|---|---|
| `require_approval` | `require_approval(*, tools: list[str], handler: ApprovalHandler \| None = None, prompt_template: str = "Approve calling {tool_name} with {arguments}?") -> Guardrail` | Interactive approval gate. |
| `max_cost` | `max_cost(*, per_episode: float, currency: str = "USD") -> Guardrail` | Cumulative cost cap. Raises `ValueError` if `per_episode <= 0`. |
| `max_tool_calls` | `max_tool_calls(*, per_run: int) -> Guardrail` | Per-run invocation cap. Raises `ValueError` if `per_run < 0`. |
| `block_tools` | `block_tools(names: list[str]) -> Guardrail` | Soft deny-list. |
| `require_parsed` | `require_parsed(*, name: str = "require_parsed") -> Guardrail` | Hard-abort when a JSON run produces `parsed is None`. No-op on chat runs. |

### Protocol (`from t0_agents import Guardrail` or `from t0_agents.guardrails import Guardrail`)

| Member | Signature | Purpose |
|---|---|---|
| `name` | `str` attribute | Appears in logs and `GuardrailLogEntry.guardrail_name`. |
| `before_run` | `async (ctx) -> None` | Per-run setup. |
| `before_tool` | `async (ctx, tool_call) -> str \| None` | Pass/soft-block/raise decision. |
| `after_tool` | `async (ctx, tool_call, result_str) -> None` | Observe the serialized result. |
| `after_run` | `async (ctx, result: AgentResult \| None) -> None` | Tear-down. Fires even on fatal abort with `result=None`. |

### Dataclasses

`GuardrailContext` (`from t0_agents.guardrails`) — mutable per-run state:

| Field | Type | Purpose |
|---|---|---|
| `agent_name` | `str` | Mirror of `AgentConfig.name`. |
| `function_name` | `str` | Mirror of `AgentConfig.function_name`. |
| `user_message` | `str` | The prompt for this run. |
| `turn` | `int = 0` | Incremented by the `on_inference` callback. |
| `tool_calls_made` | `int = 0` | Incremented by `max_tool_calls` pre-dispatch. |
| `cost_so_far` | `float = 0.0` | Cumulative `Usage.cost`; `None` is treated as `0.0`. |
| `extra` | `dict[str, Any] = {}` | Free-form; the framework never reads it. |

`GuardrailLogEntry` (`from t0_agents import GuardrailLogEntry`) — one hook
activation:

| Field | Type | Purpose |
|---|---|---|
| `guardrail_name` | `str` | The guardrail's `name`. `"__chain__"` for the aggregated pre-tool pass entry. |
| `hook` | `Literal["before_run", "before_tool", "after_tool", "after_run"]` | Which hook fired. |
| `kind` | `Literal["pass", "rejection", "block", "modification"]` | Outcome class. |
| `tool_name` | `str \| None = None` | The tool this hook ran against (for the tool hooks). |
| `detail` | `str = ""` | Human-readable explanation (e.g. the soft-block string). |

### Type aliases

`ApprovalHandler` (`from t0_agents.guardrails`):

```python
ApprovalHandler = Callable[
    [str, type[Any] | None, Any, Any],
    Awaitable[Any],
]
```

Structural alias for FastMCP's `ElicitationHandler`. Returns either an
`ElicitResult` or a raw value (the guardrail wraps raw values into
`ElicitResult(action="accept", content=value)` the same way
`fastmcp.client.elicitation.create_elicitation_callback` does).

### Advanced surfaces

`GuardedToolProvider` (`from t0_agents.guardrails`) — the runtime wrapper
`Agent.run()` installs around the caller's `tool_provider` when guardrails
are configured. Exposed for advanced composition (e.g. wrapping one
guarded provider with another); normal users never instantiate it. The
constructor is:

```python
GuardedToolProvider(
    inner: ToolProvider,
    guardrails: list[Guardrail],
    ctx: GuardrailContext,
    log_sink: list[GuardrailLogEntry],
)
```

### Exceptions (`from t0_agents`)

| Class | Parent | Raised by |
|---|---|---|
| `T0AgentsFatalError` | `T0AgentsError` | Marker base for fatal hook errors. Re-raised past the loop's broad `except Exception`. Subclass it for your own hard aborts. |
| `GuardrailBlockedError` | `T0AgentsFatalError` | Built-in guardrails (`max_cost`, `max_tool_calls`); custom guardrails that hard-abort. |

### Agent wiring

| Symbol | Default | Purpose |
|---|---|---|
| `AgentConfig.guardrails` | `list[Guardrail] = []` | Empty leaves the agent loop unwrapped — byte-identical to an agent configured without any guardrails. |
| `AgentResult.guardrail_log` | `list[GuardrailLogEntry] = []` | Populated per run. Attached even on fatal abort. |
| `run_agent(..., on_inference=...)` | `Callable[[ChatInferenceResponse], Awaitable[None]] \| None = None` | Fires after every `t0.inference()` response, **before** tool execution. `None` (default) is byte-identical to the pre-guardrail loop. Used internally to drive cost tracking; also available to user code that wants a cheap inference-level observability hook. |

---

## Writing your own guardrail

The four hooks are all optional — implement only what you need. A no-op
implementation of the other three hooks satisfies `runtime_checkable`.

### Easy — regex input validator

Block any tool call whose arguments match a pattern. Soft-blocks so the
LLM can pick another tool.

```python
import re

from t0_agents import Guardrail
from t0_agents.guardrails import GuardrailContext


def block_arguments_matching(pattern: str, *, reason: str) -> Guardrail:
    rx = re.compile(pattern)

    class _BlockArgs:
        name = f"block_arguments_matching({pattern!r})"

        async def before_run(self, ctx: GuardrailContext) -> None: ...
        async def after_tool(self, ctx, tool_call, result_str) -> None: ...
        async def after_run(self, ctx, result) -> None: ...

        async def before_tool(self, ctx, tool_call) -> str | None:
            if rx.search(repr(tool_call.arguments)):
                return f"ERROR: blocked by policy ({reason})"
            return None

    return _BlockArgs()
```

### Intermediate — audit-only PII logger

Observe every tool result after dispatch and record hits on
`ctx.extra`. `after_run` then decides what to do with the collected
evidence (log it, emit feedback, update a monitoring counter). This
pattern is valid precisely because it does not change the LLM's view of
the result — the agent keeps running.

```python
from t0_agents import Guardrail
from t0_agents.guardrails import GuardrailContext


def audit_pii() -> Guardrail:
    class _Audit:
        name = "audit_pii"

        async def before_run(self, ctx: GuardrailContext) -> None:
            ctx.extra["pii_hits"] = []

        async def before_tool(self, ctx, tool_call) -> str | None:
            return None

        async def after_tool(self, ctx, tool_call, result_str: str) -> None:
            # Bring your own classifier; simplified here for clarity.
            if "ssn" in result_str.lower() or "credit card" in result_str.lower():
                ctx.extra["pii_hits"].append(tool_call.name)

        async def after_run(self, ctx, result) -> None:
            hits = ctx.extra.get("pii_hits", [])
            if hits:
                # Ship to your monitoring sink here.
                print(f"[pii-audit] run {ctx.agent_name}: {len(hits)} hits")

    return _Audit()
```

### Advanced — stateful hard-abort with a custom fatal error

Ride the `T0AgentsFatalError` marker to build a domain-specific hard
abort. The loop re-raises any `T0AgentsFatalError` past its broad
`except Exception`, so subclasses propagate unchanged.

```python
from datetime import datetime, time

from t0_agents import Guardrail, T0AgentsFatalError
from t0_agents.guardrails import GuardrailContext


class OutOfHoursError(T0AgentsFatalError):
    """Raised when an agent runs outside allowed business hours."""


def business_hours_only(*, start: time, end: time) -> Guardrail:
    class _Hours:
        name = f"business_hours_only({start}-{end})"

        async def before_run(self, ctx: GuardrailContext) -> None:
            now = datetime.now().time()
            if not (start <= now <= end):
                raise OutOfHoursError(
                    f"agent runs are restricted to {start}–{end}; "
                    f"attempted at {now.strftime('%H:%M:%S')}"
                )
            ctx.extra["run_started_at"] = now.isoformat()

        async def before_tool(self, ctx, tool_call) -> str | None: ...
        async def after_tool(self, ctx, tool_call, result_str) -> None: ...
        async def after_run(self, ctx, result) -> None: ...

    return _Hours()
```

A runnable variant of this pattern ships at
[`examples/10_guardrails/custom_guardrail.py`](../examples/10_guardrails/custom_guardrail.py):
an `enforce_email_allowlist` factory that inspects `tool_call.arguments`,
records attempts on `ctx.extra`, raises `UnauthorizedRecipientError` on
off-list domains, and emits an audit summary from `after_run` even when
the run aborts. `main.py` composes all four built-ins (Easy) and a shared
elicitation handler for both client-side and server-side approval
(Intermediate); `custom_guardrail.py` closes the Advanced tier.

---

## The guardrail log

Every hook activation appends a `GuardrailLogEntry` to
`AgentResult.guardrail_log`. The log is populated even when the run
aborts with a fatal error — `after_run` is still invoked (via
`try`/`finally`) with `result=None` so audit guardrails always get a
chance to record the abort.

### Consuming the log

Group entries by `kind` to produce a concise audit output:

```python
from collections import Counter

result = await agent.run("...")
by_kind = Counter(entry.kind for entry in result.guardrail_log)
print(f"passes={by_kind['pass']} rejections={by_kind['rejection']}")

for entry in result.guardrail_log:
    if entry.kind in ("rejection", "block"):
        print(f"  {entry.guardrail_name} on {entry.tool_name}: {entry.detail}")
```

Shapes worth knowing:

- Every tool dispatch that passed the full chain records a single
  `kind="pass"` entry under `guardrail_name="__chain__"`; individual
  `after_tool` passes record the guardrail's own name.
- `kind="rejection"` marks a soft-block — the run continued.
- `kind="block"` is reserved for hard aborts that emit a log entry before
  raising; built-in guardrails raise directly, so this kind appears only
  from user-written guardrails that log before raising.
- `kind="modification"` is reserved for future hooks that rewrite tool
  arguments (not implemented by the built-ins).

---

## Concurrency

`GuardrailContext` is **per run**, not per agent. Two sequential
`Agent.run()` calls on the same `Agent` instance each get a fresh
context — counters start at zero, `cost_so_far` starts at zero, and
`extra` starts empty.

Under `parallel_tools=True`, multiple tool dispatches share one
`GuardrailContext`. The built-in counters mutate inside `before_tool`
**before any `await` point**, which runs atomically under asyncio's
cooperative scheduling — so `tool_calls_made += 1` cannot race.

If your custom guardrail needs more (for example a rolling window over a
`collections.deque` on `ctx.extra`), put an `asyncio.Lock` on the
mutating path:

```python
import asyncio


def _make_guard() -> Guardrail:
    lock = asyncio.Lock()

    class _Guarded:
        name = "rate_limited"

        async def before_tool(self, ctx, tool_call) -> str | None:
            async with lock:
                window = ctx.extra.setdefault("recent", [])
                window.append(datetime.utcnow())
                # ... evaluate window ...
                return None

        async def before_run(self, ctx) -> None: ...
        async def after_tool(self, ctx, tool_call, result_str) -> None: ...
        async def after_run(self, ctx, result) -> None: ...

    return _Guarded()
```

---

## Error model

- **`GuardrailBlockedError`** (from `T0AgentsFatalError`) — hard abort.
  Built by `max_cost`, `max_tool_calls`, or any user guardrail that
  raises. The loop's `_execute_single_tool_call` re-raises
  `T0AgentsFatalError` past its belt-and-braces `except Exception` trap,
  so the exception surfaces verbatim to the caller of `Agent.run()`.
- **Soft-block strings** (`before_tool` returns `str`) — not an exception.
  The string becomes the tool's error result (single `"ERROR: "` prefix
  added by the adapter serializer). The run continues.
- **Non-fatal exceptions from guardrail hooks** — bugs in your own
  guardrail. Caught by `GuardedToolProvider`, logged at `ERROR` with the
  full stack trace, treated as pass for `before_tool` / no-op for the
  other hooks. Your tool call still runs — a broken guardrail cannot
  corrupt the agent loop.
- **`T0AgentsFatalError` subclasses** — any subclass propagates with the
  same mechanics as `GuardrailBlockedError`. Use this to encode
  domain-specific hard aborts (see the Advanced example above).

See [Error handling](error-handling.md) for the complete taxonomy
covering `McpError` / `ToolError` wrapping, timeouts, and recoverable
inference status codes.

---

## Observability

Guardrails are one of the library's richest sources of structured audit
data. Three surfaces are relevant.

### The audit log

`AgentResult.guardrail_log` is the per-run activation trail. It is
populated on every run, including runs that aborted with
`GuardrailBlockedError`, and is the canonical input to manual feedback
emission (see below). For cross-run dashboards, fan the log out to your
monitoring sink inside `after_run`:

```python
async def after_run(self, ctx, result):
    await metrics_client.emit({
        "run_id": ctx.extra.get("run_id"),
        "agent": ctx.agent_name,
        "tool_calls": ctx.tool_calls_made,
        "cost_usd": ctx.cost_so_far,
        "passed": result is not None,
    })
```

### The `on_inference` callback

`run_agent(..., on_inference=callback)` fires after every
`t0.inference()` response and **before** any tool dispatch on that turn.
The library uses this internally to drive cost tracking for `max_cost`;
user code can commandeer the same hook for inference-level observability:

```python
async def record_usage(response):
    await metrics_client.emit({
        "model": response.variant_name,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": response.usage.cost or 0.0,
    })


result = await run_agent(
    t0, client, user_message, "agent",
    on_inference=record_usage,
)
```

`Agent.run()` installs its own `on_inference` internally when guardrails
are configured, so this knob is primarily for direct `run_agent` callers
(or for advanced `Agent` subclasses). `None` (the default) preserves
byte-identical behaviour with the pre-guardrail loop.

### Emitting TensorZero feedback

The framework does **not** auto-emit `t0.feedback(...)` for guardrail
outcomes. TensorZero requires every metric to be pre-declared in
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

Episode id and inference ids are preserved on `AgentResult` exactly as in
the base loop — guardrails never alter them. See
[Observability](observability.md) for the broader story on ClickHouse
replay and episode tagging.

---

## Production considerations

### Transport and `require_approval`

`require_approval` invokes its `handler` in the same process that runs
`Agent.run()`. For CLI and single-process deployments that is exactly
the same handler the developer passes to `Client(elicitation_handler=...)`
— share the callable and the UX stays consistent.

For HTTP and WebSocket deployments, the human lives in a different
process. The library ships an `ElicitationForwarder` whose `handler`
attribute is shaped like an `ApprovalHandler`; pass
`require_approval(handler=forwarder.handler, ...)` and the server wires
prompts and responses across the session. See
[Serving](serving.md) for the HTTP case,
[Streaming](streaming.md) for the WebSocket case, and
[CLI](cli.md) for the TUI case. All three surfaces rely on the same
forwarder primitive.

### `parallel_tools=True`

The built-in counters (`tool_calls_made`, `cost_so_far`) are race-safe
because they mutate before the first `await` point in `before_tool`.
Custom guardrails that mutate shared state through an `await` boundary
need an `asyncio.Lock` — see [Concurrency](#concurrency) above.

**`require_approval` is not safe under `parallel_tools=True`.** When
the model emits N tool calls in a single assistant turn, the parallel
executor fans them out via `asyncio.gather`, and every gated call enters
its `before_tool` hook concurrently. The hook awaits a human
response per call, so the user sees N approval prompts simultaneously
for the same logical action — there is no native coalescing in the
guardrail today. The supported configurations are:

- Keep `parallel_tools=False` (the default) when any tool may be wrapped
  with `require_approval`, or
- Flip `parallel_tools=True` only when **no** active guardrail awaits
  human input (the safe built-ins are `max_cost`, `max_tool_calls`,
  and `block_tools`).

Custom approval-style guardrails inherit the same caveat: if your hook
awaits an external decision (human, queue, remote policy engine), do
not pair it with `parallel_tools=True`. Cross-references: the broader
mental model around when to enable the flag lives in
[Concurrency](concurrency.md); the deployment-side checklist (rate
limits, transactional tools) lives in
[Production / Parallel vs sequential tool execution](production.md#parallel-vs-sequential-tool-execution).

### Cost-tracking caveats

`max_cost` reads `Usage.cost` directly from the gateway's response. That
field is populated only when the provider's pricing is declared in
`tensorzero.toml` or when TensorZero ships built-in rates for the
provider. Models without declared pricing report `Usage.cost = None`,
which the tracker treats as `0.0` — the cap never fires.

For a deterministic cap that does not depend on pricing configuration,
use `max_tool_calls(per_run=...)` as a belt-and-braces fallback. Declare
provider pricing in `tensorzero.toml` for real cost observability. See
[Production](production.md) for deployment-level monitoring patterns and
alerting around inference cost.

### Secret handling in `GuardrailContext.extra`

`ctx.extra` is never serialized to TensorZero — it stays in-process for
the duration of one run. But `after_run` hooks that fan out to metrics
sinks or logs can leak its content. Redact before logging, and prefer
opaque identifiers over raw user input when stashing state for cross-hook
correlation.

### `require_approval` under decline storms

A handler that declines every prompt soft-blocks the tool, returns an
error string to the LLM, and lets the model pivot. If the model keeps
trying gated tools, the run eventually either finishes (the LLM gives
up) or hits `max_turns`. Keep `max_turns` finite — the default is a sane
ceiling, but if you raised it, pair `require_approval` with
`max_tool_calls` to bound worst-case behaviour under a hostile handler.

---

## External references

- [**TensorZero inference**](https://www.tensorzero.com/docs) —
  `ChatInferenceResponse.usage` is the source of the cost data that
  drives `max_cost`. The same docs cover provider pricing configuration
  (`pricing_per_million_tokens_input` / `_output`) and
  `Usage.cost` semantics.
- [**TensorZero feedback**](https://www.tensorzero.com/docs) — metrics
  must be pre-declared in `tensorzero.toml` before `t0.feedback(...)`
  will accept them. The guardrail log surfaces every activation so the
  caller can emit the right metric name against the right inference id.
- [**FastMCP elicitation**](https://gofastmcp.com/clients/elicitation)
  — the `ElicitationHandler` signature that `ApprovalHandler` mirrors
  exactly. The same handler can power client-side
  `Client(elicitation_handler=...)` and server-side `ctx.elicit()`
  responses, plus guardrail-side `require_approval(handler=...)`.
- Library reference handlers from `t0_agents.mcp`:
  `auto_approve_handler` (accepts every binary approval) and
  `auto_decline_handler` (declines everything). Useful for smoke tests —
  `require_approval(handler=auto_approve_handler)` bypasses human input
  in CI.

---

Guardrails decide synchronously, on the same turn, before the tool dispatches. But the caller still sees nothing until the whole run finishes. When you need live visibility — token-by-token text, tool-call intent before execution, elicitation prompts as they happen, or a final-answer delta that renders progressively — switch from `Agent.run()` to `Agent.stream()`. That is the subject of the next chapter: [Streaming](streaming.md).

---

## See also

- [`examples/10_guardrails/`](../examples/10_guardrails/) — runnable demo
  composing all four built-ins on one agent, with a shared interactive
  handler for both client-side and server-side elicitation.
- [Elicitation](elicitation.md) — the FastMCP elicitation signature that
  `require_approval` rides.
- [Agent](agent.md) — `AgentConfig` reference, including the `guardrails`
  field and the per-run lifecycle diagram.
- [Error handling](error-handling.md) — full library exception taxonomy.
- [Observability](observability.md) — episode tagging, ClickHouse replay,
  and feedback emission patterns.
- [Production](production.md) — deployment, monitoring, and alerting.
- [API reference](api-reference.md) — library-wide symbol index.
