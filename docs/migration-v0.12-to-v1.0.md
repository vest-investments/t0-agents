# Migrating from v0.12 to v1.0

t0-agents 1.0 is the first stable release.  Most v0.12 code keeps
working unchanged; two surfaces — `Agent.run` / `Agent.stream` and the
JSON-mode `AgentConfig` cluster — were removed in favour of the v1.0
canonical entry points `Agent.invoke` and `OutputMode`.

This guide walks every change in order of how invasive it is to your
code.  If you only have time for one section, read
[**What changes (TL;DR table)**](#what-changes-tldr-table).

> Companion reading: [`docs/run.md`](run.md),
> [`docs/middleware.md`](middleware.md),
> [`docs/output-modes.md`](output-modes.md),
> [`docs/composition.md`](composition.md).

---

## What changes (TL;DR table)

The v0.12 surface splits into three migration tiers:

| Tier | What | Action | What happens if you don't |
|---|---|---|---|
| **Tier 0** | `run_agent`, `run_structured_inference`, `permissive_parse`, `AgentConfig(memory=, guardrails=, sub_agents=, ...)`, every `AgentResult` read-side property, `Guardrail` / `MemoryProvider` / `ToolProvider` protocols, `agent_as_tool` | Nothing — stable forever | v0.12 code keeps working. |
| **Tier 1** | `agent.run(...)`, `agent.stream(...)` | One-line migration per call site | `AttributeError: 'Agent' object has no attribute 'run'` (resp. `'stream'`). |
| **Tier 2** | `AgentConfig(function_kind=, json_mode=, output_schema=, permissive_json_parsing=, max_json_retries=)` (5 kwargs) | One-line migration per `AgentConfig` | `TypeError` at `AgentConfig(...)` construction. |

Concretely:

- If your codebase only uses chat-mode agents (`agent.run(msg)` + a
  tool provider) you have a **one-line migration per call site** —
  see [Migrating Agent.run → agent.invoke](#migrating-agentrun--await-agentinvoke).
- If your codebase uses JSON-mode (`AgentConfig(function_kind="json", ...)`),
  you have a **one-line migration per agent config** — see
  [Migrating JSON config kwargs](#migrating-json-config-kwargs--outputjsonoutput).
- If your codebase uses streaming (`agent.stream(...)`), you have a
  **one-line migration per call site** — see
  [Migrating Agent.stream → agent.invoke().events()](#migrating-agentstream--agentinvokeevents).

---

## Migrating `Agent.run()` → `await agent.invoke()`

`Agent.run(msg, **kw)` is removed in v1.0.  The canonical entry
point is `agent.invoke(msg, **kw)`, which returns a `Run[T]` handle.
Migrating is a one-liner per call site:

```python
# v0.12
result = await agent.run("What is 2 + 3?")
print(result.response)

# v1.0
result = await agent.invoke("What is 2 + 3?")
print(result.response)
```

`agent.invoke(...)` returns a `Run[T]` handle which is **awaitable** —
`await run` is the shortest path to the final `AgentResult`.  See
[`docs/run.md`](run.md) for the full Run surface (replay, fork,
cancel, events, feedback, transcript).

### What about exceptions?

`Agent.invoke(...)` raises the **same exception types** v0.12's
`Agent.run(...)` did:

```python
# v0.12
try:
    result = await agent.run(msg)
except MaxTurnsExceededError:
    ...
except GuardrailBlockedError:
    ...

# v1.0
try:
    result = await agent.invoke(msg)
except MaxTurnsExceededError:
    ...
except GuardrailBlockedError:
    ...
```

A `RunFailedError` exists in v1.0 as a **defensive fallback**: it is
raised by `await run.result()` only when the drive ended in a
`"failed"` state without a captured exception (a corner case that
shouldn't happen in well-formed agents).  In every normal failure
path, `result()` re-raises the original exception directly so your
existing `except` blocks keep working without modification.

### What you gain

Migration is not just a rename — `Run[T]` opens four capabilities
that were not reachable from the v0.12 `Agent.run` surface:

- **Live event stream.**  `async for ev in run.events()` yields
  `AgentEvent`s as the loop runs.  Two-channel consumption is supported
  — see [Migrating Agent.stream → agent.invoke().events()](#migrating-agentstream--agentinvokeevents).
- **Cooperative cancellation.**  `await run.cancel()` aborts the
  drive; a subsequent `await run.result()` raises `RunCancelledError`.
- **Replay / fork.**  `await run.replay(...)` re-runs with the same
  input; `await run.fork("follow-up")` continues from the Run's
  transcript with an extra user turn.
- **Cost and usage.**  `run.cost()` / `run.usage()` return aggregated
  totals populated by the always-on `CostTrackingMiddleware`.

See [`docs/run.md`](run.md) for the full reference.

---

## Migrating `Agent.stream()` → `agent.invoke().events()`

Same pattern.  `Agent.stream` is removed; the v1.0 equivalent is
`run.events()` on the `Run` returned by `agent.invoke(...)`:

```python
# v0.12
async for event in agent.stream("Tell me a story"):
    print(event)

# v1.0
async for event in agent.invoke("Tell me a story").events():
    print(event)
```

Two-channel consumption is new — the event buffer is replayable and
multi-subscriber, so you can iterate events on one task while another
task awaits the final result:

```python
run = agent.invoke("Tell me a story")

async def watch_events():
    async for event in run.events():
        if isinstance(event, TextDeltaEvent):
            sys.stdout.write(event.text)

async def collect_result():
    return await run.result()

result, _ = await asyncio.gather(collect_result(), watch_events())
```

See [`docs/run.md` — Two-channel consumption](run.md#two-channel-consumption).

---

## Migrating JSON config kwargs → `output=JsonOutput(...)`

The five JSON-mode `AgentConfig` kwargs are removed.  Passing any
raises `TypeError` at construction time.  Migration is one line per
`AgentConfig` build:

```python
# v0.12
config = AgentConfig(
    name="extractor",
    function_name="extract_customer",
    function_kind="json",
    json_mode="strict",
    output_schema=schema_dict,
    permissive_json_parsing=True,
    max_json_retries=2,
)

# v1.0
config = AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=JsonOutput(
        schema=schema_dict,
        json_mode="strict",
        permissive=True,
        max_retries=2,
    ),
)
```

### Full mapping table

| v0.12 kwarg (on `AgentConfig`) | v1.0 equivalent |
|---|---|
| `function_kind="chat"` (default) | omit, or `output=ChatOutput()` (also the default) |
| `function_kind="json"` | `output=JsonOutput(...)` |
| `json_mode="strict" \| "on" \| "off" \| "tool"` | `JsonOutput(json_mode=...)` |
| `output_schema=dict` | `JsonOutput(schema=dict)` |
| `permissive_json_parsing=True` (or `False`) | `JsonOutput(permissive=...)` |
| `max_json_retries=int` | `JsonOutput(max_retries=...)` |

For Pydantic users, the canonical path is now `StructuredOutput[M]`
instead of hand-rolling `output_schema` from `model_json_schema()`:

```python
from pydantic import BaseModel
from t0_agents import Agent, AgentConfig, StructuredOutput

class Customer(BaseModel):
    name: str
    email: str

config = AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=StructuredOutput(Customer),
)
agent = Agent(config, t0=gateway)
result = await agent.invoke("Sarah <sarah@example.com>").result()
print(result.value)   # Customer(name='Sarah', email='sarah@example.com')
print(result.parsed)  # {'name': 'Sarah', 'email': 'sarah@example.com'} — v0.12 alias
```

`StructuredOutput` lazily imports pydantic at construction time;
install the extra with `pip install 't0-agents[pydantic]'` if you
don't already depend on pydantic transitively.

### Per-call JSON overrides

In v0.12, `agent.run("msg", output_schema=..., json_mode=..., max_json_retries=...)`
accepted per-call JSON kwargs.  Those kwargs are removed.  On v1.0,
pass a per-call `output=` instead:

```python
# v0.12
await agent.run("msg", output_schema=other_schema, json_mode="tool")

# v1.0
await agent.invoke(
    "msg",
    output=JsonOutput(schema=other_schema, json_mode="tool"),
)
```

### `StructuredOutput` rename — direct dataclass importers

The dataclass returned by `run_structured_inference(...)` used to be
named `StructuredOutput`.  In v1.0 it is **renamed to
`StructuredInferenceResult`** so the canonical name is free for the
new Pydantic-driven `OutputMode`.

```python
# v0.12
from t0_agents.core.structured import StructuredOutput
result: StructuredOutput = await run_structured_inference(...)

# v1.0
from t0_agents.core.structured import StructuredInferenceResult
# or, the unchanged convenience import:
from t0_agents import StructuredInferenceResult
result: StructuredInferenceResult = await run_structured_inference(...)
```

The dataclass field shape is unchanged.  Most callers access the
result via attribute (`result.parsed`, `result.raw`, ...) and don't
import the type by name — those don't need any edit.

---

## What's the same (Tier 0 — stable forever)

Everything below is byte-identical to v0.12.  You do not need to
audit your codebase for any of these:

- **`mcp.run_agent`** — the underlying chat loop.  Public signature
  frozen; the `mcp/` submodule is invariant across the v1.0 refactor
  (`git diff src/t0_agents/mcp/` is empty).
- **`run_structured_inference`** — the helper that drives JSON-mode
  single-shot inference.  Signature unchanged; returns
  `StructuredInferenceResult` (see rename above).
- **`permissive_parse`** — the JSON-parsing helper.
- **`AgentConfig(memory=, guardrails=, sub_agents=, ...)`** — every
  kwarg that isn't in the Tier 2 removal list is unchanged.  v1.0
  additionally materialises a default middleware chain from these
  kwargs (see [`docs/middleware.md`](middleware.md)) but the
  user-visible behaviour is byte-identical.
- **`AgentResult.response`, `.parsed`, `.parse_strategy`,
  `.json_parse_attempts`, `.function_kind`** — preserved as
  `@property` aliases over the new `value: T` canonical field.  Code
  that reads `result.response` keeps working byte-identical.
- **`Guardrail`, `MemoryProvider`, `ToolProvider`** Protocols — the
  shape is unchanged; existing implementations satisfy the v1.0
  protocols without code changes.
- **`agent_as_tool(...)`** — the LLM-driven delegation primitive.

---

## What's new and opt-in

These features are additive — you don't have to use any of them, but
you can adopt them piecemeal at your own pace:

### `Run[T]` — first-class invocation handle

`agent.invoke(msg)` returns a `Run`.  `await run` gives you the final
`AgentResult`.  `async for ev in run.events()` gives you the live
event stream.  `await run.cancel()` cancels in-flight; `await
run.replay(...)` re-runs with a fresh episode (or the same one);
`await run.fork("follow-up")` continues from this Run's transcript.
`run.cost()` and `run.usage()` return aggregated totals.
See [`docs/run.md`](run.md).

### `Middleware` — composable cross-cutting concerns

Memory, guardrails, cost tracking, sub-agent orchestration, and JSON
retry are now middlewares.  Power users add their own (PII redactor,
observability hook, recovery policy) without forking
`core/agent.py`.  The default chain materialised by
`AgentConfig.__post_init__` reproduces v0.12 behaviour exactly.
See [`docs/middleware.md`](middleware.md).

### `OutputMode[T]` — output polymorphism

The chat/JSON dichotomy collapses into a polymorphic protocol.
Built-ins: `ChatOutput` (default), `JsonOutput`, `StructuredOutput[M]`
(Pydantic), `ToolCallOutput`, `StreamingOutput`.  Third parties can
plug in YAML extraction, Instructor-style retry, SQLAlchemy row
materialisation without forking.  See
[`docs/output-modes.md`](output-modes.md).

### `Composer` — topology primitive

Nine topology builders (`sequential`, `parallel`, `race`,
`consensus`, `map_reduce`, `conditional`, `saga`, `retry_chain`,
`circuit_breaker`) for multi-agent orchestration.  Every Composer
satisfies the `Composable[Input, T]` protocol so they nest uniformly.
Cancellation cascades to every sub-Run; `parent_episode_id` audit
threads through to ClickHouse for hierarchy reconstruction.  See
[`docs/composition.md`](composition.md).

> **Saga compensation idempotency** — if you adopt `saga(...)` for
> distributed-transaction patterns, note that compensations may be
> invoked more than once on partial failure (e.g., a raising
> compensate mid-cascade).  Make them idempotent — dedupe by resource
> id, transaction handle, or similar.  This is a classical saga
> constraint, not a t0-agents-specific quirk.

---

## Checklist

A pragmatic order to migrate one codebase from v0.12 to v1.0:

1. **Run `pytest` against v1.0**.  If green, you have no migrations
   to make.  If `TypeError` at `AgentConfig(...)` construction, you
   have JSON config kwargs to migrate — see
   [Migrating JSON config kwargs](#migrating-json-config-kwargs--outputjsonoutput).
2. **Run again**.  Every `AttributeError: 'Agent' object has no
   attribute 'run'` (resp. `'stream'`) is a Tier 1 migration — replace
   with `agent.invoke(...)` / `agent.invoke(...).events()`.
3. **Audit imports of `StructuredOutput`** from
   `t0_agents.core.structured`.  Rename to `StructuredInferenceResult`.
   Imports from `t0_agents` (the top-level package) that used
   `StructuredOutput` for the new Pydantic-driven OutputMode keep
   working unchanged.
4. **Adopt new features opt-in** if you want them — `Middleware`,
   `Composer`, `Run.replay/.fork`, `output=StructuredOutput(Model)`.
   None are required to ship 1.0.

When your tests pass, you're on v1.0.
