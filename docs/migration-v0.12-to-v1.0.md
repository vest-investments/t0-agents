# Migrating from v0.12 to v1.0

t0-agents 1.0 is the first stable release.  Most v0.12 code keeps
working unchanged; a small surface — JSON-mode configuration — was
re-cut around the new `OutputMode[T]` abstraction; the synchronous
entry points (`agent.run`, `agent.stream`) are deprecated.

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

| Tier | What | When you migrate | What happens if you don't |
|---|---|---|---|
| **Tier 0** | `run_agent`, `run_structured_inference`, `permissive_parse`, `AgentConfig(memory=, guardrails=, sub_agents=, ...)`, every `AgentResult` read-side property, `Guardrail` / `MemoryProvider` / `ToolProvider` protocols, `agent_as_tool` | Never — these are stable forever | Nothing.  v0.12 code keeps working. |
| **Tier 1** | `agent.run(...)`, `agent.stream(...)` | Before v1.1 (one minor) | `DeprecationWarning` once per call site in v1.0; **removed** in v1.1.  Stdout stays byte-identical until the removal. |
| **Tier 2** | `AgentConfig(function_kind=, json_mode=, output_schema=, permissive_json_parsing=, max_json_retries=)` (5 kwargs) | **Now** — they're already gone | `TypeError` at `AgentConfig(...)` construction. |

Concretely:

- If your codebase only uses chat-mode agents (`agent.run(msg)` + a
  tool provider) and doesn't pass any of the 5 JSON-mode kwargs to
  `AgentConfig`, **nothing breaks at all**.  You get a
  `DeprecationWarning` once per source line; everything else is
  byte-identical.
- If your codebase uses JSON-mode (`AgentConfig(function_kind="json", ...)`),
  you have a **one-line migration per agent config**.  See
  [Migrating JSON config kwargs](#migrating-json-config-kwargs--outputjsonoutput).
- If your codebase uses streaming (`agent.stream(...)`), you have a
  **one-line migration per call site**.  See
  [Migrating Agent.stream → agent.invoke().events()](#migrating-agentstream--agentinvokeevents).

---

## Migrating `Agent.run()` → `await agent.invoke()`

`Agent.run(msg, **kw)` keeps working in v1.0 — it emits a
`DeprecationWarning` once per source line (Python deduplicates by
`stacklevel + filename + lineno`) and delegates internally to
`agent.invoke(msg, **kw).result()`.  Migrating is a one-liner per
call site.

```python
# v0.12
result = await agent.run("What is 2 + 3?")
print(result.response)

# v1.0 canonical
result = await agent.invoke("What is 2 + 3?")
print(result.response)
```

`agent.invoke(...)` returns a `Run[T]` handle which is **awaitable** —
`await run` is the shortest path to the final `AgentResult`.  See
[`docs/run.md`](run.md) for the full Run surface (replay, fork,
cancel, events, feedback, transcript).

### What about exceptions?

The Tier 1 shim and the canonical surface raise the **same exception
types**:

```python
# v0.12 — works in v1.0 unchanged.
try:
    result = await agent.run(msg)
except MaxTurnsExceededError:
    ...
except GuardrailBlockedError:
    ...

# v1.0 canonical — same.
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

### Why migrate now if it still works?

The shim is removed in v1.1.  Migrating early gets you:

- Access to the live `Run` handle — replay, fork, cancel, two-channel
  consumption, feedback at the right inference id.
- `Run.cost()` / `Run.usage()` for aggregated token + spend totals
  populated by `CostTrackingMiddleware` (always-on in v1.0; was
  guardrail-gated in v0.12).
- One less `DeprecationWarning` in your test logs.

---

## Migrating `Agent.stream()` → `agent.invoke().events()`

Same pattern.  The shim emits `DeprecationWarning` once per source
line and delegates to `agent.invoke(msg).events()`.

```python
# v0.12
async for event in agent.stream("Tell me a story"):
    print(event)

# v1.0 canonical
async for event in agent.invoke("Tell me a story").events():
    print(event)
```

Two-channel consumption is new in v1.0 — the event buffer is
replayable and multi-subscriber, so you can iterate events on one
task while another task awaits the final result:

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

This is the only **breaking** part of v1.0: the five JSON-mode
`AgentConfig` kwargs are removed.  Passing any raises `TypeError` at
construction time.  Migration is one line per `AgentConfig` build:

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

### Per-call JSON overrides on `agent.run`

`Agent.run(msg, output_schema=..., json_mode=..., max_json_retries=...)`
still accepts the v0.12 per-call kwargs and translates them internally
into a synthesised `JsonOutput` so existing
`agent.run("...", output_schema=other_schema)` call sites keep
working byte-identical until the v1.1 shim removal.  On the
canonical `agent.invoke(...)` surface those kwargs **do not exist** —
use `agent.invoke(msg, output=JsonOutput(schema=...))` instead.

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

## Deprecation timeline

```
v1.0 (this release)
├── Tier 0 (run_agent, run_structured_inference, AgentResult read-side,
│           protocols, agent_as_tool, all AgentConfig kwargs not in Tier 2)
│   └── unchanged forever
│
├── Tier 1 (agent.run, agent.stream)
│   ├── works in v1.0 — emits DeprecationWarning once per call site
│   └── REMOVED in v1.1 (next minor)
│
└── Tier 2 (5 JSON config kwargs on AgentConfig)
    └── removed already — TypeError at construction

v1.1 (upcoming minor)
└── Tier 1 removed: agent.run / agent.stream no longer exist.
    Direct callers must have migrated to agent.invoke(...) by then.
```

There is **no v1.0.x** breaking change.  Patch releases (1.0.1, 1.0.2,
...) ship only bug fixes and additive surface.  The next breaking
change is v1.1 with the Tier 1 shim removal.

---

## PaaS-specific notes

The Prompt-as-a-Service plan in
`~/Descargas/prompt-as-a-service-mvp-plan-with-t0-agents.md` already
uses the canonical `agent.invoke(...).result()` path throughout
(execution API, structured inference, etc.).  Concretely:

- **Execution API handler** — no migration needed; already uses
  `Agent(config, ...).invoke(...).result()`.
- **AgentConfig construction** — if the plan uses
  `function_kind="json"` for any single-shot extraction, swap to
  `output=JsonOutput(schema=S, json_mode="strict", permissive=True,
  max_retries=2)`.
- **SSE streaming endpoints** — if the plan exposes a `/stream`
  endpoint that delegates to `agent.stream(...)`, swap that one line
  to `agent.invoke(...).events()`.
- **No Composer required for the MVP** — the plan currently runs one
  agent per request.  Post-MVP agentic features can adopt
  `Composer` topologies (sequential pipelines for triage, etc.)
  without re-architecting.
- **Rate limiting** — the plan keeps rate limiting at the application
  layer (Redis token bucket); t0-agents does not ship a
  RedisRateLimitMiddleware in v1.0.  See
  [`docs/guardrails.md` § Guardrails vs. TensorZero rate limits](guardrails.md#guardrails-vs-tensorzero-rate-limits)
  for the layered model (app-level / guardrails / gateway).

---

## Checklist

A pragmatic order to migrate one codebase from v0.12 to v1.0:

1. **Run `pytest` against v1.0**.  If green, you have no Tier 2
   migrations to make.  If `TypeError` at `AgentConfig(...)`
   construction, you have JSON config kwargs to migrate — see
   [Migrating JSON config kwargs](#migrating-json-config-kwargs--outputjsonoutput).
2. **Run again with `-W error::DeprecationWarning`**.  Every warning
   pointed at `agent.run` / `agent.stream` is a Tier 1 migration —
   replace with `agent.invoke(...)` / `agent.invoke(...).events()`.
3. **Audit imports of `StructuredOutput`** from
   `t0_agents.core.structured`.  Rename to `StructuredInferenceResult`.
   Imports from `t0_agents` (the top-level package) that used
   `StructuredOutput` for the new Pydantic-driven OutputMode keep
   working unchanged.
4. **Adopt new features opt-in** if you want them — `Middleware`,
   `Composer`, `Run.replay/.fork`, `output=StructuredOutput(Model)`.
   None are required to ship 1.0.

When the warning count is zero and your tests pass, your codebase is
v1.1-ready ahead of time.
