# Run — the first-class invocation handle

`Agent.invoke(user_message, ...)` returns a `Run[T]` — the v1.0 handle
for one invocation of an agent.  Where `agent.invoke(...)` in v0.12
returned the terminal `AgentResult`, `agent.invoke(...)` returns the
*invocation itself* — an object you can `await`, iterate, cancel,
replay, fork, attach feedback to, and introspect for cost and usage.

A `Run` is awaitable: `result = await agent.invoke("hi")` is the
shortest path from query to answer and is what most callers want.
Everything else on the handle — `events()`, `replay()`, `fork()`,
`cancel()`, `feedback()`, `cost()`, `usage()`, `transcript()` — is
opt-in for the patterns the simple `await` doesn't cover.

> This page documents the public Run surface.  For the data shape on
> the way out, see [`AgentResult[T]`](api-reference.md#agentresult-1) +
> [Output modes](output-modes.md).  For the cross-cutting concerns
> threaded through every Run, see [Middleware](middleware.md).  For
> composing Runs into topologies, see [Composition](composition.md).

---

## Why `Run` exists

v0.12 returned `AgentResult` directly from `agent.invoke(...)`:

```python
# v0.12
result = await agent.invoke("hi")
print(result.response)
```

That shape has three structural limits:

- **No handle on the live invocation.**  You can't cancel a slow LLM
  call from outside; you can't see partial events while it's still
  running; you can't fork the in-flight conversation; you can't replay
  it against a different agent.
- **Two-channel consumption is awkward.**  Streaming clients want
  events; downstream code wants the final `AgentResult`.  Either you
  call `stream(...)` and re-assemble the result yourself, or you call
  `run(...)` and miss the stream.  Both at the same time required
  custom plumbing.
- **No topology primitive.**  A "parallel of three agents that all
  cancel when the first finishes" needs a uniform handle on each
  agent's invocation — exactly what `Run` provides.

`Run[T]` solves all three, additively.  v0.12's `agent.invoke(...)` is
preserved as a deprecated shim that emits `DeprecationWarning` once
per source line and delegates to `agent.invoke(...).result()` —
existing code keeps working byte-identical until v1.1.

---

## Lifecycle

A `Run` walks through four states:

```
   pending  ──►  running  ──►  succeeded
                         ╲
                          ──►  failed
                         ╲
                          ──►  cancelled
```

- `pending` — the Run object exists but its drive task hasn't started.
  This is the visible state only when `eager=False`; the default
  `eager=True` schedules the task immediately so callers rarely see
  this state.
- `running` — drive task is in flight (executing middleware + the
  terminal inference loop).
- `succeeded` — `result()` returned an `AgentResult[T]`.  The Run is
  done; later calls to `result()` return the cached value; later
  iterations of `events()` replay the buffered events from the start.
- `failed` — the drive raised an exception.  `await run.result()`
  re-raises the **original** exception (`MaxTurnsExceededError`,
  `GuardrailBlockedError`, `ValueError`, ...).  `RunFailedError(cause)`
  exists only as a defensive fallback for the impossible case where
  the drive ended in `"failed"` state without a captured exception.
- `cancelled` — `await run.cancel()` was called.  `await run.result()`
  raises `RunCancelledError`; `events()` consumers see events emitted
  up to the cancellation point followed by end-of-stream.

The current state is exposed as `run.state`.

---

## Constructor arguments (the `invoke` kwargs)

`agent.invoke(user_message, ...)` accepts the kwargs below.  Per-call
values take precedence over the matching `AgentConfig` field; missing
kwargs inherit from the config.

| Kwarg | Type | Purpose |
|---|---|---|
| `user_message` | `str` | The user's query.  Required. |
| `output` | `OutputMode[T] \| None` | Per-call override for the output shape.  Narrows `Run[T]` via the overload.  Defaults to `config.output` (which defaults to `ChatOutput()`).  See [Output modes](output-modes.md). |
| `episode_id` | `str \| None` | Existing TensorZero episode id, or `None` to let the gateway mint a fresh one.  Use for continuing a prior conversation. |
| `tags` | `dict[str, str] \| None` | Per-call tags shallow-merged on top of `config.tags`.  Forwarded to `t0.inference(...)` and surfaced to gateway-level rate limits. |
| `system_arguments` | `dict[str, Any] \| None` | Template arguments forwarded as `input.system`.  Mutually exclusive with `system`. |
| `system` | `str \| None` | Raw system prompt forwarded as `input.system` (string form).  Mutually exclusive with `system_arguments` and with `memory_strategy='system_template'`. |
| `prior_messages` | `Sequence[Mapping[str, Any]] \| None` | TensorZero-shape `{role, content}` history to seed the inference loop.  Used by `Run.fork` to continue from a recorded transcript.  When non-None, overrides whatever the memory strategy would supply. |
| `middleware` | `Sequence[Middleware] \| None` | Per-call middleware chain that overrides `config.middleware` for this Run only.  `None` (the default) uses the config-materialised chain.  `[]` is an explicit "no middleware" override (note: `Run.cost()`/`.usage()` stop reporting when there's no `CostTrackingMiddleware`). |
| `eager` | `bool` | `True` (default) schedules the drive task immediately.  `False` defers task start to the first consumer of `result()` / `events()` — useful for queueing Runs. |
| `context` | `Mapping[str, Any] \| None` | User-supplied metadata dict copied onto the Run (readable as `run.context`).  Not threaded into the inference — purely for the caller's records. |

The return type narrows via overload: `agent.invoke(msg, output=StructuredOutput(Customer))`
returns `Run[Customer]`; without `output=`, `Run[Any]`.

---

## Two-channel consumption

A Run can be consumed two ways concurrently — the event buffer is
multi-subscriber and replayable:

```python
run = agent.invoke("Write a haiku about coffee")

async def watch_events():
    async for ev in run.events():
        if isinstance(ev, TextDeltaEvent):
            print(ev.text, end="", flush=True)

async def collect_result():
    return await run.result()

events_task = asyncio.create_task(watch_events())
result = await collect_result()
await events_task
print()
print(f"Episode: {result.episode_id}, turns: {result.turns}")
```

`result()` returns once; subsequent calls return the cached value.
`events()` can be called multiple times — the second call replays
buffered events from the start by default.  Pass `replay=False` to
get only events emitted after the call (relevant when subscribing
mid-run).

Events are bounded at 10 000 per Run; on overflow the
oldest events are dropped, `events_buffer.truncated` flips to `True`,
and a single `WARNING` is logged via `t0_agents.core._runtime`.

### When `events()` raises

For single-shot output modes (`JsonOutput`, `StructuredOutput`,
`ToolCallOutput`) `events()` raises `ConfigurationError` — there is
no streaming surface to consume.  Use `await run.result()` instead,
or wrap the base mode in `StreamingOutput(base=...)` and re-attempt
once TensorZero supports mid-stream `parsed` (Q11).

---

## `Run.replay`

Run the same input again, optionally against a different agent or
variant:

```python
original = await agent.invoke("Explain quantum entanglement")
# Try with the same agent again (e.g., to A/B different system prompts).
replay = await original.replay()
# Or against a different agent:
fancy = await original.replay(agent=fancy_agent)
# Or with a different TensorZero variant (forwarded as a tag for audit):
v2 = await original.replay(variant="gpt-4o")
```

By default replay mints a **new** `episode_id` so analytics
distinguishes replays from the original.  Pass
`replay(episode_id=original.episode_id)` to share the original
correlation key.

The optional `agent=` weakref defaults to the original via the Run's
weakref to its Agent; if the original Agent was garbage-collected,
`replay()` without `agent=` raises a `RuntimeError`.

`variant=` is forwarded as the TensorZero tag
`t0_agents.variant=<value>` so ClickHouse can audit replay provenance
without an SDK change.

---

## `Run.fork`

Continue a conversation with one extra user turn, threading the
boundary transcript into the next invocation:

```python
run1 = await agent.invoke("Plan a trip to Lisbon")
run2 = await run1.fork("Now make it budget-friendly")
print(run2.response)
```

The forked Run starts from `transcript()` of the parent — the
parent's `prior_messages` prefix + the parent's user/assistant
boundary pair — followed by the new user turn.  Multi-turn fork
chains compose transitively: `run3 = await run2.fork("Add a hike.")`
sees the full history from `run1` onwards.

> **Caveat:** in v1.0 the transcript captures only the boundary pair
> per Run (no intermediate tool turns).  This matches v0.12's
> `mcp.run_agent` contract for fork seeding and is the safe shape;
> a richer transcript is queued for a future minor once the
> middleware chain can record full message history.

`CompositeRun.fork` raises `NotImplementedError` — "extra user turn"
doesn't map cleanly onto every topology (what does it mean for a
`parallel` of voters?).  Use explicit composition instead:
`sequential(composer, follow_up_agent)`.

---

## `Run.cancel`

Cancel a Run from outside its drive task:

```python
run = agent.invoke("Generate a very long report...")
await asyncio.sleep(0.5)
await run.cancel()
try:
    await run.result()
except RunCancelledError:
    print("Cancelled")
```

`cancel()` cancels the drive task, transitions the Run to
`cancelled`, and fires guardrail `after_run(result=None)` so cleanup
hooks observe the cancellation.  `MemoryMiddleware.save` runs in
`finally` so the memory layer sees the cancellation too.

When a `Run` is the root of a Composer topology, `cancel()`
propagates to every sub-Run.  Each sub-Run gets up to
`cancel_timeout` seconds to clean up (default 30s; per-topology
override via the `cancel_timeout=` kwarg on the factory — Q3
resolution).  Overshoot logs a single WARN on
`t0_agents.compose.topology` and lets cleanup continue best-effort.

---

## `Run.feedback`

Attach TensorZero feedback to the Run's episode or to a specific
inference:

```python
run = await agent.invoke("Classify sentiment of: ...")
# Episode-level boolean (default target):
await run.feedback("task_success", True)
# Inference-level float, targeted at the last inference id:
await run.feedback("confidence", 0.87, target="inference")
```

Both `target="episode"` (default) and `target="inference"` route
through `t0.feedback(...)` — t0-agents does not maintain a parallel
metric stream (TensorZero-first invariant §2.2).

`CompositeRun.feedback` raises `NotImplementedError` because a
Composer has no single episode id of its own; route feedback to
individual sub-Runs instead (`composite._sub_runs[i].feedback(...)`,
documented as private access for now — a public surface lands when
demand justifies one per topology).

---

## `Run.cost` and `Run.usage`

Aggregated token / cost totals across every inference in the Run:

```python
run = await agent.invoke("Multi-step research query")
print(f"${run.cost():.4f}, {run.usage().input_tokens} in / {run.usage().output_tokens} out")
```

Both are populated by `CostTrackingMiddleware`, which is in the
default middleware chain.  When you override the chain with
`middleware=[]` (the explicit "no middleware"), the tracker doesn't
run and:

- `run.cost()` returns `0.0`
- `run.usage()` returns `None`

This is intentional: `middleware=[]` is an opt-out from
*every* layer including cost tracking; auto-enabling cost tracking
would violate the override contract.

`RunUsage` is a frozen dataclass with `input_tokens: int`,
`output_tokens: int`, `cost: float`.  JSON retries each fire an
inference and each contributes to the cumulative cost — that mirrors
the economic reality of the underlying TensorZero charges.

---

## `Run.transcript`

The recorded conversation boundary:

```python
run = await agent.invoke("Plan a trip")
for msg in run.transcript():
    print(msg)
```

In v1.0 `transcript()` returns `list(run.input.prior_messages) + [user_msg, assistant_response]`
— the history the Run was seeded with (via `fork()` or explicit
`prior_messages=`) plus the boundary user/assistant pair.
Intermediate tool turns are not captured — adding them
requires recording inside the middleware chain; queued for a future
minor.

The transcript is what `Run.fork(extra_message)` threads into the
next invocation's `prior_messages`, so fork chains stay transitive
even though each Run only records its boundary.

---

## `Run(agent=None)` — composer authors only

Third-party Composer authors who need to build their own topology
without a single owning Agent can construct a `Run` with `agent=None`
and a custom drive callable.  `CompositeRun` (the implementation
behind every topology factory in `compose/`) does exactly this —
the parent Run has no Agent of its own; the sub-Runs each carry
their own.

The trade-off is that `feedback()` / `replay()` / `fork()` will raise
`RuntimeError` because they require an Agent to drive against.
That's intentional — sub-Run-level methods are reachable through
`composite._sub_runs[i]`, and re-invocation of a Composer is
`composer.invoke(input)`, which is more explicit than `.replay()`
anyway.

This is an advanced surface — most users don't need to know it
exists.  Documented here for completeness.

---

## Error model

| Exception | When |
|---|---|
| `RunCancelledError` (`T0AgentsError`) | `await run.result()` after `await run.cancel()`. |
| `RunFailedError(cause)` (`T0AgentsError`) | Defensive fallback only.  Normal failure: `await run.result()` re-raises the original exception (`MaxTurnsExceededError`, `GuardrailBlockedError`, etc.). |
| `MaxTurnsExceededError`, `GuardrailBlockedError`, `ConfigurationError`, etc. | Direct re-raise from the drive. |
| `ConfigurationError` from `events()` | The Run's output mode doesn't support streaming. |

`RunFailedError(cause)` exists because mypy needs *some* type that
`run.result()` is allowed to raise besides the underlying cause
(every concrete v0.12 exception can fail a Run, and listing them all
in `raises:` blocks pollutes every caller's type-narrowing).  In
practice, normal failure paths re-raise the original exception
directly — `RunFailedError` only surfaces in the corner case where
the drive ends in `failed` state without a captured exception, which
shouldn't happen in well-formed agents.

---

## Examples

Two runnable snippets drawn from `examples/20_run_compose/`:

```python
# Two-channel consumption — events stream + final result, concurrently.
run = agent.invoke("Explain quantum entanglement.")

async def stream_to_stdout():
    async for ev in run.events():
        if isinstance(ev, TextDeltaEvent):
            print(ev.text, end="", flush=True)

await asyncio.gather(stream_to_stdout(), run.result())
print()
```

```python
# Fork + replay — multi-turn experimentation.
plan = await agent.invoke("Plan a hiking trip to Patagonia.")
cheap_plan = await plan.fork("Make it budget-friendly.")
fancy_plan = await plan.fork("Make it luxurious.")
# Replay the original against a different model:
plan_v2 = await plan.replay(variant="claude-sonnet-4-6")
```

See `examples/20_run_compose/main.py` for the full runnable file +
the TensorZero gateway it expects via `docker-compose.yml`.

---

## Related

- [Output modes](output-modes.md) — what `Run.result()` returns and how
  the type parameter `T` is inferred.
- [Middleware](middleware.md) — what runs around your inference loop.
- [Composition](composition.md) — uniformly composable Runs.
- [API reference — Run](api-reference.md#run) — exhaustive signature
  table for every method on `Run`.
