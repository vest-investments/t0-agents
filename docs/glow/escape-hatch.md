# Escape hatch — dropping down to v1.0

> Part of the glow ladder.  Glow is a thin Pythonic skin; underneath
> every `@app.agent` lives a real v1.0 [`Agent`](../agent.md) you can
> reach for whenever the decorator stands in your way.

## The wrapper is a v1.0 `Agent`

`@app.agent("name")` returns an `_AgentWrapper`.  Behind the wrapper
sits a fully-constructed `Agent` instance — same class, same
`AgentConfig`, same collaborator Protocols you'd build by hand.  It is
exposed as `wrapper._agent`:

```python
@app.agent("tutor")
async def tutor(q: str) -> str:
    """An English tutor."""
    return llm()

async with app.lifespan():
    # The high-level call:
    text = await tutor("Say hi")

    # Drop down to v1.0:
    run = tutor._agent.invoke("Say hi", middleware=[...], output=...)
    result = await run.result()
```

`tutor._agent.invoke(...)` returns a real [`Run[T]`](../run.md) —
awaitable, iterable via `.events()`, cancellable, replayable,
forkable.  Bypassing the wrapper sidesteps every glow walker (DI
resolution, MCP injection, guardrail harvesting), but the
`tool_provider` is still `app._owned_client`, so sub-agents, MCP
mounts, and `@app.tool` registrations remain visible.

## Lifespan, build, shutdown

The `App` owns a long-lived in-memory FastMCP server + matching
`Client` (`app._owned_client`).  Tools registered via `@app.tool` land
on that server; class- or instance-based `MCPServer` declarations mount
remote endpoints (optionally with per-request `headers=` / `auth=`) into
the same composed view.  The Client enters its
in-process transport task inside `app.build()`; `app.shutdown()`
closes it; `app.lifespan()` is the recommended entry point and does
`build → yield → shutdown` in a `try/finally`:

```python
async def main() -> None:
    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("tutor")
        async def tutor(q: str) -> str:
            return llm()

        async with app.lifespan():            # ← here
            print(await tutor("Say hi"))
```

Without `lifespan()` (or an explicit `build()`/`shutdown()` pair), the
first agent invocation reaches an unentered Client and the FastMCP
transport raises.  Build is idempotent; `lifespan()` is re-entrant in
the sense that consecutive calls construct and tear down cleanly.

Forward-ref failures, duplicate agent names, and mis-shaped `Depends`
declarations all surface at `build()` time rather than at the first
inference, matching FastAPI's fail-fast ergonomics.

## Drop-down recipes

### Middleware

[Middleware](../middleware.md) is a first-class glow knob at every level —
no drop-down needed: App-wide via `App(t0=t0, middleware=[...])`, per-agent
via `@app.agent("tutor", middleware=[...])`, and per-call via
`.with_(middleware=[...])`.  App- and agent-level layer (an agent-level
list replaces the App default); `.with_()` overrides that one call only.

```python
from t0_agents import CostTrackingMiddleware

app = App(t0=t0, middleware=[CostTrackingMiddleware()])   # App-wide default

@app.agent("tutor", middleware=[CostTrackingMiddleware()])
async def tutor(q: str) -> str:
    """An English tutor."""
    return llm()

async with app.lifespan():
    # per-call override, for this invocation only
    answer = await tutor("Say hi").with_(middleware=[CostTrackingMiddleware()])
```

For the raw base-API surface, `tutor.agent.invoke(...)` (the public
accessor; raises a clear error before the App is built) still works.

### Custom `OutputMode` for a single call

`.as_(custom_mode)` covers most cases, but if you need full control
over the `OutputMode` and its surrounding `AgentConfig`, drop down:

```python
from t0_agents import JsonOutput

run = tutor._agent.invoke("...", output=JsonOutput(schema=my_schema))
result = await run.result()
```

See [`docs/output-modes.md`](../output-modes.md) for the
`OutputMode[T]` Protocol and the five built-in modes.

### Streaming with custom event filtering

Glow's [seven selectors](streaming-and-events.md) cover the common
cases; for ad-hoc filtering across the full
[`AgentEvent`](../streaming.md) union, drop to `Run.events()`:

```python
run = tutor._agent.invoke("Say hi")

async for event in run.events():
    match event:
        case ToolCallEvent(name="lookup_order", arguments=args):
            ...
        case TextDeltaEvent(delta=d):
            ...
```

### Direct `Run[T]` surface — cancel, fork, replay

`FluentRun` doesn't expose `cancel()` / `fork()` / `replay()` —
they live on the underlying [`Run[T]`](../run.md):

```python
run = tutor._agent.invoke("Plan a trip")
async for ev in run.events():
    if some_condition(ev):
        await run.cancel()
        break

forked = await run.fork("Same prompt, different model")
```

## When to drop down (matrix)

| Goal                                                     | Path                                                  |
|----------------------------------------------------------|-------------------------------------------------------|
| Per-call middleware override                             | `wrapper._agent.invoke(..., middleware=[...])`        |
| `OutputMode` with custom schema or retry policy          | `wrapper._agent.invoke(..., output=...)`              |
| `cancel()` / `fork()` / `replay()`                       | `wrapper._agent.invoke(...)` → use `Run[T]` methods   |
| Full `AgentResult[T]` (`episode_id`, `inference_ids`, …) | `await wrapper(...).result()` (also works without drop-down) |
| Hand-rolled event handling across full `AgentEvent`      | `wrapper._agent.invoke(...).events()` (no selectors)  |
| Per-agent guardrails added at runtime, not decoration    | `wrapper._agent.config.guardrails.append(...)` (carefully) |
| Replace the `tool_provider` for one call                 | construct a fresh `Agent(config, t0=t0, tool_provider=other)` |

The first four are common.  The last three are rare — if you need
them, you're typically better off skipping glow for that agent and
constructing an `Agent` directly.

---

For the canonical v1.0 narratives, follow:
[Agent](../agent.md) ·
[Run](../run.md) ·
[Middleware](../middleware.md) ·
[Output Modes](../output-modes.md) ·
[Streaming](../streaming.md) ·
[Multi-Agent](../multi-agent.md).
