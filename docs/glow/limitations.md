# Known limitations of `t0_agents.glow`

> **Experimental.**  This page is the inventory of what glow does
> *not* do yet.  Most items have a deliberate deferral path (the
> v1.0 base API still does the thing; you can drop down whenever
> you need to).  Some are inherent constraints inherited from
> upstream behaviour.

For the design rationale and the surface that *does* work, see
[`getting-started.md`](getting-started.md).  Cross-links into the
v1.0 docs point at the canonical narratives.

## No streaming JSON

Glow streaming reuses v1.0 [`Run.events()`](../streaming.md), which is
chat-only — `agent.invoke(...).events()` raises `ConfigurationError`
at the first `__anext__` on agents whose `OutputMode` is
`JsonOutput` or `StructuredOutput`.  TensorZero does not emit
`output.parsed` mid-stream, so a JSON stream would yield events
whose `parsed` is permanently `None`.  Pattern: call
`await plan(query)` for the JSON path, present the structured
value on the consumer side when it lands.

See [Structured outputs — Why you cannot stream JSON
functions](../structured-outputs.md#why-you-cannot-stream-json-functions)
for the upstream rationale.  v1.1 may introduce a
streaming-JSON-capable output mode; until then, no glow workaround.

## No `AsyncIterator` tools

The plan flagged `AsyncIterator[str]` tool returns as a future
addition (an `@app.tool` whose body yields progress events for the
caller to consume via `.events()`).  That path is deferred — every
`@app.tool` today returns a single value when its body completes.
For long-running tools, plumb progress through your own logging
+ a separate observability sink; the agent loop sees only the
final string.

## No per-agent MCP subsets

Every glow agent on a given `App` sees every mounted MCP server
plus every `@app.tool` registration plus every `<sub_agent>_ask`
sub-agent tool.  There is no built-in way to expose a *subset* of
the composed tool list to a specific agent.

Workaround: spin up multiple `App` instances when isolation
matters.  Each `App` owns its own composed FastMCP and its own
in-memory `Client`, so partition logic that wants strict
visibility boundaries by creating a per-domain `App` and routing
queries between them at the application layer.  A first-class
solution (per-agent MCP visibility filters) is a v1.1 candidate.

## LLM-routed dict-composer model capability

`compose({"name": agent, ...})` installs a synthetic classifier
agent configured with `ToolCallOutput(tool_choice="required")` and
a synthetic FastMCP server holding one no-op tool per route key.
The model must honor `tool_choice="required"` with arbitrary
synthetic tool schemas.

Models known to fail: `openai/gpt-oss-20b` via Groq (M4 §3.3
documents the provider error returned during integration testing).
Models known to work: Claude Sonnet/Opus, GPT-4o, GPT-5,
Gemini-2.5-pro.  See the tutorial's
[Native composition](getting-started.md#native-composition--composespec--mergenone)
section for the boxed callout; check your gateway config before
running dict-router workloads against a smaller open-source model.

## Pydantic is a hard dep of the `[glow]` extra

`pip install 't0-agents[glow]'` pulls `pydantic>=2.0` unconditionally
— it's the FastAPI precedent (pydantic is used for type
introspection in a handful of places: the return-annotation walker
that drives `StructuredOutput` selection, the synthetic wrapper
for `list[BaseModel]` returns).  Users who can't take a pydantic
dependency should stay on the v1.0 base API (`[pydantic]` extra
is also a v1.0 add-on, but the structured surface there is opt-in;
glow always assumes it's available).

## mypy strict supported via `get_type_hints`

Glow runs `typing.get_type_hints(fn, include_extras=True)` at
decoration time.  This handles forward refs from
`from __future__ import annotations` correctly, but it requires
the resolved type to be importable from the function's module
globals at decoration time.  Local-scope forward refs (a type
defined inside an enclosing function) won't resolve; declare such
types at module scope.  The M3 §2.5 and M4 §3 follow-on
commits codified this — declaring guardrail instances or
sub-agent wrappers inside an enclosing test function reliably
trips a `NameError`-based `ConfigurationError` at decoration
time.

## `_AgentWrapper.__call__` dual-personality

The wrapper returned by `@app.agent("name")` is typed via
`ParamSpec` so the wrapped function's signature is preserved at
the static level — `mypy` sees `tutor("hi")` as
`FluentRun[str]`.  At runtime, the call body accepts exactly one
positional `str` (the user message) plus any sub-agent params
declared with `Depends(<wrapper>)`.  A typo'd kwarg may pass mypy
(because ParamSpec lifts the *whole* function signature) but
raise `TypeError` at call.  Workaround: lean on the runtime
errors — they're loud and point at the offending parameter.  A
future tightening would carve a narrower runtime signature at the
cost of losing static return-type inference for the chat case;
not done in v1 of glow.

## `05_di_tools.py` is MCP-only — no LLM round-trip

The runnable demo at
[`05_di_tools.py`](../../examples/glow/05_di_tools.py) exercises
`@app.tool` + `Depends` stripping + `app.dependency_overrides` via
a direct MCP `call_tool` round-trip.  The agent-uses-tool flow
(LLM picks the tool, glow injects the dependency, the agent
narrates the result) is *not* shown end-to-end — a future
`07_agent_with_tools.py` would close the gap.  The mechanism is
identical to what `03_streaming.py` demonstrates (an agent
calling a `@app.tool`), so the dependency-injection layer adds
nothing new at the inference loop; the gap is example coverage,
not functionality.

## `race()` is deprecated

Glow does not surface `race` as a primitive.  v1.0 still ships the
factory, but it emits a `DeprecationWarning` pointing at
`parallel(*, reducer=..., error_policy="continue")` — exactly what
`compose({...})` backs into.  Removal is slated for v1.1.  If you
need first-finisher-wins specifically (and accept the cost of the
inferences you cancel), write it with
`asyncio.wait(return_when=FIRST_COMPLETED)` in user code; glow
deliberately does not bless it as a framework primitive.

See [Composition — Advanced / not
recommended](../composition.md#advanced--not-recommended) for the
v1.0 appendix.

## Sync API

There isn't one and there won't be.  t0-agents v1.0 is async-first
(`Agent.invoke` returns an awaitable `Run[T]`), and glow is
async-first by construction.  Use `asyncio.run(main())` at the
entry point of CLI scripts; embed inside any existing async
framework (FastAPI, httpx) directly.

## Where to drop down to v1.0

When the glow surface stands in your way, the escape hatch is
always `wrapper._agent`: the v1.0 `Agent` constructed by
`@app.agent` is exposed for direct use.

```python
@app.agent("tutor")
async def tutor(q: str) -> str:
    """An English tutor."""

async with app.lifespan():
    # Drop down to v1.0:
    run = tutor._agent.invoke("Say hi", middleware=[...], output=...)
    result = await run.result()
```

That sidesteps `FluentRun`, gives you the full `Run[T]` surface,
and bypasses every glow walker (DI, MCP injection, guardrail
harvesting).  Sub-agents, MCP mounts, and `@app.tool`
registrations remain visible because the `tool_provider` is still
`app._owned_client`.

---

For the canonical v1.0 narratives, follow:
[Agent](../agent.md) ·
[Run](../run.md) ·
[Composition](../composition.md) ·
[Output Modes](../output-modes.md) ·
[Streaming](../streaming.md) ·
[Guardrails](../guardrails.md) ·
[Multi-Agent](../multi-agent.md).
