# Known limitations of `t0_agents.glow`

> This page is the inventory of what glow does
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
for the upstream rationale.  The gateway streams JSON functions as raw
string deltas only (no `parsed` mid-stream), so a future
streaming-JSON output mode would still parse client-side at the end;
until one lands, no glow workaround.

## `model=` agents cannot produce validated JSON

The functionless path routes through the gateway's built-in chat
passthrough (`tensorzero::default`), which **ignores the per-call
`output_schema`** (verified against gateways 2026.4.0 and 2026.6.0) — so
`JsonOutput`/`StructuredOutput` can never be satisfied there.
Declaring `@app.agent(model=..., ...)` on a function with a structured
return annotation raises `ConfigurationError` at first build, with a
ready-to-paste `type = "json"` TOML stanza in the message.  Drop to
`function_name=` for schema-validated output.

## Functionless inferences are invisible to `function_name`-filtered memory

Every `model=` inference is stored under `tensorzero::default`, so
`TensorZeroMemory(function_name="my_func")` filters will never see it.
Filter on the observability auto-tag instead — it is present on every
inference:

```python
TensorZeroMemory(gateway, tag_filter={"t0_agents.agent": "tutor"})
```

Add `function_name="tensorzero::default"` on top for ClickHouse
partition-pruning when the store grows.

`app.conversations(agent)` already filters on that tag for you; the recipe
above is for a `TensorZeroMemory` you construct yourself.

## `system_arguments` does not work with `model=`

TZ server-side templates (`.with_(system_arguments={...})`) need a
declared function with a system schema; the functionless passthrough
has none.  The combination raises up front (a `ValueError` from
`Agent.invoke`) instead of surfacing as an opaque gateway 400.  Use
the docstring (optionally with [`prompt_vars`
placeholders](getting-started.md#client-side-prompt-templates--prompt_vars))
or a raw `.with_(system=...)` string on the `model=` path.

## MCP tool inventory is identity-independent

`list_tools` is served by the shared per-server client and cached
across conversations (the efficiency contract for N concurrent
conversations) — `conversation(auth=...)` credentials are **not** used
to list.  A server whose tool *list* itself requires per-user auth must
carry spec-level credentials sufficient to list; per-conversation
*visibility* is the client-side `tools=` filter.  Relatedly, local
`@app.tool` functions have no spec to declare `elicits=True` on — mount
an eliciting in-memory server via `MCPServer(server=..., elicits=True)`
instead.  See [Multi-tenant apps](multi-tenant.md).

## An unreachable server serves a degraded inventory

The inventory fan-out gives each server `tool_list_timeout` seconds
(default 10, an `MCPRuntime` knob).  A server that merely *rolled* —
its old session is dead but the server answers — self-heals inside
that budget: the client is rebuilt and the list re-fetched once, so a
redeploy no longer degrades anything.  The same holds once a server
that had been *absent* comes back: the client that lost its session is
rebuilt on the next listing rather than serving stale names forever.  A
server that cannot answer in time contributes its **last-known**
inventory — so a tool added while
it is unreachable stays invisible until it answers again — and a
server that has *never* answered is excluded from that merge entirely,
with a warning.  Agents keep running on the degraded list; nothing
raises and nothing blocks.  If your app must fail loud when a specific
server is absent, assert on the merged tool list at boot.

## No `AsyncIterator` tools

The plan flagged `AsyncIterator[str]` tool returns as a future
addition (an `@app.tool` whose body yields progress events for the
caller to consume via `.events()`).  That path is deferred — every
`@app.tool` today returns a single value when its body completes.
For long-running tools, plumb progress through your own logging
+ a separate observability sink; the agent loop sees only the
final string.

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
types at module scope.  Declaring guardrail instances or
sub-agent wrappers inside an enclosing test function reliably
trips a `NameError`-based `ConfigurationError` at decoration
time — the failure surfaces before any inference call, so the
fix is always "lift the declaration to module scope".

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

## `07_di_tools.py` is MCP-only — no LLM round-trip

The runnable demo at
[`07_di_tools.py`](../../examples/glow/07_di_tools.py) exercises
`@app.tool` + `Depends` stripping + `app.dependency_overrides` via
a direct MCP `call_tool` round-trip.  The agent-uses-tool flow
(LLM picks the tool, glow injects the dependency, the agent
narrates the result) is *not* shown end-to-end.  The mechanism is
identical to what `03_streaming.py` demonstrates (an agent
calling a `@app.tool`), so the dependency-injection layer adds
nothing new at the inference loop; the gap is example coverage,
not functionality.

## Sync API

There isn't one and there won't be.  t0-agents v1.0 is async-first
(`Agent.invoke` returns an awaitable `Run[T]`), and glow is
async-first by construction.  Use `asyncio.run(main())` at the
entry point of CLI scripts; embed inside any existing async
framework (FastAPI, httpx) directly.

## Where to drop down to v1.0

When the glow surface stands in your way, the escape hatch is
[`escape-hatch.md`](escape-hatch.md): every `@app.agent` wrapper
exposes `_agent`, the v1.0 `Agent` the decorator constructed, with
the full [`Run[T]`](../run.md) / [`Middleware`](../middleware.md) /
[`OutputMode`](../output-modes.md) surface intact.

---

For the canonical v1.0 narratives, follow:
[Agent](../agent.md) ·
[Run](../run.md) ·
[Output Modes](../output-modes.md) ·
[Streaming](../streaming.md) ·
[Guardrails](../guardrails.md) ·
[Multi-Agent](../multi-agent.md).
