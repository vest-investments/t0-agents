# Getting started with `t0_agents.glow`

> **Experimental.** Public API on the
> [`experimental/glow`](https://github.com/sproutfi/t0-agents/tree/experimental/glow)
> branch is unstable.  This tutorial is the canonical entry point;
> known limitations are catalogued at [`limitations.md`](limitations.md).

`t0_agents.glow` is a Pythonic skin over the v1.0 primitives — `Agent`,
`Run[T]`, `OutputMode[T]`, `Composer`.  Where the base API takes ~25
lines to construct an `AgentConfig`, build the gateway and FastMCP
client, and orchestrate a `Run`, glow takes four: a function with
a docstring and a return-type annotation, decorated with `@app.agent`.

```python
from t0_agents.glow import App
app = App(t0=t0)

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""

print(await tutor("How do I say hello in formal Japanese?"))
```

The docstring is the system prompt; the `-> str` return annotation
selects [`ChatOutput`](../output-modes.md); `await tutor(...)` returns
the unwrapped string directly.  Everything else — MCP, tool routing,
event streaming — is opt-in.  Below is the full tour.  Each section
is short enough to read in one pass; the running examples under
[`examples/glow/`](../../examples/glow/) are the long-form companion.

Install glow with the extra:

```bash
pip install 't0-agents[glow]'
```

The extra pulls `pydantic>=2.0` (FastAPI precedent — glow uses pydantic
for type introspection in a few places).  See
[`limitations.md`](limitations.md#pydantic-is-a-hard-dep-of-the-glow-extra)
for the rationale.

---

## Toolless chat — no MCP plumbing needed

The v1.0 `Agent` requires a `tool_provider` (typically a FastMCP
`Client`) so the chat loop can advertise an empty tool list.  glow
relaxes that requirement: chat-mode agents constructed without a
`tool_provider` allocate an ephemeral in-memory FastMCP for the
invocation's duration.  Concretely, the 4-line hello-world above
runs without ever touching MCP — the M2 D5 v1.0-core unlock made
this possible for every `Agent` consumer, not just glow.

When you register tools (`@app.tool`) or mount MCP servers
(`servers=[MathMCP]`), glow switches to the App-owned in-memory
server.  Until then, the chat loop just iterates `t0.inference()`
once and returns text.  See [`01_hello.py`](../../examples/glow/01_hello.py)
for the runnable demo.

---

## Structured outputs by return type

Annotate the return type as a Pydantic model and glow picks
`StructuredOutput(M)` automatically:

```python
from pydantic import BaseModel

class TripPlan(BaseModel):
    destinations: list[str]
    cost_usd: float

@app.agent("planner", function_name="plan_trip")
async def plan(query: str) -> TripPlan:
    """Plan trips with realistic budgets."""

trip = await plan("3 days in Japan under $2000")
print(trip.destinations)        # statically typed list[str]
print(trip.cost_usd)
```

The mapping is one-to-one with [Output Modes](../output-modes.md):

| Return annotation                  | `OutputMode`                                |
| ---------------------------------- | ------------------------------------------- |
| `str` (or missing)                 | `ChatOutput()`                              |
| `dict`, `dict[str, Any]`, …        | `JsonOutput()`                              |
| `T(BaseModel)`                     | `StructuredOutput(T)`                       |
| `list[T(BaseModel)]`               | `StructuredOutput(<wrapper>)` (auto-unwrapped) |
| `Optional[T]` / `T \| None`        | as `T` (None permitted at parse)            |

Forward refs are resolved at decoration time; failures raise
`ConfigurationError` immediately, not on the first call.  Runnable
demo: [`02_structured.py`](../../examples/glow/02_structured.py).

---

## Per-call overrides with `.json()`, `.json(M)`, and `.text()`

The same agent can return three different shapes per invocation:

```python
text = await tutor("Say hi")                              # str
tips = await tutor("3 tips for learning English").json()  # dict
trip = await tutor("Plan Lisbon trip").json(TripPlan)     # TripPlan
```

`tutor(...)` returns a `FluentRun[str]`; calling `.json()` or
`.json(Model)` swaps the underlying `OutputMode` before the build
fires.  `.json(A).json(B)` is **last-wins** (dict-update precedent;
the swap logs at DEBUG).  `.as_(custom_mode)` accepts any
`OutputMode[U]` directly for fully-custom outputs.

The `await tutor("Plan Lisbon trip").result()` escape hatch returns
the full `AgentResult[T]` — `episode_id`, `inference_ids`, the
guardrail log, aggregated cost — for callers who need more than the
unwrapped value.

---

## Streaming and event selectors

`FluentRun.events(*selectors)` is the typed event stream from
[`Run.events()`](../streaming.md), filtered.  Seven module-level
selectors cover the [v1.0 event union](../../src/t0_agents/stream/events.py):
`Thought`, `ToolCall`, `Text`, `ToolResult`, `TurnEnd`, `Done`,
`Error`.  Compose them with `.where(pred)` (AND-chains predicates) and
`.map(fn)` (transforms the yielded value).

```python
from t0_agents.glow import Thought, ToolCall, Text

async for event in tutor("Explain present perfect").events(
    Thought,
    ToolCall.map(lambda tc: f"> calling {tc.name}"),
    Text,
):
    ...
```

`Text` defaults to `TextEndEvent` (the completion marker for one
text block) — for token-by-token streaming, use
`Selector(TextDeltaEvent).map(lambda e: e.delta)` instead.  Passing
no selectors yields every `AgentEvent` (parity with
`Run.events()`).  Glow only supports streaming for chat-flavoured
output modes; the [v1.0 chat-only constraint](../streaming.md) carries
through.  Demo: [`03_streaming.py`](../../examples/glow/03_streaming.py).

---

## Composers and agents are the same thing

The v1.0 [`Composable[Input, T]`](../composition.md) Protocol declares
one method: `invoke(input, /, **kwargs) -> Run[T]`.  `Agent` satisfies
it.  Every `Composer` factory (`sequential`, `parallel`, `consensus`,
…) returns a `Composer`, which is itself a `Composable`.  Composers
nest for free — `sequential(parallel(a, b), router_dict)` type-checks
and runs against the same contract.

glow's `compose()` returns a `Composer`, so anything you compose can
serve as a node of any further `compose()`.  No graph DSL, no edge
declarations:

```python
from t0_agents.glow import compose

deep_research = compose({arxiv_agent, perplexity_agent}, merge=union)
qa = compose({
    "research": compose([deep_research, summarizer]),
    "code":     code_agent,
    "default":  general_qa,
})
answer = await qa("Latest results on protein folding")
```

Each layer is a `Composable`.  Slot an agent or a composer
interchangeably wherever a node is expected.  Demo:
[`04_composition.py`](../../examples/glow/04_composition.py).

---

## Native composition — `compose(spec, *, merge=None)`

The `spec` shape selects the topology:

| Spec type                     | Topology                                                            |
| ----------------------------- | ------------------------------------------------------------------- |
| `list` / `tuple`              | `sequential(*spec)` — pipeline                                      |
| `set` / `frozenset`           | `parallel(*spec, reducer=merge, error_policy="continue")` — fan-out + merge |
| `dict[str, Composable]`       | LLM-routed dispatch via a custom `_LLMRoutedComposer`               |

The dict shape installs a synthetic FastMCP server with one no-op
tool per route key, wires a classifier `Agent` configured with
`ToolCallOutput(tool_choice="required")`, runs one inference per
call to pick the route, and dispatches the **original** user input
to `routes[picked]`.  This is *one extra inference per routing
decision*; budget accordingly.

> **Model capability requirement.**  The classifier uses
> `tool_choice="required"` with arbitrary synthetic tool schemas.
> Open-source models like `openai/gpt-oss-20b` are documented to
> return provider errors in this mode (M4 §3.3).  Use
> Claude Sonnet/Opus, GPT-4o, GPT-5, or Gemini-2.5-pro for the
> dict-router path.

---

## Sets and merging — the type-fragility risk

`compose({a, b, c})` runs every agent and merges results.  The
default reducer concatenates string `AgentResult.value`s with
`"\n---\n"`; mixed-type ensembles raise loudly and force the user
to pass an explicit `merge=`:

```python
def union(results):
    return sorted({src for r in results if not isinstance(r, BaseException) for src in r.value})

ensemble = compose({arxiv_agent, perplexity_agent}, merge=union)
```

The reducer receives raw entries: successful `AgentResult` for each
finished branch, and a `BaseException` (kept inline by
`error_policy="continue"`) for each branch that crashed.  If you
need first-finisher-wins, write it explicitly with
`asyncio.wait(return_when=FIRST_COMPLETED)` — glow does not bless
`race` as a primitive (see next section).

---

## Race is deprecated — use `parallel(*, reducer=…, error_policy=…)`

v1.0 still ships `race(*nodes)`, but it emits a `DeprecationWarning`
pointing at the same `parallel(*, reducer=..., error_policy="continue")`
factory that backs `compose({...})`.  Removal is slated for v1.1.

Why deprecated: first-finisher-cancels-the-rest pays for the
inferences anyway (the slow branches kicked off before being
cancelled), throws their work away, and the cancellation cascade is
best-effort.  glow's `set[Composable]` semantics map to fan-out +
merge precisely so users never get nudged toward the footgun.  See
[Composition appendix](../composition.md#advanced--not-recommended).

---

## `Depends` — tools and sub-agents

`Depends(provider)` is FastAPI's injection sentinel.  glow uses it
in two places:

**Inside `@app.tool` — value-style injection.**  Strip the dependency
parameter from the tool's auto-generated JSON Schema (so the LLM
never sees it) and re-inject it at call time:

```python
from t0_agents.glow import Depends

async def get_db() -> DB: ...

@app.tool
async def lookup_order(order_id: str, db: DB = Depends(get_db)) -> Order:
    return await db.orders.get(order_id)

# Tests:
app.dependency_overrides[get_db] = lambda: InMemoryDB(...)
```

Providers may themselves take `Depends` (recursive resolution); sync,
async, and sync-returning-awaitable shapes are all supported.  The
override dict is consulted on every call, not snapshotted at
registration — drop in a mock at test time without re-instantiating
the decorator.

**Inside `@app.agent` — sub-agent injection.**  When the parameter
is typed as a glow agent wrapper and the provider is itself a
glow-registered agent, glow installs it as a sub-agent on the
parent `AgentConfig.sub_agents` declaratively:

```python
@app.agent("orchestrator")
async def orchestrator(
    query: str,
    research: Agent = Depends(researcher),
    write: Agent    = Depends(writer),
) -> str:
    """Calls research → write."""
```

The sub-agents become `{name}_ask` tools on the orchestrator's MCP
server (the v1.0 [multi-agent composition](../multi-agent.md) path).
Value-style `Depends(get_db)` on `@app.agent` raises
`ConfigurationError` — the runtime only has two defined behaviours
for `@app.agent` parameters (the user message and sub-agents), so
glow rejects ambiguous shapes loudly.  Demo:
[`05_di_tools.py`](../../examples/glow/05_di_tools.py).

---

## Typed MCP

Two patterns coexist:

```python
from fastmcp import FastMCP
from t0_agents.glow import App, MCPServer

# 1. Class-based — preferred for shared servers.
class MathMCP(MCPServer, server=_math_mcp_instance, namespace="math"): ...

app = App(t0=t0, servers=[MathMCP])

# 2. Annotated — per-parameter override on a single agent.
@app.agent("calc")
async def calc(query: str, math: Annotated[Any, MCPServer(url="http://...")]) -> str:
    """Math assistant with a remote MCP server."""
```

Class-based declarations register the spec on
`__init_subclass__`; annotation-based declarations are harvested by
`@app.agent`'s parameter walk.  `App.build()` mounts every spec on
the App-owned composed FastMCP via `FastMCP.mount()` for in-memory
servers and `fastmcp.server.create_proxy(url)` for remote
HTTP/SSE/STDIO endpoints.  Remote auth headers and transport
options pass through `create_proxy(url, ...)`.

Every glow agent on a given `App` sees every mounted server — per-
agent MCP subsets are a [known limitation](limitations.md#no-per-agent-mcp-subsets);
spin up multiple `App` instances for strict isolation.  Demo:
[`06_mcp_typing.py`](../../examples/glow/06_mcp_typing.py).

---

## Guardrails as type metadata

The v1.0 [guardrail factories](../guardrails.md) (`require_approval`,
`max_cost`, `max_tool_calls`, `block_tools`, `require_parsed`) plus
M5's new `regex_guardrail` (importable from both `t0_agents.guardrails`
and `t0_agents.glow`) can be attached as `Annotated` metadata on
`@app.agent` parameter or return annotations:

```python
from typing import Annotated
from t0_agents.glow import regex_guardrail
from t0_agents.guardrails import max_cost

no_pii = regex_guardrail(r"\b\d{3}-\d{2}-\d{4}\b", "no SSN")
under_budget = max_cost(0.50)

@app.agent("support")
async def support(query: str) -> Annotated[str, no_pii, under_budget]:
    """Customer support."""
```

`@app.agent`'s decoration-time walk harvests every Annotated
metadata entry satisfying the `Guardrail` Protocol and folds it
into `AgentConfig.guardrails`.  Runtime path is unchanged — the
canonical `GuardrailsMiddleware` does the work.  `phase="before_run"`
guardrails (e.g. `regex_guardrail(pattern, msg, phase="before_run")`)
also work attached to a parameter annotation; the metadata harvest
is symmetric.

---

## Lifespan, build, shutdown

The App owns a long-lived in-memory FastMCP server + matching
Client (`app._owned_client`).  Tools registered via `@app.tool`
land on that server; class-based `MCPServer` declarations mount
remote endpoints into the same composed view.  The Client enters
its in-process transport task inside `app.build()`; `app.shutdown()`
closes it; `app.lifespan()` is the recommended entry point and does
`build → yield → shutdown` in a `try/finally`:

```python
async def main() -> None:
    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("tutor")
        async def tutor(q: str) -> str: ...

        async with app.lifespan():            # ← here
            print(await tutor("Say hi"))
```

Without `lifespan()` (or an explicit `build()`/`shutdown()` pair),
the first agent invocation reaches an unentered Client and the
FastMCP transport raises.  Build is idempotent; `lifespan()` is
re-entrant in the sense that consecutive calls construct and tear
down cleanly.  Forward-ref failures, duplicate agent names, and
mis-shaped `Depends` declarations all surface at `build()` time
rather than at the first inference, matching FastAPI's fail-fast
ergonomics.

---

## Where to go next

- **All six runnable examples** under
  [`examples/glow/`](../../examples/glow/) — boot the gateway with
  `docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`
  and run each with `uv run python examples/glow/<NN>_*.py`.
- **[Known limitations](limitations.md)** — what's deferred to v1.1.
- **[v1.0 docs](../index.md)** — the underlying primitives glow compiles
  to.  Glow's escape hatch is `wrapper._agent` (the v1.0 `Agent`
  built by the decorator); drop down to `Agent.invoke(...)` whenever
  the high-level surface is in your way.
- **Branch:**
  [`experimental/glow`](https://github.com/sproutfi/t0-agents/tree/experimental/glow)
  — the branch this tutorial documents.  Public API is unstable
  until the user opts in to merging back to `main`.
