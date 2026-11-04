# Getting started with `t0_agents.glow`

> Known limitations are catalogued at
> [`limitations.md`](limitations.md); the escape hatch to the v1.0
> `Agent` lives at [`escape-hatch.md`](escape-hatch.md).

`t0_agents.glow` is a Pythonic skin over the v1.0 primitives — `Agent`,
`Run[T]`, and `OutputMode[T]`.  Where the base API takes ~25 lines to
construct an `AgentConfig`, build the gateway and FastMCP client, and
orchestrate a `Run`, glow takes a handful: a module-level `App()` and a
function with a docstring and a return-type annotation, decorated with
`@app.agent`.

```python
from t0_agents.glow import App, llm

app = App()  # resolves TENSORZERO_URL / TENSORZERO_CONFIG at first use

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""
    return llm()

async with app:
    print(await tutor("How do I say hello in formal Japanese?"))
```

The docstring is the system prompt; the `-> str` return annotation
selects [`ChatOutput`](../output-modes.md); `await tutor(...)` returns
the unwrapped string directly.  Everything else — MCP, tool routing,
event streaming — is opt-in.  In a notebook or REPL you can skip even
the `async with`: consuming an agent auto-builds the App (one INFO log)
— the bracket buys deterministic teardown.

## Give the App a gateway

Pick whichever form fits; all four behave identically once running:

```python
app = App()                                    # deferred: TENSORZERO_URL,
                                               # else TENSORZERO_CONFIG (embedded),
                                               # else a clear error at first use
app = App(gateway="http://localhost:3000")     # HTTP gateway, App-owned
app = App(embedded="config/tensorzero.toml")   # embedded in-process — no docker
app = App(t0=t0)                               # bring your own AsyncTensorZeroGateway
```

`gateway=` / `embedded=` (and both env-var paths) are **App-owned**:
the App builds the gateway on first use and closes it on exit /
`shutdown()`.  A caller-supplied `t0=` is never entered or closed —
you own its lifecycle, exactly as before.

**Embedded, honestly**: `App(embedded=...)` runs the TensorZero gateway
in-process (`build_embedded`) with no backing databases — perfect for
dev loops and CI, with real LLM inference but *degraded observability*:
no ClickHouse means no episode queries and ephemeral feedback IDs
(spiritually `dryrun=True`).  Point it at any `tensorzero.toml` whose
providers have credentials in the environment (e.g. `GROQ_API_KEY`).
Production deployments should keep `build_http` against a managed
gateway.

Install glow with the extra:

```bash
pip install 't0-agents[glow]'
```

The extra pulls `pydantic>=2.0` (FastAPI precedent — glow uses pydantic
for type introspection in a few places).  See
[`limitations.md`](limitations.md#pydantic-is-a-hard-dep-of-the-glow-extra)
for the rationale.

The running examples under [`examples/glow/`](../../examples/glow/) are
the long-form companion to this tutorial — give them a gateway either
way (`docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`
+ `TENSORZERO_URL=http://localhost:3000`, or docker-free
`TENSORZERO_CONFIG=examples/glow/config/tensorzero.toml`) and run each
with `uv run python examples/glow/<NN>_*.py`.

## Notebooks and REPLs

Auto-build makes module-level Apps REPL-friendly, and two affordances
round it out: re-decorating a name is allowed with
`@app.agent("tutor", replace=True)` (the duplicate-name error remains
the default), and `app.remove_agent("tutor")` frees a name explicitly.
A replacement registered after the App started still gets its gateway
and `tools=` filter resolved before it first runs.

---

## Zero-TOML agents with `model=`

The hello-world above still assumes a `[functions.agent]` stanza in
`tensorzero.toml`.  With `model=` there is nothing to declare — the
gateway routes the call through its built-in passthrough function and
the agent is pure Python:

```python
@app.agent("tutor", model="groq::llama-3.3-70b-versatile")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""
    return llm()
```

`model=` accepts the `{provider}::{model}` shorthand or the name of a
`[models.*]` entry.  Tools, episodes, streaming, and tags all work
unchanged; the docstring rides along as the system prompt.  Per-model
knobs live next to it:

```python
@app.agent("planner", model="openai::gpt-4o",
           params={"chat_completion": {"temperature": 0.2}})
async def planner(q: str) -> str: ...
```

The TOML-declared production path is untouched — pin a function, a
variant, and cache behaviour explicitly when you want the full
TensorZero surface (templates, variants, experimentation):

```python
@app.agent("prod", function_name="prod_agent", variant="gpt4o_v2",
           cache={"enabled": "on", "max_age_s": 3600})
async def prod(q: str) -> str: ...
```

`model` and `function_name` are mutually exclusive (neither means the
back-compat `function_name="agent"`), `variant` needs the function
path, and a structured return annotation is rejected on `model=` —
the passthrough cannot validate JSON output; the error message carries
a ready-to-paste TOML stanza.  Two observability notes: functionless
inferences are stored under `tensorzero::default` (see
[`limitations.md`](limitations.md#functionless-inferences-are-invisible-to-function_name-filtered-memory)),
and every inference now carries a `t0_agents.agent = <name>` tag, so
per-agent segmentation works even across shared functions.

---

## Client-side prompt templates — `prompt_vars`

The docstring may contain `str.format` placeholders, filled per-call
with `.with_(prompt_vars={...})` — no server-side template required:

```python
@app.agent("tutor", model="groq::llama-3.3-70b-versatile")
async def tutor(query: str) -> str:
    """An English tutor for {level} students."""
    return llm()

await tutor("Explain gerunds.").with_(prompt_vars={"level": "advanced"})
```

Formatting happens client-side on the *effective* system prompt (an
explicit `.with_(system=...)` is formatted the same way).  A missing
placeholder raises immediately with the offending key; literal braces
only need doubling when `prompt_vars` is actually passed.  TensorZero
server-side templates remain available on declared functions via
`.with_(system_arguments={...})` — the two sides are mutually
exclusive per call.

---

## Toolless chat — no MCP plumbing needed

The v1.0 `Agent` accepts `tool_provider=None` directly: chat-mode
agents constructed without a `tool_provider` skip the MCP plumbing
entirely.  No FastMCP server is allocated, no `list_tools()`
roundtrip per turn — the loop runs a single inference and returns
the text response.  The 4-line hello-world above takes this path
by default, and the unlock lives in the core `Agent`, so every
consumer benefits, not just glow.

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
    return llm()

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
guardrail log, diagnostics — for callers who need more than the
unwrapped value; aggregate cost and token usage come from the run
handle's `.cost()` / `.usage()`.

---

## Multi-agent patterns

glow has no compose() helper of its own — three canonical patterns,
expressed in plain Python plus the v1.0 sub-agent surface, cover the
multi-agent space:

| Pattern      | Topology                              | Runnable example                                                                |
|--------------|---------------------------------------|---------------------------------------------------------------------------------|
| Sequential   | pipeline `A → B → C`                  | [`examples/glow/04_sequential.py`](../../examples/glow/04_sequential.py)        |
| Parallel     | fan-out + plain-Python reducer        | [`examples/glow/05_parallel.py`](../../examples/glow/05_parallel.py)            |
| Orchestrator | `Depends(<wrapper>)` sub-agents       | [`examples/glow/06_orchestrator.py`](../../examples/glow/06_orchestrator.py)    |

Sequential is `await b(await a(query))` plus any ad-hoc transform
in the gap.  Parallel is `asyncio.gather(a(q), b(q),
return_exceptions=True)` plus whatever reducer fits the data.
Orchestrator declares each sub-agent as a normal `@app.agent` and
threads them into the parent decorator via `Depends(<wrapper>)`
parameters; the v1.0 sub-agent installer (now in
[`t0_agents.delegation`](../multi-agent.md)) wires each as a
`<name>_ask` MCP tool on the parent's App-owned FastMCP server.

The full DI pattern — value-style `Depends(get_db)` for tools and
sub-agent injection for `@app.agent` — is documented in
[`depends-and-tools.md`](depends-and-tools.md).

---

## Where to go next

- **[Streaming and events](streaming-and-events.md)** — `FluentRun.events(*selectors)`,
  the seven typed selectors, and the `Selector(TextDeltaEvent)` recipe for
  token-by-token streaming.
- **[Depends and tools](depends-and-tools.md)** — value-style injection inside
  `@app.tool`, sub-agent injection inside `@app.agent`, and the
  `app.dependency_overrides` test hook.
- **[Typed MCP and guardrails](typed-mcp-and-guardrails.md)** — class-based
  `MCPServer` declarations, the `Annotated[...]` override, and the
  `Annotated[T, guardrail(...)]` metadata pattern.
- **[Escape hatch to v1.0](escape-hatch.md)** — when glow stands in your way:
  `wrapper._agent` is the v1.0 `Agent` the decorator built, plus
  `app.lifespan()` / `build()` / `shutdown()` for explicit lifecycle.
- **[Known limitations](limitations.md)** — what glow does not do yet
  (streaming JSON, AsyncIterator tools, sync API).
- **[v1.0 docs](../index.md)** — the underlying primitives glow compiles to.
