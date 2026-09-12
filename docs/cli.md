# CLI — `t0-agents` (Typer + Rich)

The serving chapter covered the FastAPI layer for production. The CLI is its development counterpart: `t0-agents run` drives any agent factory through a Rich-based streaming TUI, `t0-agents serve` wraps `create_app(...)` behind a uvicorn dev server with sensible defaults, and `t0-agents info` introspects a loaded agent's configuration — same stack, terminal front door.

Terminal-native way to run, serve, and inspect any `t0-agents` agent.
The `[cli]` extra ships a `t0-agents` console script with three
subcommands — `run`, `serve`, and `info` — built on
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).
It turns a declarative `Agent` into an interactive streaming REPL, a
one-shot dev HTTP server, and a config-introspection table, without
writing any glue code.

```python
from t0_agents.cli import (
    TUIRenderer,
    rich_elicitation_handler,
    make_rich_elicitation_handler,
    get_active_renderer,
    load_agent_factory,
    AgentFactory,
    main,
)
```

---

## Installation

```bash
pip install 't0-agents[cli]'      # or: uv sync --extra cli
```

The `[cli]` extra bundles every dependency the three subcommands need:
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
for the CLI / rendering surface, plus
[FastAPI](https://fastapi.tiangolo.com/),
[uvicorn](https://www.uvicorn.org/),
[Pydantic](https://docs.pydantic.dev/), and
[sse-starlette](https://github.com/sysid/sse-starlette) (shared with
the `[serve]` extra) so `t0-agents serve` is self-contained. Base
installations remain minimal — the CLI surface is strictly opt-in.

---

## Requirements

- Python ≥ 3.11.
- A TensorZero gateway the factory can reach — typically booted via the
  repo's `docker compose up -d` for local development (see
  [Getting Started](getting-started.md)).
- A provider API key in the environment (e.g. `GROQ_API_KEY`) so the
  gateway can actually run inferences.

Every subcommand enters the agent factory, which in turn opens a real
gateway connection and MCP client session — there is no Docker-free
quickstart for the CLI (the factory contract requires live resources).
`info` holds the connection open only long enough to read
`agent.config`; `run` and `serve` hold it for the whole session.

---

## Quickstart

Run the shipped example agent end-to-end. The example lives at
[`examples/agent/10_cli/`](../examples/agent/10_cli/) and exposes two tools: `add`
(plain) and `set_reminder` (elicitation).

```bash
docker compose up -d                   # boots gateway + clickhouse
export GROQ_API_KEY=gsk_...            # or whichever provider the function uses
uv sync --extra cli

uv run t0-agents info examples.10_cli.agent:assistant
```

`info` prints a Rich table with every field on the `AgentConfig` the
factory produces — a fast sanity check before starting the REPL:

```
AgentConfig · tui-assistant
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field          ┃ Value                             ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ name           │ 'tui-assistant'                   │
│ function_name  │ 'agent'                           │
│ max_turns      │ 10                                │
│ memory         │ InMemoryMemory                    │
│ tags           │ example='13_tui'                  │
│ …              │ …                                 │
└────────────────┴───────────────────────────────────┘
```

Now start the streaming REPL:

```bash
uv run t0-agents run examples.10_cli.agent:assistant
```

A cyan Rich panel appears, titled with the agent's name. Type at the
`You:` prompt; `Ctrl-D` or `Ctrl-C` quits.

```
You: What is 2 + 3?
╭─ tui-assistant ─────────────────────────────╮
│ 🔧 add({"a":2,"b":3})                       │
│ ✓ add({"a":2,"b":3})                        │
│                                             │
│ 2 + 3 is 5.                                 │
╰─────────────────────────────────────────────╯
```

Ask the agent to set a reminder and the Rich-based elicitation handler
pauses the live panel, collects each field, then resumes:

```
You: Remind me tomorrow at 9 to water the plants.
Reminder at tomorrow 9: water the plants
confirm (y/n): y
channel [email/sms/push] (email): sms
note: bring the watering can
╭─ tui-assistant ─────────────────────────────╮
│ ✓ set_reminder({"when":"tomorrow 9",...})   │
│                                             │
│ Scheduled reminder for 'tomorrow 9' via sms │
╰─────────────────────────────────────────────╯
```

---

## The three commands

| Command | What it does |
| --- | --- |
| `t0-agents run <spec>` | Interactive streaming REPL. Live-renders deltas, tool-call spinners, and elicitation prompts inside a Rich panel. |
| `t0-agents serve <spec>` | Wraps the agent in [`t0_agents.serve.create_app(..., share_client=True)`](serving.md) and runs uvicorn (non-streaming by default). |
| `t0-agents info <spec>` | Prints a tabulated `AgentConfig` — useful before booting the REPL or the server. |

All three take an **agent factory spec** in `module:attr` form — see
the next section.

---

## The agent factory convention

Every subcommand expects the `module:attr` reference to resolve to a
**zero-argument callable returning an async context manager** that
yields an `Agent`. In practice that is a function decorated with
`@asynccontextmanager`:

```python
# my_package/my_agent.py
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Awaitable
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.cli import rich_elicitation_handler
from .server import mcp  # your FastMCP server


@asynccontextmanager
async def assistant() -> AsyncIterator[Agent]:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(mcp, elicitation_handler=rich_elicitation_handler)
    async with await gateway as t0, client:
        yield Agent(
            AgentConfig(name="assistant", function_name="agent", max_turns=10),
            t0=t0,
            tool_provider=client,
        )
```

Why a factory rather than a bare `Agent` at module level? The
`fastmcp.Client` must be entered as `async with` before tool execution,
and `AsyncTensorZeroGateway.build_http` returns an awaitable that must
be `await`-ed. A zero-arg context-manager factory is the cleanest way
for the CLI to bracket both lifecycles around the REPL, the uvicorn
lifespan, or the `info` lookup. `Run.events()` ultimately calls
through to `Client.list_tools()`, which fails unless the session is
open; trying to use a pre-built `Agent` would need the CLI to reach
into private attributes to open the client.

Pointing a subcommand at a bare `Agent` produces an actionable error
with the recommended factory snippet inline:

```
Error: 'my_package.my_agent:bare_agent' resolves to a bare Agent
instance, but the CLI expects an async-context-manager factory so it
can manage the gateway and MCP client lifecycle.  Wrap your Agent
construction in an @asynccontextmanager async def, for example:

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def bare_agent():
        async with await gateway as t0, Client(mcp) as client:
            yield Agent(config, t0=t0, tool_provider=client)
```

See the [Error handling](#error-handling) section for the full loader
error taxonomy.

## Glow App specs

A spec may also resolve to a **glow `App` instance** — no factory of
your own required.  The CLI adapts it: it enters `async with app:`
around the session (gateway resolution, MCP mounts, client connection,
`tools=` filters) and drives the selected agent.

```bash
t0-agents run examples.glow.01_hello:app --agent tutor "hi"   # one-shot
t0-agents run examples.glow.01_hello:app                      # REPL (sole agent)
t0-agents serve mymodule:app --agent tutor
t0-agents info mymodule:app --agent tutor
```

- `--agent NAME` picks the agent when the App registers several; with
  exactly one registered it may be omitted.  All three subcommands
  accept it.  Ambiguous or unknown names
  produce an error listing what is available.
- The optional positional `MESSAGE` on `run` is a **one-shot**: send
  it, render the reply, exit — no REPL.  It works for classic factory
  specs too.
- `serve` wraps the selected agent in `create_app(...,
  share_client=True)` exactly like a factory spec — which means a
  **toolless** glow App (no `@app.tool`, no `MCPServer`s) cannot be
  served this way: `share_client` requires a tool provider and raises
  its standard `ConfigurationError`.  Compose
  [`create_app_from_glow`](serving.md#glow-apps) /
  [`create_app`](serving.md) directly for bespoke shapes.

---

## Concepts

### Lifecycle

Every subcommand follows the same shape:

```
t0-agents <cmd> <module:attr>
    │
    └─ asyncio.run(...)
        │
        ├─ load_agent_factory(spec)         ← validates, imports, type-checks
        │
        └─ async with factory() as agent:   ← gateway + Client opened here
              │
              ├─ agent.warmup()             ← optional readiness probe
              │
              └─ (run: renderer.consume(agent.invoke(...).events()))
                 (serve: uvicorn.Server(create_app(agent, ...)).serve())
                 (info: print(agent.config))
        │
        └─ factory() __aexit__              ← gateway + Client closed here
```

The factory's async-with block brackets the *entire* session, so
anything you set up inside it (memory backends, guardrails, the
`elicitation_handler` on the Client) persists for the whole REPL /
server lifetime and is torn down cleanly when the user presses
`Ctrl-D`, kills the server, or `info` finishes printing.

### Rendering pipeline

`t0-agents run` turns the async stream from `Run.events()` into a
live terminal frame:

```
Run.events() ──▶ AsyncIterator[AgentEvent] ──▶ TUIRenderer.consume()
                                                      │
                                                      └─ Rich Live + Panel
                                                          (refresh 20×/s)
```

The renderer folds deltas into three stacked regions — tool-call status
lines, reasoning traces (hidden by default), and the accumulating
assistant reply — and repaints the panel at 20 Hz. Every `AgentEvent`
variant is documented in [Streaming](streaming.md) § "Event reference".

### The active-renderer handoff

Rich's `Live` region redirects `stdout` / `stderr` and installs a render
hook on the `Console`. While Live is running, any `Prompt.ask` call
would paint *underneath* the panel and the 20 Hz refresh thread would
keep redrawing on top of whatever the user is typing.

The CLI solves this by exposing a process-wide reference to the
currently-active `TUIRenderer`. When `TUIRenderer.consume()` starts it
registers itself via a module-global binding (`get_active_renderer()`
reads the same binding); the zero-config `rich_elicitation_handler`
picks up that reference and calls `renderer.pause()` before each Rich
prompt, then `renderer.resume()` after. The FastMCP `Client` spawns its
receive loop in an anyio task group it opens on `__aenter__`, and that
task's `contextvars` snapshot is frozen before the REPL even runs — so
a `ContextVar`-based handoff wouldn't reach the elicitation callback.
The module-global binding, guarded by a `threading.Lock`, bypasses that
problem and lets the handler find the renderer from any task. Custom
handlers can read the binding the same way; see the Advanced tier of
[Custom renderers](#custom-renderers).

---

## API reference

Every symbol below is re-exported from `t0_agents.cli`.

### `main`

```python
def main() -> None
```

Script entry point wired to `[project.scripts] t0-agents =
"t0_agents.cli.app:main"` in `pyproject.toml`. Invokes the Typer app;
exits via `typer.Exit` / `typer.BadParameter` on user errors.

### `load_agent_factory`

```python
def load_agent_factory(spec: str) -> AgentFactory
```

Resolve a `module:attr` string to an `AgentFactory`. The CLI uses this
internally for all three subcommands; exposed publicly for tests and
for consumers that want to embed the same resolution semantics.

| Parameter | Description |
| --- | --- |
| `spec` | `"module.path:attribute"`. The attribute must be a zero-argument callable whose `()` invocation returns an async context manager yielding an `Agent`. |

**Raises:** `typer.BadParameter` if the spec is malformed, the module
is not importable, the attribute is missing, the attribute is a bare
`Agent` instance (error message points at the factory idiom), or the
attribute is not callable.

Before importing, the loader prepends the current working directory to
`sys.path` if it isn't already there. Installed console scripts do not
get `cwd` on `sys.path` the way `python script.py` does, so this lets
`t0-agents run examples.10_cli.agent:assistant` work from the repo root
without a `PYTHONPATH=.` prefix.

### `AgentFactory`

```python
AgentFactory: TypeAlias = Callable[[], AbstractAsyncContextManager[Agent]]
```

Type alias for what the spec must resolve to. Canonical form is
`@asynccontextmanager async def factory(): ...`, but any callable that
satisfies the protocol works (including a class with an async
`__aenter__` / `__aexit__` pair).

### `TUIRenderer`

```python
class TUIRenderer:
    def __init__(
        self,
        agent_name: str,
        *,
        console: Console | None = None,
        verbose: bool = False,
        refresh_per_second: float = 20.0,
    ) -> None: ...

    async def consume(
        self,
        events: AsyncIterator[AgentEvent],
    ) -> AgentResult | None: ...

    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def reset(self) -> None: ...

    agent_name: str
    console: Console
    verbose: bool
    last_episode_id: str | None
    last_result: AgentResult | None
```

Rich `Live` + `Panel` renderer. Construct once per session; call
`consume()` once per user turn. Between turns `last_episode_id` and
`last_result` persist so the REPL can thread episode continuity across
turns; per-turn text / tool / error state is reset automatically at the
start of each `consume()`.

| Parameter | Description |
| --- | --- |
| `agent_name` | Panel title. Typically `agent.config.name`. |
| `console` | Optional Rich `Console`; defaults to a freshly constructed one. Pass an explicit console if you want custom styling or a shared output stream. |
| `verbose` | If `True`, `ThoughtEvent` / `ThoughtDeltaEvent` content is rendered in italic-dim within the panel. Defaults to `False` — reasoning traces are noisy. |
| `refresh_per_second` | Live refresh rate. Higher values render deltas more smoothly at the cost of CPU. |

`consume()` returns the `AgentResult` carried by the final `DoneEvent`,
or `None` if the iterator completed without emitting one. Exceptions
from the iterator propagate after the panel stops.

`pause()` / `resume()` are idempotent and safe to call when no Live
region is active — the Rich-based elicitation handler uses them to
temporarily yield the terminal to a `Prompt.ask`.

### `get_active_renderer`

```python
def get_active_renderer() -> TUIRenderer | None
```

Thread-safe lookup of the currently-active `TUIRenderer`. Returns
`None` when no `TUIRenderer.consume()` is running. Used by
`rich_elicitation_handler` to coordinate with the live panel without
the caller having to plumb the renderer through FastMCP's
`elicitation_handler` signature; advanced handlers can call it
themselves (see [Custom renderers](#custom-renderers)).

### `rich_elicitation_handler`

```python
async def rich_elicitation_handler(
    message: str,
    response_type: type[Any] | None,
    params: ElicitRequestParams,
    context: RequestContext[Any, LifespanContextT],
) -> ElicitResult[dict[str, Any]]
```

Zero-configuration FastMCP
[`ElicitationHandler`](https://gofastmcp.com/servers/elicitation). Plug
it into a `Client` directly and the handler auto-discovers any active
`TUIRenderer`:

```python
Client(mcp, elicitation_handler=rich_elicitation_handler)
```

Outside a `TUIRenderer.consume()` context (tests, scripts, non-REPL
callers) it falls back to a plain `Console` with no pause/resume
coordination.

### `make_rich_elicitation_handler`

```python
def make_rich_elicitation_handler(
    *,
    renderer: TUIRenderer | None = None,
    console: Console | None = None,
) -> Callable[
    [str, type[Any] | None, ElicitRequestParams, RequestContext[Any, Any]],
    Awaitable[ElicitResult[dict[str, Any]]],
]
```

Factory that returns a handler closed over an explicit renderer and/or
console. Use this when you want deterministic binding (for tests, or
when multiple handlers might live in the same process):

```python
handler = make_rich_elicitation_handler(
    renderer=my_renderer,
    console=my_console,
)
Client(mcp, elicitation_handler=handler)
```

Arguments left as `None` fall back to `get_active_renderer()` at call
time, mirroring the zero-config handler's behaviour.

---

## `t0-agents run`

```
t0-agents run AGENT_SPEC [--verbose/-v] [--debug-log PATH]
```

Start an interactive streaming REPL against the factory `AGENT_SPEC`
resolves to. `Ctrl-D` or `Ctrl-C` at the `You:` prompt exits cleanly.

### Easy — just run it

```bash
uv run t0-agents run my_package.my_agent:assistant
```

Each user turn drives one `agent.invoke(...).events()` invocation. The renderer
folds the event stream into:

- A pending / success / error status line per tool call
  (`🔧 add({"a":1,"b":2})` → `✓ add({"a":1,"b":2})` or `❌ ...`).
- An accumulating assistant reply, token-by-token.
- Red exclamation-prefixed lines for recoverable `ErrorEvent`s.

### Intermediate — reasoning, continuity, debug logs

- `-v` / `--verbose` — render reasoning content (`ThoughtEvent`,
  `ThoughtDeltaEvent`) in italic-dim inside the panel. Useful with
  reasoning-model functions; ignore for chat-only models.

- **Episode continuity.** The REPL captures the last
  `DoneEvent.result.episode_id` and threads it into the next
  `agent.invoke(...).events()` call automatically, so memory state and
  TensorZero's episode tracking stay coherent across the whole session.

- `--debug-log PATH` — write `DEBUG`-level logs from every
  `t0_agents.*` logger to a file, so you can diagnose streaming /
  elicitation interactions without mixing log lines into the Live
  panel.

  ```bash
  uv run t0-agents run my_package.my_agent:assistant --debug-log repl.log
  tail -f repl.log     # in another terminal
  ```

### Advanced — swap the renderer

`TUIRenderer` is public. Bring your own renderer (Textual app, custom
`rich.layout.Layout`, headless JSON logger, …) by implementing a
coroutine that consumes `AsyncIterator[AgentEvent]`. See
[Custom renderers](#custom-renderers).

---

## `t0-agents serve`

```
t0-agents serve AGENT_SPEC [--host HOST] [--port PORT]
                           [--streaming | --no-streaming]
                           [--deployment LABEL]
```

Wraps the factory in
[`t0_agents.serve.create_app(agent, share_client=True, streaming=…,
deployment_label=…)`](serving.md) and runs uvicorn.

### Easy — local dev server

```bash
uv run t0-agents serve my_package.my_agent:assistant --port 8080
```

Boots the [non-streaming HTTP surface](serving.md):
`POST /chat`, `POST /feedback`, `GET /health`, `GET /info`. One `Agent`
instance, one `Client`, one gateway connection, reused across every
request.

### Intermediate — host, port, deployment label

```bash
uv run t0-agents serve my_package.my_agent:assistant \
    --host 127.0.0.1 --port 8080 --deployment my-demo
```

- `--host`, `--port` — standard bind flags (defaults `0.0.0.0:8080`).
- `--deployment LABEL` — merged into every inference's tags as
  `t0_agents.deployment=<label>`. Surfaces in TensorZero's
  observability pipeline; see [Observability](#observability).

### Advanced — opting in to streaming

```bash
uv run t0-agents serve my_package.my_agent:assistant --streaming
```

`--streaming` is **off by default**. Enabling it mounts
[`GET /chat/stream` (SSE) and `WS /chat/ws`](streaming.md) alongside
the non-streaming routes, but the CLI always uses `share_client=True`:
one `Client(mcp)` is reused across every connection. That is
structurally incompatible with the WebSocket elicitation route, which
needs a *per-connection* elicitation handler — one `elicitation_handler`
is allowed per `Client`, so a shared Client cannot demultiplex
elicitation across concurrent browser sessions. Every WebSocket
connection whose tools call `ctx.elicit()` will close with a
`ConfigurationError` explaining the constraint.

Use `--streaming` only when the served agent's tools never call
`ctx.elicit` — SSE works fine, and WS is simply an unused endpoint.

For full streaming + browser elicitation, drop the CLI and call
[`create_app`](serving.md) directly with a `client_factory` plus an
`elicitation_client_factory`. The reference implementation is at
[`examples/agent/07_streaming/agent_app.py`](../examples/agent/07_streaming/agent_app.py),
and the protocol reference lives in [Streaming](streaming.md).

---

## `t0-agents info`

```
t0-agents info AGENT_SPEC [--agent NAME]
```

Enter the factory, read `agent.config`, exit. Prints a single Rich
table with one row per field.

```bash
uv run t0-agents info my_package.my_agent:assistant
uv run t0-agents info mymodule:app --agent tutor
```

- `memory` renders as `type(value).__name__` (e.g. `InMemoryMemory`,
  `SqliteMemory`) rather than the full object — just enough to
  identify which backend is wired in.
- `guardrails` renders as a comma-separated list of each guardrail's
  `name`, or `—` when empty.
- `tags` renders as `key='value'` pairs, or `—` when empty.

Because `info` still opens the factory, you'll see a few seconds of
gateway setup before the table appears. The connection closes as soon
as the table is rendered.

---

## Custom renderers

The stock `TUIRenderer` paints one Rich `Panel`. Replace or wrap it
whenever the shape of your UI needs to diverge.

### Easy — bring your own `consume`

```python
from collections.abc import AsyncIterator
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent, DoneEvent,
)


async def my_consume(events: AsyncIterator[AgentEvent]) -> AgentResult | None:
    async for event in events:
        match event:
            case TextDeltaEvent(delta=delta):
                print(delta, end="", flush=True)
            case ToolCallEvent(name=name, arguments=args):
                print(f"\n[tool call] {name}({args})")
            case ToolResultEvent(name=name, is_error=err):
                print(f"[{'error' if err else 'ok'}] {name}")
            case DoneEvent(result=result):
                return result
    return None
```

Every event variant is documented in [Streaming](streaming.md)
§ "Event reference".

### Intermediate — wrap `TUIRenderer`

Subclass or compose `TUIRenderer` when you want the panel + delta
folding but need custom event handling on top:

```python
from t0_agents.cli import TUIRenderer
from t0_agents.stream.events import AgentEvent, ErrorEvent


class AuditingRenderer(TUIRenderer):
    def __init__(self, agent_name: str, audit_log: list[str]) -> None:
        super().__init__(agent_name)
        self._audit_log = audit_log

    def _update(self, event: AgentEvent) -> None:
        if isinstance(event, ErrorEvent):
            self._audit_log.append(event.message)
        super()._update(event)
```

`pause()` and `resume()` remain available so your renderer keeps
playing nicely with any elicitation handler that expects a
`TUIRenderer`-shaped surface.

### Advanced — layer a custom handler on top

Because the active `TUIRenderer` is exposed through
`get_active_renderer()`, a custom elicitation handler can coordinate
with whatever panel is running without the caller threading it
through explicitly:

```python
from rich.prompt import Prompt
from fastmcp.client.elicitation import ElicitResult
from t0_agents.cli import get_active_renderer


async def my_elicitation_handler(message, response_type, params, context):
    renderer = get_active_renderer()
    if renderer is not None:
        renderer.pause()
    try:
        answer = Prompt.ask(
            message,
            console=renderer.console if renderer else None,
        )
        return ElicitResult(action="accept", content={"value": answer})
    finally:
        if renderer is not None:
            renderer.resume()
```

For a fully runnable version that also composes on
`make_rich_elicitation_handler` (so the stock pause / resume + schema
dispatch stay in place) and writes an audit trail, see
[`examples/agent/10_cli/advanced.py`](../examples/agent/10_cli/advanced.py) — it
exposes `assistant_with_audit`, runnable via `t0-agents run
examples.10_cli.advanced:assistant_with_audit`.

The binding is a process-wide module global guarded by a
`threading.Lock`. It is NOT a `ContextVar` on purpose: FastMCP's
`Client` spawns its receive loop under an anyio task group opened at
`__aenter__` time, so a `ContextVar` snapshot taken inside the
receive-loop task would not see a `.set()` call made later by
`TUIRenderer.consume()`. The module global avoids that hazard at the
cost of a single-process, single-active-REPL constraint — not a
concern for a human-facing terminal UI.

The CLI's built-in renderer lives at
[`src/t0_agents/cli/tui.py`](../src/t0_agents/cli/tui.py) and is a
reasonable template for more ambitious custom surfaces.

---

## Elicitation in the TUI

The CLI drives FastMCP's
[elicitation protocol](https://gofastmcp.com/servers/elicitation) via
Rich prompts. See the [Elicitation](elicitation.md) doc for the
framework-wide story; this section covers the CLI-specific surface.

### Easy — zero-config handler

Wire `rich_elicitation_handler` into your `Client`:

```python
from t0_agents.cli import rich_elicitation_handler

client = Client(mcp, elicitation_handler=rich_elicitation_handler)
```

The handler automatically:

- Renders a binary request (empty-schema form or no form) as
  `rich.prompt.Confirm` — `y` → `accept`, `n` → `decline`.
- Renders a schema-driven form request as one prompt per property:
  `IntPrompt` for `integer`, `FloatPrompt` for `number`, `Confirm` for
  `boolean`, `Prompt` with `choices=` for `enum`, plain `Prompt` for
  `string`.
- Coerces `enum` answers back to the original Python type — if the
  server declared `enum: [1, 2, 3]` the handler returns the `int`, not
  the string the user typed.
- Pauses any active `TUIRenderer` around each prompt, then resumes, so
  the Live panel doesn't repaint over your typing.
- Collapses `EOFError` / `KeyboardInterrupt` into
  `ElicitResult(action="cancel", content=None)` so a piped-stdin
  exhaustion ends cleanly rather than bubbling as a tool error.

### Intermediate — explicit renderer binding

Use `make_rich_elicitation_handler(renderer=..., console=...)` when you
want deterministic binding instead of relying on the active-renderer
lookup (tests, shared-process setups, multiple concurrent handlers):

```python
from t0_agents.cli import TUIRenderer, make_rich_elicitation_handler

renderer = TUIRenderer("assistant")
handler = make_rich_elicitation_handler(renderer=renderer, console=renderer.console)

client = Client(mcp, elicitation_handler=handler)
```

The handler relies on a strict subset of JSON Schema that FastMCP
validates at server construction time: a flat `type: "object"` whose
properties are primitives (`string`, `integer`, `number`, `boolean`)
plus `enum` / `const`, with `description` and `default` preserved per
field. Nested objects and arrays-of-objects are rejected server-side,
so the handler can iterate top-level properties without worrying about
recursion. For scalar responses, FastMCP wraps the field in
`{"value": ...}`; the handler produces the same shape.

### Advanced — URL flows and non-FastMCP servers

- **URL-mode elicitation.** When the server issues an out-of-band flow
  (OAuth, payment confirmation, any "go complete this elsewhere"
  experience), `params` is an `ElicitRequestURLParams` rather than a
  form. The handler prints the message and URL, asks a single
  `Confirm("Have you completed the out-of-band flow?")`, and returns
  `accept` / `decline` with `content=None`. No schema introspection is
  attempted.

- **Non-FastMCP servers.** When pointing the CLI at a server outside
  FastMCP's validated-schema guarantees, any unknown field `type`
  falls back to a plain string `Prompt.ask` with a logged warning.
  The run continues; the answer is returned as `str`.

- **Cancellation semantics.** `Ctrl-C` at any prompt returns
  `ElicitResult(action="cancel", content=None)`. Handler exceptions
  from inside `_run_prompt` are logged locally and re-raised so FastMCP
  can wrap them as `ErrorData` for the server side.

- **Layered handlers.** A custom handler can *wrap*
  `make_rich_elicitation_handler()` rather than reinventing the
  prompt dispatch — the wrapper adds behaviour (logging, consent gates,
  side-channel notifications) while the stock handler keeps driving
  the actual prompts. See
  [`examples/agent/10_cli/advanced.py`](../examples/agent/10_cli/advanced.py) for a
  runnable audit-logging wrapper.

---

## Error handling

The loader is the single source of user-facing errors for all three
subcommands. Every failure surfaces as `typer.BadParameter`, which
Typer prints as a single-line `Error:` message and exits non-zero —
scriptable and CI-friendly.

| Condition | Message shape |
| --- | --- |
| Missing `:` in spec | `"Spec must be of the form 'module:attr', got '<spec>'. Example: 'examples.10_cli.agent:assistant'."` |
| Empty module or attr name | `"Spec must have a non-empty module and attribute, got '<spec>'."` |
| `ModuleNotFoundError` | `"Could not import module '<name>': <exc>. Make sure the module is importable…"` |
| Generic `ImportError` | `"Failed to import module '<name>': <exc>."` |
| `AttributeError` on the module | `"Module '<name>' has no attribute '<attr>'."` |
| Attribute is a bare `Agent` instance | Multi-line error with an `@asynccontextmanager` snippet inline. |
| Attribute is not callable | `"'<spec>' resolves to <type>, but the CLI expects a callable that returns an async context manager yielding an Agent."` |

Errors raised inside the factory itself (gateway unreachable, missing
API key, MCP server crash) bubble up as normal exceptions — Typer
prints the traceback and exits non-zero. For the general
cross-submodule error taxonomy, see
[Error handling](error-handling.md).

---

## Production considerations

The CLI is designed for interactive development and quick deployment,
not for long-lived production serving. A few notes for picking the
right surface:

- **`t0-agents run` is interactive only.** There is no way to drive it
  from a pipe or a cron job because the REPL owns `stdin`. For
  headless automation, call `Agent.invoke` or `Run.events` directly
  from Python (see [Agent](agent.md)), or run `t0-agents serve` behind
  a request-driven caller.

- **`t0-agents serve` is a dev server.** It is the same FastAPI app
  you'd build with [`create_app(share_client=True)`](serving.md), so
  the [Serving](serving.md) § "Production notes" applies verbatim —
  the CLI is just a one-liner wrapper around that. For multi-tenant
  production, wire `client_factory` / `session_provider` directly into
  `create_app` instead of sharing a single `Client` across requests.

- **Terminal requirements.** Rich `Live` depends on an ANSI-capable
  terminal and correct `stdout` size detection. CI environments
  without a TTY (e.g. `docker run` with no `-t`) will still work but
  render as a scrolling log rather than a refreshing panel. Set
  `TERM=dumb` if you want fully plain output.

- **Diagnostics.** `--debug-log PATH` on `t0-agents run` is the
  supported way to capture `t0_agents.*` DEBUG logs during a session
  without contaminating the Live panel. Combine with `RICH_TRACEBACK=1`
  for formatted tracebacks on any uncaught exception.

See also [Production](production.md) for the cross-cutting deployment
story.

---

## Observability

- **Episode threading.** The REPL captures each turn's
  `DoneEvent.result.episode_id` and threads it into the next
  `agent.invoke(...).events()` call automatically. Memory load/save stays
  keyed by the same episode; TensorZero sees one long trace per REPL
  session rather than one per user turn.

- **Deployment tag.** `t0-agents serve --deployment <label>` merges
  `t0_agents.deployment=<label>` into every inference's tags, so
  ClickHouse queries can filter by deployment surface.

- **Inference IDs.** `AgentResult.inference_ids` lists every
  `inference_id` the turn produced, exposed through both `DoneEvent`
  and `renderer.last_result`. Hand them to `t0.feedback(...)` for
  metric emission — see [Agent](agent.md) § "Observability" for the
  full feedback pipeline.

- **Debug logs.** `--debug-log` writes a timestamped file of every
  `t0_agents.*` log line at `DEBUG` level. Useful for bisecting
  elicitation-vs-streaming interactions or tracking down tool timeouts.

See also [Observability](observability.md) for the cross-cutting
surface (custom tags, ClickHouse hooks, reflection queries).

---

## Troubleshooting

- **`t0-agents: command not found`** — the `[cli]` extra isn't
  installed. Run `uv sync --extra cli` or `pip install 't0-agents[cli]'`.

- **`could not import module 'examples.10_cli.agent'`** — the factory's
  module isn't on `PYTHONPATH`. The loader auto-prepends the current
  working directory, so running from the repo root usually fixes it.
  For installed packages, make sure the module is importable via
  `python -c "import examples.10_cli.agent"` first.

- **Prompt garbled by the Live panel (custom handler)** — your handler
  isn't pausing the active renderer. Use
  `make_rich_elicitation_handler(renderer=...)` or call
  `get_active_renderer().pause()` / `.resume()` explicitly; see
  [Custom renderers](#custom-renderers) § Advanced.

- **WebSocket connections close immediately with
  `ConfigurationError`** — `t0-agents serve --streaming` uses a shared
  `Client`, which cannot demultiplex per-connection elicitation. Drop
  `--streaming` if the tools never call `ctx.elicit`; otherwise call
  [`create_app`](serving.md) directly with `client_factory` +
  `elicitation_client_factory`.

- **Browser client needs elicitation** — the CLI's Rich handler only
  works in the same terminal the REPL runs in. For browser-hosted
  serving, see [`examples/agent/07_streaming/`](../examples/agent/07_streaming/)
  for the `SessionProvider`-based WS wiring.

---

## External references

- [Typer](https://typer.tiangolo.com/) — the CLI framework. We rely on
  it for argument parsing, help rendering, and the
  `typer.BadParameter` convention the loader uses for user-facing
  errors.
- [Rich](https://rich.readthedocs.io/) — for `Live`, `Panel`,
  `Console`, `Prompt`, `Confirm`, `IntPrompt`, `FloatPrompt`, and the
  elicitation rendering. `Live` is the surface every renderer hooks
  into; see the [Live documentation](https://rich.readthedocs.io/en/stable/live.html)
  for what it does with `stdout` / `stderr` redirection.
- [FastMCP elicitation](https://gofastmcp.com/servers/elicitation) —
  the protocol our handlers satisfy. FastMCP validates elicitation
  schemas at server construction time, which gives the Rich handler
  the flat-object + primitives + `enum` / `const` guarantees it relies
  on.
- [TensorZero streaming](https://www.tensorzero.com/docs/gateway/guides/inference#streaming-inference)
  — the stream the CLI drives through `Run.events()`. Stable
  per-content-block ids are what the renderer uses to correlate
  deltas into coherent UI regions.

---

## Example

A runnable end-to-end walkthrough lives at
[`examples/agent/10_cli/`](../examples/agent/10_cli/):

- `server.py` — FastMCP server exposing `add` and `set_reminder`
  (structured-elicitation tool).
- `agent.py` — the stock `@asynccontextmanager` factory wired with
  `rich_elicitation_handler` (Easy / Intermediate tiers).
- `advanced.py` — an audit-logging elicitation handler layered on
  `make_rich_elicitation_handler` + `get_active_renderer`, exposed as
  `assistant_with_audit` (Advanced tier).
- `README.md` — install + run instructions for both factories, with
  sample session transcripts and audit-log output.

```bash
docker compose up -d
uv sync --extra cli
uv run t0-agents run examples.10_cli.agent:assistant
uv run t0-agents run examples.10_cli.advanced:assistant_with_audit
```

---

Interactive exploration via the REPL is how you build confidence in an agent's behaviour. Automating that confidence into regressions — so a refactor two weeks from now catches what you saw by hand today — is the subject of the final chapter: `t0_agents.testing` provides `MockGateway` for deterministic unit tests (no Docker, no API keys) and `EmbeddedGatewayHelper` for real-inference integration tests without a long-lived gateway. See [Testing](testing.md).
