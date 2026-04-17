# CLI — `t0-agents` (Typer + Rich)

Milestone **M6 — TUI Agent**. Install the `[cli]` extra to get a
terminal-native way to run, serve, and inspect any `t0-agents` agent.

```bash
pip install 't0-agents[cli]'        # or: uv sync --extra cli
```

The extra brings in Typer and Rich alongside the streaming + serving
dependencies (`fastapi`, `uvicorn[standard]`, `pydantic`, `sse-starlette`)
so every subcommand works out of the box.

## Three commands

| Command | What it does |
| --- | --- |
| `t0-agents run <spec>` | Interactive REPL. Live-streams deltas, tool calls, and elicitation prompts in a Rich panel. |
| `t0-agents serve <spec>` | Wraps the agent in `t0_agents.serve.create_app(..., share_client=True)` and runs uvicorn (non-streaming by default). |
| `t0-agents info <spec>` | Prints a tabulated `AgentConfig` — useful before booting the streaming REPL. |

All three take an **agent factory spec** in `module:attr` form — see
below.

## The agent factory convention

Every subcommand expects the `module:attr` reference to resolve to a
**zero-argument callable returning an async context manager** that
yields an `Agent`. In practice this means a function decorated with
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
`fastmcp.Client` must be entered as an `async with` before tool
execution, and `AsyncTensorZeroGateway.build_http` may need to be
`await`-ed. A zero-arg context-manager factory is the cleanest way for
the CLI to bracket both lifecycles around the REPL / uvicorn lifespan.

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

## `t0-agents run`

```bash
uv run t0-agents run my_package.my_agent:assistant
uv run t0-agents run my_package.my_agent:assistant --verbose
```

You'll see a cyan Rich `Panel` labelled with the agent's name. Type at
the `You:` prompt; `Ctrl-D` (or `Ctrl-C`) quits.

Each user turn drives one `agent.stream(...)` invocation. The renderer
folds the event stream into:

- A pending / success / error status line per tool call
  (`🔧 add({"a":1,"b":2})` → `✓ add({"a":1,"b":2})`).
- An accumulating assistant reply, token-by-token.
- Optional reasoning-model thoughts, visible only with `--verbose`.
- Red exclamation-prefixed lines for recoverable `ErrorEvent`s.

The REPL threads the last `DoneEvent.result.episode_id` into the next
turn automatically, so memory / TensorZero state stays coherent across
the whole session.

## `t0-agents serve`

```bash
uv run t0-agents serve my_package.my_agent:assistant \
    --host 127.0.0.1 --port 8080 --deployment my-demo
uv run t0-agents serve my_package.my_agent:assistant --streaming
```

Wraps the factory in `create_app(agent, share_client=True, streaming=...,
deployment_label=...)` and runs uvicorn. `--streaming` is **off by
default** — the subcommand serves the Phase D `POST /chat` /
`/feedback` / `/health` / `/info` surface against the factory's shared
`Client`.

Why `--no-streaming` is the safe default: the CLI uses
`share_client=True`, which means every request reuses the same
`Client(mcp)` opened inside your factory. The streaming WebSocket route
needs a fresh elicitation handler per connection (FastMCP Q3 — one
handler per Client), and the SessionFactory raises
`ConfigurationError` whenever shared mode is asked for an elicitation
handler. Opting in to `--streaming` boots the SSE route fine, but every
WS connection that fires `ctx.elicit()` will close with that error.
Use `--streaming` only when the served agent's tools never call
`ctx.elicit`.

For full streaming + browser elicitation, drop the CLI and call
`create_app` directly with a `client_factory` plus an
`elicitation_client_factory` — see
[`examples/12_streaming_http/agent_app.py`](../examples/12_streaming_http/agent_app.py),
[`serving.md`](serving.md), and [`streaming.md`](streaming.md).

## `t0-agents info`

```bash
uv run t0-agents info my_package.my_agent:assistant
```

Enters the factory just long enough to read `agent.config`, exits, and
prints a single table with one row per `AgentConfig` field. Memory
providers render as `type(value).__name__`; guardrail lists render as a
comma-separated list of `Guardrail.name`. Good for a quick "what is
this agent configured with?" sanity check.

## Custom renderers

The stock `TUIRenderer` fits in one Rich `Panel`. Swap it for a
Textual app, a custom `rich.layout.Layout`, or anything else that
consumes an `AsyncIterator[AgentEvent]`:

```python
async def consume(stream: AsyncIterator[AgentEvent]) -> AgentResult | None:
    async for event in stream:
        match event:
            case TextDeltaEvent(id=_, delta=delta):
                my_widget.append(delta)
            ...
    return my_widget.last_result
```

The CLI's built-in renderer lives at
[`src/t0_agents/cli/tui.py`](../src/t0_agents/cli/tui.py) and is a
reasonable template for a custom one. All event types are documented in
[`streaming.md`](streaming.md) § "Event reference".

## Elicitation in the TUI

The CLI ships a Rich-based `rich_elicitation_handler` that drives
FastMCP's elicitation protocol (see [`elicitation.md`](elicitation.md)).
Wire it into your `Client` in the factory:

```python
from t0_agents.cli import rich_elicitation_handler

client = Client(mcp, elicitation_handler=rich_elicitation_handler)
```

The handler:

- Renders `response_type=None` requests as `rich.prompt.Confirm` —
  maps `y` to `accept`, `n` to `decline`.
- Renders schema-driven requests as one prompt per property:
  `IntPrompt` for `integer`, `FloatPrompt` for `number`, `Confirm` for
  `boolean`, `Prompt` with `choices=` for `enum`, plain `Prompt` for
  `string`.
- Coerces enum answers back to the original Python type — if the
  server declared `enum: [1, 2, 3]` the handler returns the `int`, not
  the string the user typed.
- Falls back to `Prompt.ask(str)` with a logged warning for schema
  types outside the FastMCP Q5 subset (useful when pointing the CLI at
  a non-FastMCP MCP server).
- Pauses the TUI's `Live` panel while prompting, then resumes — a
  `Live` region redirects `stdout` / `stderr` by default and would
  otherwise scramble the prompt. If you build your own handler + custom
  renderer, call `renderer.pause()` / `renderer.resume()` around each
  prompt or use `make_rich_elicitation_handler(renderer=...)` which
  does it for you.

Schema guarantees are the same flat-object subset the Phase F
[`02-open-questions.md`](internal/agentic-feature/02-open-questions.md)
FastMCP Q5 resolution formalised: primitives + enum + const, no nested
objects, description and default preserved.

## Troubleshooting

- **`t0-agents: command not found`** — the `[cli]` extra isn't
  installed. Run `uv sync --extra cli` or
  `pip install 't0-agents[cli]'`.
- **`could not import module`** — the factory's module isn't on
  `PYTHONPATH` / isn't in the current working directory. The CLI uses
  `importlib.import_module`; make sure your project layout is
  importable (e.g. run from the repo root, or `pip install -e .`).
- **Prompt garbled by the Live panel** — your custom elicitation
  handler isn't pausing the renderer. Use
  `make_rich_elicitation_handler(renderer=...)` or call
  `renderer.pause()` / `renderer.resume()` explicitly.
- **Browser client needs elicitation** — the CLI's Rich handler only
  works in the same terminal the REPL runs in. For a browser-hosted
  serving surface, wire a `SessionProvider` that routes elicitation
  through WebSockets (Phase E) — see `examples/12_streaming_http/`.

## Example

A runnable end-to-end walkthrough lives at
[`examples/13_tui/`](../examples/13_tui/):

- `server.py` — FastMCP server exposing `add` and `set_reminder`
  (structured-elicitation tool).
- `agent.py` — the `@asynccontextmanager` factory.
- `README.md` — install + run instructions.

```bash
docker compose up -d
uv sync --extra cli
uv run t0-agents run examples.13_tui.agent:assistant
```
