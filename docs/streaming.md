# Streaming an Agent — typed events, SSE, WebSocket

Guardrails constrain what the agent may do. Streaming reveals what the agent *is* doing. Both are complementary — a streamed run honours every configured guardrail and surfaces the approval/block events as part of the same stream the caller consumes for text deltas and tool calls.

`Run.events()` returns an async iterator of **typed `AgentEvent`
variants** — one per assistant-text delta, one per tool call, one per
thought chunk, plus terminal consolidation events.  The `stream/`
submodule defines the event model; the `serve/` package exposes it over
HTTP via Server-Sent Events (one-way push) and WebSocket (bidirectional,
required when a tool calls `ctx.elicit()`).

```python
from t0_agents import Agent, AgentConfig, DoneEvent, TextDeltaEvent, ToolCallEvent
```

> **Chat-only.**  `Run.events(...)` supports chat-flavoured agents
> only.  For an agent whose `AgentConfig.function_kind == "json"`, the
> stream raises `ConfigurationError` at first `__anext__` — TensorZero
> does not emit `output.parsed` mid-stream, so a JSON stream would
> yield events whose `parsed` is permanently `None`.  See
> [Structured outputs — Why you cannot stream JSON functions](structured-outputs.md#why-you-cannot-stream-json-functions)
> for the rationale and recommended pattern (call `Agent.invoke(...)` and
> present the JSON on the consumer side once it lands).
>
> **Glow companion.**  `FluentRun.events(*selectors)` filters this
> stream through typed selectors (`Thought`, `ToolCall`, `Text`, …);
> see [`docs/glow/streaming-and-events.md`](glow/streaming-and-events.md).

## Why stream?

- **UX** — tokens render as they arrive; tool calls surface as inline
  spinners; thinking blocks (reasoning models) show the model's work
  without waiting for the final answer.
- **Elicitation over the wire** — `ctx.elicit()` prompts the user
  mid-tool, so an HTTP agent can ask "Delete task T-103?" and wait for
  a browser click before proceeding.
- **Backpressure visibility** — the stream stops progressing when the
  client is slow; no hidden buffering.
- **Drop-in** — `Agent.invoke()` is byte-identical to the non-streaming
  path.  Add streaming when you need it; keep batch callers untouched.

## Install

Streaming primitives (the event dataclasses, `TurnStreamAssembler`,
`StreamingAgentLoop`, and `Run.events()`) ship in the base
distribution — no new dependency, no extra:

```bash
pip install t0-agents
```

The HTTP endpoints (SSE at `/chat/stream`, WebSocket at `/chat/ws`)
live in `t0_agents.serve` and need the `[serve]` extra, which pulls
FastAPI, Uvicorn, Pydantic, and `sse-starlette>=2.0`:

```bash
pip install 't0-agents[serve]'
```

The TUI renderer that consumes the same events lives in
`t0_agents.cli` behind the `[cli]` extra — see
[`docs/cli.md`](cli.md).

## Quickstart

### Easiest — `stream_text()`, the text one-liner

Most consumers want exactly one thing: the assistant's text, streamed.

```python
async for token in agent.invoke(query).stream_text():
    print(token, end="", flush=True)
```

`Run.stream_text()` filters the event stream down to the text deltas —
and it is **reasoning-model safe**: OpenAI o-series / `gpt-oss` route
the final answer through thought summaries and emit *no* text deltas,
so with `fallback=True` (the default) a turn that streamed no non-blank
delta yields the final `result.response` once after the stream drains.
Pass `fallback=False` for the raw delta filter.  Fatal errors raise out
of the iterator — reach for `events(on_error="event")` when you need a
terminal event instead.

### Easy — consume a mock stream (no Docker, no gateway)

This snippet covers the shape of `Run.events()` without booting
TensorZero.  `fake_event_stream` from `t0_agents.testing.streaming_mocks`
wraps a list of `AgentEvent`s into the same `AsyncIterator[AgentEvent]`
shape the real method returns, so you can design a renderer or test a
consumer end-to-end:

```python
import asyncio

from t0_agents import DoneEvent, TextDeltaEvent, TextEndEvent
from t0_agents.core.types import AgentResult
from t0_agents.testing.streaming_mocks import fake_event_stream


async def main() -> None:
    scripted = [
        TextDeltaEvent(id="t1", delta="Tokyo "),
        TextDeltaEvent(id="t1", delta="is sunny, "),
        TextDeltaEvent(id="t1", delta="22°C."),
        TextEndEvent(id="t1"),
        DoneEvent(result=AgentResult(
            response="Tokyo is sunny, 22°C.",
            episode_id="demo",
            inference_ids=["i1"],
            turns=1,
        )),
    ]

    async for event in fake_event_stream(scripted):
        match event:
            case TextDeltaEvent(delta=s):
                print(s, end="", flush=True)
            case DoneEvent(result=r):
                print(f"\n[done in {r.turns} turn]")


asyncio.run(main())
```

Runs without Docker, without an API key, and without an MCP server.
Use it as the template for writing unit tests of anything that consumes
`AsyncIterator[AgentEvent]` — TUI renderers, SSE routes, reducers,
background workers.

### Real — `Run.events()` against a running gateway

Once you have `docker compose up` running and a `GROQ_API_KEY` in your
environment, the full loop looks like this:

```python
import sys

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, DoneEvent, TextDeltaEvent, ToolCallEvent
from my_mcp_server import mcp


async def chat(query: str) -> None:
    async with (
        await AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
        ) as t0,
        Client(mcp) as tools,
    ):
        agent = Agent(
            AgentConfig(name="assistant", function_name="agent"),
            t0=t0,
            tool_provider=tools,
        )

        async for event in agent.invoke(query).events():
            match event:
                case TextDeltaEvent(delta=text):
                    sys.stdout.write(text)
                    sys.stdout.flush()
                case ToolCallEvent(name=name):
                    print(f"\n[calling {name}]")
                case DoneEvent(result=r):
                    print(f"\n[done in {r.turns} turns, episode {r.episode_id}]")
```

### What just happened

1. `Run.events()` starts a turn: it lists MCP tools, builds a T0
   request, and calls `t0.inference(stream=True)`.
2. TensorZero delivers chunks — one per content-block delta.
   `TurnStreamAssembler` folds them into `AgentEvent`s *as they arrive*
   (`TextDeltaEvent`, `ToolCallDeltaEvent`, `ThoughtDeltaEvent`).
3. When the chunk iterator ends, the assembler flushes consolidation
   events — one `TextEndEvent` per text block, one `ToolCallEvent`
   (with parsed args) per tool call, one `ThoughtEvent` per thought —
   then the runner emits `TurnEndEvent(turn=N)`.
4. If the assistant emitted tool calls, the runner executes them via
   the same `execute_tool_calls` path `Agent.invoke()` uses, yields one
   `ToolResultEvent` per call, and opens the next turn.
5. The first tool-call-free turn is terminal: the runner builds an
   `AgentResult` and yields `DoneEvent(result=...)`.

`Agent.invoke()`'s non-streaming path goes through the same turn shape —
both loops drive the shared turn engine (`core/turn_engine.py`), so the
streaming loop is an additive layer, not a replacement.

## Concepts

### Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  Run.events(message, *, episode_id=None, tags=None)             │
│      │                                                            │
│      │  memory.load()  +  guardrail before_run hooks              │
│      ▼                                                            │
│  StreamingAgentLoop.stream(t0, tool_provider, ...)                │
│      │                                                            │
│      │  for turn in range(max_turns):                             │
│      │      t0.inference(stream=True) ── AsyncIterator[ChatChunk] │
│      │      TurnStreamAssembler.assemble(...)                     │
│      │      → yields AgentEvent as each chunk arrives             │
│      │      execute_tool_calls[_parallel](...)                    │
│      ▼                                                            │
│  DoneEvent(result=AgentResult(response, episode_id, ...))         │
└───────────────────────────────────────────────────────────────────┘
```

`stream/` has zero new third-party dependencies on top of the rest of
the library.  It composes `core/` (for `AgentResult` and the shared
turn engine, `core/turn_engine.py` — validation, inference kwargs,
corrective retry, the post-inference sequence, final-text mining, the
max-turns guard) and `mcp/` (for `convert_tools` and
`execute_tool_calls`) — and nothing else.

### Event taxonomy

Every `AgentEvent` is a `@dataclass(slots=True)` carrying a
`type: Literal[...]` discriminator so consumers can `match` without
`isinstance`.  The shapes fall into three groups:

1. **Delta events** fire *as chunks arrive* —
   `TextDeltaEvent`, `ToolCallDeltaEvent`, `ThoughtDeltaEvent`, and
   `ToolProgressEvent` (server-side `ctx.report_progress(...)`
   notifications relayed while a tool executes).  Consumers that
   render a live UI typically only listen to these.
2. **Consolidation events** fire once per block *after* its final
   delta — `TextEndEvent`, `ToolCallEvent` (with parsed `arguments`
   dict), `ThoughtEvent`.  Consumers that want the assembled state
   (e.g. a reducer that writes to a database) listen to these.
3. **Control events** frame the run — `ToolResultEvent` (after each
   tool call finishes), `TurnEndEvent(turn=N)` (one per turn),
   `DoneEvent(result=AgentResult(...))` (always last on success),
   `ErrorEvent` (recoverable anomalies), and `RunErrorEvent` (the
   terminal failure under `on_error="event"`; see
   [Error handling](#error-handling)).

### Correlation key

Every event that references a block carries the **stable `id`** the
TensorZero chunk emitted — `TextChunk.id`, `ToolCallChunk.id`,
`ThoughtChunk.id`.  A `TextDeltaEvent` and its matching `TextEndEvent`
share one `id`; two `ToolCallDeltaEvent`s for the same call share one
`id` and are accumulated into a single `ToolCallEvent` with that
`id`.  Consumers rely on that stability to implement reducers, to pair
start/end markers, and to update the right DOM node when rendering.

### Lifecycle per turn

```
list_tools()  →  t0.inference(stream=True)
                          │
                          ▼
                 AsyncIterator[ChatChunk]
                          │
           TurnStreamAssembler.assemble(chunks)
                          │
                          ▼
              AsyncIterator[AgentEvent]
                          │
  delta events as chunks arrive ── consolidation events at end
                          │
                          ▼
                     TurnEndEvent
                          │
           any ToolCallEvents? ── yes ──→ execute_tool_calls → ToolResultEvents → next turn
                          │ no
                          ▼
                        DoneEvent
```

### Design decisions

- **Consolidation at turn end, not per-chunk.**  TensorZero's SDK
  currently emits one block delta per chunk, so a per-chunk diff
  strategy would fire false-positive end-of-block markers.  The
  assembler tracks insertion order and flushes consolidation events
  once — right before `TurnEndEvent`.
- **Per-session MCP `Client` when elicitation is on the table.**
  FastMCP's `elicitation_handler` is a *single instance* on each
  `Client`; sharing one Client across concurrent callers would route
  every elicitation to the same handler.  The `serve/` WebSocket path
  builds a fresh `Client` per connection; the SSE path is one-way so
  it can share.  The `SessionProvider` section below covers the
  advanced wiring.
- **`Agent.invoke()` stays byte-identical.**  The streaming loop is an
  additive submodule; `run_agent` and the non-streaming `Agent.invoke`
  path are unchanged.  Tests that exercise the non-streaming contract
  remain valid verbatim.
- **TensorZero-first.**  The framework never invents a parallel
  transport — it uses `t0.inference(stream=True)` directly.  Every
  `AgentEvent` is a small semantic upgrade over the underlying chunk
  (parsed tool args, correlation-preserving end markers); nothing is
  duplicated.

## API reference

### `Run.events`

```python
async def events(
    self,
    *,
    replay: bool = False,
    on_error: Literal["raise", "event"] = "raise",
) -> AsyncIterator[AgentEvent]
```

The streaming view of a `Run`.  Shares memory hydration, guardrail
scaffolding, and tool execution with the non-streaming path; on
success the terminal event is always a `DoneEvent` wrapping an
`AgentResult` (including `guardrail_log` when guardrails are
configured).

| Argument | Purpose |
|---|---|
| `replay` | `True` re-yields from the earliest still-buffered event; `False` (default) subscribes at the live cursor. |
| `on_error` | `"raise"` (default) re-raises drive-time failures out of the iterator.  `"event"` converts the failure into a single terminal `RunErrorEvent`, so **every stream ends with exactly one `DoneEvent \| RunErrorEvent`**.  Cancellation is never converted — `RunCancelledError` raises in both modes. |

Raises (with the default `on_error="raise"`):

- `MaxTurnsExceededError` — `config.max_turns` is exhausted without a
  tool-call-free turn.
- `GuardrailBlockedError` — any configured guardrail hard-aborts the
  run.
- `TensorZeroError` — a non-recoverable gateway error.  (Recoverable
  errors surface inline as `ErrorEvent`; see
  [Error handling](#error-handling).)
- `RunCancelledError` — the run was cancelled (both modes).

### `Run.stream_text`

```python
async def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]
```

The text one-liner (see [Quickstart](#quickstart)): yields each
`TextDeltaEvent.delta` as it arrives; with `fallback=True` a turn that
streamed no non-blank delta yields `result.response` once after the
drain — covering reasoning models (`response_source == "thought"`) and
any other no-text turn that still produced an answer.  The library
never invents copy for a genuinely empty turn.

### Event types

| Event | When | Fields |
|---|---|---|
| `TextDeltaEvent` | Each assistant-text chunk | `id`, `delta` |
| `TextEndEvent` | Once per text block after its final delta | `id` |
| `ToolCallDeltaEvent` | Each partial tool-call delta (name/args) | `id`, `raw_name_delta`, `raw_arguments_delta` |
| `ToolCallEvent` | Once per tool call, args parsed | `id`, `name`, `arguments: dict[str, Any]` |
| `ToolResultEvent` | Once per tool, after the executor returns | `id`, `name`, `result: str`, `is_error: bool`, `structured: dict \| None` (raw structured payload, additive) |
| `ThoughtDeltaEvent` | Each reasoning-model thought chunk | `id`, `delta` |
| `ThoughtEvent` | Once per thought block after its final delta | `id`, `text` |
| `ToolProgressEvent` | Each server-side `ctx.report_progress(...)` while a tool runs | `id` (matches the `ToolCallEvent`), `progress: float`, `total: float \| None`, `message: str \| None` |
| `TurnEndEvent` | Once per turn, after all blocks are assembled | `turn: int` |
| `DoneEvent` | Always last on success | `result: AgentResult` |
| `ErrorEvent` | Recoverable anomaly (bad JSON, unknown chunk, retry) | `message: str`, `detail: dict[str, Any]` |
| `RunErrorEvent` | Terminal failure, only under `on_error="event"` | `error_type: str`, `message: str`, `detail: dict[str, Any]`, `exception` (in-process only) |

`AgentEvent` is the discriminated union of all twelve variants.  Every
dataclass is `slots=True`; JSON serialisation at the HTTP boundary is
`dataclasses.asdict(event)`, which round-trips cleanly to the browser.

### Codec — `event_to_dict` / `event_from_dict`

For anything that ships events across a process boundary — a bus, a
replay log, SSE relays — `t0_agents.stream.codec` provides the
versioned dict codec:

```python
from t0_agents import event_from_dict, event_to_dict

payload = event_to_dict(event)        # {"v": 1, "type": "...", ...fields}
event = event_from_dict(payload)      # round-trips; ValueError on unknown type/version
```

Every field of every variant round-trips except
`RunErrorEvent.exception` (in-process only; decodes as `None`) and
non-primitive `AgentResult.value` payloads, which degrade to plain JSON
data (a pydantic model encodes via `model_dump()`).  Unknown ``type``
discriminators and codec-version mismatches raise `ValueError` — an
outdated payload fails loudly instead of degrading silently.

### Status synthesis — `synthesize_statuses`

`t0_agents.stream.status` ports the production status-hint policy as
pure functions: `synthesize_statuses(events, known_servers=...)`
re-emits every event and interleaves opaque `Status` hints —
`"Calling {tool}…"` per tool call (identity decomposed against the
known server prefixes, longest first) and `"Thinking…"` per reasoning
block (deduplicated within a block, suppressed after a tool call until
a terminal event re-arms it).  Compose it at the endpoint/consumer
boundary — never inside a worker or on a bus.  Glow exposes it as
`FluentRun.statuses()`.

### `StreamingAgentLoop` — advanced

The loop that `Run.events()` delegates to.  Exposed so frameworks
can drive the streaming turn loop directly — for example, in a renderer
that pre-dates an `Agent` wrapper:

```python
from t0_agents import StreamingAgentLoop

loop = StreamingAgentLoop()
async for event in loop.stream(
    t0=t0, tool_provider=client,
    user_message="hi", function_name="agent",
    max_turns=20, tool_timeout=30.0, parallel_tools=False,
):
    ...
```

The full signature accepts every knob `run_agent` does —
`episode_id`, `dryrun`, `tool_choice`, `tags`, `namespace`,
`system_arguments`, plus an `on_inference` hook that fires once per
turn with a `ChatInferenceResponse`-shaped object (used internally by
guardrails' cost tracker).

### `TurnStreamAssembler` — advanced

The per-turn fold from `AsyncIterator[ChatChunk]` to
`AsyncIterator[AgentEvent]`.  Useful on its own for tests that
hand-craft chunks:

```python
from t0_agents import TurnStreamAssembler

assembler = TurnStreamAssembler()
async for event in assembler.assemble(chunks):
    ...
```

A fresh assembler is built per turn inside `StreamingAgentLoop`; reuse
across turns would mix block state.

## Consumption tiers

### Easy — minimal consumer

The snippet in [Quickstart](#quickstart) is the canonical pattern:
`match` on the event type, render the field(s) you care about, bail
out on `DoneEvent`.  For a terminal UI `print(text, end="",
flush=True)` is enough; for a web UI, forward each `TextDeltaEvent` to
the client as soon as it arrives.

### Intermediate — REPL with Rich

[`examples/agent/10_cli/`](../examples/agent/10_cli/) wraps `Run.events()` in a
`rich.live.Live` display.  `t0_agents.cli.TUIRenderer` folds the event
iterator into a live panel; `rich_elicitation_handler` provides
browser-free `Prompt.ask()` modals when a tool calls `ctx.elicit()`.

```bash
uv run t0-agents run examples.10_cli.agent:assistant
```

See [`docs/cli.md`](cli.md) for the factory convention and the three
CLI subcommands (`run`, `serve`, `info`).

### Advanced — HTTP with SSE, WebSocket, and elicitation

[`examples/agent/07_streaming/`](../examples/agent/07_streaming/) ships a
full browser demo — a FastAPI app, a vanilla-JS client, a FastMCP
server with elicitation-gated tools, and `InMemoryMemory` for
multi-turn conversations.  The recipe below wires the four HTTP
endpoints.

```python
from t0_agents.serve import create_app

def _client_factory():
    return Client(mcp)                                 # fresh per session

def _elicitation_client_factory(handler):
    return Client(mcp, elicitation_handler=handler)    # WS streaming

app = create_app(
    agent,
    client_factory=_client_factory,
    elicitation_client_factory=_elicitation_client_factory,
    streaming=True,
)
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Non-streaming single-shot agent call ([docs/serving.md](serving.md)) |
| `POST` | `/feedback` | Emit TensorZero feedback ([docs/serving.md](serving.md)) |
| `GET` | `/chat/stream` | SSE — one-way `AgentEvent` push |
| `WS` | `/chat/ws` | Bidirectional — required for elicitation |

Pick:

- **SSE (`/chat/stream`)** when the client only reads (dashboards,
  read-only chat views, CLI `curl --no-buffer`).  No elicitation
  channel — tools that call `ctx.elicit()` fall through to the
  forwarder's default decline.
- **WebSocket (`/chat/ws`)** when the client can send frames back —
  required whenever a tool may call `ctx.elicit()`.

#### SSE frame shape

```
event: text_delta
data: {"id":"t1","delta":"hola ","type":"text_delta"}

event: done
data: {"type":"done","result":{"response":"...","episode_id":"..."}}
```

SSE query parameters:

| Param | Type | Required | Purpose |
|---|---|---|---|
| `message` | `str` | yes | The user's query. |
| `episode_id` | `str` | no | Continue an existing TensorZero episode. |
| `tags` | JSON-encoded `dict[str, str]` | no | Shallow-merge on top of `AgentConfig.tags` for this request only.  Invalid JSON or non-object values surface as the first SSE frame — an `ErrorEvent` with `detail.kind == "invalid_tags"` — and the stream terminates. |

```bash
curl --no-buffer \
  'http://localhost:8080/chat/stream?message=hi&tags=%7B%22env%22%3A%22staging%22%7D'
```

#### WebSocket frame shapes

Client → server:

```json
{"type":"user_message","message":"hi","episode_id":null,"tags":{}}
{"type":"elicitation_response","elicitation_id":"uuid","payload":{"action":"accept","content":{}}}
```

`tags` in a `user_message` frame shallow-merges on top of
`AgentConfig.tags` for *that frame only* — the only way to vary tags
per frame when one `Agent` is reused across a WS connection's
lifetime.

Server → client: every `AgentEvent` serialised as JSON (its `type`
field is the discriminator), plus:

```json
{"type":"elicitation_request","elicitation_id":"uuid","message":"Delete task T-103?","schema":{...},"response_type":null}
{"type":"ws_error","message":"...","detail":{...}}
```

#### Elicitation round-trip over WebSocket

1. Client sends `user_message`.
2. Server streams `text_delta` / `tool_call` events.
3. A tool calls `ctx.elicit("...?", response_type=None)`.
4. Server pushes `elicitation_request` with a fresh `elicitation_id`.
5. Client renders a modal, user clicks Accept / Decline.
6. Client sends `elicitation_response` with the same `elicitation_id`.
7. Server resolves the forwarder's Future, the tool returns, streaming
   resumes until `done`.

If the WS disconnects while a tool is waiting, the forwarder
auto-declines the pending request — no hanging tool.

#### Browser client recipe

A 180-line vanilla-JS client that covers every event is in
[`examples/agent/07_streaming/static/app.js`](../examples/agent/07_streaming/static/app.js).
The core pattern:

```js
const ws = new WebSocket("ws://localhost:8080/chat/ws");
ws.onmessage = (m) => {
  const frame = JSON.parse(m.data);
  switch (frame.type) {
    case "text_delta": appendToAssistant(frame.delta); break;
    case "tool_call": showSpinner(frame.id, frame.name); break;
    case "tool_result": replaceSpinner(frame.id, frame.result); break;
    case "elicitation_request": openModal(frame); break;
    case "done": recordEpisode(frame.result.episode_id); break;
  }
};
ws.send(JSON.stringify({type: "user_message", message: "hi"}));
```

#### `SessionProvider` for auth or multi-tenant wiring

For authentication, per-tenant routing, telemetry spans, or bespoke
elicitation wiring, implement `SessionProvider` directly instead of
passing the convenience factories.  It's the framework's single
extension point for per-session `Client` construction:

```python
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from fastmcp import Client
from fastmcp.client.elicitation import ElicitationHandler
from t0_agents.serve import SessionProvider, create_app


class AuthedSessionProvider:
    """Decorates the Client with per-session auth + telemetry hooks."""

    def __init__(self, mcp, tracer):
        self._mcp = mcp
        self._tracer = tracer

    def build_client(
        self,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AbstractAsyncContextManager[Client]:
        @asynccontextmanager
        async def _mgr():
            token = self._tracer.start_span("mcp.session")
            try:
                client = Client(self._mcp, elicitation_handler=elicitation_handler)
                async with client:
                    yield client
            finally:
                self._tracer.end_span(token)

        return _mgr()


app = create_app(agent, session_provider=AuthedSessionProvider(mcp, tracer),
                 streaming=True)
```

The Protocol is small on purpose: one method, the same shape for the
non-streaming `/chat` path and the streaming WS path.  Any object that
implements `build_client(elicitation_handler=...)` satisfies it
(`@runtime_checkable` — `isinstance(x, SessionProvider)` works at
construction time).

For the simple path, pass `client_factory=...` (and optionally
`elicitation_client_factory=...`) to `create_app`; the framework wraps
them in a `CallableSessionProvider` adapter.

## Error handling

The streaming surface encodes three tiers of error:

| Kind | Surface | Examples |
|---|---|---|
| Non-fatal | `ErrorEvent` yielded inline; stream continues | invalid tool-call JSON (emits `ErrorEvent` + a `ToolCallEvent` with `arguments={}` so the `id` is still observable; the call is **not** dispatched — the model receives the same `ERROR: invalid JSON arguments...` result as on the non-streaming path). A complete object followed only by closing noise (`}` / `]` / whitespace) is **salvaged** instead: no `ErrorEvent`, the `ToolCallEvent` carries the recovered dict, the tool runs with it, and the wire's `arguments_json` is its canonical form — see [error-handling.md](error-handling.md); unknown TensorZero chunk type; recoverable `TensorZeroError` corrective retry |
| Fatal, default | Raised out of the async iterator (`on_error="raise"`) | `MaxTurnsExceededError`, `GuardrailBlockedError`, non-recoverable `TensorZeroError`, or any `T0AgentsFatalError` subclass |
| Fatal, terminal-event | One terminal `RunErrorEvent` ends the stream (`on_error="event"`) | the same failures, converted: `error_type` carries the exception class name, `detail` carries structured extras (`{"max_turns": 20}`), `exception` carries the original for in-process consumers.  **Invariant: exactly one `DoneEvent \| RunErrorEvent` per stream.**  Cancellation always raises. |

Glow flips the default to `on_error="event"` (a UI stream should never
raise mid-iteration); core keeps `"raise"` for backward compatibility.

The recoverable-retry contract is shared with `run_agent` — when the
gateway returns a 4xx status whose body matches one of the
`RECOVERABLE_INFERENCE_PATTERNS` (owned by `core.turn_engine`,
re-exported from `mcp.loop`), the
streaming loop appends a corrective-message turn and keeps going, and
the consumer sees an `ErrorEvent` announcing the retry followed by
the next turn's events.

On the WebSocket path, a client disconnect cancels the in-flight
stream task and calls `forwarder.cancel_all(action="decline")`,
draining any pending elicitations so waiting tools return cleanly
instead of hanging.  Stream cancellation *mid-response* (the client
aborts a running turn without disconnecting) is deferred — clients
should close and reconnect for the next turn.  One stream per WS
connection at a time.

See [`docs/error-handling.md`](error-handling.md) for the full
library-wide exception hierarchy and the
`T0AgentsError` / `T0AgentsFatalError` split.

## Production considerations

- **Transport**.  Use SSE for read-only streams (dashboards, CLI
  smoke-tests) and WebSocket when any tool may call `ctx.elicit()`.
  The one-way SSE path can share a single `Client` across the
  process, but the default is per-session for symmetry with WS.
- **Per-session `Client`**.  FastMCP's `elicitation_handler` binds
  once per Client, so a shared Client cannot route replies to the
  right user.  The WS route always builds a fresh `Client` per
  connection via `elicitation_client_factory` (or via
  `SessionProvider.build_client`).
- **Backpressure**.  `Run.events()` does not buffer.  Each event is
  yielded synchronously as the TensorZero SDK delivers a chunk.  On
  the WS route, events are sent one-per-`ws.send_json()` call — if
  the client is slow, the WS transport back-pressures the `send_json`
  coroutine, which back-pressures the `async for` loop, which
  back-pressures the agent's inference stream.
- **Sticky sessions**.  Because one `Agent` + `Client` pair serves
  one WS connection, a front-end load balancer must stick a given
  client to the same backend pod for the duration of the socket.
  Round-robin routing will break elicitation mid-tool.
- **Auth and multi-tenant routing**.  The `SessionProvider` Protocol
  is the extension point.  Implement it to inject JWT headers into
  the MCP client, pick a per-tenant MCP server, attach telemetry
  spans, or all three.

See [`docs/production.md`](production.md) for the library-wide
production checklist (gateway HA, ClickHouse sizing, secret
management, etc.).

## Observability

Every streaming run produces the same telemetry as its non-streaming
analogue:

- **Episode continuity** — `DoneEvent.result.episode_id` is the
  stable identifier the caller threads back through the next
  `Run.events(episode_id=...)` call.  TensorZero attaches episode
  metadata to every inference for ClickHouse-backed analytics.
- **Per-turn inference IDs** — `DoneEvent.result.inference_ids` is
  the ordered list of T0 inference IDs, one per turn (including
  recoverable-retry placeholders).  Attach feedback to a specific
  turn via `POST /feedback` with the corresponding ID.
- **Guardrail log** — if the agent is configured with guardrails,
  `DoneEvent.result.guardrail_log` carries one `GuardrailLogEntry`
  per hook activation.  The log survives every abort path, including
  `GuardrailBlockedError` — see
  [`docs/guardrails.md`](guardrails.md).
- **Deployment tagging** — pass `deployment_label="..."` to
  `create_app`; it adds `t0_agents.deployment=<label>` to every
  inference call so ClickHouse queries can slice by deployment.

See [`docs/observability.md`](observability.md) for the canonical
dashboards, ClickHouse queries, and feedback-emission patterns.

## External references

- **TensorZero streaming inference** —
  [tensorzero.com/docs](https://www.tensorzero.com/docs/).  The
  `t0.inference(stream=True)` call and the `ChatChunk` / `Usage`
  types this submodule folds into `AgentEvent`s.
- **FastMCP elicitation** — [gofastmcp.com](https://gofastmcp.com/).
  The `ctx.elicit()` primitive that the WebSocket path bridges to a
  browser modal; the `elicitation_handler` single-instance constraint
  that drives the per-session `Client` mandate.
- **Server-Sent Events** —
  [MDN: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).
  The one-way `text/event-stream` protocol used by `GET /chat/stream`.
- **WebSocket protocol** —
  [RFC 6455](https://www.rfc-editor.org/rfc/rfc6455).  The
  bidirectional framing that `WS /chat/ws` builds on.

---

Streaming makes a single agent's work visible. When a single agent is not enough — one researches, one writes, one reviews — the composition layer lets an orchestrator delegate to others as ordinary MCP tools, each with its own memory, guardrails, and (if you need it) its own stream. That is the subject of the next chapter: [Multi-Agent Composition](multi-agent.md).

## See also

- [`examples/agent/07_streaming/`](../examples/agent/07_streaming/) —
  runnable browser demo covering every event type and the
  elicitation round-trip.
- [`examples/agent/10_cli/`](../examples/agent/10_cli/) — streaming REPL with
  Rich and `rich_elicitation_handler`.
- [`docs/elicitation.md`](elicitation.md) — the HITL contract the WS
  route bridges.
- [`docs/serving.md`](serving.md) — non-streaming `POST /chat` and
  `POST /feedback` endpoints.
- [`docs/cli.md`](cli.md) — `t0-agents run`, the TUI consumer of the
  same event stream.
