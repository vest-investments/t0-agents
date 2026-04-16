# Streaming an Agent — SSE and WebSocket

`Agent.stream()` yields **typed events** as the LLM produces tokens and
tools run.  The `t0_agents.serve` layer exposes two streaming endpoints
that bridge those events to the browser: SSE (one-way push) and
WebSocket (bidirectional, required when a tool calls `ctx.elicit()`).

Ships in the same `[serve]` extra:

```bash
pip install 't0-agents[serve]'
```

## Why stream?

- **UX** — tokens render as they arrive; tool calls surface as inline
  spinners; thinking blocks (reasoning models) show the model's work
  without waiting for the final answer.
- **Elicitation over the wire** — `ctx.elicit()` prompts the user
  mid-tool, so an HTTP agent can ask "Delete task T-103?" and wait for a
  browser click before proceeding.
- **Back-pressure visibility** — the stream stops progressing when the
  client is slow; no hidden buffering.

## `Agent.stream()` cookbook

```python
from t0_agents import Agent, AgentConfig, DoneEvent, TextDeltaEvent, ToolCallEvent
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from my_mcp_server import mcp

async with (
    await AsyncTensorZeroGateway.build_http(...) as t0,
    Client(mcp) as client,
):
    agent = Agent(AgentConfig(name="assistant", function_name="agent"),
                  t0=t0, tool_provider=client)

    async for event in agent.stream("What's the weather in Tokyo?"):
        match event:
            case TextDeltaEvent(delta=text):
                sys.stdout.write(text); sys.stdout.flush()
            case ToolCallEvent(name=name):
                print(f"\n[calling {name}]")
            case DoneEvent(result=r):
                print(f"\n[done in {r.turns} turns, episode {r.episode_id}]")
```

Every non-streaming caller keeps working — `Agent.run()` is untouched.

## Event types

| Event | When | Fields |
|---|---|---|
| `TextDeltaEvent` | Each assistant-text chunk | `id`, `delta`, `type` |
| `TextEndEvent` | End of a text block | `id`, `type` |
| `ToolCallDeltaEvent` | Each partial tool-call delta (name+args) | `id`, `raw_name_delta`, `raw_arguments_delta` |
| `ToolCallEvent` | Assembled tool call ready to dispatch | `id`, `name`, `arguments` |
| `ToolResultEvent` | Tool finished executing | `id`, `name`, `result`, `is_error` |
| `ThoughtDeltaEvent` | Reasoning-model thought chunk | `id`, `delta` |
| `ThoughtEvent` | End of a thought block | `id`, `text` |
| `TurnEndEvent` | End of one agent turn | `turn` |
| `DoneEvent` | Run completed (always last) | `result: AgentResult` |
| `ErrorEvent` | Recoverable error (retry, bad JSON, unknown chunk) | `message`, `detail` |

The `id` fields are **stable correlation keys** — every delta for a
single content block shares the same id, so consumers can accumulate or
pair them (see contract §3.3).

## Server-side: SSE or WebSocket?

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

This wires **four** new route behaviours on the same FastAPI app:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Non-streaming (Phase D) |
| `POST` | `/feedback` | Emit TensorZero feedback (Phase D) |
| `GET` | `/chat/stream` | SSE — one-way AgentEvent push |
| `WS` | `/chat/ws` | Bidirectional — required for elicitation |

Pick:

- **SSE (`/chat/stream`)** when the client only needs to read (dashboards,
  read-only chat views, CLI `curl --no-buffer`). No elicitation channel.
- **WebSocket (`/chat/ws`)** when the client can send as well as receive
  — required whenever a tool may call `ctx.elicit()`.

### SSE frame shape

```
event: text_delta
data: {"id":"t1","delta":"hola ","type":"text_delta"}

event: done
data: {"type":"done","result":{"response":"...","episode_id":"..."}}
```

#### SSE query parameters

| Param | Type | Required | Purpose |
|---|---|---|---|
| `message` | `str` | yes | The user's query. |
| `episode_id` | `str` | no | Continue an existing TensorZero episode. |
| `tags` | JSON-encoded `dict[str, str]` | no | Shallow-merge on top of `AgentConfig.tags` for this request only (matches the `tags` field of `POST /chat`'s body). Invalid JSON or non-object values surface as the first SSE frame — an `ErrorEvent` with `detail.kind == "invalid_tags"` — and the stream terminates. |

Example:

```bash
curl --no-buffer \
  'http://localhost:8080/chat/stream?message=hi&tags=%7B%22env%22%3A%22staging%22%7D'
```

### WebSocket frame shapes

Client → server:

```json
{"type":"user_message","message":"hi","episode_id":null,"tags":{}}
{"type":"elicitation_response","elicitation_id":"uuid","payload":{"action":"accept","content":{}}}
```

`tags` in a `user_message` frame shallow-merges onto `AgentConfig.tags`
for *that frame only* — the same behaviour as `tags` in the `POST /chat`
body and `?tags=<json>` on `GET /chat/stream`.  Override values win on
key collisions; the configured deployment label is preserved across
every frame.  This is the **only** way to vary tags per frame, because
one `Agent` instance is shared across the lifetime of a WS connection.

Server → client: every `AgentEvent` serialised as JSON (its `type` field
is the discriminator), plus:

```json
{"type":"elicitation_request","elicitation_id":"uuid","message":"Delete task T-103?","schema":{...},"response_type":null}
{"type":"ws_error","message":"...","detail":{...}}
```

### Elicitation round-trip over WS

1. Client sends `user_message`.
2. Server streams `text_delta` / `tool_call` events.
3. A tool calls `ctx.elicit("...?", response_type=None)`.
4. Server pushes `elicitation_request` with a fresh `elicitation_id`.
5. Client renders a modal, user clicks Accept.
6. Client sends `elicitation_response` with the same `elicitation_id`.
7. Server resolves the forwarder's Future, the tool returns, streaming
   resumes until `done`.

If the WS disconnects while a tool is waiting on elicitation, the
forwarder auto-declines the pending request — no hanging tool.

## Browser client recipe

A tiny vanilla-JS client that covers every event is in
[`examples/12_streaming_http/static/app.js`](../examples/12_streaming_http/static/app.js)
(~180 lines).

Key patterns:

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

## Advanced: `SessionProvider`

For authentication, multi-tenant routing, telemetry, or custom
elicitation wiring, implement the `SessionProvider` Protocol yourself.
It's the framework's single extension point for per-session `Client`
construction:

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
    ) -> AbstractAsyncContextManager:
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

When you only need the simple path, pass `client_factory=...` (and
optionally `elicitation_client_factory=...`) to `create_app`; the
framework wraps them in a `CallableSessionProvider` adapter for you.

## Backpressure

`Agent.stream()` does not buffer.  Each event is yielded synchronously
as the TensorZero SDK delivers a chunk.  On the WS route, events are
sent one-per-`ws.send_json()` call — if the client is slow, the WS
transport back-pressures the `send_json` coroutine, which back-pressures
the `async for` loop, which back-pressures the agent's inference stream.

If the client disconnects mid-stream, the route catches
`WebSocketDisconnect`, cancels the active stream task, and
`forwarder.cancel_all()` drains pending elicitations.

Stream cancellation mid-response (client aborting a running turn) is
deferred — clients should close and reconnect for the next turn.  One
stream per WS connection at a time.

## Regression — `Agent.run()` unchanged

`Agent.run()` remains byte-identical to Phase D; streaming is additive.
The non-streaming `/chat` route continues to decline elicitation
attempts (the HTTP contract has no return channel).

## See also

- [`examples/12_streaming_http/`](../examples/12_streaming_http/) —
  runnable demo with a browser client covering every event type.
- [`docs/elicitation.md`](elicitation.md) — the HITL contract the WS
  route bridges.
- [`docs/serving.md`](serving.md) — non-streaming Phase D surface.
