# Streaming and event selectors

> Part of the glow ladder.  Backs the chat-only streaming path on top
> of the [`Run.events()`](../streaming.md) surface.  For
> JSON-shaped agents, streaming is deferred — see
> [`limitations.md` — No streaming JSON](limitations.md#no-streaming-json).

## `stream_text()` — the text one-liner

Most UIs want exactly one thing: the assistant's text, streamed.

```python
async for token in tutor("Explain present perfect").stream_text():
    print(token, end="", flush=True)
```

`FluentRun.stream_text()` yields every text delta as it arrives, and it
is **reasoning-model safe**: OpenAI o-series / `gpt-oss` route the final
answer through thought summaries and stream *no* text deltas at all —
when a turn streamed no non-blank delta, the final `result.response` is
yielded once after the stream drains (`fallback=False` restores the raw
delta filter).  Fatal errors raise out of the iterator; use
`events()` below when you need a terminal event instead.

## `events(*selectors)` — the typed stream

`FluentRun.events(*selectors)` is the typed event stream from
[`Run.events()`](../streaming.md), filtered.  The module-level
selectors cover the event union — one selector per
[`*Event`](../../src/t0_agents/stream/events.py) class:

| Selector       | Underlying event                     | Yields                             |
|----------------|--------------------------------------|------------------------------------|
| `Thought`      | `ThoughtEvent`                       | extended-reasoning thinking blocks |
| `ToolCall`     | `ToolCallEvent`                      | LLM tool invocations               |
| `ToolProgress` | `ToolProgressEvent`                  | server-side tool progress (id-correlated to the call) |
| `Text`         | `TextDeltaEvent` (carries the text)  | token-by-token assistant text      |
| `TextEnd`      | `TextEndEvent` (completion marker)   | one event per text block finished  |
| `ToolResult`   | `ToolResultEvent`                    | tool execution results             |
| `TurnEnd`      | `TurnEndEvent`                       | one event per inference turn       |
| `Done`         | `DoneEvent`                          | final loop completion              |
| `Error`        | `ErrorEvent`                         | recoverable inline anomalies       |
| `RunError`     | `RunErrorEvent`                      | the terminal failure event         |

> **v2 breaking change** — `Text` used to select `TextEndEvent`, whose
> only payload is a block `id`.  It now selects `TextDeltaEvent`, the
> events that actually carry text.  Migration: consumers that wanted
> the completion marker switch to `TextEnd`; consumers that built a
> manual `Selector(TextDeltaEvent)` can now just use `Text`.

Compose them with `.where(pred)` (AND-chains predicates) and `.map(fn)`
(transforms the yielded value).

```python
from t0_agents.glow import Thought, ToolCall, Text

async for event in tutor("Explain present perfect").events(
    Thought,
    ToolCall.map(lambda tc: f"> calling {tc.name}"),
    Text.map(lambda d: d.delta),
):
    ...
```

Passing no selectors yields every `AgentEvent`.  Glow only supports
streaming for chat-flavoured output modes; the
[chat-only constraint](../streaming.md) carries through — calling
`.events()` on a `JsonOutput` / `StructuredOutput` agent raises
`ConfigurationError` at the first `__anext__`.

## Errors: glow streams never raise mid-iteration

`FluentRun.events()` defaults to `on_error="event"` — a fatal failure
(loop guard, guardrail block, gateway error) ends the stream with a
single terminal
[`RunErrorEvent`](../../src/t0_agents/stream/events.py) instead of
raising out of your render loop.  **Every glow stream ends with exactly
one `DoneEvent | RunErrorEvent`.**

```python
from t0_agents.glow import RunError

async for event in tutor(question).events():
    match event:
        case RunErrorEvent(error_type=etype, message=msg):
            render_error(f"{etype}: {msg}")
        case _:
            render(event)
```

This is a deliberate v2 default flip **in glow only** — core
`Run.events()` keeps `on_error="raise"`.  Pass `on_error="raise"` to
restore the raising contract; cancellation raises in both modes.  Note
that selector filters apply to the terminal event too — include
`RunError` (or use no selectors) if you need to observe it.
`ErrorEvent` remains the *recoverable* inline anomaly; the stream
continues after it.

## `statuses()` — user-facing hints, namespace-aware

`FluentRun.statuses()` interleaves opaque
[`Status`](../../src/t0_agents/stream/status.py) hints into the event
stream — `"Calling fred_get_series…"` per tool call (with the tool
identity decomposed against the App's mounted MCP namespaces:
`server="fred"`, `tool="get_series"`) and `"Thinking…"` per reasoning
block, deduplicated and suppressed exactly like the production policy
this generalises.

```python
from t0_agents.stream.status import Status

async for item in tutor(question).statuses():
    match item:
        case Status(text=text):
            set_status_line(text)
        case _:
            render(item)
```

## `.where(pred)` for conditional emission

Predicates AND-chain; the selector yields only when every predicate
returns true.

```python
from t0_agents.glow import ToolCall

# Only emit calls to "lookup_order"
order_calls = ToolCall.where(lambda tc: tc.name == "lookup_order")

async for tc in tutor("Where is order #42?").events(order_calls):
    print(tc.arguments)
```

Combine `.where()` and `.map()` in any order — they compose left-to-right.

## Runnable demo

[`examples/glow/03_streaming.py`](../../examples/glow/03_streaming.py)
exercises `stream_text()`, then `Thought`, `ToolCall.map(...)`, and
`Text.map(...)` against a live gateway + FastMCP server.  Boot the
shared gateway with
`docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`
before running.

---

For the canonical streaming narrative (`Run.events()`, the codec,
status synthesis, SSE, WebSocket, the full `AgentEvent` union), see
[`docs/streaming.md`](../streaming.md).
