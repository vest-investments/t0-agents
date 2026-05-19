# Streaming and event selectors

> Part of the glow ladder.  Backs the chat-only streaming path on top
> of the v1.0 [`Run.events()`](../streaming.md) surface.  For
> JSON-shaped agents, streaming is deferred — see
> [`limitations.md` — No streaming JSON](limitations.md#no-streaming-json).

`FluentRun.events(*selectors)` is the typed event stream from
[`Run.events()`](../streaming.md), filtered.  Seven module-level
selectors cover the v1.0 event union — one selector per
[`*Event`](../../src/t0_agents/stream/events.py) class:

| Selector     | Underlying event                          | Yields                          |
|--------------|-------------------------------------------|---------------------------------|
| `Thought`    | `ThoughtEvent`                            | extended-reasoning thinking blocks |
| `ToolCall`   | `ToolCallEvent`                           | LLM tool invocations             |
| `Text`       | `TextEndEvent` (completion marker)        | one event per text block finished |
| `ToolResult` | `ToolResultEvent`                         | tool execution results           |
| `TurnEnd`    | `TurnEndEvent`                            | one event per inference turn      |
| `Done`       | `DoneEvent`                               | final loop completion             |
| `Error`      | `ErrorEvent`                              | tool / inference failures         |

Compose them with `.where(pred)` (AND-chains predicates) and `.map(fn)`
(transforms the yielded value).

```python
from t0_agents.glow import Thought, ToolCall, Text

async for event in tutor("Explain present perfect").events(
    Thought,
    ToolCall.map(lambda tc: f"> calling {tc.name}"),
    Text,
):
    ...
```

`Text` defaults to `TextEndEvent` (the completion marker for one text
block).  For **token-by-token** streaming, build a selector for
`TextDeltaEvent` directly and map it to the delta string:

```python
from t0_agents.glow import Selector
from t0_agents.stream.events import TextDeltaEvent

async for delta in tutor("Explain present perfect").events(
    Selector(TextDeltaEvent).map(lambda e: e.delta),
):
    print(delta, end="", flush=True)
```

Passing no selectors yields every `AgentEvent` (parity with
`Run.events()`).  Glow only supports streaming for chat-flavoured
output modes; the [v1.0 chat-only constraint](../streaming.md) carries
through — calling `.events()` on a `JsonOutput` / `StructuredOutput`
agent raises `ConfigurationError` at the first `__anext__`.

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
exercises `Thought`, `ToolCall.map(...)`, and `Text` against a live
gateway + FastMCP server.  Boot the shared gateway with
`docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`
before running.

---

For the canonical streaming narrative (`Run.events()`, SSE, WebSocket,
the full `AgentEvent` union), see [`docs/streaming.md`](../streaming.md).
