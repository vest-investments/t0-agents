# Phase E — Streaming

> Status: **ready to start**. Phase deps: A, D complete. Open
> questions resolved: T0 Q1 (per-block deltas, stable IDs), FastMCP
> Q2 (build own forwarder).
> Delivers: **Milestone M5 — Live Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md) §3.3
  (`Agent.stream`), correlation-key clause
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q1, FastMCP Q2
- [`../problemas.md`](../problemas.md) — Problem 2
- [`phase-a-agent-core.md`](phase-a-agent-core.md),
  [`phase-d-serving.md`](phase-d-serving.md)

---

## 1. Goal

Two-level streaming:

- **Token-level**: `TextDeltaEvent` while the LLM emits the final
  response — letting browser/TUI clients render text as it arrives.
- **Event-level**: `ToolCallDeltaEvent` / `ToolCallEvent`,
  `ToolResultEvent`, `ThoughtDeltaEvent` for spinner / progress UX.

Plus serving-layer upgrades: SSE for one-way push and WebSocket for
bidirectional traffic (so elicitation can reach the user).

After Phase E:

```python
async for event in agent.stream("What's the weather?"):
    match event:
        case TextDeltaEvent(id=id_, delta=d):
            stream_to_ui(id_, d)
        case ToolCallEvent(name=n, arguments=a):
            show_spinner(f"Calling {n}…")
        case ToolResultEvent(name=n):
            hide_spinner()
        case DoneEvent(result=r):
            finalize(r)
```

Server-side (extends Phase D):

- `GET /chat/stream` (SSE) — push-only event stream.
- `WS /chat/ws` — bidirectional channel: client sends `{"type":
  "user_message", ...}` and `{"type": "elicitation_response", ...}`;
  server pushes `AgentEvent` JSON.

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/stream/` with:
  - `events.py` — full `AgentEvent` union.
  - `assembler.py` — fold the SDK chunk stream into `AgentEvent` stream.
  - `runner.py` — `StreamingAgentLoop`: streaming variant of
    `run_agent` that emits events.
- Extension on `Agent`: `Agent.stream(...)` method.
- Extension to `serve/`:
  - `streaming_routes.py` — SSE + WS endpoints.
  - `ws_protocol.py` — typed WS message envelopes.
- Completion of `core/elicitation_forwarder.py` (real Future-based
  resolver wired to WS).
- Tests under `tests/test_stream/`, plus serve-streaming integration.
- Example `examples/12_streaming_http/` with a tiny browser client
  (HTML + vanilla JS).
- Public doc page `docs/streaming.md`.

### 2.2 Out of scope

- Stream cancellation mid-response (deferred). Server hangs up,
  client reconnects fresh.
- Backpressure beyond a bounded send queue. If the queue fills, the
  Agent run pauses; document the buffer ceiling.
- Multiplexing multiple agents over one WS. One agent per connection.

### 2.3 Non-goals

- Do not invent a custom token format. Always serialise `AgentEvent`s
  as JSON over SSE/WS. Browsers and tooling can decode generically.
- Do not break the non-streaming `run()`. `stream()` is additive.

---

## 3. Constraints (from the contract)

1. `mcp/` stays standalone. Phase E adds a streaming entry point in
   `stream/runner.py` rather than rewriting `run_agent`. The non-
   streaming path stays byte-identical.
2. `stream/` depends only on `core/` and `mcp/`. `serve/streaming_routes.py`
   depends on `stream/` and `serve/`.
3. **`AgentEvent` correlation key** (contract §3.3): every delta event
   carries the underlying chunk's `id`, so consumers can pair
   `TextDeltaEvent`s with their terminal `TextEndEvent`, accumulate
   `ToolCallDeltaEvent`s into a final `ToolCallEvent`, etc.
4. TensorZero-first: streaming uses `t0.inference(stream=True)` —
   no parallel transport.
5. Per-session Client (FastMCP Q3) carries over from Phase D and is
   doubly important for WS, since each WS connection must own its
   elicitation handler.

---

## 4. Module additions

### 4.1 `src/t0_agents/stream/events.py` (new)

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Union

from t0_agents.core.types import AgentResult


@dataclass(slots=True)
class TextDeltaEvent:
    id: str          # block id from TextChunk.id (correlation key)
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndEvent:
    id: str
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ToolCallDeltaEvent:
    id: str          # ToolCallChunk.id
    raw_name_delta: str = ""
    raw_arguments_delta: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(slots=True)
class ToolCallEvent:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultEvent:
    id: str          # matches the ToolCallEvent's id
    name: str
    result: str      # serialised tool result (string per existing contract)
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class ThoughtDeltaEvent:
    id: str
    delta: str
    type: Literal["thought_delta"] = "thought_delta"


@dataclass(slots=True)
class ThoughtEvent:
    id: str
    text: str
    type: Literal["thought"] = "thought"


@dataclass(slots=True)
class TurnEndEvent:
    """Emitted at the end of each turn — useful for client-side
    'thinking…' indicators that should reset between turns.
    """
    turn: int
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class DoneEvent:
    result: AgentResult
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    """Surfaces non-fatal errors that didn't abort the run.
    Fatal errors raise out of the async iterator; this is for
    things like recoverable TensorZeroError retries (which today log
    a sentinel inference_id).
    """
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: Literal["error"] = "error"


AgentEvent = Union[
    TextDeltaEvent, TextEndEvent,
    ToolCallDeltaEvent, ToolCallEvent,
    ToolResultEvent,
    ThoughtDeltaEvent, ThoughtEvent,
    TurnEndEvent, DoneEvent, ErrorEvent,
]
```

### 4.2 `src/t0_agents/stream/assembler.py` (new)

```python
from collections.abc import AsyncIterator

from tensorzero.types import (
    ChatChunk, TextChunk, ToolCallChunk, ThoughtChunk,
)

from t0_agents.stream.events import (
    AgentEvent, TextDeltaEvent, TextEndEvent,
    ToolCallDeltaEvent, ToolCallEvent,
    ThoughtDeltaEvent, ThoughtEvent,
)


class TurnStreamAssembler:
    """Folds T0 chunk stream into AgentEvent stream for ONE turn.

    Per T0 Q1 answer:
    - Each chunk's content list may have multiple deltas of mixed types.
    - chunks share stable ids per content block (the correlation key).
    - usage arrives in a final chunk.
    - [DONE] terminates the stream.

    The assembler emits delta events as chunks arrive AND emits
    consolidation events (TextEndEvent, ToolCallEvent, ThoughtEvent)
    when a block is complete (no more deltas for that id).

    Block completion is detected when:
    - the next chunk doesn't include this id, OR
    - the final usage chunk arrives.

    """

    async def assemble(
        self, chunks: AsyncIterator[ChatChunk]
    ) -> AsyncIterator[AgentEvent]:
        text_acc: dict[str, str] = {}
        tool_acc: dict[str, dict] = {}
        thought_acc: dict[str, str] = {}
        seen_ids_in_chunk: set[str] = set()

        async for chunk in chunks:
            seen_ids_in_chunk.clear()

            for block in chunk.content:
                seen_ids_in_chunk.add(block.id)
                if isinstance(block, TextChunk):
                    text_acc[block.id] = text_acc.get(block.id, "") + block.text
                    yield TextDeltaEvent(id=block.id, delta=block.text)
                elif isinstance(block, ToolCallChunk):
                    state = tool_acc.setdefault(block.id, {"raw_name": "", "raw_arguments": ""})
                    state["raw_name"] += block.raw_name
                    state["raw_arguments"] += block.raw_arguments
                    yield ToolCallDeltaEvent(
                        id=block.id,
                        raw_name_delta=block.raw_name,
                        raw_arguments_delta=block.raw_arguments,
                    )
                elif isinstance(block, ThoughtChunk):
                    if block.text:
                        thought_acc[block.id] = thought_acc.get(block.id, "") + block.text
                        yield ThoughtDeltaEvent(id=block.id, delta=block.text)

            # Consolidation: emit *End / *Event for ids that did NOT
            # appear in this chunk and were last seen previously.
            # (Implementation detail: track previous_seen_ids and diff.)
            ...

        # Final flush: emit consolidation events for everything still
        # in the accumulators.
        for tid, state in tool_acc.items():
            import json
            yield ToolCallEvent(
                id=tid,
                name=state["raw_name"],
                arguments=json.loads(state["raw_arguments"]) if state["raw_arguments"] else {},
            )
        for tid in text_acc:
            yield TextEndEvent(id=tid)
        for tid, text in thought_acc.items():
            yield ThoughtEvent(id=tid, text=text)
```

### 4.3 `src/t0_agents/stream/runner.py` (new)

```python
class StreamingAgentLoop:
    """Streaming analogue of mcp.loop.run_agent.

    Same loop structure: list_tools → inference → execute tool calls →
    repeat. Difference: each inference is stream=True, and the
    assembled events are yielded as they arrive.

    Memory + guardrail hooks integrate identically to Agent.run.
    """

    async def stream(
        self,
        t0,
        tool_provider,
        user_message,
        function_name,
        *,
        episode_id=None,
        max_turns=20,
        ...,
    ) -> AsyncIterator[AgentEvent]:
        ...
        for turn in range(max_turns):
            chunks = await t0.inference(
                function_name=function_name,
                input={...},
                stream=True,
                ...,
            )
            assembler = TurnStreamAssembler()

            tool_calls_collected = []

            async for event in assembler.assemble(chunks):
                if isinstance(event, ToolCallEvent):
                    tool_calls_collected.append(event)
                yield event

            yield TurnEndEvent(turn=turn + 1)

            if not tool_calls_collected:
                break

            # Execute tool calls — same path as run_agent
            for tc in tool_calls_collected:
                tr = await tool_provider.call_tool(tc.name, tc.arguments)
                yield ToolResultEvent(id=tc.id, name=tc.name, result=str(tr))

        yield DoneEvent(result=AgentResult(...))
```

### 4.4 `Agent.stream()` (extends `core/agent.py`)

```python
async def stream(
    self,
    user_message: str,
    *,
    episode_id: str | None = None,
) -> AsyncIterator[AgentEvent]:
    # Memory hydration + guardrail before_run hooks (same as run())
    # Then drive StreamingAgentLoop instead of run_agent.
    ...
```

### 4.5 `core/elicitation_forwarder.py` — full implementation (Phase E)

```python
import asyncio
from dataclasses import dataclass
from uuid import uuid4

from fastmcp.client.elicitation import ElicitResult


class ElicitationForwarder:
    """Future-based bridge from FastMCP elicitation handler to a
    network channel (WebSocket).

    For each ctx.elicit() call:
    1. handler() is invoked by FastMCP.
    2. handler creates a Future, stores it under a fresh elicitation_id.
    3. handler emits ElicitationRequestEvent on the agent's event
       channel (the WS).
    4. handler awaits the Future.
    5. The WS-side route receives the client's response, looks up
       the Future by elicitation_id, and resolves it.

    Cancellation: if the WS disconnects while a Future is pending,
    the resolver auto-resolves with action="decline" and surfaces an
    ErrorEvent.
    """

    def __init__(self, *, on_request, default_timeout: float = 300.0):
        self._pending: dict[str, asyncio.Future[ElicitResult]] = {}
        self._on_request = on_request   # async callable that pushes to WS
        self._default_timeout = default_timeout

    async def handler(self, message, response_type, params, context):
        elicit_id = str(uuid4())
        fut: asyncio.Future[ElicitResult] = asyncio.get_event_loop().create_future()
        self._pending[elicit_id] = fut

        await self._on_request({
            "type": "elicitation_request",
            "elicitation_id": elicit_id,
            "message": message,
            "schema": params.requestedSchema if params else None,
        })

        try:
            return await asyncio.wait_for(fut, timeout=self._default_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return ElicitResult(action="decline")
        finally:
            self._pending.pop(elicit_id, None)

    async def resolve(self, elicit_id: str, payload: dict) -> None:
        fut = self._pending.get(elicit_id)
        if fut and not fut.done():
            action = payload.get("action", "accept")
            content = payload.get("content", {})
            fut.set_result(ElicitResult(action=action, content=content))
```

### 4.6 `serve/streaming_routes.py` (new)

SSE and WS endpoints. Both use a per-connection `Agent` (FastMCP Q3).

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import EventSourceResponse
from sse_starlette.sse import EventSourceResponse


def build_streaming_router(agent_factory) -> APIRouter:
    router = APIRouter()

    @router.get("/chat/stream")
    async def chat_stream(message: str, episode_id: str | None = None):
        async def event_stream():
            async with agent_factory.session() as session_agent:
                async for event in session_agent.stream(message, episode_id=episode_id):
                    yield {
                        "event": event.type,
                        "data": json.dumps(asdict(event)),
                    }
        return EventSourceResponse(event_stream())

    @router.websocket("/chat/ws")
    async def chat_ws(ws: WebSocket):
        await ws.accept()
        async with agent_factory.session(elicitation_forwarder=...) as session_agent:
            try:
                while True:
                    incoming = await ws.receive_json()
                    match incoming.get("type"):
                        case "user_message":
                            async for event in session_agent.stream(
                                incoming["message"],
                                episode_id=incoming.get("episode_id"),
                            ):
                                await ws.send_json({"type": event.type, **asdict(event)})
                        case "elicitation_response":
                            await session_agent.elicitation_forwarder.resolve(
                                incoming["elicitation_id"], incoming.get("payload", {}),
                            )
            except WebSocketDisconnect:
                pass

    return router
```

### 4.7 `[serve]` extra additions

```toml
[project.optional-dependencies.serve]
sse-starlette = ">=2.0"   # Adds EventSourceResponse on top of FastAPI
```

---

## 5. Tests

Location: `tests/test_stream/` and `tests/test_serve/test_streaming.py`.

### 5.1 Unit (offline)

- [ ] `test_assembler_text_only` — feed mocked TextChunks, assert
      `TextDeltaEvent`s + `TextEndEvent`.
- [ ] `test_assembler_tool_call_streaming` — partial `ToolCallChunk`s
      accumulate into one `ToolCallEvent` with parsed JSON args.
- [ ] `test_assembler_thought_chunks` — same for thoughts.
- [ ] `test_assembler_mixed_blocks_in_chunk` — text + thought + tool
      call in one chunk; correlation IDs preserved.
- [ ] `test_assembler_invalid_tool_args_json` — emits a
      `ToolCallEvent` with empty args + `ErrorEvent` (does not
      crash).
- [ ] `test_streaming_agent_loop_terminates` — happy path.
- [ ] `test_elicitation_forwarder_resolve` — Future resolves with
      action/content; pending dict empties.
- [ ] `test_elicitation_forwarder_timeout` — auto-decline.
- [ ] `test_elicitation_forwarder_disconnect` — Future auto-declined
      on cancellation.

### 5.2 Integration (live gateway)

- [ ] `test_agent_stream_e2e` — drive `Agent.stream` against the live
      gateway; assert the event sequence ends with `DoneEvent`.
- [ ] `test_sse_endpoint` — `httpx.AsyncClient.stream("GET", ...)`
      reads SSE frames, parses events.
- [ ] `test_ws_endpoint_with_elicitation` — start WS connection,
      send user message that triggers a `ctx.elicit()` tool, respond
      via WS, assert agent completes.

### 5.3 Regression

- [ ] `Agent.run()` (non-streaming) byte-identical to Phase D.
- [ ] All earlier phases' tests untouched.

---

## 6. Example

Location: `examples/12_streaming_http/`.

- `agent_app.py` — wires `Agent` + `create_app` (with streaming
  router enabled).
- `static/index.html` — minimal browser client: textbox, "send"
  button, live-rendered response area, elicitation modal popup.
- `static/app.js` — WebSocket client that handles every
  `AgentEvent.type`.
- `README.md` — boot instructions, screenshot description, walkthrough
  of an elicitation round-trip.

---

## 7. Documentation

- New: `docs/streaming.md`.
- Sections:
  - "Why stream?" — the UX argument.
  - "`Agent.stream()` cookbook" — Python consumption.
  - "Event types" — reference of every `AgentEvent` variant.
  - "Server-side: SSE vs WebSocket" — when to pick which.
  - "Browser client recipe" — minimal JS example.
  - "Elicitation over WS" — round-trip diagram.
  - "Backpressure" — the bounded queue contract.
- Update `developer-guide.md` §4 (public API) — add `stream`,
  `AgentEvent`, `ElicitationForwarder`.
- `CHANGELOG.md` — `0.6.0`.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/stream/__init__.py`.
- [ ] `src/t0_agents/stream/events.py`.
- [ ] `src/t0_agents/stream/assembler.py`.
- [ ] `src/t0_agents/stream/runner.py`.
- [ ] `src/t0_agents/core/agent.py` — `stream()` method.
- [ ] `src/t0_agents/core/elicitation_forwarder.py` — full Future-
      based resolver.
- [ ] `src/t0_agents/serve/streaming_routes.py`.
- [ ] `src/t0_agents/serve/ws_protocol.py`.
- [ ] `pyproject.toml` — `sse-starlette` added under `[serve]`.

### 8.2 Tests

- [ ] All §5.1 unit tests.
- [ ] All §5.2 integration tests.

### 8.3 Example

- [ ] `examples/12_streaming_http/` complete with browser client.

### 8.4 Docs

- [ ] `docs/streaming.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `phase-e-implementation-log.md` with M5 release-notes snippet
      and the T0 Q1 chunk-correlation finding logged for posterity.

---

## 9. Acceptance criteria (Milestone M5)

1. `from t0_agents import AgentEvent` (re-exported from `stream`).
2. `Agent.stream()` works against the live gateway.
3. `examples/12_streaming_http/` browser client renders deltas live
   and supports an elicitation round-trip.
4. SSE and WS endpoints documented in `docs/streaming.md`.
5. Phase A `run()` behaviour byte-identical (regression check).

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Tool-call argument JSON arrives malformed across chunks | low (per T0 docs) | Defensive `json.loads(..., default=None)`; emit `ErrorEvent` on failure. |
| Browsers buffering SSE differently | medium | Test with curl `--no-buffer` and Chrome/Firefox. WS path is the recommended channel. |
| WebSocket reconnection mid-elicitation | high | On WS close, forwarder auto-declines pending elicitations and surfaces `ErrorEvent`s. Document the contract. |
| Memory write happens after stream consumer disconnects | low | Run completion is bound to the AgentLoop, not the consumer; memory persists regardless. |

---

## 11. Estimated surface

- **New files:** 5 src + 4 test + 4 example + 1 doc.
- **Modified files:** 3 (`core/agent.py`, `core/elicitation_forwarder.py`,
  `pyproject.toml`).
- **Net new public symbols:** ~12 (`AgentEvent` + every variant;
  `Agent.stream`; `ElicitationForwarder.resolve`).
- **New deps:** 1 under `[serve]` (`sse-starlette`).
