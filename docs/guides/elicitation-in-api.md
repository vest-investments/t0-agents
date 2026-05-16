# Elicitation in your `t0-agents` API

> **Audience.** You maintain a Python API (FastAPI assumed; the
> principles port to anything ASGI) that uses `t0-agents` as its
> agent core. The API instantiates an `Agent` per request, streams
> agent output to the user over **Server-Sent Events**, and
> connects to one or more **external** FastMCP servers via HTTP.
> One of those servers exposes elicitation-capable tools, and you
> need the human-in-the-loop prompt to reach the end user — and
> the user's reply to flow back — through your existing SSE
> channel.
>
> This guide is self-contained for the **API side**. It covers
> everything you need to wire up: the SSE + POST pattern, the
> `ElicitationForwarder` bridge, per-session lifecycle, the
> handler design, both common patterns (binary confirm, dynamic
> survey) from the API angle, sticky-session and timeout
> constraints, tests, and pitfalls.
>
> For the **server-side** code (the `@mcp.tool async def ...`
> that calls `ctx.elicit(...)`) see the companion guide
> [`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md).

---

## 1. What this guide covers

- The SSE-down + POST-up pattern that makes elicitation work over
  SSE, with the same expressiveness WebSocket gives you.
- `t0_agents.ElicitationForwarder`: what it is, how to wire it,
  why it solves the multiplex problem out of the box.
- Per-session state for the API: forwarder registry, Client
  construction, agent lifecycle, cleanup on disconnect.
- The wire format between the API and the user-facing UI.
- The handler choices: forwarder for production, scripted handlers
  for tests, `require_approval` guardrail for tools you do not
  control.
- The two main patterns end-to-end (confirm an action, dynamic
  survey) from the API perspective — what the UI shows, what to
  POST back, how to instruct the LLM via prompt.
- Sticky sessions, proxy buffering, timeouts, disconnect handling.
- Tests with `httpx`. Production checklist.

Out of scope: how the FastMCP server implements `ctx.elicit()`,
which transport the server uses, what `response_type` shapes the
MCP spec allows. All of that is in
[`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md).

---

## 2. The architecture

```
[external MCP servers]            [your API process]            [user browser]

  server A (plain tools)              GET /chat/stream  ◄────── EventSource
  server B (plain tools)              ── mint session_id
  server C (elicit tools) ◄──MCP HTTP─ Client(elicitation_handler=
                                         forwarder.handler)
                                       Run.events(user_message)

  tool: ctx.elicit("Approve?")  ──────► forwarder.handler runs
                                          emit SSE event ────────► UI shows prompt
                                          await Future                ↓
                                                                  user clicks
                                       POST /elicitation/{id}  ◄──── fetch()
                                          forwarder.resolve(id)
  tool resumes  ◄──── elic response ──   Future resolves
  returns                                Agent continues
                                          ... more SSE events ...
                                       SSE closes on DoneEvent
```

Two channels, both keyed by the same `session_id`:

- **SSE down**: agent events + elicitation prompts.
- **POST up**: elicitation responses.

The bridge is `ElicitationForwarder`, an object that lives for the
duration of one SSE stream and demultiplexes elicitation prompts
to/from `asyncio.Future`s indexed by `elicitation_id`.

---

## 3. `ElicitationForwarder` in 30 seconds

`t0-agents` ships this class precisely for the SSE / WebSocket /
async-resolver case. Construct it with an `on_request` coroutine
that takes the elicitation envelope and ships it to wherever the
human will see it (your SSE queue, your WebSocket, a Slack
message, …). It exposes three methods:

| Method | Purpose |
|---|---|
| `forwarder.handler` | The coroutine you pass to `Client(elicitation_handler=...)`. FastMCP calls it whenever a tool elicits. It pushes the envelope through `on_request`, awaits a Future, returns the resolved `ElicitResult`. |
| `await forwarder.resolve(elicitation_id, payload)` | Called from your POST handler when the user replies. Looks up the matching Future and resolves it. |
| `await forwarder.cancel_all()` | Called in your SSE `finally` block when the user disconnects. Drains every pending Future with `action="decline"` so no tool hangs until timeout. |

You build one forwarder per session (= per SSE stream). FastMCP's
`Client` binds exactly one `elicitation_handler` at construction,
which is why each session needs a fresh `Client` and therefore a
fresh forwarder.

---

## 4. Per-session state

Keep a dictionary keyed by `session_id`, locked for concurrent
access. The POST endpoint looks up the forwarder for the session
the user is responding to.

```python
# src/your_api/state.py
from __future__ import annotations
import asyncio

from t0_agents import ElicitationForwarder

_FORWARDERS: dict[str, ElicitationForwarder] = {}
_LOCK = asyncio.Lock()


async def register(session_id: str, forwarder: ElicitationForwarder) -> None:
    async with _LOCK:
        _FORWARDERS[session_id] = forwarder


async def unregister(session_id: str) -> None:
    async with _LOCK:
        _FORWARDERS.pop(session_id, None)


def get(session_id: str) -> ElicitationForwarder | None:
    return _FORWARDERS.get(session_id)
```

The lock matters only for the rare race where a request finishes
while another arrives — not for correctness of normal operation,
but to avoid `RuntimeError` on dict mutation. Reads do not need
the lock.

This dict is **in process memory**, which dictates your deployment
topology — see §10.

---

## 5. The SSE endpoint

This is the one big piece of code. It owns the agent lifecycle for
one user request: build the forwarder, build the Client (every MCP
server in `mcpServers`, one shared handler), build the Agent, drive
`Run.events(...)`, forward events to the user, clean up on exit
or disconnect.

```python
# src/your_api/routes/chat.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Awaitable, cast
from uuid import uuid4

from fastapi import APIRouter, Request
from fastmcp import Client
from sse_starlette.sse import EventSourceResponse
from tensorzero import AsyncTensorZeroGateway
from t0_agents import (
    Agent,
    AgentConfig,
    AgentEvent,
    DoneEvent,
    ElicitationForwarder,
    ErrorEvent,
)

from your_api.state import register, unregister

logger = logging.getLogger(__name__)
router = APIRouter()

# One Client constructor per request — the elicitation_handler binds at __init__.
MCP_CONFIG: dict[str, Any] = {
    "mcpServers": {
        "tools_a":      {"url": "http://mcp-a:8000/mcp"},
        "tools_b":      {"url": "http://mcp-b:8000/mcp"},
        "elicit_tools": {"url": "http://mcp-elicit:8000/mcp"},
    }
}


@router.get("/chat/stream")
async def chat_stream(
    request: Request, user_message: str
) -> EventSourceResponse:
    session_id = str(uuid4())

    async def generate() -> AsyncIterator[dict[str, str]]:
        # 1. Outbound queue carries everything we send to the user.
        outbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        # 2. Forwarder: on_request pushes onto the outbound queue.
        async def push(envelope: dict[str, Any]) -> None:
            await outbound.put({"type": "elicitation_request", **envelope})

        forwarder = ElicitationForwarder(on_request=push, default_timeout=600.0)
        await register(session_id, forwarder)

        # 3. First event: tell the UI its session_id so it can POST.
        yield {"event": "open", "data": json.dumps({"session_id": session_id})}

        async def run_agent() -> None:
            try:
                gateway = cast(
                    Awaitable[AsyncTensorZeroGateway],
                    AsyncTensorZeroGateway.build_http(
                        gateway_url="http://tensorzero:3000"
                    ),
                )
                # ONE handler serves all MCP servers in the config.
                client: Client[Any] = Client(
                    MCP_CONFIG, elicitation_handler=forwarder.handler
                )
                async with await gateway as t0, client:
                    agent = Agent(
                        AgentConfig(
                            name="api_agent",
                            function_name="agent",
                            tool_timeout=None,  # human is the bottleneck — see §11
                        ),
                        t0=t0,
                        tool_provider=client,
                    )
                    async for event in agent.invoke(user_message).events():
                        await outbound.put(
                            {"type": "agent_event", **_serialise_event(event)}
                        )
            except Exception as exc:  # noqa: BLE001 — surface anything
                logger.exception("Agent run failed for %s", session_id)
                await outbound.put(
                    {"type": "agent_error", "message": str(exc)}
                )
            finally:
                # Sentinel — tells the drain loop to stop.
                await outbound.put(None)

        runner = asyncio.create_task(run_agent())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(outbound.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Heartbeat — defeats proxy idle timeouts.
                    yield {"event": "ping", "data": "{}"}
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    continue

                if item is None:
                    break  # agent finished
                yield {"data": json.dumps(item)}

        except asyncio.CancelledError:
            logger.info("SSE %s cancelled — client disconnected", session_id)
            await forwarder.cancel_all()  # unblock in-flight elicit
            runner.cancel()
            raise
        finally:
            try:
                await runner
            except (asyncio.CancelledError, Exception):
                pass
            await unregister(session_id)

    return EventSourceResponse(generate())


def _serialise_event(event: AgentEvent) -> dict[str, Any]:
    """Project an AgentEvent into a JSON-serialisable dict."""
    if isinstance(event, DoneEvent):
        return {
            "kind": "done",
            "response": event.result.response,
            "episode_id": event.result.episode_id,
            "turns": event.result.turns,
        }
    if isinstance(event, ErrorEvent):
        return {"kind": "error", "message": event.message, "detail": event.detail}
    # TextDelta, ToolCallDelta, ToolResult, etc.
    return {
        "kind": event.__class__.__name__,
        **{k: _safe(v) for k, v in event.__dict__.items()},
    }


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return repr(value)
```

Three properties of this design that matter operationally:

- **One forwarder per session**. The forwarder owns the Futures.
  Per-session isolation is mandatory because `Client` binds one
  handler per instance.
- **One `Client` per session**. Never shared across requests. The
  cost of building a fresh Client + opening N MCP connections per
  request is real but acceptable for HITL flows. If you need to
  share clients across requests in non-HITL paths, do it on a
  separate code path with a different non-elicitation Client.
- **`forwarder.cancel_all()` in the `finally` block**. Without it,
  a disconnected user leaves tool tasks hanging on a Future until
  `default_timeout` (600s here) expires, holding resources and
  potentially leaking external side effects.

---

## 6. The POST endpoint

The other half of the wire: receives the user's reply, resolves the
matching Future.

```python
# src/your_api/routes/elicitation.py
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from your_api.state import get

router = APIRouter()


@router.post("/chat/{session_id}/elicitation/{elicitation_id}")
async def reply(
    session_id: str,
    elicitation_id: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, bool]:
    """User's response to a pending elicitation.

    Payload shape:
        {"action": "accept" | "decline" | "cancel",
         "content": {...}}      # required only when action == "accept"
    """
    forwarder = get(session_id)
    if forwarder is None:
        raise HTTPException(404, "Session not found or already closed")
    await forwarder.resolve(elicitation_id, payload)
    return {"ok": True}
```

The forwarder validates `action`, coerces unknown values to
`"decline"`, and ignores unknown `elicitation_id`s (the user could
have raced the timeout). Returning 404 for a missing session is the
right signal to the UI — show "session ended" rather than retry.

---

## 7. The wire format

Three event types flow over the SSE channel. The UI dispatches on
`event` (the SSE event name) and on `data.type` (your payload tag).

### 7.1 `open` event (always first)

```
event: open
data: {"session_id": "ab12..."}
```

The UI saves `session_id` to use in the POST URL.

### 7.2 Agent events

```
event: message      (the default; SSE event name omitted)
data: {"type": "agent_event", "kind": "TextDeltaEvent",
       "id": "txt-1", "delta": "Hello"}
data: {"type": "agent_event", "kind": "ToolCallEvent",
       "id": "tc-1", "name": "delete_resource",
       "arguments": {"resource_id": "alpha-7"}}
data: {"type": "agent_event", "kind": "done",
       "response": "Done — kept alpha-7 in place.",
       "episode_id": "ep-1", "turns": 3}
```

See [`streaming.md`](../streaming.md) for the full event catalogue.

### 7.3 Elicitation request

When a tool calls `ctx.elicit()`, the forwarder pushes:

```
data: {
  "type": "elicitation_request",
  "elicitation_id": "ab12-eli-7",
  "message": "Delete resource alpha-7? This cannot be undone.",
  "schema": {"type": "object", "properties": {}},   // MCP requestedSchema
  "response_type": null                              // null | "str" | "int" | "bool" | <type name>
}
```

The UI renders a prompt, asks the user, then POSTs:

```
POST /chat/ab12.../elicitation/ab12-eli-7
{"action": "accept", "content": {}}
```

### 7.4 Content format by response_type

The `content` payload of an `"accept"` follows MCP's wire format.
Match it on the UI side:

| `response_type` field on the prompt | `content` shape | Example |
|---|---|---|
| `null` (binary approval) | `{}` (empty object — **not** `None`) | `{"action": "accept", "content": {}}` |
| `"str"` | `{"value": "<text>"}` | `{"action": "accept", "content": {"value": "Alice"}}` |
| `"int"` | `{"value": 42}` | `{"action": "accept", "content": {"value": 42}}` |
| `"bool"` | `{"value": true}` | `{"action": "accept", "content": {"value": true}}` |
| array of strings (constrained choice) | `{"value": "<choice>"}` | `{"action": "accept", "content": {"value": "medium"}}` |
| dataclass name (structured input) | flat object with one key per field | `{"action": "accept", "content": {"name": "Alice", "age": 30}}` |

Scalar wrapping (`{"value": ...}`) is **not optional**. The server
side unwraps it transparently, but the UI must produce it.

---

## 8. Handler design — beyond the forwarder

`forwarder.handler` is the production handler. Two alternatives
are worth knowing.

### 8.1 Scripted handler — for tests

```python
from typing import Any
from fastmcp.client.elicitation import ElicitResult


async def auto_approve_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
    """Approve binary approvals; decline structured input. Tests only."""
    if response_type is None:
        return ElicitResult(action="accept", content={})
    return ElicitResult(action="decline")
```

`t0-agents` ships this exact handler plus a sibling
`auto_decline_handler` at `t0_agents.mcp`:

```python
from t0_agents.mcp import auto_approve_handler, auto_decline_handler
```

Both are intended for tests and smoke prototypes only — they
provide no human input.

### 8.2 `require_approval` guardrail — for tools you do not control

When the elicitation-capable tool lives in a third-party MCP server
that you cannot modify, you can still gate it on the client side:

```python
from t0_agents import Agent, AgentConfig
from t0_agents.guardrails import require_approval

agent = Agent(
    AgentConfig(
        name="api_agent",
        function_name="agent",
        tool_timeout=None,
        guardrails=[
            require_approval(
                tools=["delete_resource", "drop_index"],
                handler=forwarder.handler,
                prompt_template="Approve calling {tool_name} with {arguments}?",
            ),
        ],
    ),
    t0=t0,
    tool_provider=client,
)
```

How it differs from server-side `ctx.elicit()`:

| Property | Server-side `ctx.elicit()` | `require_approval` guardrail |
|---|---|---|
| Where the prompt is authored | Tool author | API author (you) |
| Per-tool granularity | Tool body decides | Listed by name in `tools=[...]` |
| Shows arguments | Whatever message the tool crafts | Rendered via `prompt_template`, `{tool_name}` + `{arguments}` available |
| Works on tools you do not own | Only if author wrote it | Yes — pure client-side |
| Decline result | Whatever string the tool returns | `"ERROR: user declined to run <tool>"` injected into the loop |

You can reuse the **same** `forwarder.handler` here, so the user
experience is identical whether the prompt came from a tool or from
the guardrail.

---

## 9. The two patterns end-to-end

### 9.1 Confirm an action

This is the basic pattern: an MCP server tool calls
`ctx.elicit(..., response_type=None)`, the API forwards the prompt
to the user, the user accepts or declines, the tool returns
"Deleted" or "Cancelled", the LLM adapts.

From the API's side, nothing pattern-specific to do — the wire-up
in §5 already covers it. Two concerns to be deliberate about:

**Prompt instruction to the agent.** Tell the model in your
`tensorzero.toml` system template how to react to a decline:

```
When a destructive action is declined by the user, do not retry
the same action. Acknowledge the decline in your reply and propose
an alternative or report the unchanged state of the world.
```

Without this, models tend to silently drop the failed attempt from
their final summary — worse UX than a visible "I did not delete
alpha-7 because you declined."

**UI rendering of the prompt.** A binary approval has
`response_type: null` in the SSE envelope. The UI renders the
`message`, two buttons (Approve / Decline), and POSTs
`{"action": "accept", "content": {}}` or `{"action": "decline"}`.
That is the whole spec.

### 9.2 Dynamic survey

The most natural way to implement an LLM-driven survey in this
architecture: keep question generation in the **agent's prompt**
and have the MCP server expose a single tool that iterates
`ctx.elicit()` over a passed-in list.

The agent's prompt asks the LLM to **emit the question list as the
arguments of the tool call**. No separate inference, no
`question_generator` sub-agent, no closure-over-gateway gymnastics
on the server side.

#### Server side (recap)

Companion guide [`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md)
§4.5 has the full code. Shape recap — a single tool:

```python
@mcp.tool
async def ask_questions(questions: list[QuestionSpec], ctx: Context) -> str:
    """Iterate elicitations over a caller-supplied list of QuestionSpec.

    Returns JSON {"answers": [...], "complete": true|false}.
    """
    ...
```

where `QuestionSpec` is a flat dataclass with `prompt: str`,
`type: str` (`"str" | "int" | "bool" | "choice"`), and
`options: list[str] | None`.

#### API side — instruct the LLM via prompt

Add a directive to your main agent's `tensorzero.toml` system
template:

```
When you need information from the user that is not in the
conversation, call `ask_questions` with a JSON list of 1–6
QuestionSpec objects. Each object has:
  - "prompt": one short question, ≤ 80 chars
  - "type":   "str" | "int" | "bool" | "choice"
  - "options": only when "type" == "choice", a list of ≥ 2 strings

The tool returns a JSON object with "answers" and "complete".
If "complete" is false, the user declined a question — adapt
your plan rather than retrying the same list.
```

The LLM authors the questions in its tool-call arguments. The
server iterates the elicitations. The collected answers come back
to the loop as a normal tool result.

#### Wire-up

Same SSE endpoint as §5. The `ask_questions` tool lives in the
MCP server config; its `ctx.elicit()` calls route through
`forwarder.handler` just like any binary approval.

The UI sees a sequence of `elicitation_request` events with
varying `response_type` values (`"str"`, `"int"`, `"bool"`, or
the constrained-choice list). For each one, render the prompt,
collect the answer, POST.

#### When prompt-driven generation is not enough

If you want stricter control over question generation — separate
A/B testing, dedicated prompt versioning, structured output
validation against a JSON schema — there are two paths:

- **Dedicated TensorZero function on the API side.** Add
  `[functions.question_generator]` to your `tensorzero.toml`.
  Before each `agent.invoke()`, call
  `t0.inference(function_name="question_generator", input={...})`,
  parse the response, and pass the questions as
  `system_arguments={"questions": [...]}` so the main agent sees
  them in its system template. Then instruct the agent to call
  `ask_questions(questions=questions_from_system)`.
- **Server-side question generation.** Move the question-generator
  inference into the MCP server itself. See
  [`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md)
  §4.6 — the server opens its own TensorZero gateway via a
  FastMCP `lifespan` and the survey tool calls it directly.

For most use cases the prompt-driven shape is enough and avoids
extra moving parts.

---

## 10. Sticky sessions — the deployment constraint

The forwarder dict in §4 lives in process memory. The POST endpoint
must reach the **same process** that holds the SSE generator.

| Topology | What to do |
|---|---|
| Single replica | Nothing — same process trivially |
| Multi-replica behind a load balancer | Enable sticky sessions: AWS ALB cookie stickiness on the target group; GKE `sessionAffinity: ClientIP` on the Service; nginx `ip_hash` or `sticky` directive |
| Multi-replica without stickiness | Move forwarder registry to shared state (Redis pub/sub + polling). Significant rewrite, loses the in-memory Future ergonomics. Not covered. |

Session lifetime is short — minutes, maybe an hour at most. Cookie
stickiness or IP-hash is usually sufficient.

---

## 11. Timeouts — four of them, coordinated

| Where | Default | Recommended for HITL |
|---|---|---|
| `AgentConfig.tool_timeout` (client side) | `30.0` | `None` — yield to the server-side timeout. |
| `@mcp.tool(timeout=...)` (server side) | ~30s | `300.0`–`1800.0` — bound to expected human latency. |
| `ElicitationForwarder(default_timeout=...)` | `300.0` | `600.0` if humans browse before answering. The forwarder declines on timeout, so this protects against orphaned Futures rather than hard-failing the tool. |
| Upstream proxy read timeout (nginx, ALB, CloudFront) | varies — often 60s | Comfortably above `default_timeout`. Otherwise the SSE stream is killed mid-elicit. |

The tightest of these wins. If your proxy times out at 60s, no
amount of `default_timeout=600.0` on the forwarder will help — the
SSE stream dies first.

---

## 12. Disconnect handling

When the user closes the tab:

1. `EventSourceResponse` cancels the SSE generator.
2. The `asyncio.CancelledError` propagates into the `try/finally`
   in §5.
3. `await forwarder.cancel_all()` declines every pending Future.
4. Tools waiting on `ctx.elicit()` observe `action="decline"`,
   their normal decline branch runs, they return a string.
5. The agent loop sees a decline-shaped tool result, finishes the
   turn (the model may not even run another turn if the user is
   gone), and the `Run.events(...)` async-iterator completes.
6. `runner` task completes normally.
7. `unregister(session_id)` removes the forwarder from the dict.

You do not need to track disconnects manually — `cancel_all` plus
the existing `result.action == "decline"` branches in your tools
do all the work.

---

## 13. Production checklist

- [ ] **Sticky sessions** configured at the LB (or single replica).
- [ ] **Proxy buffering disabled** — `proxy_buffering off;` in
      nginx, `X-Accel-Buffering: no` header. Otherwise no event
      reaches the user until the agent finishes.
- [ ] **Proxy read timeout ≥ `forwarder.default_timeout`**.
- [ ] **Heartbeat events** every N seconds (the example does 15s).
- [ ] **`AgentConfig.tool_timeout=None`**.
- [ ] **`forwarder.cancel_all()` in SSE `finally`**.
- [ ] **Per-session `Client`** — fresh
      `Client(MCP_CONFIG, elicitation_handler=...)` per request.
- [ ] **MCP servers on a stateful transport** (streamable-http or
      sse+http). Verify with the server team —
      [`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md) §3.
- [ ] **UI handles 404 on the elicitation POST** — session ended,
      show a "your turn was lost" message.
- [ ] **Structured logging** — include `session_id`,
      `elicitation_id`, and `episode_id` (from `DoneEvent`) for
      cross-referencing in ClickHouse.
- [ ] **System template instructs the agent** to respect declines
      and surface them in the final reply (see §9.1).

---

## 14. Testing the API

An integration test that drives both endpoints with `httpx`:

```python
# tests/test_chat_stream.py
import asyncio
import json

import httpx
import pytest

from your_api.app import app


@pytest.mark.asyncio
async def test_elicitation_round_trip() -> None:
    """User asks; agent triggers elicitation; user accepts; agent finishes."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream(
            "GET", "/chat/stream",
            params={"user_message": "delete alpha-7"},
        ) as r:
            assert r.status_code == 200
            session_id: str | None = None
            saw_done = False

            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])

                if payload.get("session_id"):
                    session_id = payload["session_id"]

                elif payload.get("type") == "elicitation_request":
                    elic_id = payload["elicitation_id"]
                    # POST in a parallel task — must not block the SSE drain
                    asyncio.create_task(
                        ac.post(
                            f"/chat/{session_id}/elicitation/{elic_id}",
                            json={"action": "accept", "content": {}},
                        )
                    )

                elif payload.get("kind") == "done":
                    saw_done = True
                    break

            assert saw_done
```

For unit-testing without a real MCP server, drive
`Agent` against a `MockGateway` from `t0_agents.testing` with a
scripted assistant turn that emits the destructive tool call, then
assert on `DoneEvent.result.response` — see
[`testing.md`](../testing.md).

For unit-testing `ElicitationForwarder` itself, construct one
directly with an `on_request` that appends envelopes to a list,
then call `handler` and `resolve` in sequence. The class
docstring at `t0_agents.core.elicitation_forwarder` is the
reference.

---

## 15. Pitfalls

| # | Pitfall | Fix |
|---|---|---|
| 1 | Reverse proxy buffers the stream — UI sees nothing until the agent finishes | `X-Accel-Buffering: no` header; `proxy_buffering off;` in nginx; streaming-enabled origin response in CloudFront/ALB. |
| 2 | Proxy idle timeout kills the stream mid-elicitation | Bump the proxy's read timeout above `forwarder.default_timeout`; emit heartbeats. |
| 3 | `tool_timeout=30.0` fires while the user reads the prompt | `tool_timeout=None` on `AgentConfig`. |
| 4 | Sharing one `Client` across sessions — different users see each other's prompts | Build a fresh `Client(MCP_CONFIG, elicitation_handler=forwarder.handler)` per request. |
| 5 | `tool_choice={"specific": ...}` plus elicitation triggers provider validation failure on the terminal turn | Omit `tool_choice`. Guide tool selection through the prompt instead. |
| 6 | Multi-pod deployment without sticky sessions — POST lands on the wrong pod and 404s | Enable sticky sessions at the LB (§10). |
| 7 | UI re-opens `EventSource` on a transient network blip and starts a new session — original session is orphaned | The `finally` block calls `cancel_all`; tools observe decline and complete cleanly. Continuity across reconnects is out of scope here. |
| 8 | POST endpoint hits 404 because the SSE finished before the user clicked | UI must treat 404 as "session ended" and not retry. The forwarder validates this correctly. |
| 9 | `outbound.put(None)` sentinel forgotten on a code path that exits the agent runner — SSE hangs forever | The `finally` block of `run_agent` in §5 always pushes the sentinel. Preserve that invariant if you refactor. |
| 10 | Logging `result.data` for structured-input elicitations dumps PII to your log aggregator | Mask sensitive fields before logging. |

---

## 16. Cross-references

- [`elicitation-in-mcp-server.md`](elicitation-in-mcp-server.md) —
  the companion guide for the FastMCP server side. Required reading
  for whoever owns the server with the elicit tool, and useful
  context for you.
- [`elicitation.md`](../elicitation.md) — protocol-level reference:
  full handler signature, content-format table, scalar wrapping,
  all constraints.
- [`serving.md`](../serving.md) — `t0-agents`' own built-in HTTP
  layer (`create_app`). Uses an equivalent `ElicitationForwarder`
  wire-up over **WebSocket** instead of SSE. Read
  `src/t0_agents/serve/streaming_routes.py` (~70 lines) for the WS
  variant of this same pattern.
- [`streaming.md`](../streaming.md) — full `AgentEvent` catalogue
  that the SSE stream surfaces.
- [`guardrails.md`](../guardrails.md) — `require_approval` and the
  other built-in guardrail factories.
- [`testing.md`](../testing.md) — `MockGateway`, `MockToolProvider`
  and other primitives for testing the agent loop without TensorZero
  or a real MCP server.
- [`production.md`](../production.md) — broader checklist for a
  production deployment of the whole stack.
