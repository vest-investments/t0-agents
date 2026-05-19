# Serving an Agent over HTTP

You have built, composed, and instrumented an agent. `t0_agents.serve` is how you expose it to the outside world. The transport adds no new primitives — it drives the same `Agent.invoke()` and `Run.events()` you already know, behind four HTTP endpoints and an opt-in streaming surface.

`t0_agents.serve` wraps an [`Agent`](agent.md) in a [FastAPI](https://fastapi.tiangolo.com/)
application. A single line — `app = create_app(agent, client_factory=...)` —
turns a working agent script into a service with four non-streaming
endpoints (`POST /chat`, `POST /feedback`, `GET /health`, `GET /info`);
setting `streaming=True` additionally mounts Server-Sent Events
(`GET /chat/stream`) and a bidirectional WebSocket (`WS /chat/ws`) for
live deltas and browser-side elicitation. The full streaming event model
lives in [Streaming](streaming.md); this page covers the HTTP foundation
and the session-wiring layer that both transports share.

## Install

The HTTP layer ships behind the `[serve]` extra:

```bash
pip install 't0-agents[serve]'
```

The extra pulls four dependencies:

| Package | Purpose |
|---|---|
| [`fastapi`](https://fastapi.tiangolo.com/) | HTTP framework |
| [`uvicorn[standard]`](https://www.uvicorn.org/) | ASGI server |
| [`pydantic`](https://docs.pydantic.dev/) | Request/response models |
| [`sse-starlette`](https://github.com/sysid/sse-starlette) | Used only when `streaming=True` |

`t0-agents` without the extra remains fully usable for the bridge
(`from t0_agents.mcp import run_agent`) and the declarative `Agent` —
nothing in `t0_agents.serve` leaks into those import paths.

## Quickstart

Requirements: Docker + `GROQ_API_KEY` (or any provider the gateway is
configured for), plus an MCP server. Throughout the examples we assume
`my_mcp_server.mcp` exposes a few tools and the gateway function
`agent` is declared in `tensorzero.toml`.

```python
# agent_app.py
from typing import cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.serve import create_app

from my_mcp_server import mcp


_t0 = cast(
    AsyncTensorZeroGateway,
    AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
        async_setup=False,
    ),
)
_agent = Agent(
    AgentConfig(name="assistant", function_name="agent"),
    t0=_t0,
    tool_provider=Client(mcp),      # placeholder; factory mints real clients per request
)
app = create_app(_agent, client_factory=lambda: Client(mcp))
```

Run with:

```bash
docker compose up -d
uvicorn agent_app:app --host 0.0.0.0 --port 8080
```

Hit the four endpoints:

```bash
curl -s -X POST localhost:8080/chat \
    -H 'content-type: application/json' \
    -d '{"message": "hello"}'

curl -s localhost:8080/health
# {"gateway":"ok","tool_provider":"per-session"}
```

### What just happened

1. The module imports run `AsyncTensorZeroGateway.build_http(...,
   async_setup=False)` so the gateway handle exists at import time —
   exactly what `uvicorn agent_app:app` needs.
2. `Agent(...)` is constructed with a placeholder `Client(mcp)` that
   satisfies the `ToolProvider` protocol check but is never used by the
   serving layer (per-request clients come from `client_factory`).
3. `create_app(agent, client_factory=...)` returns a `FastAPI`
   application with one router wired in and a lifespan hook that calls
   `agent.warmup()` on startup.
4. For every `POST /chat`, the `SessionFactory` calls `client_factory()`
   to mint a fresh `fastmcp.Client`, enters its async context, runs the
   agent once, and closes the client deterministically on exit — even
   if the HTTP caller disconnected mid-request.
5. `/health` reports `"per-session"` for the tool provider: there is no
   single persistent client to probe in this mode, and that value is a
   healthy signal. See [`HealthResponse.tool_provider`
   semantics](#get-health).

## Concepts

### Architecture

```
┌─────────────────┐   ┌───────────────────────────────────────────────┐   ┌──────────────────┐
│  HTTP caller    │   │                  FastAPI app                  │   │ TensorZero       │
│                 │   │  ┌─────────────────────────────────────────┐  │   │ gateway          │
│  curl / SDK     │ ─►│  │  /chat   /feedback  /health   /info     │  │──►│ (inference,      │
│  browser / CLI  │   │  │        /chat/stream (SSE)               │  │   │  observability)  │
│                 │   │  │        /chat/ws     (WS bidirectional)  │  │   └──────────────────┘
└─────────────────┘   │  └────────────────┬────────────────────────┘  │
                      │                   │                           │
                      │            SessionFactory                     │
                      │                   │                           │   ┌──────────────────┐
                      │  per-session Client / Agent (with fresh       │──►│  FastMCP server  │
                      │  AgentConfig.tags merge for this request)     │   │  (tools)         │
                      │                                               │   └──────────────────┘
                      └───────────────────────────────────────────────┘
```

Each HTTP request — or each SSE stream, or each open WebSocket — is a
**session**. Inside one session the framework owns three things: a
fresh `Client` (via the configured factory/provider), a per-session
`AgentConfig` with shallow-merged tags, and an `Agent` rebuilt around
those values so concurrent callers cannot stomp each other's state.

### Request lifecycle

For `POST /chat` with a tagged payload, end to end:

1. FastAPI validates the JSON body against
   [`ChatRequest`](#pydantic-models).
2. `SessionFactory.session(extra_tags=req.tags)` enters a context.
3. The factory shallow-merges the request tags onto `AgentConfig.tags`
   (which already contains the `deployment_label`, if any), copies the
   config with `dataclasses.replace`, and picks the right session
   mode:
   - **per-session mode** (`client_factory=`): call the factory,
     `async with client` open/close around step 4.
   - **shared mode** (`share_client=True`): reuse the caller-owned
     `Client` directly; no open/close here.
   - **provider mode** (`session_provider=`): delegate to
     `provider.build_client(elicitation_handler=None)`.
4. A fresh `Agent` (of the same class the caller passed, including
   subclasses) is built for this request with the merged config and the
   session client.
5. `await agent.invoke(req.message, episode_id=req.episode_id)` drives the
   non-streaming loop.
6. The client is closed as the context exits. The response is
   serialised into [`ChatResponse`](#pydantic-models) — including the
   `guardrail_log` if any guardrails were configured.

### Why a fresh Client per session

`fastmcp.Client` binds exactly one `elicitation_handler` at
construction. If a single `Client` served many concurrent users,
every `ctx.elicit()` call — no matter which request triggered it —
would hit the same handler with no built-in way to route the answer
back to the originating caller. The serving layer defaults to
per-session construction so elicitation stays demultiplexed, and
offers `share_client=True` as an opt-out for deployments whose tools
never call `ctx.elicit()`.

The same invariant applies across transports: `POST /chat` owns one
Client per HTTP request, `GET /chat/stream` owns one Client per SSE
stream, `WS /chat/ws` owns one Client per WebSocket connection. The
abstraction is the same (`SessionFactory.session(...)`); only the
enclosing context manager differs.

### Concurrency and statelessness

The non-streaming routes keep no server-side session state: any pod
can serve any request, conversation continuity is the caller's
responsibility via `episode_id`. Running many `uvicorn` workers
behind a standard load balancer works without sticky sessions.
The streaming surface changes that contract — see
[Production considerations](#production-considerations).

## API reference

### `create_app`

```python
def create_app(
    agent: Agent,
    *,
    session_provider: SessionProvider | None = None,
    client_factory: Callable[[], Client] | None = None,
    elicitation_client_factory: Callable[[ElicitationHandler], Client] | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
    streaming: bool = False,
) -> FastAPI
```

Build a FastAPI app around `agent`. The session mode is selected by
**exactly one** of `session_provider`, `client_factory`, or
`share_client=True` — any other combination raises
[`ConfigurationError`](error-handling.md) at app construction:

| Combination | Outcome |
|---|---|
| only `client_factory=...` | per-session mode — one fresh client per request |
| only `share_client=True` | shared mode — reuses the agent's tool provider verbatim |
| only `session_provider=...` | provider mode — call `provider.build_client(...)` per session |
| `client_factory` + `share_client` | `ConfigurationError` ("mutually exclusive") |
| none / two+ of the three | `ConfigurationError` ("requires exactly one of …") |
| `elicitation_client_factory` without `client_factory` | `ConfigurationError` ("additive to `client_factory`") |
| `share_client=True` on an agent built without a `tool_provider` (typically `function_kind="json"`) | `ConfigurationError` (nothing to share) — pick `client_factory=...` or `session_provider=...`, or construct the agent with an explicit `tool_provider` |

The rest of the knobs:

| Argument | Default | Purpose |
|---|---|---|
| `elicitation_client_factory` | `None` | Callable receiving the per-connection `ElicitationHandler` and returning a `Client`. Only valid with `client_factory`. Used by the WebSocket path when `streaming=True` to bridge `ctx.elicit()` into the browser. |
| `deployment_label` | `None` | When set, merged into `AgentConfig.tags` as `t0_agents.deployment=<label>` before any session is built, so every inference record (and every feedback record that ferries it back) carries the same observability tag. |
| `streaming` | `False` | Mount `GET /chat/stream` and `WS /chat/ws` in addition to the four non-streaming routes. Leave `False` for batch-only deployments; `True` is the path described in [Streaming](streaming.md). |

The returned `FastAPI` object is ordinary — compose any middleware you
would add to a hand-rolled app (see [Production
considerations](#production-considerations)). The caller runs it with
`uvicorn` (or an equivalent ASGI runner); `create_app` does not start
the server.

### `mount_agent`

```python
def mount_agent(
    parent: FastAPI,
    agent: Agent,
    prefix: str,
    *,
    session_provider: SessionProvider | None = None,
    client_factory: Callable[[], Client] | None = None,
    elicitation_client_factory: Callable[[ElicitationHandler], Client] | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
    streaming: bool = False,
) -> None
```

Mount a per-agent sub-application under `prefix` on an existing
FastAPI app. Each sub-app owns its own lifespan, its own
`SessionFactory`, and its own `deployment_label`, so one process can
host several agents on disjoint URL prefixes without leaking state
between them.

### `SessionFactory`

Advanced: the per-session `(Client, Agent)` builder `create_app`
instantiates internally. Construct it directly only when you need a
`SessionFactory` outside a FastAPI app (for example in a custom ASGI
framework); most callers should stop at `create_app`.

```python
SessionFactory(
    *,
    base_config: AgentConfig,
    t0: AsyncTensorZeroGateway,
    session_provider: SessionProvider | None = None,
    client_factory: Callable[[], Client] | None = None,
    elicitation_client_factory: Callable[[ElicitationHandler], Client] | None = None,
    shared_tool_provider: ToolProvider | None = None,
    agent_class: type[Agent] = Agent,
)
```

Exactly one of `session_provider`, `client_factory`, or
`shared_tool_provider` must be passed — same XOR validation as
`create_app`. `agent_class` defaults to `Agent` but is set to
`type(agent)` by `create_app`, so subclasses survive per-request
reconstruction.

The one method callers typically interact with:

```python
@asynccontextmanager
async def session(
    self,
    *,
    extra_tags: dict[str, str] | None = None,
    elicitation_handler: ElicitationHandler | None = None,
) -> AsyncIterator[Agent]
```

Yields an `Agent` bound to a session-scoped tool provider. `extra_tags`
is shallow-merged on top of `base_config.tags`. Passing
`elicitation_handler` in shared mode raises `ConfigurationError` —
streaming elicitation requires a per-session client.

`SessionFactory` also exposes `startup()` / `shutdown()` async hooks
(no-ops today; the lifespan machinery calls them) and
`probe_tool_provider()` for `/health`.

### `SessionProvider` Protocol

```python
@runtime_checkable
class SessionProvider(Protocol):
    def build_client(
        self,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AbstractAsyncContextManager[Client]: ...
```

One-method hook for session-scoped `Client` construction. Implement
it directly when `client_factory` + `elicitation_client_factory` are
not expressive enough — for example, to inject auth, route per tenant,
attach telemetry spans, or register a bespoke elicitation handler.
`elicitation_handler is None` signals the non-streaming `POST /chat`
path; a non-`None` handler signals the WebSocket streaming path, and
the provider must register it on the returned `Client` or raise
`ConfigurationError`.

### `CallableSessionProvider`

The default adapter `create_app` uses when `client_factory` (and
optionally `elicitation_client_factory`) is passed. Useful directly if
you are plumbing a `SessionFactory` by hand:

```python
from t0_agents.serve import CallableSessionProvider, SessionFactory

provider = CallableSessionProvider(
    client_factory=lambda: Client(mcp),
    elicitation_client_factory=lambda handler: Client(mcp, elicitation_handler=handler),
)
```

### `ClientFactory`, `ElicitationClientFactory`

Type aliases exposed for precise annotations:

```python
ClientFactory = Callable[[], Client]
ElicitationClientFactory = Callable[[ElicitationHandler], Client]
```

### `Agent.warmup()`

```python
class Agent:
    async def warmup(self) -> None: ...
```

Lifecycle hook the FastAPI lifespan invokes on startup. The default is
a no-op — TensorZero's SDK does not expose a public health method, so
any built-in probe would either couple to a per-request client or burn
provider tokens. Override in a subclass to front-load one-time checks:

```python
class WarmableAgent(Agent):
    async def warmup(self) -> None:
        # e.g. a lightweight gateway ping:
        await self._t0.list_inferences(limit=1)
```

Subclasses pass straight through to `create_app`: the session factory
captures `type(agent)` and rebuilds the same class per request.

### `ElicitationForwarder`

Re-exported at the package root (`from t0_agents import
ElicitationForwarder`). Bridges an in-process FastMCP elicitation
handler to an out-of-process resolver (typically a WebSocket frame).
Without an `on_request` callback the forwarder auto-declines every
elicitation, which is what happens on the non-streaming `POST /chat`
path: a tool that calls `ctx.elicit()` sees an explicit decline
instead of hanging. The streaming WebSocket route wires
`on_request=ws.send_json` and resolves futures from incoming frames —
the full lifecycle (pending map, `resolve`, `cancel_all`) lives in
[Streaming](streaming.md).

### Endpoints

| Method | Path | Request | Response | Streaming-only |
|---|---|---|---|---|
| `POST` | `/chat` | `ChatRequest` | `ChatResponse` | no |
| `POST` | `/feedback` | `FeedbackRequest` | `FeedbackResponse` | no |
| `GET` | `/health` | — | `HealthResponse` | no |
| `GET` | `/info` | — | `InfoResponse` | no |
| `GET` | `/chat/stream` | query params: `message`, `episode_id?`, `tags?` (JSON) | SSE stream of `AgentEvent` frames | yes (`streaming=True`) |
| `WS` | `/chat/ws` | `WSUserMessage`, `WSElicitationResponse` | `AgentEvent` frames, `WSElicitationRequest`, `WSErrorFrame` | yes (`streaming=True`) |

The streaming endpoints and their frame shapes are documented in
[Streaming](streaming.md).

#### `GET /health`

`HealthResponse.tool_provider` takes one of three string values:

| Value | Meaning |
|---|---|
| `"ok"` | Shared mode and the underlying `Client.is_connected()` returned true |
| `"down"` | Shared mode and the client reports disconnected, or `is_connected()` raised |
| `"per-session"` | Per-session or provider mode — no persistent client exists to probe, and each request builds its own |

Clients comparing on equality with `"ok"` must treat `"per-session"`
as a healthy state. The value is free-form `str` so subclasses of the
router can report richer states (`"degraded"`, `"starting"`) without
a schema change.

#### `POST /feedback`

A direct passthrough to
[`AsyncTensorZeroGateway.feedback(...)`](https://www.tensorzero.com/docs/gateway/api-reference#post-feedback).
The caller is responsible for pre-declaring `metric_name` in
`tensorzero.toml`; gateway validation happens upstream. Error mapping
differs from `/chat` — see [Error handling](#error-handling).

#### Pydantic models

All six request/response models live in `t0_agents.serve.protocol`:

- `ChatRequest(message: str, episode_id: str | None = None, tags:
  dict[str, str] | None = None)`
- `ChatResponse(response: str, episode_id: str | None = None,
  inference_ids: list[str] = [], turns: int = 0, response_source: str
  = "text", guardrail_log: list[dict[str, Any]] = [])` — mirrors
  `AgentResult` with `guardrail_log` converted from dataclass
  instances to plain dicts.
- `FeedbackRequest(metric_name: str, value: Any, inference_id: str |
  None = None, episode_id: str | None = None, tags: dict[str, str] |
  None = None, dryrun: bool | None = None)` — six fields match the
  gateway's `feedback(...)` kwargs.
- `FeedbackResponse(feedback_id: str)`
- `HealthResponse(gateway: str, tool_provider: str)`
- `InfoResponse(name: str, function_name: str, namespace: str | None =
  None, tags: dict[str, str] | None = None, has_memory: bool,
  guardrails: list[str] = [])` — the non-secret subset of
  `AgentConfig`. `has_memory` is a boolean rather than the backend
  class name so swapping implementations does not leak type names.

## Feature tiers

### Easy — one-line serve

Covered in the [Quickstart](#quickstart) above: `create_app(agent,
client_factory=lambda: Client(mcp))` plus `uvicorn`. Fits in ~15 lines
of Python and a `docker compose up` for the gateway.

### Intermediate — multi-surface hosting, deployment labels, memory continuity

[`examples/agent/09_serving/`](../examples/agent/09_serving/) is the
runnable version. It adds:

- **Memory continuity across requests.** `AgentConfig(memory=InMemoryMemory())`
  hydrates prior turns when the caller re-sends an `episode_id`, so
  `client.sh`'s follow-up (`"what code did I give you?"`) answers
  correctly. For process-restart-safe persistence, swap to
  `SqliteMemory(".db")` from the `[memory-sqlite]` extra — see
  [Memory](memory.md).
- **Deployment label tagging.** `create_app(agent,
  client_factory=..., deployment_label="demo")` adds
  `tags["t0_agents.deployment"]="demo"` to every inference, observable
  via `GET /info` and queryable in ClickHouse.
- **Feedback round-trip.** `curl -X POST /feedback -d '{"metric_name":
  "accuracy", "value": 1.0, "inference_id": "..."}'` emits a metric
  correlated with a specific turn of a specific request.

**Multi-surface hosting** on one `uvicorn` process:

```python
from fastapi import FastAPI
from fastmcp import Client

from t0_agents.serve import mount_agent


parent = FastAPI()
mount_agent(
    parent, mobile_agent, "/mobile",
    client_factory=lambda: Client(mobile_mcp),
    deployment_label="mobile",
)
mount_agent(
    parent, analytics_agent, "/analytics",
    client_factory=lambda: Client(analytics_mcp),
    deployment_label="analytics",
)
# uvicorn module:parent --port 8080
```

`/mobile/chat`, `/analytics/chat`, and their siblings now route to
their respective agents. Each mounted sub-app runs its own lifespan
and its own `SessionFactory`; tags and `deployment_label`s stay
isolated per prefix.

**Sharing a client** — for agents whose tools never call
`ctx.elicit()`, skip the per-request open/close overhead:

```python
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp import Client

from t0_agents import Agent, AgentConfig
from t0_agents.serve import create_app


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with Client(mcp) as client:
        agent = Agent(AgentConfig(name="assistant", function_name="agent"),
                      t0=t0, tool_provider=client)
        nested = create_app(agent, share_client=True)
        yield {"inner_app": nested}
```

In shared mode the framework never calls `__aenter__` / `__aexit__`
on the client on your behalf — you own its lifecycle. Elicitation
attempts are declined by the forwarder with an explicit error; the
LLM sees it as a tool failure rather than a hang.

### Advanced — custom session provider, streaming, live elicitation

The `SessionProvider` Protocol is the single extension point for
per-session `Client` construction. Implement it directly whenever
`client_factory` + `elicitation_client_factory` are not enough —
authentication, per-tenant routing, telemetry spans, or bespoke
elicitation wiring all live here:

```python
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastmcp import Client
from fastmcp.client.elicitation import ElicitationHandler

from t0_agents.serve import SessionProvider, create_app


class AuthedSessionProvider:
    """Inject per-session auth and telemetry spans."""

    def __init__(self, mcp, tracer) -> None:
        self._mcp = mcp
        self._tracer = tracer

    def build_client(
        self,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AbstractAsyncContextManager[Client]:
        @asynccontextmanager
        async def _session():
            span = self._tracer.start_span("mcp.session")
            try:
                client = Client(self._mcp, elicitation_handler=elicitation_handler)
                async with client:
                    yield client
            finally:
                self._tracer.end_span(span)

        return _session()


app = create_app(
    agent,
    session_provider=AuthedSessionProvider(mcp, tracer),
    streaming=True,
)
```

The Protocol is `@runtime_checkable` — `isinstance(x,
SessionProvider)` works at `SessionFactory` construction time and
raises `ConfigurationError` on mismatch.

**Streaming and live elicitation.** Add `streaming=True` to
`create_app(...)` to mount Server-Sent Events (`GET /chat/stream`)
and a bidirectional WebSocket (`WS /chat/ws`) alongside the
non-streaming routes. The WS path is the only browser-compatible
human-in-the-loop transport today: `ctx.elicit()` emits a
`WSElicitationRequest` frame, the browser responds with a
`WSElicitationResponse`, and the `ElicitationForwarder` resolves the
pending future. The full event model, frame shapes, backpressure
contract, and browser cookbook live in [Streaming](streaming.md);
[`examples/agent/07_streaming/`](../examples/agent/07_streaming/) ships
a runnable vanilla-JS client that covers every frame type.

## Error handling

Each endpoint has its own exception-to-status-code contract.

### `POST /chat`

| Python exception | HTTP status | Notes |
|---|---|---|
| `MaxTurnsExceededError` | `408 Request Timeout` | Loop exhausted `AgentConfig.max_turns` without a tool-call-free turn |
| `GuardrailBlockedError` | `409 Conflict` | A guardrail hard-aborted the run; `AgentResult.guardrail_log` is *not* returned on this path (the run did not complete) |
| `ConfigurationError` | `400 Bad Request` | Typically a misconfigured `SessionFactory` surfacing at session entry |
| `TensorZeroError` | `502 Bad Gateway` | Any gateway failure — regardless of upstream status code, since the `/chat` payload does not map 1:1 to a gateway request |
| Other `T0AgentsError` | `400 Bad Request` | Catch-all for framework-raised errors not covered above |

### `POST /feedback`

The `/feedback` body maps one-to-one to the gateway's `feedback(...)`
kwargs (`metric_name`, `value`, `inference_id`, `episode_id`, `tags`,
`dryrun`). That makes a 4xx from the gateway honestly a 4xx from the
HTTP caller's perspective — "unknown metric" or "inference_id not
found" is a caller error, not an upstream failure.

| Upstream | HTTP status |
|---|---|
| Gateway returns 4xx (e.g. 404 for unknown metric) | Forwarded verbatim — `404 Not Found`, `400 Bad Request`, etc. |
| Gateway returns 5xx, or `TensorZeroError.status_code` is absent | `502 Bad Gateway` |
| Success | `200 OK` with `FeedbackResponse(feedback_id=...)` |

`/chat` does *not* use this passthrough rule because the `/chat`
request body (`{"message": "..."}`) does not map directly to a
gateway request — a 4xx from the gateway would mislead the caller
into thinking their `/chat` payload was invalid.

### `GET /health`, `GET /info`

Always `200 OK` under normal operation. Uncaught exceptions surface as
FastAPI's default `500 Internal Server Error`.

### Streaming error surface

On `/chat/stream` (SSE) and `/chat/ws` (WebSocket), `MaxTurnsExceededError`,
`GuardrailBlockedError`, and other `T0AgentsError` subclasses surface
as `ErrorEvent` / `WSErrorFrame` frames instead of HTTP status codes —
the stream stays open long enough to transmit the error, then
terminates cleanly. See [Streaming — Error
handling](streaming.md#error-handling).

See [Error handling](error-handling.md) for the library-wide
exception hierarchy and the `T0AgentsError` / `T0AgentsFatalError`
split.

## Production considerations

### Statelessness and scaling

The four non-streaming routes keep no server-side session state.
`episode_id` is the caller's responsibility — the gateway correlates
inferences in ClickHouse on its own. Any pod can serve any request,
and horizontally scaling the service is as simple as running more
`uvicorn` workers behind a standard load balancer.

### Sticky sessions with `streaming=True`

The WebSocket elicitation path requires sticky sessions. When a tool
calls `ctx.elicit()`, the framework allocates an in-memory future keyed
by `elicitation_id`; the `WSElicitationResponse` that resolves it must
reach the same pod. Plan load-balancer affinity when enabling
streaming (GKE Ingress supports `sessionAffinity: CLIENT_IP`; most
cloud LBs expose a similar knob). SSE streams on `/chat/stream` are
one-way — they do not require sticky sessions but benefit from the
same affinity setting when one client opens many streams in sequence.

### Timeouts

Long-running non-streaming `POST /chat` calls interact poorly with
proxy-level idle-connection cutoffs (for example GCP HTTP(S) Load
Balancer closes idle connections after ~30 s). Match `uvicorn
--timeout-keep-alive` / `--timeout-graceful-shutdown` to bracket your
upstream LB's idle timeout. Agents whose runs may exceed 30 s should
use `streaming=True` so the SSE / WS transport emits progress frames
and stays below the LB's idle threshold.

### BYO middleware — auth, CORS, rate limiting

`create_app` returns a plain `FastAPI` instance — compose any
middleware you would add to a hand-rolled service. For an API-key
gate:

```python
import os

from fastapi import Depends, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware

from t0_agents.serve import create_app


app = create_app(agent, client_factory=lambda: Client(mcp))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(req: Request) -> None:
    if req.headers.get("x-api-key") != os.environ["SERVICE_API_KEY"]:
        raise HTTPException(status_code=401, detail="bad api key")


app.router.dependencies.append(Depends(require_api_key))
```

For rate limiting, plug [`slowapi`](https://slowapi.readthedocs.io/),
[`fastapi-limiter`](https://pypi.org/project/fastapi-limiter/), or a
reverse-proxy bucket in front of `uvicorn`. TensorZero's
`[rate_limiting.rules]` in `tensorzero.toml` are complementary: they
bucket at the gateway level (cross-request, cross-pod), whereas HTTP
middleware buckets per caller. See
[Guardrails](guardrails.md) for per-run quotas (`max_cost`,
`max_tool_calls`) that apply regardless of transport.

### Logging

The library installs a `NullHandler` on `t0_agents.*` loggers;
applications attach their own handler in the runner (for example
`uvicorn --log-config logging.json`). Every `Agent.invoke` log record
carries `agent_name` / `function_name` metadata as `extra` fields.

See [Production](production.md) for the library-wide production
checklist — gateway HA, ClickHouse sizing, secret management.

## Observability

Every HTTP session emits the same telemetry as its in-process
counterpart:

- **`deployment_label`** — the canonical "which surface made this
  call?" tag. Set it once at `create_app`/`mount_agent` time; it
  reaches ClickHouse as `tags["t0_agents.deployment"]` on every
  inference and every feedback record. Use it to split cost and
  latency by surface (mobile vs analytics vs batch workers).
- **`AgentConfig.namespace`** — the experimentation key TensorZero
  applies when sampling variants. Namespace and `deployment_label` are
  orthogonal: namespace routes to a variant-sampling group, the label
  is metadata. Unknown namespaces fall back to the default config
  silently.
- **Per-request `tags`** — `ChatRequest.tags` shallow-merge on top of
  `AgentConfig.tags` for *this* request only; the merged set is
  observable on every inference the request produces.
- **Episode IDs** — `ChatResponse.episode_id` is the correlation key
  threaded back through subsequent requests. All turns of a multi-
  turn conversation appear under the same episode in ClickHouse.
- **Inference IDs** — `ChatResponse.inference_ids` is the ordered
  list of per-turn IDs. `POST /feedback` with an `inference_id`
  attaches the metric to that specific turn; with an `episode_id` it
  attaches to the run as a whole.

See [Observability](observability.md) for canonical ClickHouse
queries and feedback-emission patterns.

## External references

- **[FastAPI](https://fastapi.tiangolo.com/)** — the ASGI web
  framework `create_app` returns an instance of; middleware,
  dependencies, and mount semantics come from here.
- **[Uvicorn](https://www.uvicorn.org/)** — the ASGI server pulled
  in by the `[serve]` extra; any ASGI-compatible runner works, but
  Uvicorn is the tested default.
- **[Pydantic v2](https://docs.pydantic.dev/)** — request/response
  validation; every model in `t0_agents.serve.protocol` and
  `t0_agents.serve.ws_protocol` is a `BaseModel`.
- **[sse-starlette](https://github.com/sysid/sse-starlette)** —
  `EventSourceResponse` is used by `GET /chat/stream` when
  `streaming=True`.
- **[TensorZero inference API](https://www.tensorzero.com/docs/gateway/api-reference#post-inference)**
  — the request shape `Agent.invoke` issues through the gateway.
- **[TensorZero feedback API](https://www.tensorzero.com/docs/gateway/api-reference#post-feedback)**
  — what `POST /feedback` is a passthrough to.
- **[FastMCP](https://gofastmcp.com/)** — `Client(mcp)` is the tool
  provider the session factory builds per request; the elicitation
  handler constraint that motivates per-session construction is
  documented there.

---

The FastAPI layer covered above is the right target for production deployments. For development, quick demos, and local exploration, the `[cli]` extra ships a `t0-agents` console script that wraps the same `create_app(...)` behind `t0-agents serve`, adds an interactive `run` REPL that consumes `Run.events()` directly, and an `info` command that introspects any loaded agent factory. That is the subject of the next chapter: [CLI](cli.md).

## See also

- [`examples/agent/09_serving/`](../examples/agent/09_serving/) — runnable
  single-agent demo with memory continuity and deployment tagging.
- [`examples/agent/07_streaming/`](../examples/agent/07_streaming/) —
  runnable browser demo covering SSE, WebSocket, and live elicitation
  when `streaming=True`.
- [Streaming](streaming.md) — `Run.events()` event model, SSE / WS
  frame shapes, `SessionProvider` recipes for streaming.
- [Agent](agent.md) — the `Agent` class and its `AgentConfig` fields.
- [Memory](memory.md) — cross-request memory via `AgentConfig.memory`.
- [Guardrails](guardrails.md) — policy layer applied inside
  `Agent.invoke`.
- [Multi-agent composition](multi-agent.md) — sub-agents registered
  via `AgentConfig.sub_agents`.
- [CLI](cli.md) — `t0-agents serve` wraps `create_app` with
  `share_client=True` defaults for quick deployments.
- [Observability](observability.md) — ClickHouse queries, dashboards,
  feedback patterns.
- [Error handling](error-handling.md) — library-wide exception
  hierarchy and recovery patterns.
- [Production](production.md) — broader production checklist.
