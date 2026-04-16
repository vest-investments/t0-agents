# Serving an Agent over HTTP

`t0_agents.serve` wraps an `Agent` in a FastAPI application with four
endpoints.  Install the extra to pull in FastAPI, Uvicorn, and
Pydantic:

```bash
pip install 't0-agents[serve]'
```

## Quick start

```python
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.serve import create_app

from my_mcp_server import mcp

t0 = AsyncTensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
    async_setup=False,
)
agent = Agent(
    AgentConfig(name="assistant", function_name="agent"),
    t0=t0,
    tool_provider=Client(mcp),
)
app = create_app(agent, client_factory=lambda: Client(mcp))
```

Run with:

```bash
uvicorn my_module:app --host 0.0.0.0 --port 8080
```

Hit the four endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/chat` | `{"message": "...", "episode_id": "?", "tags": {}}` | `AgentResult` JSON |
| `POST` | `/feedback` | `{"metric_name": "...", "value": ..., "inference_id"?: "...", "episode_id"?: "...", "tags"?: {}, "dryrun"?: false}` | `{"feedback_id": "..."}` |
| `GET` | `/health` | — | `{"gateway": "ok", "tool_provider": "ok" \| "down" \| "per-session"}` |
| `GET` | `/info` | — | Redacted `AgentConfig` snapshot |

`AgentResult` over the wire mirrors the in-process dataclass with
`guardrail_log` serialized as a list of plain dicts.

## Multi-agent in one process

Expose several agents on disjoint URL prefixes:

```python
from fastapi import FastAPI
from t0_agents.serve import mount_agent

app = FastAPI()
mount_agent(app, mobile_agent,     "/mobile",
            client_factory=lambda: Client(mobile_mcp),
            deployment_label="mobile")
mount_agent(app, analytics_agent,  "/analytics",
            client_factory=lambda: Client(analytics_mcp),
            deployment_label="analytics")
```

`/mobile/chat`, `/analytics/chat`, and their siblings now run against the
same gateway.  Each mounted sub-app owns its own
`SessionFactory` and lifespan; no state leaks across prefixes.

## Configuration knobs

| Argument | Default | Purpose |
|---|---|---|
| `client_factory` | *(required unless `share_client=True`)* | Zero-arg callable that mints a fresh `fastmcp.Client` for every session |
| `share_client` | `False` | Reuse `agent.tool_provider` across every session.  Mutually exclusive with `client_factory`; see trade-off below |
| `deployment_label` | `None` | When set, merged into `AgentConfig.tags` as `t0_agents.deployment=<label>` before every inference.  Doubles as the canonical "which surface made this call" signal on every observability record (resolved T0 Q3) |

Passing neither or both raises `ConfigurationError` at app construction —
there is no silent fallback.

## Per-surface deployment recipe

A common pattern is "one agent server per surface, one gateway
underneath":

```
Mobile app  ──► mobile server  (tags=t0_agents.deployment=mobile)
Analytics   ──► analytics server (tags=t0_agents.deployment=analytics,
                                  namespace=analytics)
                              │
                              ▼
                   one TensorZero gateway
                   (config/tensorzero.toml)
```

* Set **`deployment_label`** on each surface so ClickHouse can split
  cost and latency.
* Set **`namespace`** on `AgentConfig` when a surface needs different
  variant-sampling groups (`[functions.agent.namespaces.analytics]`
  in `tensorzero.toml`).  Unknown namespaces fall back to the default
  config silently (resolved T0 Q3).
* Keep **one `tensorzero.toml`** for all surfaces.  Namespace-specific
  experimentation blocks are opt-in and do not break the default
  sampling.

## Concurrency model — why fresh Clients per session

`fastmcp.Client` binds exactly one `elicitation_handler` at
construction time (`elicitation_callback` in
`_session_kwargs`).  Sharing one Client across users would collapse
every `ctx.elicit()` onto the single handler with no way to route the
response back to the originating request — the resolution of FastMCP
Q3.

`SessionFactory.session()` enforces the safe default:

```python
async with session_factory.session(extra_tags=tags) as agent:
    result = await agent.run(message, episode_id=episode_id)
```

The `async with` block owns the Client's lifetime.  In Phase D this
block is a single HTTP request; in Phase E (streaming) the same
primitive scopes one SSE stream or one WebSocket connection without a
shape change.

### `share_client=True` trade-off

For deployments that never use elicitation (read-only agents,
analytics assistants), skip the per-request open/close overhead:

```python
client = Client(mcp)
async with client:            # you own the lifecycle
    agent = Agent(cfg, t0=t0, tool_provider=client)
    app = create_app(agent, share_client=True)
    # uvicorn ...
```

* No factory is called; `agent.tool_provider` is reused verbatim.
* Any elicitation attempt is declined by the Phase-D forwarder with an
  explicit error — the LLM sees it as a tool failure, not a hang.
* The caller retains full control of the Client's lifecycle; the
  framework does *not* call `__aenter__` / `__aexit__` on your behalf
  in shared mode.

## Streaming and live elicitation — Phase E

HTTP request/response cannot round-trip an elicitation prompt back to
the caller; the Phase D default forwarder declines explicitly.  Phase E
(M5 — Live Agent) adds SSE + WebSocket transports and swaps in a live
`ElicitationForwarder.resolve()` resolver.  The public API for wiring a
forwarder into a `Client` is already exposed:

```python
from t0_agents import ElicitationForwarder

forwarder = ElicitationForwarder()
client = Client(mcp, elicitation_handler=forwarder.handler)
```

Code written against Phase D keeps working after Phase E lands; the
only difference is which `handler` the forwarder installs.

## Auth, rate limiting, observability — BYO middleware

`create_app` returns a standard FastAPI application — compose any
middleware you would add to a hand-rolled service.

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware

app = create_app(agent, client_factory=...)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

async def require_api_key(req: Request) -> None:
    if req.headers.get("x-api-key") != os.environ["SERVICE_API_KEY"]:
        raise HTTPException(status_code=401, detail="bad api key")

# Install on every route (including /chat /feedback /health /info)
app.router.dependencies.append(Depends(require_api_key))
```

For rate limits, plug `slowapi` / `fastapi-limiter` / a reverse-proxy
bucket in front of uvicorn.  Gateway-side rate limits (TensorZero
v2026.4.0 `[rate_limiting.rules]`) are complementary — gateway limits
are cross-request and persist across workers, whereas Phase C's
`max_cost` and `max_tool_calls` guardrails are per-run and in-memory.

## Deployment notes (GKE, cloud LBs)

* **Phase D is stateless.**  No server-side session store, no sticky
  cookies needed.  Any pod can serve any request.
* **Phase E will require sticky sessions** once SSE / WebSocket
  transports land — an `/elicit/<id>` answer must reach the pod
  holding the in-memory `Future`.  Plan load-balancer affinity now
  (GKE Ingress: `session_affinity: CLIENT_IP` or a header-based
  policy).
* **Match uvicorn vs. LB timeouts.**  Default GCP HTTP(S) LB closes
  idle connections after ~30s; uvicorn's `--timeout-keep-alive` /
  `--timeout-graceful-shutdown` should bracket that.  For agents
  whose runs may exceed 30s, wait for Phase E streaming — long-running
  non-streaming HTTP calls will be cut silently by most production LBs.
* **Structured logs.**  The framework installs a `NullHandler` on
  `t0_agents.*` loggers; attach your own handler in the runner
  (e.g. `uvicorn --log-config logging.json`).  Every log record carries
  `agent_name` / `function_name` metadata once the request reaches
  `Agent.run`.

## See also

* `docs/agent.md` — the underlying `Agent` class.
* `docs/memory.md` — persistent state across requests.
* `docs/guardrails.md` — declarative policy layer.
* `docs/observability.md` — how `namespace` and `tags` surface in
  ClickHouse.
* `examples/11_http_agent/` — runnable single-agent demo.
