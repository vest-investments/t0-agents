# Phase D — Serving (HTTP, non-streaming)

> Status: **ready to start**. Phase deps: A, B, C complete.
> Open questions resolved: T0 Q3 (namespace + tags orthogonal),
> T0 Q6 (no episode close → idle GC), FastMCP Q2 (build own
> forwarder), FastMCP Q3 (per-session Client mandatory).
> Delivers: **Milestone M4 — Hosted Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md) §3.5
  (concurrency), §7 (resolved Qs)
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q3, Q6;
  FastMCP Q2, Q3
- [`../problemas.md`](../problemas.md) — Problem 5
- [`phase-a-agent-core.md`](phase-a-agent-core.md),
  [`phase-b-memory.md`](phase-b-memory.md),
  [`phase-c-guardrails.md`](phase-c-guardrails.md)

---

## 1. Goal

Expose an `Agent` as an HTTP microservice. One line of code turns a
script into a service. Request/response only in this phase; SSE +
WebSocket streaming lands in Phase E. Designed around the user's
"deploy specific agents per surface" use case: one process can host
one agent (most common), and a process can equally host multiple
named agents on disjoint URL prefixes.

After Phase D:

```python
from fastapi import FastAPI
from t0_agents import Agent, AgentConfig
from t0_agents.serve import create_app, mount_agent

# Single-agent server
agent = Agent(AgentConfig(name="assistant", function_name="agent"), t0=t0, tool_provider=...)
app = create_app(agent)        # FastAPI app with /chat, /feedback, /health
# uvicorn this:app --host 0.0.0.0 --port 8080

# Multi-agent server (mounting on existing FastAPI)
app = FastAPI()
mount_agent(app, mobile_agent, prefix="/mobile")
mount_agent(app, analytics_agent, prefix="/analytics")
```

Endpoints v1 (non-streaming):

| Method | Path | Body / Query | Returns |
| --- | --- | --- | --- |
| `POST` | `/chat` | `{"message": "...", "episode_id": "?"}` | `AgentResult` JSON. |
| `POST` | `/feedback` | `{"metric_name": "...", "inference_id" \| "episode_id": "...", "value": ...}` | `{"feedback_id": "..."}`. |
| `GET` | `/health` | — | `{"gateway": "ok", "tool_provider": "ok"}`. |
| `GET` | `/info` | — | `AgentConfig` redacted summary (name, function_name, namespace, no secrets). |

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/serve/` with:
  - `app.py` — `create_app(agent)`, `mount_agent(app, agent, prefix)`.
  - `routes.py` — request/response handlers.
  - `protocol.py` — Pydantic models for request/response bodies.
  - `session.py` — per-request `Client` factory + `Agent` factory
    (the per-session Client mandate from FastMCP Q3).
  - `lifespan.py` — startup/shutdown hooks; warm the gateway connection.
- Extension: nothing on `AgentConfig`. The serving layer wraps an
  existing `Agent` instance.
- New `core/elicitation_forwarder.py` (per FastMCP Q2) — Future-based
  forwarder; Phase E uses it for live WS-piped elicitation. Phase D
  ships the module but defaults to `auto_decline_handler` for the
  non-streaming server (elicitation tools won't work over plain HTTP
  request/response). Documented limitation.
- `[serve]` extra: `fastapi`, `uvicorn[standard]`, `pydantic`.
- Tests under `tests/test_serve/` — using `httpx.AsyncClient` against
  the FastAPI app.
- Example `examples/11_http_agent/`.
- Public doc page `docs/serving.md`.

### 2.2 Out of scope

- SSE / WebSocket streaming → Phase E.
- Elicitation forwarding to a live human → Phase E (depends on
  WS).
- Auth / authz. The framework provides extension points (FastAPI
  middleware composes naturally); no built-in JWT/OIDC.
- Rate limiting. Same — bring-your-own middleware.
- Multi-process worker pooling. Single-process default; production
  deploys add Gunicorn + Uvicorn workers as standard FastAPI.

### 2.3 Non-goals

- Do not invent a "session ID" header. Use `episode_id` for
  conversation continuity (idiomatic with TensorZero) and rely on
  the caller to track it.
- Do not auto-emit feedback. Same rationale as Phase C (T0 Q5).

---

## 3. Constraints (from the contract)

1. `mcp/` stays standalone.
2. `serve/` depends on `core/`, `memory/` (optional), `guardrails/`
   (optional). It does NOT import `stream/` (Phase E adds that
   dependency separately).
3. **Per-session `Client` mandatory.** Per FastMCP Q3 + contract
   §3.5: any agent with a tool that uses elicitation must have its
   own Client per session. Phase D enforces this by default —
   `session.py` constructs one Client per request. Sharing is opt-in
   via a `share_client=True` flag (`session.py:42`) for non-elicitation
   workloads.
4. TensorZero-first: `namespace` and `tags` reach the gateway on
   every `inference()` call (resolved T0 Q3). Both are ferried from
   `AgentConfig`; the serving layer additionally auto-adds
   `tags["t0_agents.deployment"]` if a deployment label is configured
   at app construction.

---

## 4. Module additions

### 4.1 `src/t0_agents/serve/protocol.py` (new)

```python
from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    message: str
    episode_id: str | None = None
    tags: dict[str, str] | None = None    # additive to AgentConfig.tags


class ChatResponse(BaseModel):
    response: str
    episode_id: str
    inference_ids: list[str]
    turns: int
    response_source: str
    guardrail_log: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    metric_name: str
    value: Any
    inference_id: str | None = None
    episode_id: str | None = None
    tags: dict[str, str] | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str


class HealthResponse(BaseModel):
    gateway: str
    tool_provider: str


class InfoResponse(BaseModel):
    name: str
    function_name: str
    namespace: str | None = None
    tags: dict[str, str] | None = None
    has_memory: bool
    guardrails: list[str]
```

### 4.2 `src/t0_agents/serve/session.py` (new)

```python
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client


class SessionFactory:
    """Manages per-request lifecycle of Client + Agent.

    Default: per-session Client (FastMCP Q3 mandate).
    Opt-out: share_client=True only when the developer asserts no
    elicitation tools are in play.
    """

    def __init__(self, agent_factory, *, share_client: bool = False):
        self._agent_factory = agent_factory
        self._share_client = share_client
        self._shared_client: Client[Any] | None = None

    @asynccontextmanager
    async def get(self):
        if self._share_client:
            if self._shared_client is None:
                self._shared_client = await self._agent_factory.build_client()
            yield self._agent_factory.attach(self._shared_client)
            return

        client = await self._agent_factory.build_client()
        try:
            async with client:
                yield self._agent_factory.attach(client)
        finally:
            ...
```

### 4.3 `src/t0_agents/serve/routes.py` (new)

Standard FastAPI routes wrapping the SessionFactory:

```python
from fastapi import APIRouter, HTTPException

from t0_agents.serve.protocol import (
    ChatRequest, ChatResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, InfoResponse,
)


def build_router(agent, session_factory) -> APIRouter:
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        async with session_factory.get() as session_agent:
            # Merge per-request tags into per-Agent tags
            session_agent.config.tags = {
                **(agent.config.tags or {}),
                **(req.tags or {}),
            }
            try:
                result = await session_agent.run(
                    req.message, episode_id=req.episode_id
                )
            except T0AgentsError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return ChatResponse(...)

    @router.post("/feedback", response_model=FeedbackResponse)
    async def feedback(req: FeedbackRequest):
        # Direct passthrough to t0.feedback — gateway validates metric
        ...

    @router.get("/health", response_model=HealthResponse)
    async def health():
        ...

    @router.get("/info", response_model=InfoResponse)
    async def info():
        ...

    return router
```

### 4.4 `src/t0_agents/serve/app.py` (new)

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI


def create_app(agent, *, deployment_label: str | None = None) -> FastAPI:
    """Entry point: wraps an Agent in a FastAPI app."""
    if deployment_label:
        agent.config.tags = {
            **(agent.config.tags or {}),
            "t0_agents.deployment": deployment_label,
        }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await agent.warmup()  # health-check the gateway
        yield
        # Nothing to clean up at app level — sessions self-clean

    app = FastAPI(lifespan=lifespan, title=f"t0-agents · {agent.config.name}")
    session_factory = SessionFactory(...)
    app.include_router(build_router(agent, session_factory))
    return app


def mount_agent(app: FastAPI, agent, prefix: str, **kwargs):
    """Mount a per-agent sub-app under prefix."""
    sub_app = create_app(agent, **kwargs)
    app.mount(prefix, sub_app)
```

### 4.5 `src/t0_agents/core/elicitation_forwarder.py` (new — Phase D ships skeleton, Phase E completes)

```python
import asyncio
from dataclasses import dataclass


@dataclass
class ElicitationRequest:
    elicitation_id: str
    message: str
    response_type: type | None
    params: dict


class ElicitationForwarder:
    """Future-based bridge between an in-process FastMCP elicitation
    handler and an out-of-process channel (WS in Phase E).

    Phase D ships an inert version: forwarder.handler() is set on the
    Client to auto_decline_handler. Phase E swaps in a real
    WS-piped resolver.
    """

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}

    async def handler(self, message, response_type, params, context):
        from fastmcp.client.elicitation import ElicitResult
        return ElicitResult(action="decline")  # Phase D default

    # Phase E will add resolve(), forward(), wait_for(), ...
```

---

## 5. Tests

Location: `tests/test_serve/`.

### 5.1 Unit (offline; FastAPI TestClient)

- [ ] `test_chat_round_trip` — POST `/chat` → 200 + valid response;
      mock `Agent.run` to assert kwargs forwarded.
- [ ] `test_feedback_round_trip` — POST `/feedback` → 200; mock
      `t0.feedback` to assert call.
- [ ] `test_health_endpoint` — both subsystems ok / fail cases.
- [ ] `test_info_endpoint` — redacts no-secrets, lists guardrails by
      name.
- [ ] `test_per_session_client_default` — multiple concurrent
      requests get distinct `Client` instances.
- [ ] `test_share_client_opt_in` — opt-in flag yields one shared
      Client; doc-test that elicitation is incompatible.
- [ ] `test_namespace_and_tags_forwarded` — both reach the
      `t0.inference` mock with merged values.
- [ ] `test_deployment_label_auto_tag` — `create_app(..., deployment_label="mobile")`
      adds `t0_agents.deployment=mobile` to every inference's tags.

### 5.2 Integration (live gateway)

- [ ] `test_live_chat` — boot uvicorn in a subprocess,
      `httpx.AsyncClient` POSTs `/chat`, asserts a non-empty response.
- [ ] `test_episode_continuity_over_http` — first POST returns an
      `episode_id`; second POST uses it; assistant remembers.
- [ ] `test_concurrent_users` — 10 simultaneous `/chat` POSTs with
      different `episode_id`s — all succeed without cross-contamination.

### 5.3 Regression

- [ ] Earlier phases' tests untouched.
- [ ] No new top-level dep — only `[serve]` extra.

---

## 6. Example

Location: `examples/11_http_agent/`.

- `server.py` — same FastMCP server pattern (no elicitation tools in
  this example to keep it clean; elicitation-over-HTTP comes in M5).
- `agent_app.py` — wires `Agent` + `create_app`, exposes `app`.
- `run.sh` — `uvicorn agent_app:app --host 0.0.0.0 --port 8080`.
- `client.sh` — `curl` examples for each endpoint.
- `README.md` — covers single-process and `mount_agent` patterns;
  documents that elicitation tools require Phase E (M5).

---

## 7. Documentation

- New: `docs/serving.md`.
- Sections:
  - "One-line server" — quickstart.
  - "Multi-agent in one process" — `mount_agent`.
  - "Configuration knobs" — deployment_label, share_client.
  - "Per-surface deployment recipe" — explicit walkthrough of the
    user's mobile-vs-analytics use case (different deployment labels,
    same gateway, optional namespace per surface).
  - "Concurrency model" — per-session Client, why elicitation needs
    it.
  - "What about streaming?" — pointer to Phase E.
  - "Auth, rate limiting, observability" — bring-your-own middleware
    examples.
- Update `developer-guide.md` §4 (public API), §12 (knobs).
- `CHANGELOG.md` — `0.5.0`; call out the `[serve]` extra.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/serve/__init__.py`.
- [ ] `src/t0_agents/serve/app.py` (`create_app`, `mount_agent`).
- [ ] `src/t0_agents/serve/routes.py`.
- [ ] `src/t0_agents/serve/protocol.py`.
- [ ] `src/t0_agents/serve/session.py`.
- [ ] `src/t0_agents/serve/lifespan.py`.
- [ ] `src/t0_agents/core/elicitation_forwarder.py` (skeleton; Phase
      E completes the real resolver).
- [ ] `src/t0_agents/core/agent.py` — add `Agent.warmup()` (calls
      `t0` `health`-style ping), no-op if already healthy.
- [ ] `pyproject.toml` — `[serve] = ["fastapi>=0.115.0", "uvicorn[standard]>=0.32.0", "pydantic>=2.0"]`.
- [ ] `uv run ruff check src/`, `uv run mypy src/` green.

### 8.2 Tests

- [ ] All §5.1 unit tests with FastAPI TestClient.
- [ ] §5.2 integration tests against live gateway.

### 8.3 Example

- [ ] `examples/11_http_agent/` complete and runnable.

### 8.4 Docs

- [ ] `docs/serving.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `phase-d-implementation-log.md` with M4 release-notes snippet
      and the "Phase E pre-requisite" tracking note (forwarder
      skeleton landed).

---

## 9. Acceptance criteria (Milestone M4)

1. `from t0_agents.serve import create_app, mount_agent` works.
2. A user follows `docs/serving.md` and exposes their agent on
   `:8080` with `uvicorn`. `curl POST /chat` works.
3. The two-surface example (mobile + analytics) demonstrates that
   `tags["t0_agents.deployment"]` reaches ClickHouse — verified via
   `t0` UI / SQL.
4. Concurrent users do not cross-contaminate.
5. Phases A–C tests still green.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Per-session Client overhead (cold start per request) | medium | Document; benchmark in §11. Provide `share_client=True` opt-out for non-elicitation workloads. |
| Long-running tools time out the HTTP request | medium | Default `httpx`/`uvicorn` timeout is generous; document the knob. Tools needing >30s should use Phase E (streaming) for progress. |
| Elicitation tools used over plain HTTP fail silently | medium | `auto_decline_handler` is the Phase D default → tools fail with explicit error string. Doc warns that elicitation needs Phase E. |
| FastAPI version drift breaks routes | low | Pin minimum version in extra; lifecycle hook compatible with FastAPI 0.115+. |

---

## 11. Estimated surface

- **New files:** 6 src + 4 test + 5 example + 1 doc.
- **Modified files:** 3 (`core/agent.py` for `warmup`,
  `pyproject.toml`, `CHANGELOG.md`).
- **Net new public symbols:** 3 (`create_app`, `mount_agent`,
  `ElicitationForwarder`).
- **New deps:** 3, all under `[serve]` extra (`fastapi`,
  `uvicorn[standard]`, `pydantic`).
