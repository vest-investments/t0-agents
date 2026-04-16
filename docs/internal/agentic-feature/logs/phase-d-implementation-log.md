# Phase D Implementation Log

Phase: D (Serving — HTTP, non-streaming)
Milestone: M4 — Hosted Agent
Date: 2026-04-16

## Divergences from Spec

### D1: `Agent.warmup()` ships as a no-op stub

**Spec (§4.4, §8.1)** calls for `Agent.warmup()` to issue a "t0
health-style ping" during FastAPI startup.

**Actual:** The method is an async no-op by default.  Subclasses may
override it to add custom probes; the framework does not ship an
active probe today.

**Reason.** The TensorZero SDK v2026.4.0 exposes no public health API
— validated at `.venv/lib/python3.12/site-packages/tensorzero/tensorzero.pyi:1017`.
The only alternatives are:

1. `list_inferences(limit=1)` — hits ClickHouse directly; not a
   gateway probe.
2. `inference(function_name=..., dryrun=True)` — burns provider
   tokens on every boot.
3. Raw HTTP GET to `gateway_url/health` — requires knowing the URL
   outside the SDK and couples `warmup` to the transport.

None of these is a clean "gateway-only" probe.  `docker-compose.yml:37-45`
already runs a native `/health` healthcheck on the gateway container,
and the first `/chat` request surfaces any real failure with a full
`TensorZeroError` trace.  A no-op with a subclass hook is the minimum
honest surface.

**Impact.** None on the contract (§3 still permits lifespan-phase
hooks).  The phase spec's wording will be amended in the next
revision; the behaviour remains forward-compatible with Phase E (which
may add a per-forwarder probe).

### D2: `create_app` requires explicit XOR of `client_factory` / `share_client`

**Spec (§4.4)** sketches `create_app(agent, *, deployment_label=None)`
and builds `SessionFactory(...)` without spelling out the mode
selection.

**Actual:** `create_app` requires *exactly one* of `client_factory=...`
or `share_client=True`.  Supplying both or neither raises
`ConfigurationError` at app construction — no silent fallback.

**Reason.** Phase D must enforce the FastMCP Q3 / contract §3.5
per-session-Client mandate by default, but also support the
`share_client=True` opt-out for non-elicitation workloads.  A silent
default in either direction creates surprise (either per-request
overhead the deployer did not ask for, or elicitation-routing bugs
they cannot diagnose).  An explicit XOR pushes the mode decision into
the caller's source where it can be reviewed.

**Impact.** Documented in `docs/serving.md` ("Configuration knobs").

### D3: `SessionFactory.agent_class` preserves `Agent` subclasses

**Spec (§4.2)** sketches `SessionFactory` yielding an Agent without
specifying how subclass identity survives the per-session copy.

**Actual:** `SessionFactory.__init__` accepts an
`agent_class: type[Agent] = Agent` kwarg; the session context manager
instantiates `self._agent_class(session_config, t0=..., tool_provider=...)`
instead of hard-coding `Agent(...)`.  `create_app` passes `type(agent)`
so a caller's subclass survives per-request reconstruction.

**Reason.** `SessionFactory` must hand each request a freshly-built
Agent so concurrent `AgentConfig.tags` merges do not leak across
callers (Plan decision "Per-request tags merge → `dataclasses.replace`").
Without `agent_class`, subclass overrides of `run`, `warmup`, or the
memory-injection helpers are silently lost — a regression anyone
subclassing `Agent` would hit.  Preserving `type(agent)` is one-line
and keeps the test helper pattern clean.

**Impact.** Public behaviour unchanged for the default path
(`create_app(Agent(...))`); subclasses now round-trip correctly.
`tests/test_serve/test_chat.py` relies on this to inject
`MaxTurnsExceededError` / `GuardrailBlockedError` via tiny subclasses
without pulling `unittest.mock`.

### D5 (post-merge): `/feedback` forwards 4xx `TensorZeroError` as-is

**Observed during manual validation (2026-04-16).**  Calling
`POST /feedback` with a metric that is not declared in
`tensorzero.toml` returned HTTP 502 with a detail string containing
`"TensorZeroError (status code 404): Unknown metric: ..."`.  The
gateway's 404 is a caller-side error (the `metric_name` field in the
`/feedback` payload is wrong), but the framework wrapped it as 502 Bad
Gateway.  Similarly, "inference_id not found" would be 404, not 502.

**Fix.**  `routes.py::feedback` inspects `TensorZeroError.status_code`
(available on the installed SDK — see Phase C log D5 for precedent)
and passes through any 4xx verbatim; 5xx and missing status codes
still map to 502.

**Rationale.**  `/feedback` fields (`metric_name`, `value`,
`inference_id`) map 1:1 to the caller's request body, so a 4xx from
the gateway is honestly a 4xx from the HTTP caller's perspective.
`/chat` keeps its unconditional 502 — its body (`{"message": ...}`)
does not map 1:1 to gateway request shape, and a gateway 400 would
mislead the caller into thinking their `/chat` payload was bad.

**Impact.**  Behaviour change in the HTTP status code returned by
`/feedback` on unknown metric / unknown inference.  Unit tests cover
both the 4xx passthrough (`tests/test_serve/test_feedback.py::test_feedback_forwards_gateway_4xx`)
and the 5xx wrap (`::test_feedback_wraps_gateway_5xx_as_502`).  The
integration test's accepted set is broadened to `{200, 404, 502}`.
Documented in the router's docstring.

### D6 (post-merge): `examples/11_http_agent` configures `InMemoryMemory`

**Observed during manual validation (2026-04-16).**  `client.sh`'s
second `/chat` call sends the first call's `episode_id` back, expecting
the LLM to remember "the code 42".  Without `AgentConfig.memory`, the
LLM has no prior context in its messages list — `episode_id`
correlates inferences in ClickHouse but does not replay them into new
runs.  The LLM hallucinated a TypeScript snippet instead of answering
with "42".

**Fix.**  `examples/11_http_agent/agent_app.py` now attaches
`InMemoryMemory()` to `AgentConfig` so the continuation demo works
out of the box.  No extras required (unlike `SqliteMemory`, which is
behind `[memory-sqlite]`).  README note points at
`docs/memory.md` for the swap to `SqliteMemory` in production.

**Rationale.**  M4 "Hosted Agent" includes A+B+C+D per the milestone
table (`01-implementation-plan.md` §3) — the example should therefore
demonstrate memory continuity as part of the M4 value proposition.

**Impact.**  None on the library code.  Example only; documented in
`examples/11_http_agent/README.md` "Highlights".

### D4: `/health.tool_provider` reports `"per-session"` for factory mode

**Spec (§4.3)** describes the route returning `"ok"` / `"down"`; the
plan's D1 section documents `fastmcp.Client.is_connected()` as the
probe source.

**Actual:** When `SessionFactory` is built from a `client_factory`
(per-session mode), there is no persistent `Client` to probe.  The
route returns `"per-session"` — an honest signal that the tool
provider's readiness is decided on each request.  `"ok"` / `"down"`
still apply in shared mode (`share_client=True`), where the caller
owns a single `Client` whose `is_connected()` is stable across
requests.

**Reason.** Opening a throw-away `Client` inside `/health` purely to
probe connectivity is deceptive (it proves a *new* client can open,
not that *this* request's client would).  Per-session mode's health
is "one per request", and the response shape says exactly that.

**Impact.** `HealthResponse.tool_provider` type stays `str`; the Phase
D docs in `docs/serving.md` document the three values (`ok`, `down`,
`per-session`).  Clients that compare on equality with `"ok"` must
treat `"per-session"` as a healthy state — noted in the docs.

## Resolved Open Questions

No open questions from `02-open-questions.md` were newly answered
during Phase D.  The four questions the spec referenced (T0 Q3, T0 Q6,
FastMCP Q2, FastMCP Q3) were already resolved before the phase began
and are honoured in the design:

- **T0 Q3** — `namespace` and `tags` are orthogonal; `deployment_label`
  ferries `tags["t0_agents.deployment"]` without touching namespace.
- **T0 Q6** — No episode close API; the serving layer stores no
  server-side session state, so there is nothing to GC.  Callers
  re-send `episode_id` on each request to continue a conversation.
- **FastMCP Q2** — No built-in remote elicitation abstraction;
  `core/elicitation_forwarder.py` (skeleton in Phase D, complete in
  Phase E) implements the `Future`-per-`elicitation_id` pattern
  directly.
- **FastMCP Q3** — Per-session Client mandatory; `share_client=True`
  is the documented opt-out for non-elicitation workloads.

## Contract Amendments

None.  `00-integration-contract.md` is unchanged:

- §2.1 (`mcp/` standalone): the `mcp/` submodule is untouched.
  `from t0_agents.mcp import run_agent` continues to work without
  `[serve]` installed.
- §2.4 (TensorZero-first): no parallel systems introduced.  `/feedback`
  is a direct passthrough to `AsyncTensorZeroGateway.feedback`;
  `deployment_label` ferries through `tags`; no custom episode store.
- §2.5 (elicitation-only HITL): `ElicitationForwarder` defaults to
  `ElicitResult(action="decline")`, matching the existing
  `auto_decline_handler` exactly.  No parallel HITL channel.
- §3 (`Agent` public contract): `AgentConfig` is unchanged.
  `Agent.warmup()` is a new instance method with a `None` return;
  existing callers are unaffected.
- §3.5 (concurrency): `SessionFactory` enforces the per-session-Client
  default the section already requires.
- §6.1 (frozen `run_agent`): no changes.
- §6.2 (frozen `AgentResult`): no changes.
- §6.3 (frozen `mcp/` exports): no changes.

## Release Notes Snippet

> **v0.5.0 — Hosted Agent (M4)**
>
> HTTP serving layer for `Agent`: `from t0_agents.serve import
> create_app, mount_agent`.  One line wraps an existing agent in a
> FastAPI app with `/chat`, `/feedback`, `/health`, `/info`.
> `create_app(agent, client_factory=lambda: Client(mcp))` is the
> default (one fresh Client per HTTP request — honours the FastMCP
> per-session mandate); `share_client=True` is the opt-out for
> read-only agents without elicitation.  `mount_agent(app, agent,
> "/mobile", deployment_label="mobile")` serves several agents on
> disjoint URL prefixes against the same gateway.  Ships behind the
> new `[serve]` extra (`pip install 't0-agents[serve]'`).  Elicitation
> over plain HTTP is declined with an explicit error result — Phase E
> (M5) adds SSE + WebSocket streaming for live HITL.  See
> `docs/serving.md` and `examples/11_http_agent/`.
