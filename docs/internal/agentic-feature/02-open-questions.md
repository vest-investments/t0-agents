# 02 — Open Questions

> Questions whose answers inform or unblock specific phases. Each
> question is grouped by **upstream library** (`tensorzero`, `fastmcp`)
> and carries:
>
> - **What** — the specific thing we need to know.
> - **Why** — the phase(s) it blocks and the concrete design decision it
>   informs.
> - **Default** — what we'll ship if no answer arrives; the cheapest
>   option that keeps the phase moving.
> - **Answer** — filled in once the user responds. The answer may amend
>   [`00-integration-contract.md`](00-integration-contract.md); when it
>   does, note the amended section under "Impact".

**Related**

- [`00-integration-contract.md`](00-integration-contract.md) — contract
  that answers may amend.
- [`01-implementation-plan.md`](01-implementation-plan.md) — phase
  ordering and dependencies.

---

## Status (2026-04-15)

**All 12 questions answered.** Initial answers came from upstream
documentation review; a follow-up validation pass against the installed
SDK source code (`.venv/lib/python3.12/site-packages/{fastmcp,tensorzero}/`)
confirmed nine answers verbatim and corrected two:

| # | Source-validated finding | Effect |
| --- | --- | --- |
| **T0 Q7** | `AsyncTensorZeroGateway.build_embedded(...)` IS available in v2026.4.0 (`tensorzero.pyi:1017`). All backend-DB URLs are `Optional`. | Phase H ships **both** `MockGateway` (for unit tests) **and** an embedded-gateway test helper. |
| **FastMCP Q1** | `FastMCP.add_tool` does NOT accept a `name=` kwarg directly (`fastmcp/server/server.py:1438`). Name override requires `Tool.from_function(handler, name=...)` first. | Phase G uses the `Tool.from_function(...)` + `add_tool(tool)` pattern; same simplicity, one extra construction call. |

The remaining ten answers (T0 Q1–Q6, FastMCP Q2–Q5) match the installed
SDK exactly. No contract amendment is required by the corrections.

---

## TensorZero

### T0 Q1 — Streaming chunk semantics

**Blocks.** Phase E (streaming), indirectly Phase D (WS channel).

**What.** When `AsyncTensorZeroGateway.inference(stream=True)` is used
and a single turn produces mixed content (e.g. a `Thought` block and a
`ToolCall` block), do chunks arrive:

- interleaved per content block (one chunk = one block delta), or
- as a single terminal `ChatChunk` with a fully-assembled `content` list?

Additionally: how is `finish_reason` delivered when the turn ends with
tool calls (no user-visible text)? Is there one chunk per tool call, or
accumulation with a final "done" chunk?

**Why.** Determines whether `AgentEvent` can emit `ThoughtDeltaEvent`
and `ToolCallEvent` in real time or has to buffer the whole turn and
emit at the boundary. The former is what users actually want (live
"Calling `search_papers`…" feedback); the latter is cheaper to
implement.

**Default.** Buffer the whole turn. Emit `TextDeltaEvent` for every
intra-turn text chunk (natural when streaming assistant text), and
`ToolCallEvent` once per turn after the final chunk assembles the
complete call. Ship this conservative chunk assembler; revisit once we
read real server-sent chunks.

**Answer.** Chunks arrive **interleaved per content block delta** — not
as a single terminal assembled chunk. The SSE stream emits a JSON
message per delta, and each message carries a `content` list of block
chunks. Three delta types are documented:

- `type: "text"` → `{ id, text }` (delta string for a text block)
- `type: "tool_call"` → `{ id, raw_name, raw_arguments }` (all strings,
  streamed incrementally)
- `type: "thought"` → `{ id, text, signature? }` (reasoning models,
  e.g. DeepSeek R1)

`usage` (token counts + cost) arrives in a final chunk; content-bearing
chunks include `raw_chunk` when `include_raw_response` is set. The
stream terminates with a `[DONE]` sentinel. There is **no explicit
`finish_reason` field** in the native TensorZero SSE format — tool-call
turns simply stop emitting `tool_call`-typed deltas before the final
usage chunk.

**Impact.** The conservative default (buffer + emit at boundary) is
still valid, but **real-time per-block emission is achievable without
buffering**: each delta chunk already carries a stable `id` that
identifies the content block, so `ThoughtDeltaEvent` and streaming
`ToolCallEvent` (name + args as they arrive) can be forwarded directly.
Recommend revisiting Phase E to implement the real-time path; no
contract amendment needed, but the `AgentEvent` stream shape in §4 of
the contract should note the block `id` as a stable correlation key.

*(source: `gateway-api-reference.md` → "API Reference: Inference" →
Streaming response, Chat Function streaming example —
https://www.tensorzero.com/docs/gateway/api-reference/inference)*

---

### T0 Q2 — Mid-episode `additional_tools` mutation

**Blocks.** Phase G (multi-agent composition).

**What.** Within a single episode, is it safe to pass a different
`additional_tools` list on each `inference()` call? Specifically:

- Does variant stickiness break if the tool set changes?
- Does TensorZero log a warning / error?
- Are there known prompt-level issues (e.g. the LLM sees different
  tool schemas across turns)?

**Why.** Phase G wants to expose sub-agents as tools dynamically — they
should appear in the parent's tool list and disappear once the
delegation completes. If the gateway requires a stable tool list per
episode, the design has to pre-declare every sub-agent up front and
gate via `tool_choice`.

**Default.** Assume tool lists are stable per *run* (one `Agent.run()`
call). Sub-agents declared via `AgentConfig.sub_agents` are advertised
for the entire run; nothing gets added or removed mid-run. Dynamic
per-turn tool changes are a future enhancement.

**Answer.** The documentation treats `additional_tools` as a **per-call
parameter** with no per-episode stability requirement. The MCP
integration example passes `additional_tools=t0_tools` on every
`inference()` call inside a `while True` loop over the same `episode_id`
without any noted caveat. Variant stickiness is governed separately:
*"During an episode, multiple inference requests to the same function
will receive the same variant (unless fallbacks are necessary)"* — this
guarantee is about which **variant** is sampled, not about which tools
are passed. The gateway does not validate or warn on tool-set
differences between turns in the same episode.

However, the docs are silent on the **LLM-level** consequence (the
model seeing inconsistent schemas across turns in its conversation
history). That risk is real and sits outside the gateway layer.

**Impact.** The default (stable tool list per run) remains a safe
conservative choice for Phase G. The gateway itself does not block
dynamic per-turn `additional_tools`, so the dynamic sub-agent pattern
**is technically feasible** without pre-declaration. If Phase G pursues
it, document the model-side caveat in `docs/multi-agent.md`.

*(source: `gateway-api-reference.md` → `additional_tools` parameter;
`gateway-guides.md` → MCP integration example —
https://www.tensorzero.com/docs/gateway/api-reference/inference,
https://github.com/tensorzero/tensorzero/tree/main/examples/mcp-model-context-protocol)*

---

### T0 Q3 — Deployment-per-surface: `namespace` vs `tags`

**Blocks.** Phase D (serving) — specifically the "one server per
deployment surface" use case (`mobile agent server`, `analytics agent
server`, all talking to the same gateway).

**What.** When a deployment wants to isolate its traffic on the
gateway — for experimentation, cost accounting, or routing — should the
framework set:

- `namespace="mobile"` (triggers per-surface experimentation config), or
- `tags={"surface": "mobile"}` (observability only), or
- both?

**Why.** `namespace` is a stickier binding — it routes to different
variant sampling groups declared in `tensorzero.toml`. `tags` are
free-form observability metadata. Coupling the framework's "surface"
concept to `namespace` forces every deployer to declare per-namespace
experimentation, which may be over-reach.

**Default.** Ship both knobs on `AgentConfig`:

- `namespace: str | None = None` — forwarded verbatim; documented as
  the experimentation-isolation knob.
- `tags: dict[str, str] | None = None` — forwarded verbatim; documented
  as the observability knob. The serving layer auto-populates
  `tags["t0_agents.deployment"]` with a caller-configurable value for
  the common case.

**Answer.** The documentation confirms the two knobs are **fully
orthogonal**:

- `namespace` selects a namespace-specific experimentation config block
  in `tensorzero.toml`. If no matching block exists the gateway
  **silently falls back** to the default config — it does not error.
  The value is automatically stored as the `tensorzero::namespace` tag
  on every inference record, so it is also queryable for cost
  accounting without a separate `tags` entry. Restriction: if a variant
  uses a model scoped to a different namespace, the request fails.
- `tags` are free-form `Map(String, String)` written verbatim to the
  inference row; purely observability, no routing effect.

The default (ship both knobs) is therefore **correct as designed**.
`namespace` is the right knob for per-surface experimentation
isolation; `tags` is additive for richer observability. Because an
unknown namespace falls back gracefully, deployers who omit a
per-namespace TOML block will not see errors.

**Impact.** No contract amendment needed. The behaviour of both
parameters is exactly what §3 of the contract assumes. Worth noting
in `docs/deployment.md` that `namespace` doubles as a free
observability tag (`tensorzero::namespace`) so teams need not duplicate
it in `tags`.

*(source: `experimentation.md` → "Scope experiments with namespaces";
`gateway-api-reference.md` → `namespace`, `tags` parameters;
`gateway-configuration-reference.md` → namespace-specific
experimentation —
https://www.tensorzero.com/docs/experimentation/scope-experiments-with-namespaces,
https://www.tensorzero.com/docs/gateway/api-reference/inference)*

---

### T0 Q4 — Memory injection path (BLOCKS FINAL SPEC FOR B)

**Blocks.** Phase B — final spec (skeleton starts without).

**What.** To inject persisted memory into a prompt, which is preferred:

- **(a) Template-driven.** Require variants to declare a
  `templates.system.*` MiniJinja template that accepts a
  `memory_context` argument; the framework passes
  `input.system.arguments={"memory_context": "..."}` per call.
- **(b) Message-driven.** Prepend a synthetic
  `Message(role="user", content="Previous context: ...")` to the
  conversation ahead of the real user message.
- **(c) Both, with `memory_strategy` choice.** Developer picks per
  agent.

**Why.** (a) is prompt-engineering-clean and keeps memory invisible to
the conversation history, but forces every memory-using function to
maintain a templated variant with the right placeholder. (b) works with
any function but mixes memory into `messages` which may confuse the
model and ClickHouse analytics.

**Default.** Ship **(c)**. Default to **(b)** for the simple case where
the developer hasn't declared a template — since it doesn't constrain
TOML. Document (a) as the recommended production path with a matching
example variant under `config/tensorzero.toml`.

**Answer.** Option **(a) is fully supported** by the documented template
system. Variants declare `templates.system.path` pointing to a MiniJinja
file; the function declares a matching `schemas.system.path` JSON Schema
to validate arguments. At inference time the caller passes:

```python
input={
    "system": {"arguments": {"memory_context": "..."}},
    "messages": [...]
}
```

The schema enforces the `memory_context` field, catching misconfigured
variants at startup. The legacy equivalent (`system_template` +
`system_schema`) is also still accepted.

Option **(b)** (prepend a `tensorzero::raw_text` user message) is
equally valid — `tensorzero::raw_text` content blocks bypass templates
and inject text directly, which is exactly what a prepended memory
message needs.

There is **no built-in memory primitive** in TensorZero; both strategies
are implemented at the framework layer.

**Impact.** The default plan (c) is confirmed viable. Recommend
documenting (a) as the production path with `templates.system.path =
"functions/.../system_template.minijinja"` and a `memory_context`
variable in the schema. Contract §3.2 amendment for
`memory_strategy: Literal["system_template", "prepended_message"]`
stands as planned.

*(source: `gateway-configuration-reference.md` → `system_template`,
`templates`, `schemas`; `gateway-create-a-prompt-template.md`;
`gateway-api-reference.md` → `tensorzero::raw_text` content block type —
https://www.tensorzero.com/docs/gateway/create-a-prompt-template,
https://www.tensorzero.com/docs/gateway/configuration-reference)*

---

### T0 Q5 — Feedback for guardrail outcomes

**Blocks.** Phase C (guardrails).

**What.** When a user rejects a tool via elicitation, should the
framework auto-emit feedback so future variants can learn?

- Metric name: `action_approved` (bool) or something else?
- Level: inference-level (tied to the turn that proposed the rejected
  tool) or episode-level (broader signal)?
- Enabled by default, or opt-in?

**Why.** Makes guardrail rejections discoverable in ClickHouse without
the developer wiring them by hand. Risk: if "rejection rate" isn't a
useful metric for most agents, auto-emission adds noise and forces a
metric declaration in `tensorzero.toml` that many users don't want.

**Default.** Do not auto-emit. Document the manual path in
`docs/guardrails.md`:

```python
if guardrail_rejected:
    await t0.feedback(
        metric_name="action_approved",
        inference_id=result.inference_ids[-1],
        value=False,
    )
```

**Answer.** The feedback API (`POST /feedback`) requires every metric to
be **pre-declared in `tensorzero.toml`** with an explicit `type` and
`level`. There is no dynamic or ad-hoc metric creation at feedback time;
passing an undeclared metric name will fail. This means auto-emission of
guardrail feedback would require every deployer to add a
`[metrics.action_approved]` block to their config, which is the exact
over-reach the default is trying to avoid.

Additionally, the docs make clear feedback can be at `inference` or
`episode` level as configured in the metric — the level is a property of
the metric definition, not the feedback call. Inference-level (tied to
the specific turn that proposed the tool) is the right granularity for
guardrail signals.

**Impact.** **Default confirmed: do not auto-emit.** The manual path
documented in the default is correct. Document in `docs/guardrails.md`
that developers must add `[metrics.action_approved]` (type `boolean`,
level `inference`) to their `tensorzero.toml` if they want this signal,
then call `t0.feedback(...)` manually. No contract amendment needed.

*(source: `gateway-api-reference.md` → "API Reference: Feedback" →
`POST /feedback`; `gateway-configuration-reference.md` →
`[metrics]` section —
https://www.tensorzero.com/docs/gateway/api-reference/feedback,
https://www.tensorzero.com/docs/gateway/configuration-reference)*

---

### T0 Q6 — Episode lifecycle terminator

**Blocks.** Phase B (memory snapshot timing), Phase D (session close
semantics).

**What.** Is there an explicit API to mark an episode as "closed"
(analogous to `DELETE /episodes/{id}`), or does TensorZero treat
episodes as indefinitely open until some GC sweep?

**Why.** Memory snapshot timing. We want to persist the full episode
when a conversation ends, not on every turn. Serving layer also needs
to know when to tear down per-episode state.

**Default.** Persist on every `Agent.run()` call, upsert keyed by
`episode_id`. No explicit close needed. Serving layer GCs session
state on WS disconnect + 15-minute idle timeout.

**Answer.** There is **no episode close / terminate API** in TensorZero.
Episodes are implicitly defined by the set of inferences that share an
`episode_id`; the gateway creates one on first use and never needs an
explicit close signal. The documented episode lifecycle is: *first
inference* (no `episode_id` provided → gateway generates one and returns
it) → *subsequent inferences* (caller passes the `episode_id`) →
episode is "done" when the application stops sending inferences with
that ID. No `DELETE /episodes/{id}` or equivalent endpoint exists.

**Impact.** **Default confirmed.** Persist memory on every
`Agent.run()` call. The serving layer's WS-disconnect + idle-timeout GC
is the correct teardown mechanism. No contract amendment needed.

*(source: `gateway-guides.md` → "Episodes" guide;
`gateway-api-reference.md` → `episode_id` parameter —
https://www.tensorzero.com/docs/gateway/guides/episodes)*

---

### T0 Q7 — Embedded / in-memory gateway for CI

**Blocks.** Phase H (agent testing framework) — ability to run
integration-style tests in CI without Docker.

**What.** Does `AsyncTensorZeroGateway.build_embedded(...)` run:

- Without ClickHouse (in-memory or SQLite backend)?
- Without any external provider (mock LLM responses)?
- As a realistic target for CI that runs `t0_agents.testing` suites?

**Why.** The testing framework needs a zero-infra target so contributor
CI can assert agent behaviour without booting containers. If
`build_embedded` doesn't work for this, the testing framework ships
gateway mocks instead.

**Default.** Ship gateway mocks in `t0_agents.testing.mocks` that
simulate `AsyncTensorZeroGateway` well enough to exercise the loop.
Document that integration tests still require Docker. Re-evaluate
embedded mode once a user request surfaces.

**Answer.** _Doc-only review said `build_embedded` is "not documented"._
**Source-code validation contradicts that.** The installed
`tensorzero` SDK (v2026.4.0) ships
`AsyncTensorZeroGateway.build_embedded(...)` as a public classmethod —
file `tensorzero/tensorzero.pyi:1017-1035`:

```python
@classmethod
def build_embedded(
    cls,
    *,
    config_file: Optional[str] = None,
    clickhouse_url: Optional[str] = None,
    postgres_url: Optional[str] = None,
    valkey_url: Optional[str] = None,
    timeout: Optional[float] = None,
    async_setup: bool = True,
) -> Union[Awaitable["AsyncTensorZeroGateway"], "AsyncTensorZeroGateway"]: ...
```

`TensorZeroGateway.build_embedded` exists symmetrically at line 545.
All backend-DB URLs are `Optional`, so the embedded mode can run with
no observability backend at all (degraded mode — feedback IDs are
ephemeral, mirroring `dryrun=True` semantics). This makes embedded mode
a realistic CI target: a config file is enough to drive real LLM
inference (against any provider whose API key is in the env) without
spinning up Docker / ClickHouse.

**Impact.** **Default revised: ship BOTH `MockGateway` (deterministic
unit-test path) AND an `embedded_gateway` test helper that wraps
`build_embedded`.** Phase H now covers two complementary tools:

- `MockGateway` for tests asserting exact LLM responses / tool-usage
  (no provider call).
- `EmbeddedGatewayHelper` (or just a fixture) for integration tests
  that exercise real T0 inference without Docker — useful for
  contributors running `pytest` locally and for CI that has provider
  API keys.

No contract amendment needed. `CONTRIBUTING.md` should document both
paths and when to pick which.

*(source validated against installed SDK: `tensorzero/tensorzero.pyi:1017-1035`
in `.venv/lib/python3.12/site-packages/tensorzero/`. Doc URL was
incorrect at the time of review — the public docs may not yet describe
`build_embedded`, but the SDK does ship it. File a doc PR upstream as a
follow-up.)*

---

## FastMCP

### FastMCP Q1 — Runtime tool registration

**Blocks.** Phase G (agent-as-tool).

**What.** To expose a sub-agent as an MCP tool at runtime without
forcing the developer to write a `@mcp.tool` decoration for it, is
there a supported API like `mcp.add_tool(name, schema, handler)` on a
`FastMCP` instance? Or is the tool surface frozen at decoration time?

**Why.** The agent-as-tool pattern ideally lets the framework register
a handler `async def _invoke_sub_agent(args) -> str: return await
sub_agent.run(args["message"])` at runtime, without boilerplate. If
FastMCP has no public runtime API, we build a dedicated in-process
`FastMCP` server per sub-agent and use `mount()` — more moving parts
but uses only public surface.

**Default.** Build a dedicated in-process `FastMCP` server per
sub-agent. One server per sub-agent, mounted under a namespace derived
from `AgentConfig.name`. Revisit once a public runtime-registration API
is documented.

**Answer.** `FastMCP.add_tool(tool: Tool | Callable[..., Any]) -> Tool`
is a **public API** that supports runtime registration. Validated in
the installed package at
`fastmcp/server/server.py:1438`:

```python
def add_tool(self, tool: Tool | Callable[..., Any]) -> Tool: ...
```

**Caveat (source-validated).** The signature does **not** accept a
`name=` kwarg. Tool name comes from either:

- the wrapped callable's `__name__`, or
- a pre-built `Tool` instance constructed via `Tool.from_function(fn,
  name="...")`.

So the runtime registration pattern for a sub-agent is:

```python
from fastmcp.tools import Tool

async def _invoke_sub_agent(message: str) -> str:
    return (await sub_agent.run(message)).response

tool = Tool.from_function(_invoke_sub_agent, name=f"{sub_agent.config.name}_ask")
mcp.add_tool(tool)
```

`remove_tool(name, version)` also exists (line 1452, marked deprecated
in 3.x — favoured replacement is the `LocalProvider.remove_tool` path).
The tool surface is **not frozen** at decoration time.

**Impact.** **Default plan (dedicated server + mount) is unnecessary
complexity.** Use `Tool.from_function(handler, name=...)` +
`mcp.add_tool(tool)` on the parent server. Avoids the extra `FastMCP`
instance and mount overhead. No contract amendment needed; Phase G
spec uses this pattern explicitly.

*(source: `python-sdk-fastmcp-server-server.md` → `add_tool`,
`remove_tool`, `tool` methods —
https://gofastmcp.com/python-sdk/fastmcp-server-server.md.
Source-validated in
`.venv/lib/python3.12/site-packages/fastmcp/server/server.py:1438`.)*

---

### FastMCP Q2 — Remote elicitation across HTTP/WS

**Blocks.** Phase D (serving — elicitation forwarding), indirectly E.

**What.** Is there a documented FastMCP pattern for a "remote"
elicitation handler — one that runs locally in the agent process but
resolves asynchronously based on input arriving from outside (e.g. a
WebSocket message from the browser client)?

**Why.** The serving phase has to pipe `ctx.elicit()` requests to the
user through the browser. Two implementation shapes exist:

- If FastMCP has a pattern (e.g. an `ElicitationChannel` abstraction),
  we copy it for consistency.
- If not, we build our own: the agent HTTP server registers a
  per-session `asyncio.Future` keyed by an elicitation ID, the
  elicitation handler awaits it, the client POSTs back to resolve.

**Default.** Build our own. Design the forwarder as a small
`core/elicitation_forwarder.py` module so Phase D and any future
remote-channel phase share it.

**Answer.** There is **no documented `ElicitationChannel` or
"remote handler" abstraction** in FastMCP. The only documented
client-side pattern is `elicitation_handler` — an `async` coroutine
passed to `Client(...)` at construction time, called inline when the
server invokes `ctx.elicit()`. It is expected to block until it can
return an `ElicitResult`.

The only documented relay mechanism for out-of-band elicitation is in
`fastmcp.server.tasks.elicitation` (`relay_elicitation`,
`handle_task_input`), which is specifically for **background Docket
tasks** and relies on Redis as the transport — not a general-purpose
pattern for browser WebSocket channels.

**Impact.** **Default confirmed: build our own forwarder.** The
`asyncio.Future`-per-elicitation-ID pattern described in the default
is the correct approach — it is not superseded by any FastMCP
abstraction. Implement `core/elicitation_forwarder.py` as planned.

*(source: `clients.md` → "User Elicitation" →
Handler Template, Response Actions —
https://gofastmcp.com/clients/elicitation.md;
`python-sdk-fastmcp-server-tasks-elicitation.md` →
`relay_elicitation`, `handle_task_input` —
https://gofastmcp.com/python-sdk/fastmcp-server-tasks-elicitation.md)*

---

### FastMCP Q3 — Concurrency of a single `fastmcp.Client`

**Blocks.** Phase D (serving concurrency model).

**What.** If one agent process hosts the `Agent` over HTTP and serves
many concurrent users, is it safe to share one `Client(mcp)` instance
across sessions, or must each user get their own?

Relevant sub-questions:

- Is `call_tool` safe to issue concurrently on the same `Client`?
- Does the elicitation handler registration persist across
  concurrent users, or does each session need its own `Client` to have
  its own handler?

**Why.** Drives connection pooling and session lifecycle in Phase D.
One shared client is simpler; per-session clients are safer. Answer
decides the default.

**Default.** One `Client` per session, created on session open, closed
on session close. Safe by default; revisit for performance if sharing
becomes necessary.

**Answer.** The answer is **split**:

1. **`call_tool` concurrency**: `Client` is documented as supporting
   reentrant context managers with multiple concurrent `async with
   client:` blocks via reference counting and background session
   management. The design explicitly solves race conditions from
   concurrent `call_tool` / `list_tools` pairs. `call_tool` **is safe
   to call concurrently** on a shared `Client` instance.

2. **Elicitation handler isolation**: `elicitation_handler` is a
   constructor argument on `Client` — it is a **single handler shared
   by all concurrent calls on that client**. If one `Client` is shared
   across users, every elicitation request from every user is routed
   to the same handler, with no built-in mechanism to associate a
   request with a specific user session. Per-user routing requires
   either per-session `Client` instances (each with its own handler)
   or a multiplexing handler that reads the `context` parameter to
   dispatch.

3. **Session/context mixing**: The docs warn that sharing an
   already-connected client across concurrent requests "may cause
   context mixing in concurrent scenarios. Use only in single-threaded
   situations or with explicit synchronization."

**Impact.** **Default confirmed: one `Client` per session.** The
`call_tool` concurrency guarantee alone is not sufficient to share a
client across users, because elicitation handlers would be incorrectly
shared. Per-session `Client` creation is the correct and safe default.
No contract amendment needed.

*(source: `python-sdk-fastmcp-client-client.md` → `Client` class
docstring, reentrant context manager description —
https://gofastmcp.com/python-sdk/fastmcp-client-client.md;
`clients.md` → "User Elicitation" →
https://gofastmcp.com/clients/elicitation.md;
`servers-providers.md` → "Shared Sessions" warning —
https://gofastmcp.com/servers/providers.md)*

---

### FastMCP Q4 — MCP Resources as a memory primitive

**Blocks.** Phase B (optional alternative).

**What.** Is it idiomatic to expose persistent memory via `@mcp.resource`
rather than as an in-framework `Memory` class? That way the LLM "reads
memory" through MCP primitives — list resources, fetch a specific
resource by URI — rather than through a framework-injected context
prefix.

**Why.** Elegant for RAG-style "memory as knowledge base". Potentially
overloads the mental model for simple session state (which is closer
to a message history than to a knowledge base).

**Default.** Keep `Memory` as a t0-agents concept (not MCP resources)
for Phase B. Document the MCP-resources alternative in `docs/memory.md`
as "Bringing your own RAG via MCP Resources" so developers can opt in
when their use case fits.

**Answer.** The `@mcp.resource` pattern is **technically valid** for
this use case. Resources support list + read via `client.list_resources()`
and `client.read_resource(uri)`, and the `ResourcesAsTools` transform
can expose them to tool-only clients that cannot issue
`resources/read` requests directly. Nothing in the docs discourages
this pattern.

However, the docs present resources as **general-purpose data sources**
(files, configs, dynamic data) — there is no documented idiom
specifically for session memory or conversation history. The docs do not
address memory primitives at all; they are silent on the pattern.

**Impact.** **Default confirmed: keep `Memory` as a t0-agents concept.**
The resource-based alternative is viable for RAG/knowledge-base use
cases and should be documented in `docs/memory.md` as an opt-in path,
as planned. No contract amendment needed.

*(source: `servers-resources.md` —
https://gofastmcp.com/servers/resources.md;
`python-sdk-fastmcp-server-transforms-resources_as_tools.md` →
`ResourcesAsTools` —
https://gofastmcp.com/python-sdk/fastmcp-server-transforms-resources_as_tools.md)*

---

### FastMCP Q5 — Elicitation schema guarantees

**Blocks.** Phase F (TUI demo — rich rendering of elicitation
prompts).

**What.** Is `params.requestedSchema` guaranteed to follow JSON Schema
draft-07 with the standard keywords (`type`, `properties`, `enum`,
`default`, `description`, `required`), or does FastMCP normalise /
strip / rewrite anything before it reaches the handler?

Specifically for the TUI demo:

- Are `enum` values always present for constrained-choice fields?
- Is `description` preserved for rich prompt labels?
- Do nested schemas (objects within objects) arrive flat, or
  hierarchically?

**Why.** The TUI demo renders each schema field using Rich's prompts
(`Confirm.ask`, `Prompt.ask` with `choices=`). Reliable schema
guarantees mean we can write a generic renderer; unreliable means
we fall back to text input.

**Default.** Assume JSON Schema draft-07 subset. Trust `type`,
`properties`, `enum`, `default`, `description`, `required`. Fail soft
on unknown constructs — when a field type isn't recognisable, fall
back to `Prompt.ask(str)` with a warning. Document the assumption in
`docs/elicitation.md` with a pointer to this question.

**Answer.** FastMCP enforces a **strict, documented subset** of JSON
Schema for elicitation. The guarantees are stronger than draft-07 and
are enforced at server side via `validate_elicitation_json_schema()`.
Specific guarantees:

- **Schema shape**: always a flat JSON object (`"type": "object"`) with
  primitive properties. Nested objects and arrays of objects are
  **rejected** at server construction time — they cannot appear in
  `requestedSchema`.
- **Field types**: only `string`, `number`/`integer`, `boolean`, `enum`
  (via `Literal`, Python `Enum`, or list shorthand), and `const`
  (for `Literal` with a single value). No other types reach the
  handler.
- **`enum` values**: always present as an inline `{"enum": [value, ...]}`
  for constrained-choice fields. FastMCP's `ElicitationJsonSchema`
  generator **always inlines enums** and never uses `$ref`/`$defs`.
- **`description`**: preserved when declared via `Field(description=...)`
  on a Pydantic model or dataclass.
- **`default`**: preserved when declared via `Field(default=...)`.
  Supported for `string`, `integer`, `number`, `boolean`, and `enum`
  fields.
- **Scalar wrapping**: when `response_type` is a scalar (`str`, `int`,
  `bool`), FastMCP wraps it in `{"type": "object", "properties":
  {"value": {...}}}`. The handler sees a single-property object; the
  `value` key is unwrapped server-side before returning `result.data`.

**Impact.** **Default assumption is correct and can be tightened.**
The TUI renderer can rely on all four keywords (`type`, `enum`,
`description`, `default`) and can treat nested-object schemas as
impossible (they are blocked by the server). The fail-soft fallback to
`Prompt.ask(str)` is still worth keeping for schemas originating from
non-FastMCP servers. No contract amendment needed; update
`docs/elicitation.md` to document the guaranteed flat-object shape.

*(source: `servers-elicitation.md` → "Schema and Response Types",
"Default Values", "Constrained Options" —
https://gofastmcp.com/servers/elicitation.md;
`python-sdk-fastmcp-server-elicitation.md` →
`validate_elicitation_json_schema`, `ElicitationJsonSchema`,
`enum_schema` —
https://gofastmcp.com/python-sdk/fastmcp-server-elicitation.md)*

---

## How to answer

1. Respond inline in this file under the **Answer** heading for each
   question you've resolved. Short is fine — one to three sentences
   plus a code/config pointer is ideal.
2. If the answer forces a contract amendment, note which section of
   [`00-integration-contract.md`](00-integration-contract.md) needs
   updating under **Impact**. The phase that picks up the answer
   applies the amendment in the same PR.
3. Leave unanswered questions as `_pending_`. Phases that depend on
   them will either ship with the default or wait, as recorded in
   [`01-implementation-plan.md`](01-implementation-plan.md) §4.

## How to add a new question

If during any phase a new uncertainty surfaces:

1. Add a new section under the appropriate upstream library.
2. Number it with the next available ID (e.g. `T0 Q8`).
3. Update `01-implementation-plan.md` §4 with the new blocker.
4. If the open question invalidates a default already shipped, log it
   in that phase's implementation log as a follow-up.
