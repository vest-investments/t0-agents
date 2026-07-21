# Multi-tenant apps — identity per conversation

Mount servers **once**, then give every user their own
`conversation(...)`.  One long-lived `App` serves any number of
identities; nothing about a user binds at App construction.

```python
from t0_agents.core.elicitation_forwarder import ElicitationForwarder
from t0_agents.glow import App, MCPServer, ServerCredentials
from t0_agents.memory import TensorZeroMemory

class CustomersMCP(
    MCPServer,
    url="https://customers.internal/mcp",
    namespace="customers",
    auth=service_auth,          # process-level service credential
    elicits=True,               # its tools may ask the human to confirm
): ...

app = App(t0=t0, servers=[CustomersMCP, PublicDataMCP])

@app.agent("assistant", memory=default_memory, memory_strategy="messages")
async def assistant(message: str) -> str:
    """A per-user assistant."""

# Per user message / session — NO App rebuild:
forwarder = ElicitationForwarder(on_request=push_to_this_user)
conv = assistant.conversation(
    id=conversation_id,                      # resume across messages
    user=user_id,                            # write tags + read filter
    auth={"customers": {"X-User-Id": user_id, "X-User-Token": token}},
    elicitation_handler=forwarder,
    memory=TensorZeroMemory(t0, function_name="chat",
                            tag_filter={"user_id": user_id}),
)
reply = await conv.send(message)
```

Identity has **three legs**, and they land together — shipping auth
alone would leave HITL unroutable and memory shared across tenants:

| Leg | Kwarg | What it scopes |
|-----|-------|----------------|
| auth | `auth={namespace: credentials}` | The headers/`httpx.Auth` each mounted server sees for THIS conversation |
| elicitation | `elicitation_handler=` | Where `elicits=True` servers' `ctx.elicit(...)` prompts land |
| memory | `memory=` | The provider every send loads from / saves to (tenant `tag_filter`) |

Plus `tools=` — per-conversation tool visibility with the
`@app.agent(tools=)` grammar, intersected with the agent's own filter
(arm a sensitive tool for one confirmed conversation instead of a whole
process).

A conversation that passes none of these behaves exactly as before —
the identity dimension is pay-for-what-you-use.

## Credentials — `auth=`

`auth=` maps mounted server **namespaces** to this conversation's
credentials.  Three value shapes:

```python
auth={"customers": {"X-User-Id": uid}}                  # headers-only (the common case)
auth={"customers": my_httpx_auth}                       # auth-only
auth={"customers": ServerCredentials(                   # both + explicit pool key
    headers={"X-User-Id": uid}, auth=my_httpx_auth, key=uid)}
```

Overlay headers **merge over** the spec's static headers (overlay keys
win); an overlay `auth` **replaces** the spec's.  A namespace that
matches no mounted server raises immediately — credentials that
silently apply to nothing are a security bug, not an inconvenience.

## The pool model — what scales with what

The runtime pools warm clients by `(server, identity)`:

- **Static/public servers** (no overlay): every conversation shares the
  server's one warm client — exactly the single-tenant behaviour.
- **Overlaid servers**: one client per *identity* — the pool key is a
  fingerprint of the resolved credentials (or the explicit
  `ServerCredentials(key=...)`), so two conversations of the same user
  share a client, across agents and across sends.
- **`elicits=True` servers**: one client per *conversation* — see below.

Connections scale with `identities x overlaid servers +
conversations-that-elicit`, never `conversations x servers`.  Pool
entries idle out (default 15 min) and the pool is size-capped (default
1024; the oldest idle entry is evicted at the cap, and a pool full of
in-flight clients fails loud rather than tearing one down under a live
call).  Tune via `MCPRuntime(pool_idle_ttl=..., pool_max_size=...)` if
you construct the runtime directly.

### Credential rotation

Rotation needs no plumbing: reopening `conversation(id=..., auth=...)`
with new material fingerprints to a **new** pool entry (the stale one
idles out), and `httpx.Auth` objects are per-request-fresh by
construction (`auth_flow` runs on every request), so an auth object
reading a mutable token rotates without any pool churn at all.

## HITL — `elicits=True` + `elicitation_handler=`

The MCP protocol has no elicit↔call correlation a client can use (the
SDK discards `related_request_id`), so a shared client **cannot** route
a confirmation prompt to the right conversation.  Declare
`elicits=True` on servers whose tools call `ctx.elicit(...)`; the
runtime gives each conversation its own private client for those
servers — routing by construction, not by bookkeeping.

- Pass an `ElicitationForwarder` (recommended): the in-stream
  `ElicitationRequestedEvent.id` **is** the forwarder's
  `elicitation_id`, so the id your out-of-band resolver answers and the
  id on the event stream are the same thing.
- Pass any bare `ElicitationHandler`: events carry a glow-minted id
  (self-consistent Requested/Resolved pairing).
- `conv.elicitation_handler` is rebindable between turns; the client's
  trampoline re-resolves it at elicit time.
- A server that elicits **without** declaring `elicits=True` rides a
  handler-less shared client: the elicit fails loud (the client never
  advertises the capability) — it can never reach another
  conversation's handler.

An App-level `App(elicitation_handler=...)` still works exactly as
before for single-tenant apps, and serves as the fallback when a
conversation sets none.

## Migrating from one-App-per-tenant

The previous pattern rebuilt the App (mounts + handshakes) per user
message because identity bound at declaration time:

```python
# BEFORE: per-message rebuild
app, agent = build_app(customer_id=..., customer_token=...,
                       elicitation_handler=fwd.handler)   # fresh mounts every time
async with app:
    reply = await agent.conversation(id=cid).send(msg)
```

```python
# AFTER: module-level App, identity per conversation
APP, AGENT = build_app()          # once per process

conv = AGENT.conversation(
    id=cid,
    auth={"customers": {"X-Customer-Id": customer_id,
                        "X-Customer-Token": customer_token}},
    elicitation_handler=fwd,
    memory=TensorZeroMemory(t0, tag_filter={"customer_id": customer_id}),
    tools=[...],                  # if arming differs per conversation
)
reply = await conv.send(msg)
```

Checklist:

1. Move per-user `headers=` off the `MCPServer` spec into
   `conversation(auth=...)`; keep process-level service auth on the spec.
2. Declare `elicits=True` on servers that confirm actions.
3. Pass the per-conversation forwarder as `elicitation_handler=`
   (the forwarder object, not `.handler`, to get id-unified events).
4. Move the tenant `tag_filter` memory into `conversation(memory=...)`;
   keep a safe default on the agent declaration.
5. Delete the per-message `async with app:` — the App now lives for the
   process (`async with app.lifespan():` once, at startup).

## Caveats

- **Inventory is identity-independent.**  `list_tools` is served by the
  shared per-server client and cached across conversations — a server
  whose *tool list* itself requires per-user auth must carry spec-level
  credentials sufficient to list.  Per-conversation *visibility* is the
  `tools=` filter, applied client-side.
- **One forwarder per conversation.**  A forwarder observed by two
  conversations keeps only the last binding (its documented
  one-channel contract) — mint one per conversation.
- **Local `@app.tool` functions cannot declare `elicits=True`** (there
  is no spec to declare it on).  Mount an eliciting in-memory server
  via `MCPServer(server=..., elicits=True)` instead.
