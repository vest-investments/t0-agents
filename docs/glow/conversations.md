# Conversations — saying a second thing

A single call is one turn:

```python
answer = await tutor("What's a gerund?")
```

A **conversation** is many turns that share one TensorZero *episode*, so the
model sees what was already said:

```python
conv = tutor.conversation()
await conv.send("What's a gerund?")
async for token in conv.send("Give me three examples").stream_text():
    print(token, end="")
```

`send()` returns the **same** `FluentRun[R]` a direct call does — every surface
is intact:

```python
a1 = await conv.send("What's a gerund?")                 # unwrapped str
data = await conv.send("As JSON").json(GerundReport)     # structured output
async for tok in conv.send("More").stream_text(): ...    # streamed text
tagged = await conv.send("More").with_(tags={"topic": "grammar"})
```

## The identity: `conv.id == episode_id`

Every conversation *is* its episode. The id is a UUIDv7 (TensorZero's
requirement), minted when you open the conversation and stable for its life:

```python
conv = tutor.conversation()
conv.id                     # e.g. "0192f0a1-....": the TensorZero episode_id
await conv.send("hi")       # runs under episode_id == conv.id
```

Store `conv.id` and resume later — it *is* the episode:

```python
conv = tutor.conversation(id=saved_id, hydrate=True)
```

`id=` must be a UUIDv7 returned by a prior run; `hydrate=True` reconstructs the
transcript from that episode before the first send (see [Resuming](#resuming)).

## The transcript

`conv.transcript` is the in-process record of the turns this handle has
completed — one `{"role", "content"}` pair each, oldest first:

```python
await conv.send("What's a gerund?")
await conv.send("Three examples")
conv.transcript
# [{"role": "user", "content": "What's a gerund?"},
#  {"role": "assistant", "content": "A gerund is …"},
#  {"role": "user", "content": "Three examples"},
#  {"role": "assistant", "content": "1. …"}]
```

It records **each turn's final assistant reply**, not the intermediate
tool-calling turns within a turn — the limitation inherited from
[`Run.transcript()`](../run.md). A turn is recorded once its run completes
**successfully**, via whichever path drained it (`await`, `.stream_text()`,
`.events()`); a turn that failed is not recorded.

## How history reaches the model

TensorZero does **not** rebuild prior turns from the `episode_id` alone — the
`episode_id` groups inferences for observability; history is threaded into the
prompt as `prior_messages`. `send()` handles this for you, keyed on the agent's
`memory_strategy`:

- **`memory_strategy="messages"`** — `send()` passes only `episode_id`. The
  memory provider weaves the full history via
  [`load_history`](../memory.md); passing `prior_messages` as well would
  *suppress* that read (a caller-supplied history wins), so `send()` deliberately
  does not. This is the recommended setup for a memory-backed agent.
- **any other strategy** (including the default) — `send()` seeds the
  in-process `transcript` as `prior_messages`, the same way
  [`Run.fork()`](../run.md) continues a run. Multi-turn therefore works for
  *any* agent, memory-backed or not.

You never choose between these — configure the agent's `memory_strategy` and
`send()` does the right thing.

## Tenancy: `user=` / `tags=`

Reads and writes must agree on their tags or a scoped memory silently reads
nothing. `conversation()` closes that trap by taking the tags **once** and
applying them to both sides — the tags stamped on every `send()` *and* the read
filter used to hydrate and to list:

```python
conv = tutor.conversation(user="cust-42")            # -> tags {"user": "cust-42"}
conv = tutor.conversation(tags={"customer_id": cid}) # any key you like
```

`user=` is sugar for `tags={"user": ...}`; use `tags=` for other keys.

## Sequential sends

Turns on one episode must be sequential — you cannot start the next `send()`
until the previous one has been consumed:

```python
conv.send("first")        # not awaited / not drained
conv.send("second")       # ConfigurationError: the previous send() has not
                          # been consumed
```

Await each send (or drain its stream) before the next.

Consuming a **derived** handle counts. `send()` returns the root `FluentRun`,
and every fluent override — `.with_(...)`, `.json(...)`, `.text()`, `.as_(...)`
— derives a new one; the chain shares a single consumption record, so draining
whichever handle you end up with consumes the send. This is also how in-stream
elicitation events find the run you actually drive: a HITL prompt raised during
`conv.send(q).with_(...)` lands on that derived run's stream.

## Resuming

`hydrate=True` reloads a stored conversation's transcript from the gateway
(thoughts stripped, as on every cross-turn replay). It runs automatically before the first
`send()`; call it explicitly to inspect the transcript first:

```python
conv = tutor.conversation(id=saved_id, hydrate=True)
await conv.hydrate()          # transcript now reconstructed
print(len(conv.transcript))
await conv.send("continue")   # continues the reloaded conversation
```

## Listing conversations

`app.conversations(agent, tags=…)` lists an agent's recent conversations,
newest first:

```python
recent = await app.conversations(tutor, tags={"user": "cust-42"}, limit=20)
for c in recent:
    print(c.conversation_id, c.title, c.turn_count, c.updated_at)
```

Each row is a `ConversationSummary` (`conversation_id`, `title`, `preview`,
`turn_count`, `updated_at` in epoch-ms).

The listing is scoped to `tutor` by the `t0_agents.agent` tag every inference
carries, so agents sharing one tensorzero.toml function stay separate and a
`model=` agent is not drowned in the passthrough's shared log. `tags=` narrows
further, and a key collision resolves the caller's way — the same precedence
the write side uses.

By default this is **dev-grade**: it
groups a bounded window of recent inferences by episode client-side (there is
no server-side per-episode aggregation in TensorZero through 2026.6.0,
its last release), so a
conversation whose latest turn falls outside the window is not listed, and
`turn_count` is exact only under `memory_strategy="messages"`. See
[`TensorZeroMemory.list_conversations`](../memory.md) for the precise guarantees.

### Production listing — the `ConversationIndex` seam

For real scale, implement the `ConversationIndex` protocol
(`t0_agents.core.conversation`) over your own store and hand it in:

```python
from t0_agents.core.conversation import ConversationIndex, ConversationSummary

class RedisInbox:                       # your infrastructure, your store
    async def list_conversations(
        self, *, tags=None, limit=20
    ) -> list[ConversationSummary]:
        ...

recent = await app.conversations(tutor, tags={"user": uid}, index=RedisInbox())
```

The library owns `conversation()` / `send()` / `transcript` / `hydrate` and the
dev-grade default; the cross-conversation *inbox* stays yours. The protocol is
the only coupling point — a tag-scoped `list_conversations` — so a production
index plugs in with no change to the library.
