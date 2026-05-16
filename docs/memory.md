# Memory

The `run_agent` loop is stateless: every `Agent.invoke()` call starts with a fresh message history, and continuity is carried only by the optional `episode_id` argument. To let an agent *remember* earlier turns — conversations with the same user last week, facts it learned in a previous process, decisions it made before a redeploy — wire a `MemoryProvider` into `AgentConfig`.

Cross-run persistence for an `Agent`, keyed by TensorZero's episode id.

The memory subsystem lets an `Agent` remember earlier turns — both within
a single process (ephemeral, dict-backed) and across processes
(persistent, via SQLite or a custom backend). Configuration is a single
field on `AgentConfig`; the `Agent` handles load + save automatically
around every `.run()` / `.stream()` call.

```python
from t0_agents import Agent, AgentConfig, MemoryProvider
from t0_agents.memory import (
    InMemoryMemory,
    SqliteMemory,
    MemoryRecord,
    render_memory_summary,
)
```

---

## Installation

`InMemoryMemory`, `TensorZeroMemory`, `MemoryRecord`, and
`render_memory_summary` ship with the base distribution.
`TensorZeroMemory` is included because the `tensorzero` SDK is already
a base dependency — there is nothing to install separately.
`SqliteMemory` lives behind an opt-in extra because it pulls
[`aiosqlite`](https://pypi.org/project/aiosqlite/):

```bash
# Base (InMemoryMemory + TensorZeroMemory)
pip install t0-agents

# With SQLite persistence
pip install 't0-agents[memory-sqlite]'

# Or with uv
uv pip install 't0-agents[memory-sqlite]'
```

The store primitives themselves do not require Docker or a running
TensorZero gateway — you can exercise them in plain `asyncio`. The
Docker-required `Agent` flow is introduced in the second quickstart
below.

---

## Quickstart — store primitive (no Docker)

Exercise `InMemoryMemory` on its own to see the shape of a record, what
`render_memory_summary` returns, and how `search` ranks matches. Runs
in under a second with no external dependencies.

```python
import asyncio

from t0_agents.memory import (
    InMemoryMemory,
    MemoryRecord,
    render_memory_summary,
)


async def main() -> None:
    store = InMemoryMemory()

    await store.save(MemoryRecord(
        episode_id="demo",
        user_message="My name is Alice.",
        agent_response="Nice to meet you, Alice.",
        turns=1,
    ))
    await store.save(MemoryRecord(
        episode_id="demo",
        user_message="My favourite colour is teal.",
        agent_response="Got it — teal.",
        turns=1,
    ))

    records = await store.load("demo")
    print(f"Records: {len(records)}")
    print(render_memory_summary(records))

    hits = await store.search("colour")
    print(f"\nSearch 'colour' → {len(hits)} match(es)")


asyncio.run(main())
```

Output:

```
Records: 2
Previous conversation (2 turns):
- User: My name is Alice.
  Assistant: Nice to meet you, Alice.
- User: My favourite colour is teal.
  Assistant: Got it — teal.

Search 'colour' → 1 match(es)
```

Swap `InMemoryMemory()` for `SqliteMemory("./memory.db")` and the same
script now writes to disk — every method signature is identical because
both classes satisfy the `MemoryProvider` Protocol.

---

## Quickstart — `Agent` with memory (Docker required)

The real value shows up when memory wraps an agent loop. Boot the
TensorZero gateway, then run two `agent.invoke()` calls on the same
`episode_id` — the second one sees the first turn as context.

```bash
# Boot the gateway (from the repo root); requires GROQ_API_KEY in .env
docker compose up -d
```

```python
import asyncio
from collections.abc import Awaitable
from typing import cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import SqliteMemory


mcp = FastMCP("demo")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[object] = Client(mcp)

    async with await gateway as t0, client:
        agent = Agent(
            AgentConfig(
                name="assistant",
                function_name="agent",
                memory=SqliteMemory("./agent_memory.db"),
            ),
            t0=t0,
            tool_provider=client,
        )

        # Turn 1 — fresh episode
        r1 = await agent.invoke("My name is Alice. Please remember that.")
        print(f"Episode: {r1.episode_id}")
        print(f"Reply:   {r1.response}\n")

        # Turn 2 — reuse the episode id so memory loads the first turn
        r2 = await agent.invoke(
            "What is my name?",
            episode_id=r1.episode_id,
        )
        print(f"Reply:   {r2.response}")


asyncio.run(main())
```

The second call injects the first exchange into the prompt before the
LLM sees the new question, so the model can answer `"Alice"` even
though the gateway itself is stateless. After the run returns, the
record is committed to `./agent_memory.db` — restart the process and
the same `episode_id` still loads it.

A complete runnable version, with both `InMemoryMemory` and
`SqliteMemory` exercised in one script and a simulated process
restart, lives at
[`examples/09_memory/`](../examples/09_memory/).

---

## Concepts

### Memory is a lifecycle, not a tool

The LLM does not see `MemoryProvider` as a callable tool. Memory is
infrastructure the `Agent` hides around `run_agent`: it hydrates state
into the prompt before inference and persists the outcome after.
Because memory wraps the call rather than participating in it, you can
add, remove, or swap backends without touching the `tensorzero.toml`
configuration or any tool schemas.

### Lifecycle of one `agent.invoke()`

```
agent.invoke(message, episode_id=<id or None>)
   │
   ├─ 1. Load          memory.load(episode_id)  ─►  List[MemoryRecord]
   │                   (skipped when episode_id is None or
   │                    when memory_strategy="messages")
   │
   ├─ 2. Inject        render_memory_summary(records)
   │                   │
   │                   ├─ strategy "prepended_message":
   │                   │    summary prepended to the user message
   │                   │
   │                   ├─ strategy "system_template":
   │                   │    summary passed via
   │                   │    input.system.memory_context
   │                   │
   │                   └─ strategy "messages":
   │                        memory.load_history(episode_id) ─► list[Message]
   │                        threaded through input.messages verbatim;
   │                        records are NOT used on this path
   │
   ├─ 3. Run           run_agent(...)        ← regular loop, no memory knowledge
   │
   └─ 4. Save          memory.save(MemoryRecord(
                         episode_id        = result.episode_id,
                         user_message      = original user_message,
                         agent_response    = result.response,
                         turns             = result.turns,
                         inference_ids     = result.inference_ids,
                       ))
                       (skipped when result.episode_id is falsy —
                        never persists under an empty key)
```

`Run.events()` runs the same four phases around the streaming loop;
load and save sit at the outer edge, so every consumer of an
`AsyncIterator[AgentEvent]` sees memory hydrated into the prompt and
persisted after `DoneEvent`.

### Why two injection strategies ship

TensorZero offers two code paths for feeding context into an inference:

- **`input.system`** — the variant's MiniJinja system
  template is rendered with a `memory_context` variable. Requires you
  to edit `tensorzero.toml`, but keeps memory out of the message
  history (the template is a server-side concern).
- **`tensorzero.RawText` content blocks** — inject memory as part of
  the first user message. No TOML changes; works with any variant,
  including off-the-shelf ones you did not author.

The `memory_strategy` field picks between the two so you can start
with the no-config `"prepended_message"` path and upgrade to a
template when you want memory separated from the conversation.

### `"auto"` resolves to `"prepended_message"` today

The default `"auto"` currently resolves to `"prepended_message"`
because the TensorZero SDK does not yet expose an API to introspect a
function's active variant at runtime. When that API lands, `"auto"`
will prefer `"system_template"` whenever the variant declares a
`memory_context` system property and fall back to the message path
otherwise. Callers who want `"system_template"` today set it
explicitly.

### Persistence is per run, not per session

TensorZero has no explicit "close an episode" API — episodes are
defined implicitly by inferences that share an `episode_id`. The
`Agent` therefore writes a record on every `.run()` / `.stream()` call
that produces a truthy `episode_id`, rather than waiting for an
end-of-session signal that never comes. The side effect: memory is
always current, and each turn is independently recoverable if the
process crashes.

### Failure is non-fatal

If `memory.load()` raises, the `Agent` logs a warning and runs with an
empty context. If `memory.save()` raises, the `Agent` logs a warning
and returns the unchanged `AgentResult`. Memory is a *side effect* of
a run, not an output: losing a write must not invalidate an otherwise
successful inference. If you want hard failure on persistence errors,
wrap your backend and raise a `T0AgentsFatalError` subclass (see
[Error handling](#error-handling)).

### `search` is a seam, not a built-in retriever

`MemoryProvider.search(query, *, limit)` is part of the Protocol, but
the core loop never calls it. It exists so that custom backends — a
vector store, Elasticsearch, Redis with RediSearch, a local embeddings
index — expose a uniform query surface, and your application code can
run retrieval on the same object it hands the `Agent`. This library
does not ship a retriever; see
[Bringing your own backend](#bringing-your-own-backend) below for the
Advanced tier.

---

## API reference

### `MemoryProvider` (Protocol)

```python
@runtime_checkable
class MemoryProvider(Protocol):
    async def load(self, episode_id: str) -> list[Any]: ...
    async def save(self, record: Any) -> None: ...
    async def search(self, query: str, *, limit: int = 5) -> list[Any]: ...
```

`@runtime_checkable`, so `isinstance(obj, MemoryProvider)` works
without inheritance. The `Agent` itself calls `isinstance` at
construction time and raises `ConfigurationError` if `config.memory`
does not satisfy the Protocol.

| Method | Purpose | Called by the `Agent`? |
|---|---|---|
| `load` | Return every record for an episode, oldest first. | Yes, before each run when an `episode_id` was supplied. |
| `save` | Append a new record. | Yes, after each run with a truthy `result.episode_id`. |
| `search` | Query across episodes. | No — this is a Protocol slot for callers and custom retrievers. |

The Protocol types the record as `Any` to keep `core/` free of a
dependency on `memory/`; concrete backends return and accept
`MemoryRecord` instances.

### `MemoryRecord`

```python
@dataclass
class MemoryRecord:
    episode_id: str
    user_message: str
    agent_response: str
    turns: int
    inference_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = field(default_factory=dict)
```

| Field | Meaning |
|---|---|
| `episode_id` | TensorZero episode id. Primary grouping key. |
| `user_message` | The unmodified user input. Never the injected variant. |
| `agent_response` | The agent's final reply for that turn.  For chat agents this is the joined terminating text; for JSON agents it is the raw JSON string of the final attempt (parsed or not) — see [Structured outputs](structured-outputs.md#structured-agent). |
| `turns` | Number of inference turns that produced `agent_response`.  For JSON runs this equals `1 + json_parse_attempts` (one initial inference plus retries). |
| `inference_ids` | TensorZero inference ids for cross-reference with ClickHouse.  For JSON runs the list contains every retry attempt's id in order. |
| `created_at` | Timezone-aware UTC datetime. Populated at construction if omitted. |
| `extra` | Free-form dict for application metadata (labels, tags, traces). |

### `render_memory_summary`

```python
def render_memory_summary(
    records: list[MemoryRecord],
    *,
    max_records: int = 10,
) -> str
```

Turns records into a plain-text summary suitable for prompt injection.
Returns `""` when `records` is empty, so callers can concatenate without
a branch. When `len(records) > max_records`, keeps the most recent
`max_records` entries and includes the total turn count in the header.

### `InMemoryMemory`

```python
class InMemoryMemory:
    def __init__(self) -> None: ...
    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]: ...
```

Dict-backed, ephemeral. One process, no persistence, no locking —
concurrent `save` calls against the *same* episode are not protected by
any explicit synchronisation. Because Python lists are thread-safe for
single `append` operations under the GIL and the store is only touched
from asyncio tasks running on one event loop, the typical single-process
agent pattern is safe in practice. If you spawn threads that call
`save` concurrently, pick a backend with a real lock.

`search` performs a case-insensitive substring match across
`user_message` and `agent_response`, ordered most-recent-first, and
returns up to `limit` results.

### `SqliteMemory`

Requires the `memory-sqlite` extra. If the import fails, the symbol is
`None` at `t0_agents.memory.SqliteMemory` rather than raising, so your
module can import conditionally.

```python
class SqliteMemory:
    def __init__(self, db_path: str) -> None: ...
    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]: ...
```

Each method opens its own `aiosqlite` connection and closes it in an
`async with` block. That means concurrent `Agent.invoke()` calls on the
same process do not contend on a shared connection object; SQLite's own
file-level locking handles the cross-call coordination.

Table and index are created on first use via a cached `CREATE TABLE IF
NOT EXISTS` path. The schema:

```sql
CREATE TABLE memory_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id     TEXT    NOT NULL,
    user_message   TEXT    NOT NULL,
    agent_response TEXT    NOT NULL,
    turns          INTEGER NOT NULL,
    inference_ids  TEXT    NOT NULL,      -- JSON-encoded list[str]
    extra          TEXT    NOT NULL DEFAULT '{}',  -- JSON-encoded dict
    created_at     TEXT    NOT NULL       -- ISO-8601 timestamp
);
CREATE INDEX idx_memory_episode ON memory_records(episode_id);
```

`load` returns rows ordered by `id ASC` (insertion order). `search`
orders by `id DESC` (most recent first) and uses SQLite's
`LIKE ... COLLATE NOCASE` for case-insensitive matching.

### `TensorZeroMemory`

Reads cross-turn memory from TensorZero's `list_inferences` API. Ships
in the base distribution because `tensorzero` is already a base
dependency.

```python
class TensorZeroMemory:
    DEFAULT_LOAD_LIMIT = 10

    def __init__(
        self,
        gateway: AsyncTensorZeroGateway,
        *,
        function_name: str | None = None,
        tag_filter: dict[str, str] | None = None,
        extra_filters: InferenceFilter | None = None,
        load_limit: int = DEFAULT_LOAD_LIMIT,
        experimental_search: bool = False,
    ) -> None: ...

    # MemoryProvider Protocol
    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]: ...

    # Native-history extension
    async def load_history(
        self, episode_id: str,
    ) -> list[Message] | None: ...
```

| Parameter | Behaviour |
|---|---|
| `gateway` | An open `AsyncTensorZeroGateway`. Must outlive the memory instance — `load`/`load_history` issue calls on it. |
| `function_name` | Filter inferences by TensorZero function name. Strongly recommended for partition-pruning on ClickHouse-backed observability stores; `None` queries across functions. |
| `tag_filter` | Equality-only convenience: each `(key, value)` pair becomes an `InferenceFilterTag(=)` and the pairs are AND-combined. Pair with the same tags on `AgentConfig.tags` so reads and writes share a tenant scope. |
| `extra_filters` | Raw `InferenceFilter` passthrough for non-equality predicates (`InferenceFilterTime`, `InferenceFilterOr`, manual AND/NOT trees). AND-combined with `tag_filter` when both are set. |
| `load_limit` | Maximum number of inferences `load` may return. Defaults to `10`, matching `render_memory_summary`'s `max_records`. |
| `experimental_search` | When `True`, `search` forwards the query to TZ's `search_query_experimental`. The endpoint is flagged experimental in the SDK; defaults to `False` (no-op). |

`load` reverses the gateway's DESC-by-timestamp results to oldest-
first and produces one `MemoryRecord` per chat inference that emitted
text. `MemoryRecord.turns` is always `1` and `MemoryRecord.inference_ids`
contains a single id — this backend exposes one record per inference,
not per agent run.

`load_history` reuses the most recent inference's `input.messages` as
the cumulative history (TZ accumulates within an episode). Stored
content blocks are converted to runtime equivalents (`Text`,
`RawText`, `ToolCall`, `ToolResult`, `Thought`); `Template`, `File`,
and `Unknown` blocks are dropped with a single warning per call.

`save` is a no-op: TZ wrote the row at inference time. `search`
returns `[]` unless opted in via `experimental_search=True`.

### `AgentConfig` fields

```python
@dataclass(slots=True)
class AgentConfig:
    # …
    memory: MemoryProvider | None = None
    memory_strategy: Literal[
        "auto", "system_template", "prepended_message", "messages",
    ] = "auto"
    # …
```

| Field | Type | Default | Behaviour |
|---|---|---|---|
| `memory` | `MemoryProvider \| None` | `None` | When `None`, the `Agent` skips load and save entirely. When set, `Agent.__init__` validates the instance against the Protocol and raises `ConfigurationError` on mismatch. |
| `memory_strategy` | `Literal["auto", "system_template", "prepended_message", "messages"]` | `"auto"` | How loaded records are injected. Validated at `AgentConfig.__post_init__`; an invalid value raises `ConfigurationError`, as does picking `"messages"` against a backend that does not implement `load_history`. |

---

## Memory injection strategies

| Strategy | How memory reaches the LLM | When to use |
|---|---|---|
| `"auto"` (default) | Currently resolves to `"prepended_message"`. | You haven't decided yet, or you want the framework to pick the right path once TensorZero exposes variant introspection. |
| `"prepended_message"` | Text summary is prepended to the user message, separated by `---`. | Works with any TensorZero function or variant. No TOML changes required. |
| `"system_template"` | Summary is passed to the gateway as `input.system.memory_context` (the dict is the schema-described arguments object directly). | Your variant declares a MiniJinja system template with a `memory_context` variable. Keeps memory out of the message history. |
| `"messages"` | The prior conversation flows through `input.messages` as a TZ-native message history. No string rendering. | Pair with `TensorZeroMemory` (or any backend that implements `load_history`). Cleanest multi-turn semantics; preserves tool calls and tool results across turns; sidesteps the unbounded summary growth that `"prepended_message"` exhibits when the backend re-reads its own writes. |

### Using `"system_template"`

Declare a system template on the variant:

```toml
# tensorzero.toml
[functions.assistant.variants.with_memory]
type = "chat_completion"
model = "gpt_oss_20"
templates.system.path = "functions/assistant/with_memory/system.minijinja"
```

```jinja
{# functions/assistant/with_memory/system.minijinja #}
You are a helpful assistant.
{% if memory_context %}

Previous conversation:
{{ memory_context }}
{% endif %}
```

Then set the strategy explicitly:

```python
AgentConfig(
    name="assistant",
    function_name="assistant",
    memory=SqliteMemory("./memory.db"),
    memory_strategy="system_template",
)
```

The `Agent` will call
`run_agent(..., system_arguments={"memory_context": <summary>})` on
every run with loaded records. When no records match the
`episode_id`, no injection happens at all — neither the template
nor the user message is modified.

#### Combining memory with caller-supplied template variables

`Agent.invoke` and `Run.events` accept their own `system_arguments`
kwarg, which is shallow-merged with whatever the memory strategy
injects:

```python
await agent.invoke(
    "question",
    episode_id=ep,
    system_arguments={"max_turns": 5, "user_name": "Alice"},
)
```

Use this when the function's system template declares variables
beyond `memory_context` (e.g. `{{ max_turns }}`, `{{ user_name }}`).
On collision, **caller keys take precedence** — the explicit caller
value beats the implicit memory injection. Caller `None` (the
default) is byte-identical to today's behaviour: only the memory
strategy contributes.

#### Incompatible with raw `system="<str>"`

`memory_strategy="system_template"` writes a dict to `input.system`
via the configured template, so it cannot be combined with the raw
`system` string kwarg on `Agent.invoke` / `Run.events` — the two
shapes are mutually exclusive at the gateway. The `Agent` rejects
that combination up front with `ValueError`. If you need to pass a
raw system prompt at runtime (the prompt-as-a-service shape — see
[`system` vs `system_arguments`](api-reference.md#runtime-system-prompt-system-vs-system_arguments)),
switch to `memory_strategy="prepended_message"` (the default) or
`memory_strategy="messages"`. Both leave `input.system` untouched
and remain compatible with `system="<str>"`.

---

## Built-in backends

### `InMemoryMemory` — ephemeral

Ideal for tests, prototypes, and single-process scripts where losing
state at process exit is fine.

```python
from t0_agents.memory import InMemoryMemory

agent = Agent(
    AgentConfig(name="assistant", function_name="agent", memory=InMemoryMemory()),
    t0=t0,
    tool_provider=client,
)
```

### `SqliteMemory` — persistent, single-node

The right default for single-process deployments and local development
where "survive a restart" matters but "coordinate across nodes" does
not.

```python
from t0_agents.memory import SqliteMemory

agent = Agent(
    AgentConfig(
        name="assistant",
        function_name="agent",
        memory=SqliteMemory("./agent_memory.db"),
    ),
    t0=t0,
    tool_provider=client,
)
```

The DB file and table are created automatically on first write.

---

## Reading memory from TensorZero observability

`TensorZeroMemory` is the third reference backend, and it has a
different shape from the first two: it does not *store* anything. The
TensorZero gateway already persists every chat inference to ClickHouse
at the moment the call is made, so a "memory" that wraps the
gateway's own observability log is simultaneously the cheapest backend
to operate and the most aligned with TensorZero's design — episodes
are *defined* as the unit of cross-inference correlation. This backend
just exposes that log as a `MemoryProvider`.

The trade-off is that every `Agent.invoke` with an `episode_id` adds one
HTTP round-trip to the gateway before the inference (the read of the
prior history) — the gateway is the source of truth, so each turn
re-reads it. With the recommended `memory_strategy="messages"` that
read is one `list_inferences(limit=1)` call; with the records-shaped
strategies it is `limit=10` by default. Locally the round-trip is
~30–50 ms; against a remote TZ deployment, plan accordingly.

### Quickstart

```python
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import TensorZeroMemory


mcp = FastMCP("demo")


async def main() -> None:
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    ) as t0, Client(mcp) as client:
        agent = Agent(
            AgentConfig(
                name="assistant",
                function_name="agent",
                memory=TensorZeroMemory(
                    t0,
                    function_name="agent",
                    tag_filter={"customer_id": "alice"},
                ),
                memory_strategy="messages",
                tags={"customer_id": "alice"},
            ),
            t0=t0,
            tool_provider=client,
        )

        r1 = await agent.invoke("My name is Alice. Please remember.")
        r2 = await agent.invoke(
            "What is my name?", episode_id=r1.episode_id,
        )
        print(r2.response)  # → "Your name is Alice."
```

Pair the `tag_filter` on the memory with the same key/value on
`AgentConfig.tags`. The gateway tags every inference with the agent's
configured tags; the memory's filter scopes reads to those same tags.
With matching tags on both sides a single `customer_id` (or any other
domain key) gives you tenant isolation by construction.

### Why pick `memory_strategy="messages"`

The two earlier strategies render memory into a string — either
prepended to the user message or passed as a `system_arguments` field.
That round-trip is fine for in-process backends like `InMemoryMemory`
and `SqliteMemory` because they store the *original* user message
(not the prepended one) and the next read returns clean records.

`TensorZeroMemory` reads from TZ, which records what the gateway
*received* — i.e. the rendered, prepended user message. With the
default `"prepended_message"` strategy, every subsequent turn would
prepend a summary that itself contains the previous summaries,
turning N turns into O(2^N) tokens worth of nested history. The
`"messages"` strategy bypasses string rendering entirely:
`load_history` returns a TZ-native `list[Message]` — already in the
shape the gateway accepts — and the agent threads it through
`input.messages`. Tool calls and tool results from prior turns are
preserved alongside the user/assistant text, so the model can reason
about its own earlier work.

If you do choose `"system_template"` (because your variant has a
MiniJinja template that ships with the function), `TensorZeroMemory`
works against it without any further changes. Only the
`"prepended_message"` path has the unbounded-growth pitfall.

### Filters

Two complementary kwargs: `tag_filter` for the equality-only common
case (it covers `customer_id`, `tenant`, `env`, etc.) and
`extra_filters` for everything else.

```python
from tensorzero import InferenceFilterTime

memory = TensorZeroMemory(
    t0,
    function_name="agent",
    tag_filter={"customer_id": "alice"},
    extra_filters=InferenceFilterTime(
        comparison_operator=">=",
        time="2026-01-01T00:00:00Z",
    ),
)
```

Both are AND-combined when both are populated; either is accepted
alone. `tag_filter` builds one `InferenceFilterTag(=)` per dict
entry. For `!=` predicates, OR/NOT trees, or anything else the
`InferenceFilter` union supports, build the filter manually and pass
it via `extra_filters` (or subclass `TensorZeroMemory` and override
the private `_build_filters` method if you want full control).

### Reading and saving semantics

| Method | Behaviour |
|---|---|
| `load(episode_id)` | Issues `list_inferences(limit=load_limit, order_by=DESC)` and returns one `MemoryRecord` per chat inference that emitted text. JSON-output and tool-call-only inferences are skipped. Reversed to oldest-first to match `InMemoryMemory` / `SqliteMemory`. `MemoryRecord.turns` is always `1` and `MemoryRecord.inference_ids` is a single-element list — this backend exposes one record per inference, not per agent run. |
| `load_history(episode_id)` | Used by `memory_strategy="messages"`. Issues `list_inferences(limit=1)` and reuses the most recent inference's `input.messages` as the cumulative history (TZ accumulates within an episode), then appends the assistant turn from `output`. Returns `None` for empty episodes. |
| `save(record)` | No-op. The gateway already persisted the row at inference time; calling `save` again would duplicate the observability entry. |
| `search(query, *, limit=5)` | Returns `[]` by default. Pass `experimental_search=True` to the constructor to opt in; the query is forwarded to TZ's `search_query_experimental` field. The gateway flags that endpoint as experimental and warns it may cause full-table scans, so it is not appropriate for low-latency request paths. |

### Read-after-write timing

TensorZero's writer is asynchronous on the gateway side. Under heavy
local-Docker load, the row for an inference may not be visible to
`list_inferences` for a fraction of a second. Production paths read
the *prior* turn's history, not the inference that just happened, so
this is not normally observable. Tests that immediately read after
writing should sleep ~1 s before the read; never sleep in production
code — make the consumer of `load_history` resilient to occasional
empty results instead.

### Lifetime

`TensorZeroMemory` keeps a reference to the gateway. The gateway must
outlive the memory instance — close the `AsyncTensorZeroGateway` only
after every `Agent.invoke` / `Run.events` that depends on the memory has
completed. Inside `t0_agents.serve`, build the memory inside the
per-session `SessionFactory` so each request gets its own scope.

---

## Bringing your own backend

Any object that satisfies `MemoryProvider` qualifies. The three tiers
below walk from a ten-line in-process dict to a production-grade Redis
client to a RAG alternative driven by MCP Resources.

### Easy — minimal custom store

If you want memory isolated per `Agent` instance but keyed differently
(for example, partitioned by tenant), implement the three methods
directly against an `asyncio.Lock`-guarded dict. `InMemoryMemory`
itself is a 40-line reference at
[`src/t0_agents/memory/in_memory.py`](../src/t0_agents/memory/in_memory.py).
For a runnable no-Docker walkthrough of the store primitives on their
own, see [`examples/09_memory/easy.py`](../examples/09_memory/easy.py).

```python
import asyncio
from collections import defaultdict

from t0_agents.memory import MemoryRecord


class DictMemory:
    def __init__(self) -> None:
        self._store: dict[str, list[MemoryRecord]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        async with self._lock:
            return list(self._store[episode_id])

    async def save(self, record: MemoryRecord) -> None:
        async with self._lock:
            self._store[record.episode_id].append(record)

    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]:
        q = query.lower()
        results: list[MemoryRecord] = []
        async with self._lock:
            for records in self._store.values():
                for r in records:
                    hay = f"{r.user_message}\n{r.agent_response}".lower()
                    if q in hay:
                        results.append(r)
                        if len(results) >= limit:
                            return results
        return results
```

### Intermediate — Redis backend

A Redis implementation fits in under 50 lines once you pick a
serialisation format. This one stores records as JSON in per-episode
lists and scans for search matches — swap in RediSearch for production
scale.

```python
import json
from datetime import UTC, datetime

import redis.asyncio as redis

from t0_agents.memory import MemoryRecord


class RedisMemory:
    def __init__(self, url: str = "redis://localhost:6379") -> None:
        self._redis = redis.from_url(url)

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        raw = await self._redis.lrange(f"memory:{episode_id}", 0, -1)
        return [self._deserialize(r) for r in raw]

    async def save(self, record: MemoryRecord) -> None:
        data = json.dumps({
            "episode_id": record.episode_id,
            "user_message": record.user_message,
            "agent_response": record.agent_response,
            "turns": record.turns,
            "inference_ids": record.inference_ids,
            "created_at": record.created_at.isoformat(),
            "extra": record.extra,
        })
        await self._redis.rpush(f"memory:{record.episode_id}", data)

    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]:
        # Simplified: scan all keys. For production, use RediSearch.
        results: list[MemoryRecord] = []
        async for key in self._redis.scan_iter(match="memory:*"):
            raw = await self._redis.lrange(key, 0, -1)
            for item in raw:
                rec = self._deserialize(item)
                hay = f"{rec.user_message}\n{rec.agent_response}".lower()
                if query.lower() in hay:
                    results.append(rec)
                    if len(results) >= limit:
                        return results
        return results

    @staticmethod
    def _deserialize(raw: bytes) -> MemoryRecord:
        d = json.loads(raw)
        return MemoryRecord(
            episode_id=d["episode_id"],
            user_message=d["user_message"],
            agent_response=d["agent_response"],
            turns=d["turns"],
            inference_ids=d["inference_ids"],
            created_at=datetime.fromisoformat(d["created_at"]).replace(
                tzinfo=UTC,
            ),
            extra=d.get("extra", {}),
        )
```

### Advanced — RAG via a vector store

For retrieval-augmented memory, drive `search` with your vector
retriever and leave `load` / `save` for transactional history.

```python
from t0_agents.memory import MemoryRecord, SqliteMemory


class RagMemory:
    """SQLite for history; vector store for semantic search."""

    def __init__(self, db_path: str, retriever) -> None:
        self._history = SqliteMemory(db_path)
        self._retriever = retriever  # any object with .query(text, k)

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        return await self._history.load(episode_id)

    async def save(self, record: MemoryRecord) -> None:
        await self._history.save(record)
        await self._retriever.index(
            text=f"{record.user_message}\n{record.agent_response}",
            metadata={"episode_id": record.episode_id},
        )

    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]:
        hits = await self._retriever.query(query, k=limit)
        records: list[MemoryRecord] = []
        for hit in hits:
            ep_records = await self._history.load(hit.metadata["episode_id"])
            records.extend(ep_records)
        return records[:limit]
```

This pattern keeps the library vendor-neutral — pick your own
embeddings and vector store — while giving the `Agent` a familiar
`MemoryProvider` seam to call into.

For a runnable no-Docker example of a custom `MemoryProvider` written
from scratch — implementing per-tenant namespacing and per-episode LRU
eviction in under 70 lines — see
[`examples/09_memory/advanced.py`](../examples/09_memory/advanced.py).

### Advanced — memory as MCP Resources

If the memory you care about is a knowledge base the *LLM* should
read (not context the framework should inject), expose it as MCP
Resources or as MCP tools instead of plugging it into
`MemoryProvider`. The runnable example at
[`examples/09_memory/server.py`](../examples/09_memory/server.py)
ships a FastMCP server with `remember(key, value)` / `recall(key)`
tools the LLM invokes explicitly — a different pattern, useful when
you want the model in control of what it stores and retrieves.

See the
[FastMCP Resources docs](https://gofastmcp.com/servers/resources)
for the resource-based variant.

---

## Error handling

Persistence failures are non-fatal by design. The `Agent` wraps both
calls in a `try / except Exception`, logs a warning, and continues.

| Failure | Effect | Log level | Can I observe it? |
|---|---|---|---|
| `memory.load` raises | Run proceeds with an empty context. | `WARNING` (with traceback). | Yes — `logging.getLogger("t0_agents")` or a child logger. |
| `memory.load_history` raises (`memory_strategy="messages"`) | Run proceeds with no prior messages. | `WARNING` (with traceback). | Yes — same logger. |
| `memory.save` raises | `AgentResult` is still returned unchanged. | `WARNING` (with traceback). | Yes — same logger. |

If you want hard failure instead, wrap your backend and raise
`T0AgentsFatalError` (or a subclass you define). Any exception
subclassed from `T0AgentsFatalError` propagates past the `Agent`'s
non-fatal handler — this is the same contract guardrails use for
hard aborts.

```python
from t0_agents import T0AgentsFatalError
from t0_agents.memory import MemoryRecord, MemoryProvider


class MemoryUnavailable(T0AgentsFatalError):
    pass


class StrictBackend:
    def __init__(self, inner: MemoryProvider) -> None:
        self._inner = inner

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        try:
            return await self._inner.load(episode_id)
        except Exception as exc:
            raise MemoryUnavailable(
                f"Cannot load memory for {episode_id}: {exc}",
            ) from exc

    async def save(self, record: MemoryRecord) -> None:
        try:
            await self._inner.save(record)
        except Exception as exc:
            raise MemoryUnavailable(
                f"Cannot save memory for {record.episode_id}: {exc}",
            ) from exc

    async def search(
        self, query: str, *, limit: int = 5,
    ) -> list[MemoryRecord]:
        return await self._inner.search(query, limit=limit)
```

For the broader error taxonomy and how memory warnings sit alongside
`MaxTurnsExceededError`, `GuardrailBlockedError`, `ConfigurationError`,
and tool-level `McpError` / `ToolError` wrapping, see
[Error handling](error-handling.md).

---

## Production considerations

### Concurrency model

- `InMemoryMemory` has no locking. Safe under a single asyncio event
  loop; not safe across threads. Use it for tests and single-process
  agents only.
- `SqliteMemory` opens a fresh connection per method call and relies on
  SQLite's file-level locking. Safe for multiple `Agent` instances in
  one process. Across processes writing the same file, contention
  manifests as `database is locked` retries (aiosqlite default
  behaviour); for heavy write traffic, move to Redis or Postgres.
- `TensorZeroMemory` holds an `AsyncTensorZeroGateway` reference and
  delegates concurrency to whatever the gateway provides. Each
  `load`/`load_history` is one HTTP round-trip; safety across
  concurrent `Agent.invoke` calls follows the gateway's connection pool.
  Build the memory inside the per-session `SessionFactory` in
  `t0_agents.serve` so the gateway lifetime always wraps the memory
  lifetime, never the other way round.
- Custom backends are whatever their driver guarantees. Document the
  concurrency model where you document the backend.

### Durability and backups

`SqliteMemory` writes one `INSERT` per `save` and commits immediately,
so a clean shutdown loses nothing. A hard crash during the commit may
lose the in-flight record, but never corrupts the DB file thanks to
SQLite's journalling. For on-disk backups, any standard SQLite snapshot
tool works — the Agent never holds a long-lived connection.

For stronger durability (cross-node replication, point-in-time
recovery), swap in a Postgres or Redis implementation.

`TensorZeroMemory` inherits its durability from TensorZero — the
gateway is the writer, ClickHouse is the store. Whatever your TZ
deployment promises (single-node Docker, a managed cloud cluster, a
self-managed replicated ClickHouse) is what your memory promises. No
in-process state to back up.

### Size management

Neither built-in backend bounds record count. `render_memory_summary`
keeps only the last 10 records when it builds the prompt injection, so
runaway storage does not poison the LLM context — but the DB itself
grows forever unless you prune. Patterns that work:

- Periodic `DELETE FROM memory_records WHERE created_at < ?` (SQLite).
- Application-level retention policies in a wrapping
  `MemoryProvider` that calls `save` and periodically prunes old
  rows.
- Per-tenant episode namespacing (below) so you can drop an entire
  tenant with one `DELETE WHERE episode_id LIKE ?`.

### Episode-id namespacing

`episode_id` is a TensorZero-owned string — you pass `None` for a fresh
episode and the gateway returns the allocated id on the first
inference. If you want to isolate memory per tenant, per user, or per
deployment, wrap the backend with a namespace prefix rather than trying
to reuse the id across domains:

```python
class NamespacedMemory:
    def __init__(self, inner: MemoryProvider, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        return await self._inner.load(f"{self._prefix}:{episode_id}")
    # save / search follow the same pattern
```

The prefix lives entirely in your store; the `Agent` still forwards
TensorZero's raw `episode_id` in and out.

For deployment topology, sticky sessions, and the broader runtime
picture, see [Production](production.md).

---

## Observability

Memory does not emit its own TensorZero inference records — it only
reads and writes `MemoryRecord` rows. The linkage to the gateway's
analytics is transitive:

- `AgentResult.episode_id` is the single key that ties a stored
  record to every `ChatInferenceResponse` logged to ClickHouse for
  that episode.
- `AgentResult.inference_ids` (which the `Agent` copies into the
  saved `MemoryRecord`) resolves each turn of the run back to its
  individual inference row.
- `MemoryRecord.extra` is a free-form dict intended for application
  metadata — labels, trace ids, experiment arms. It round-trips
  through both built-in backends (JSON-encoded in SQLite), so you can
  use it to attach whatever your dashboards need without extending
  the Protocol.

The rendered summary that reaches the LLM is visible as part of the
prompt. If you serve your agent over HTTP or expose streaming events,
`render_memory_summary(records)` is the exact string the model
receives — log it from your client if you want parity with what the
gateway sees.

For the end-to-end observability story (`t0.feedback`, ClickHouse
tables, per-episode analytics), see
[Observability](observability.md).

---

Memory hydrates the prompt with whatever the agent remembers. Once the context is in place, you often want to constrain what the agent may *do* with it — require approval for expensive tools, cap cost or tool-call budgets per run, block dangerous tool names, encode a domain rule of your own. That policy layer is the subject of the next chapter: [Guardrails](guardrails.md).

---

## External references

- TensorZero — [Episodes & inference routing](https://www.tensorzero.com/docs/gateway/guides/episodes/)
  · [`input.system` schema](https://www.tensorzero.com/docs/gateway/api-reference/inference/)
  · [`tensorzero.RawText` content blocks](https://www.tensorzero.com/docs/gateway/api-reference/inference/#message-content-blocks)
- FastMCP — [`Client` interface](https://gofastmcp.com/clients/overview)
  · [Resources](https://gofastmcp.com/servers/resources) (for the
  resource-driven memory alternative)
- [MiniJinja](https://docs.rs/minijinja/latest/minijinja/) — the template engine
  TensorZero renders `templates.system.path` through when
  `memory_strategy="system_template"` is active.
- [`aiosqlite`](https://aiosqlite.omnilib.dev/) — the async SQLite
  driver `SqliteMemory` depends on when the `memory-sqlite` extra is
  installed.
