# Memory

The memory subsystem gives an `Agent` the ability to remember state
between `.run()` calls — within a single process (in-memory) or across
processes (persistent via SQLite or a custom backend).

## Installation

```bash
# InMemoryMemory ships with the base package
pip install t0-agents

# SqliteMemory requires the memory-sqlite extra
pip install 't0-agents[memory-sqlite]'
```

## Quick start

```python
from t0_agents import Agent, AgentConfig
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

# Session 1
r1 = await agent.run("Remind me to call John tomorrow at 3pm.")

# Session 2 — pass the episode_id to load prior context
r2 = await agent.run(
    "What was I supposed to do today?",
    episode_id=r1.episode_id,
)
```

## What is `MemoryProvider`?

`MemoryProvider` is a `@runtime_checkable` Protocol defined in
`t0_agents.core.types`. Any object implementing these three async
methods satisfies it:

```python
class MemoryProvider(Protocol):
    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]: ...
```

The `Agent` calls `load` before every run (if an `episode_id` is
provided) and `save` after. `search` is not called by the core loop
but is part of the protocol so custom backends expose a uniform query
surface.

## Built-in backends

### `InMemoryMemory`

Dict-backed, ephemeral. Useful for tests and single-process applications.

```python
from t0_agents.memory import InMemoryMemory

mem = InMemoryMemory()
agent = Agent(AgentConfig(..., memory=mem), t0=t0, tool_provider=client)
```

### `SqliteMemory`

Persistent, file-backed via `aiosqlite`. Requires the `memory-sqlite`
extra.

```python
from t0_agents.memory import SqliteMemory

mem = SqliteMemory("./agent_memory.db")
agent = Agent(AgentConfig(..., memory=mem), t0=t0, tool_provider=client)
```

The database file and table are created automatically on first use.

## Memory injection strategies

The `memory_strategy` field on `AgentConfig` controls how loaded memory
records are injected into the LLM prompt:

| Strategy | Behaviour |
|----------|-----------|
| `"auto"` (default) | Resolves to `"prepended_message"`. |
| `"prepended_message"` | Prepends a text summary to the user message. Works with any TensorZero function/variant. |
| `"system_template"` | Passes memory via `input.system.arguments.memory_context`. Requires a MiniJinja system template with a `memory_context` variable in `tensorzero.toml`. |

### When to use `"system_template"`

If your variant declares a system template:

```toml
[functions.assistant.variants.with_memory]
type = "chat_completion"
model = "gpt_oss_20"
templates.system.path = "functions/assistant/with_memory/system.minijinja"
```

```jinja
{# system.minijinja #}
You are a helpful assistant.
{% if memory_context %}

Previous conversation:
{{ memory_context }}
{% endif %}
```

Then set `memory_strategy="system_template"` to inject memory through
the template rather than the user message.

## Writing your own backend

Implement the three methods of `MemoryProvider`. Here's a Redis sketch:

```python
import json
from datetime import UTC, datetime

import redis.asyncio as redis

from t0_agents.memory import MemoryProvider, MemoryRecord


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
        self, query: str, *, limit: int = 5
    ) -> list[MemoryRecord]:
        # Simplified: scan all keys. For production, use RediSearch.
        results: list[MemoryRecord] = []
        async for key in self._redis.scan_iter(match="memory:*"):
            raw = await self._redis.lrange(key, 0, -1)
            for item in raw:
                rec = self._deserialize(item)
                if query.lower() in rec.user_message.lower():
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
            created_at=datetime.fromisoformat(d["created_at"]),
            extra=d.get("extra", {}),
        )
```

## Bringing your own RAG via MCP Resources

For RAG-style memory (semantic search over a knowledge base), consider
exposing your retriever as an MCP Resource rather than a
`MemoryProvider`. The LLM reads memory through MCP primitives
(`list_resources`, `read_resource`) instead of framework-injected
context. See the [FastMCP Resources docs](https://gofastmcp.com/servers/resources.md)
for details.

## Error handling

Memory operations are non-fatal. If `load` or `save` raises an
exception, the `Agent` logs a warning and continues. The run result is
still valid even if memory persistence failed.
