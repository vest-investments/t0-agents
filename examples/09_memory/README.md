# Example 09 — Agent with Memory (Milestone M2)

Demonstrates `MemoryProvider` integration with `Agent`. The agent
remembers context across successive `.run()` calls, both in-memory
(ephemeral) and via SQLite (persistent across process restarts).

## Quick start

```bash
# 1. Boot the TensorZero gateway
docker compose up -d          # from repo root

# 2. Install the memory-sqlite extra
pip install 't0-agents[memory-sqlite]'

# 3. Run the example
python main.py
```

## What this shows

| Backend | Persistence | Use case |
|---------|-------------|----------|
| `InMemoryMemory` | Process lifetime | Tests, prototyping |
| `SqliteMemory` | Disk (`.db` file) | Single-node production, local dev |

Both backends implement the `MemoryProvider` Protocol, so swapping one
for the other is a single-line change.

## Memory injection strategies

The `memory_strategy` field on `AgentConfig` controls how stored context
is injected into the prompt:

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| `"auto"` (default) | Resolves to `"prepended_message"` | Most cases |
| `"prepended_message"` | Prepends a text summary to the user message | Works with any TensorZero function/variant |
| `"system_template"` | Passes memory via `input.system.arguments.memory_context` | When your variant declares a MiniJinja system template with a `memory_context` variable |

## Writing your own backend

Implement the `MemoryProvider` Protocol:

```python
from t0_agents.memory import MemoryProvider, MemoryRecord

class RedisMemory:
    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]: ...
```

See `docs/memory.md` for a complete Redis sketch.
