# Phase B — Memory (Repository Pattern)

> Status: **ready to start**. Phase dep: A complete. Open questions
> resolved: T0 Q4 (both injection paths supported), T0 Q6 (no episode
> close API; persist per `run()`), FastMCP Q4 (Memory stays a
> framework concept; MCP Resources documented as opt-in).
> Delivers: **Milestone M2 — Stateful Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md) §3.2,
  §4 (how memory composes)
- [`../01-implementation-plan.md`](../01-implementation-plan.md)
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q4, T0 Q6,
  FastMCP Q4
- [`../problemas.md`](../problemas.md) — Problem 3
- [`phase-a-agent-core.md`](phase-a-agent-core.md)

---

## 1. Goal

Give `Agent` the ability to remember state between `.run()` calls —
both within a single process (in-memory) and across processes
(persistent). The API is **Repository-pattern**: the framework defines
a `MemoryProvider` Protocol; two reference implementations ship
(`InMemoryMemory`, `SqliteMemory`). Developers writing their own
backend (Redis, MongoDB, Postgres) implement the Protocol — we don't
ship every backend ourselves.

After Phase B:

```python
from t0_agents import Agent, AgentConfig
from t0_agents.memory import SqliteMemory

agent = Agent(
    AgentConfig(
        name="personal_assistant",
        function_name="agent",
        memory=SqliteMemory("./agent_memory.db"),
    ),
    t0=t0,
    tool_provider=client,
)

# Session 1
await agent.run("Remind me to call John tomorrow at 3pm.")

# Process exits, restarts tomorrow morning

# Session 2 — same episode_id carried by the memory row
result = await agent.run("What was I supposed to do today?", episode_id="...")
# result.response contains "Call John at 3pm"
```

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/memory/` with:
  - `base.py` — `MemoryProvider` Protocol, `MemoryRecord` dataclass.
  - `in_memory.py` — `InMemoryMemory` reference backend (dict-backed).
  - `sqlite.py` — `SqliteMemory` backend behind `[memory-sqlite]`
    extra.
- Extension of `AgentConfig` with `memory: MemoryProvider | None =
  None`.
- Integration in `core/agent.py` — hydration before `run_agent`,
  persistence after.
- `memory_strategy` field on `AgentConfig` to pick injection path (see
  §4.3 — default decided once T0 Q4 answered).
- Tests under `tests/test_memory/`.
- Example `examples/09_memory/` with both backends.
- Public doc page `docs/memory.md`.
- Implementation log.

### 2.2 Out of scope (reserved for later)

- RAG engine / vector retrieval. If a developer wants semantic search
  memory, they implement `MemoryProvider.search` against their own
  vector store. Not our job to pick `sentence-transformers` vs OpenAI
  embeddings.
- Redis / Mongo / Postgres backends. Ship as example implementations
  under `docs/memory.md` appendix or as community contributions.
- Compression, summarisation, or memory-eviction policies. Leave that
  to the developer's `MemoryProvider` implementation.
- Memory sharing across agents (multi-agent composition, Phase G).

### 2.3 Non-goals

- Do not encode memory semantics in TensorZero config. Memory stays
  out of `tensorzero.toml`; it's a framework-level concept.
- Do not change `AgentResult`. Memory is a side effect of `run()`;
  callers inspect memory via `memory.load(episode_id)` if they need to.

---

## 3. Constraints (from the contract)

1. `mcp/` stays standalone. `run_agent` unchanged by this phase.
2. `memory/` depends only on `core/`. No other sibling imports.
3. `memory-sqlite` extra ships `aiosqlite` as the only new dep.
4. TensorZero-first: memory content enters T0 via standard primitives
   (system template args **or** prepended messages — see T0 Q4).
5. Async-first: every `MemoryProvider` method is `async def`.

---

## 4. Module additions

### 4.1 `src/t0_agents/memory/base.py` (new)

```python
# sketch — finalised after T0 Q4
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class MemoryRecord:
    """One stored interaction.

    The exact shape is intentionally narrow — it's what's needed to
    rehydrate a conversation or render a memory summary, not the full
    ChatInference raw log (which lives in ClickHouse).
    """
    episode_id: str
    user_message: str
    agent_response: str
    turns: int
    inference_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = field(default_factory=dict)  # user-extensible


@runtime_checkable
class MemoryProvider(Protocol):
    """Repository-style memory interface.

    The Agent calls `load` before every run and `save` after. `search`
    is optional; default backends implement it as a simple substring
    scan. Complex backends (vector stores) override.
    """

    async def load(self, episode_id: str) -> list[MemoryRecord]: ...
    async def save(self, record: MemoryRecord) -> None: ...
    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]: ...
```

### 4.2 `src/t0_agents/memory/in_memory.py` (new)

```python
from collections import defaultdict

from t0_agents.memory.base import MemoryProvider, MemoryRecord


class InMemoryMemory:
    """Ephemeral, dict-backed MemoryProvider.

    Useful for tests and single-process applications that want session-
    but-not-cross-process continuity.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryRecord]] = defaultdict(list)

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        return list(self._store[episode_id])

    async def save(self, record: MemoryRecord) -> None:
        self._store[record.episode_id].append(record)

    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        results = []
        for records in self._store.values():
            for r in records:
                if query.lower() in r.user_message.lower() or query.lower() in r.agent_response.lower():
                    results.append(r)
                    if len(results) >= limit:
                        return results
        return results
```

### 4.3 `src/t0_agents/memory/sqlite.py` (new, behind `[memory-sqlite]`)

Table schema (final — same shape regardless of `memory_strategy`
because the strategy affects how memory is *injected* into prompts,
not how it's *stored*):

```sql
CREATE TABLE IF NOT EXISTS memory_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    turns INTEGER NOT NULL,
    inference_ids TEXT NOT NULL,  -- JSON array
    extra TEXT NOT NULL DEFAULT '{}', -- JSON object
    created_at TEXT NOT NULL       -- ISO-8601
);
CREATE INDEX IF NOT EXISTS idx_memory_episode ON memory_records(episode_id);
```

Implementation uses `aiosqlite`, connection pooling per instance.
ImportError guard at top of module per contract §2.3.

### 4.4 `src/t0_agents/core/config.py` (extend)

```python
@dataclass(slots=True)
class AgentConfig:
    ...
    memory: MemoryProvider | None = None
    memory_strategy: Literal["auto", "system_template", "prepended_message"] = "auto"
```

`"auto"` resolution rule (final, per T0 Q4):

1. On the **first** `run()` call against a given function/variant,
   the framework introspects the system template via
   `t0.get_function_config(function_name)` (if available) and inspects
   whether the system schema declares a `memory_context` property.
2. If yes → `system_template`. If no → `prepended_message`. Cached for
   the lifetime of the `Agent`.
3. Developers can override at construction time by passing the
   strategy explicitly, skipping introspection.

If the introspection API is unavailable (e.g. older T0 SDK), `"auto"`
defaults to `"prepended_message"` (the strategy that requires no TOML
support).

### 4.5 `src/t0_agents/core/agent.py` (extend)

Integrate memory into `Agent.run`:

```python
async def run(self, user_message, *, episode_id=None):
    memory_records: list[MemoryRecord] = []
    if self._config.memory and episode_id:
        memory_records = await self._config.memory.load(episode_id)

    # Prepare prompt prefix according to memory_strategy (see §4.6)
    injected_user_message, injected_kwargs = self._prepare_memory_injection(
        user_message, memory_records
    )

    result = await run_agent(..., **injected_kwargs)

    if self._config.memory:
        await self._config.memory.save(MemoryRecord(
            episode_id=result.episode_id,
            user_message=user_message,
            agent_response=result.response,
            turns=result.turns,
            inference_ids=result.inference_ids,
        ))

    return result
```

### 4.6 Memory injection helper (final per T0 Q4)

Two paths, chosen by `memory_strategy`:

- **`"system_template"`** — pass `input.system = {"arguments":
  {"memory_context": <rendered summary>}}` to every `t0.inference(...)`
  call for this run. Requires `run_agent` to accept a new
  `system_arguments: dict[str, Any] | None = None` kwarg — a contract
  §6.1-compliant widening of `run_agent`. The variant TOML must declare
  matching `templates.system.path` + `schemas.system.path` with a
  `memory_context` property:

  ```toml
  [functions.assistant]
  type = "chat"
  schemas.system.path = "functions/assistant/system_schema.json"

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

- **`"prepended_message"`** — prepend a `tensorzero.RawText` content
  block (the SDK type at `tensorzero/types.py:110-112`, mapping to the
  `tensorzero::raw_text` block kind) ahead of the user message:

  ```python
  from tensorzero import RawText

  messages = [
      Message(
          role="user",
          content=[
              RawText(value=f"Previous conversation:\n{summary}\n---"),
              # original user message follows
          ],
      ),
      Message(role="user", content=user_message),
  ]
  ```

  `RawText` bypasses MiniJinja templating, so it survives any system
  template or variant. No TOML changes required. Slight cost: the
  prepended text shows up in ClickHouse `messages` rows, which may
  pollute analytics queries that aggregate user inputs.

Summary format (default, replaceable via `MemoryProvider.summarize`
override — covered in §6 of the public docs):

```
Previous conversation (3 turns):
- User: Remind me to call John tomorrow at 3pm.
  Assistant: Reminder set for tomorrow at 3pm: Call John.
- ...
```

### 4.7 `src/t0_agents/memory/__init__.py`

```python
from t0_agents.memory.base import MemoryProvider, MemoryRecord
from t0_agents.memory.in_memory import InMemoryMemory

try:
    from t0_agents.memory.sqlite import SqliteMemory
except ImportError:
    SqliteMemory = None  # type: ignore[assignment]

__all__ = ["InMemoryMemory", "MemoryProvider", "MemoryRecord", "SqliteMemory"]
```

`SqliteMemory` re-export is `None` when `aiosqlite` isn't installed,
so `from t0_agents.memory import MemoryProvider` still works even
without the extra.

---

## 5. Tests

Location: `tests/test_memory/`.

### 5.1 Unit tests (offline)

For each backend (`InMemoryMemory`, `SqliteMemory`):

- [ ] `test_round_trip` — save then load returns the record.
- [ ] `test_multiple_episodes_isolated` — different `episode_id`s
      don't cross-contaminate.
- [ ] `test_search_substring` — search finds a record.
- [ ] `test_concurrent_saves` — two simultaneous saves on the same
      episode don't clobber each other.

Agent-integration tests (still offline, using `FakeT0`/`FakeMCPClient`
and `InMemoryMemory`):

- [ ] `test_agent_with_memory_hydrates_and_persists` — `Agent.run()`
      twice, second run sees the first in its injected context.
- [ ] `test_agent_without_memory_unchanged` —
      `memory=None` path identical to Phase A.
- [ ] `test_memory_strategy_system_template` — when
      `memory_strategy="system_template"`, `run_agent` receives
      `system_arguments={"memory_context": ...}`.
- [ ] `test_memory_strategy_prepended_message` — when
      `memory_strategy="prepended_message"`, the `user_message`
      passed to `run_agent` is rewritten.

### 5.2 Integration tests (live gateway)

Marked `@pytest.mark.integration`:

- [ ] `test_agent_remembers_across_runs` — two successive `Agent.run`
      calls, second asks about the first. Use `SqliteMemory` backed
      by a temp file.
- [ ] `test_agent_remembers_across_processes` — write session 1 in one
      process (or one pytest function), read in another; assert
      episode-level continuity. Use in-process subprocess or a
      teardown/setup boundary.

### 5.3 Regression

- [ ] `tests/test_mcp/` unchanged.
- [ ] `tests/test_core/` unchanged (Phase A tests) — except possibly
      the parity tests, which gain a `memory=None` explicit
      parameter if Python typing requires it.

---

## 6. Example

Location: `examples/09_memory/`.

- `server.py` — two tools: `remember(key, value)` (writes to a
  framework-agnostic store), `recall(key)` (reads).
  *Caveat: these tools are separate from `MemoryProvider` — the
  example intentionally shows the distinction between "LLM-visible
  tools" and "framework-managed memory".*
- `main.py` — two backends side by side:

  ```python
  from t0_agents.memory import InMemoryMemory, SqliteMemory

  # Session 1 — in-memory (ephemeral)
  agent = Agent(AgentConfig(..., memory=InMemoryMemory()), t0=t0, tool_provider=client)

  # Session 2 — SQLite (persistent)
  agent = Agent(AgentConfig(..., memory=SqliteMemory("./agent.db")), ...)
  ```

- `README.md` — explains the strategy choice, points at T0 Q4 answer.

---

## 7. Documentation

- New: `docs/memory.md` — public-facing guide.
- Sections:
  - "What is `MemoryProvider`?" — Protocol signature.
  - "Built-in backends" — `InMemoryMemory`, `SqliteMemory`.
  - "Writing your own" — the Redis / Mongo / Postgres pattern, with
    a short Redis sketch inline.
  - "Memory injection strategies" — explain `memory_strategy` and
    when to pick which.
  - "Bringing your own RAG via MCP Resources" — FastMCP Q4 pointer.
- Update `docs/internal/agentic-feature/developer-guide.md` §4 (public
  API), §12 (knobs).
- `CHANGELOG.md` — `0.3.0` release line; call out the
  `memory-sqlite` extra.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/memory/__init__.py`.
- [ ] `src/t0_agents/memory/base.py` — Protocol + `MemoryRecord`.
- [ ] `src/t0_agents/memory/in_memory.py`.
- [ ] `src/t0_agents/memory/sqlite.py` — behind `[memory-sqlite]`.
- [ ] `src/t0_agents/core/config.py` — `memory` + `memory_strategy`
      fields.
- [ ] `src/t0_agents/core/agent.py` — memory hooks in `run()`.
- [ ] `src/t0_agents/mcp/loop.py` — `system_arguments: dict[str, Any] | None = None`
      kwarg added; forwarded into `input.system.arguments` when set.
      Defaults to `None` so existing behaviour is preserved (contract
      §6.1 widening).
- [ ] `pyproject.toml` — `[project.optional-dependencies]` adds
      `memory-sqlite = ["aiosqlite>=0.20.0"]`.
- [ ] `uv run ruff check src/` green.
- [ ] `uv run mypy src/` green in strict mode.

### 8.2 Tests

- [ ] `tests/test_memory/test_in_memory.py`.
- [ ] `tests/test_memory/test_sqlite.py`.
- [ ] `tests/test_memory/test_agent_integration.py`.
- [ ] Offline pytest green. Integration pytest green with `docker
      compose up -d`.

### 8.3 Example

- [ ] `examples/09_memory/server.py`.
- [ ] `examples/09_memory/main.py`.
- [ ] `examples/09_memory/README.md`.

### 8.4 Docs

- [ ] `docs/memory.md`.
- [ ] `developer-guide.md` §4, §12.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `docs/internal/agentic-feature/logs/phase-b-implementation-log.md`
      with the T0 Q4 decision recorded, any contract amendments, and
      the M2 release-notes snippet.

### 8.6 Review

- [ ] `mcp/` public API still byte-identical (or widened only by
      `system_arguments` kwarg).
- [ ] `InMemoryMemory` and `SqliteMemory` pass the same behavioural
      tests (Liskov sanity).
- [ ] Example runs locally end-to-end across a simulated "process
      restart".

---

## 9. Acceptance criteria (Milestone M2)

1. `from t0_agents.memory import InMemoryMemory, SqliteMemory,
   MemoryProvider` works.
2. `examples/09_memory/` demonstrates cross-session continuity and
   runs green.
3. An external user can implement `MemoryProvider` against Redis in
   under 80 lines (confirmed by a reference snippet in
   `docs/memory.md`).
4. `uv run pytest -m "not integration"` green.
5. Phase-A acceptance criteria still hold (M1 is not regressed).
6. T0 Q4 answer is recorded in both
   [`../02-open-questions.md`](../02-open-questions.md) and the phase
   log; any required contract amendment is applied in the same PR.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| T0 Q4 answer comes late, phase stalls | medium | Proceed with `memory_strategy="prepended_message"` path since it requires no T0 changes. Revisit template path when answer arrives. |
| SQLite write contention with concurrent `run()` calls | medium | Use `aiosqlite` connection-per-call pattern or a single-writer queue. Benchmark in tests. |
| Memory summary blows past the model's context window | medium | Surface the summarisation hook in `MemoryProvider` — default implementation shows last N turns; developers override. |
| Developers expect automatic episode closure | low | Answer T0 Q6; in the meantime document "memory upserts on every run". |
| `MemoryRecord.extra` becomes a dumping ground | low | Document it as "backend-owned metadata" and reserve no fields ourselves. |

---

## 11. Estimated surface

- **New files:** ~9 (4 src, 3 test, 3 example) plus `docs/memory.md`.
- **Modified files:** 4 (`core/config.py`, `core/agent.py`,
  `pyproject.toml`, `CHANGELOG.md`), possibly
  `mcp/loop.py`.
- **Net new public symbols:** 4 (`MemoryProvider`, `MemoryRecord`,
  `InMemoryMemory`, `SqliteMemory`).
- **New deps:** 1, under extra (`aiosqlite`).
