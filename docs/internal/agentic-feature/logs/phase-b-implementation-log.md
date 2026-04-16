# Phase B Implementation Log

Phase: B (Memory)
Milestone: M2 — Stateful Agent
Date: 2026-04-15

## Divergences from Spec

### D1: `MemoryProvider` Protocol location

**Spec (§4.1)** places `MemoryProvider` in `memory/base.py`.

**Actual:** Defined in `core/types.py` alongside `ToolProvider`. Re-exported
from `memory/__init__.py` so `from t0_agents.memory import MemoryProvider`
still works.

**Reason:** `AgentConfig` (in `core/config.py`) declares `memory:
MemoryProvider | None` and `Agent.__init__` (in `core/agent.py`) does an
`isinstance` check. Both are in `core/`, and importing from `memory/` would
create `core/ → memory/` — the wrong direction per contract §2.2. Placing
the Protocol in `core/types.py` follows the established pattern
(`ToolProvider` is already there) and keeps the dependency graph clean.

**Impact:** None on callers. The import path `t0_agents.memory.MemoryProvider`
works identically to what the spec described.

### D2: `"auto"` memory_strategy always resolves to `"prepended_message"`

**Spec (§4.4)** describes `"auto"` introspecting the function's system schema
via `t0.get_function_config(function_name)` to detect a `memory_context`
property.

**Actual:** `get_function_config` does not exist in the TensorZero SDK
v2026.4.0 (validated by grepping the installed package — zero matches). The
spec anticipated this fallback: "If the introspection API is unavailable,
`auto` defaults to `prepended_message`."

**Resolution:** `Agent._resolve_memory_strategy()` returns
`"prepended_message"` when strategy is `"auto"`. A `# TODO` comment documents
the future introspection path for when the SDK gains that API.

**Impact:** None. Developers who want `system_template` can set it explicitly.

### D3: `datetime.now(UTC)` instead of `datetime.utcnow()`

**Spec (§4.1)** uses `datetime.utcnow()` in `MemoryRecord.created_at`
default factory.

**Actual:** Uses `datetime.now(UTC)` (from `datetime.UTC`).

**Reason:** `datetime.utcnow()` is deprecated since Python 3.12. The project
targets Python 3.11+ and ruff rule UP017 flags the deprecated form. Using
`datetime.UTC` is future-proof and ruff-clean.

**Impact:** None. The datetime is timezone-aware (`tzinfo=UTC`) which is
strictly better than the naive UTC datetime that `utcnow()` returns.

## Contract Amendments

None. The integration contract (`00-integration-contract.md`) is unchanged.
All changes are within the scope already documented in §2.4 (memory injection
path), §3.2 (memory fields on AgentConfig), and §4 (composition sketch).

## Release Notes Snippet

> **v0.3.0 — Stateful Agent (M2)**
>
> New memory subsystem: `InMemoryMemory` (ephemeral) and `SqliteMemory`
> (persistent, behind `[memory-sqlite]` extra) implement the
> `MemoryProvider` Protocol. Configure `AgentConfig(memory=...)` to
> persist state across `Agent.run()` calls. New `system_arguments`
> kwarg on `run_agent` enables system-template-driven memory injection.
> See `docs/memory.md` and `examples/09_memory/`.
