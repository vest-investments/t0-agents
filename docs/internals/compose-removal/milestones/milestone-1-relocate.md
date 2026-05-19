# M1 — `relocate`

> **Status:** Not started.
> **Branch:** `experimental/glow` (topic branch: `compose-removal/m1-relocate`).
> **Dependencies:** none.
> **Blocks:** M2 (M2 deletes `compose/`; the sub-agent code must be
> safely in `delegation/` first).

## Purpose

Migrate every internal call site off `compose/` topology primitives,
and **move** the sub-agent orchestration files out of `compose/` into
their new home `src/t0_agents/delegation/`. After M1, the only modules
that still import from `compose/` are the topology files themselves
(`builtins.py`, `_runtime.py`, `topology.py`) and `glow/compose.py` —
both slated for deletion in M2.

This milestone is **purely a relocation** — no public-API symbols are
removed, no tests are deleted, no docs are rewritten. That all happens
in M2. The deliberate split keeps M1's risk surface small: an `import`
typo in a relocated file is recoverable before M2 starts; a deletion
during the same milestone would compound the failure modes.

## Scope

### In

- **Audit internal usage** of `compose/` primitives:
  - `git grep -n "from t0_agents.compose" src/ tests/ examples/`
  - `git grep -nE "from t0_agents import.*\b(sequential|parallel|race|conditional|consensus|map_reduce|saga|retry_chain|circuit_breaker|Composer|Composable|ComposerError|CompositeRun)\b" src/ tests/ examples/`
  - For each non-test call site: replace with `asyncio.gather` /
    awaits / try-except / `sub_agents` as appropriate.
  - Tests that exercise topology primitives directly: **leave alone**
    (M2 deletes them).

- **Create `src/t0_agents/delegation/`** with:
  - `delegation/__init__.py` — re-exports `agent_as_tool`,
    `install_sub_agents`, `bind_parent_episode_id`,
    `get_parent_episode_id`, `update_parent_episode_id`,
    `default_tool_name`, `assert_unique_names`. (`_OrchestrationToolProvider`
    stays underscore-prefixed; importers inside the package use
    `from t0_agents.delegation.orchestrator import _OrchestrationToolProvider`.)
  - `delegation/orchestrator.py` ← move from `compose/orchestrator.py`
    unchanged (including `_OrchestrationToolProvider`,
    `_PARENT_EPISODE_ID` ContextVar, `PARENT_EPISODE_ID_TAG`).
  - `delegation/agent_as_tool.py` ← move from `compose/delegation.py`
    (renamed; contents unchanged).
  - `delegation/routing.py` ← move from `compose/routing.py` unchanged.

- **Update imports** in core:
  - `src/t0_agents/core/middleware/builtins.py` — every
    `from t0_agents.compose.orchestrator import ...` →
    `from t0_agents.delegation.orchestrator import ...`. The
    `SubAgentMiddleware` class body itself is unchanged.
  - `src/t0_agents/core/agent.py` —
    `from t0_agents.compose.orchestrator import install_sub_agents` →
    `from t0_agents.delegation import install_sub_agents`. The call
    site in `Agent.__init__` is unchanged.

- **Update `src/t0_agents/__init__.py`:**
  - `agent_as_tool`, `SubAgentMiddleware`, `SubAgentError` continue
    to be re-exported in `__all__`. Their origin module changes:
    - `agent_as_tool`: `t0_agents.compose` → `t0_agents.delegation`.
    - `SubAgentMiddleware`: stays at `t0_agents.core.middleware` (no
      change — the class itself didn't move, only its imports did).
    - `SubAgentError`: stays at `t0_agents.core.exceptions` (no change).
  - The 13 topology symbols (`Composable`, `Composer`, `ComposerError`,
    `circuit_breaker`, `conditional`, `consensus`, `map_reduce`,
    `parallel`, `race`, `retry_chain`, `saga`, `sequential` — and
    `CompositeRun` if exported) **stay** in `__all__` for now. M2
    removes them.

- **Move sub-agent tests:**
  - `tests/test_compose/test_orchestrator.py` →
    `tests/test_delegation/test_orchestrator.py`.
  - `tests/test_compose/test_delegation.py` →
    `tests/test_delegation/test_agent_as_tool.py`.
  - `tests/test_compose/test_compat_sub_agents.py` →
    `tests/test_delegation/test_compat.py`.
  - `tests/test_compose/test_integration.py` →
    `tests/test_delegation/test_integration.py`.
  - Create `tests/test_delegation/__init__.py` (empty). If
    `tests/test_compose/conftest.py` exists and has fixtures relevant
    to the moved tests, copy those to
    `tests/test_delegation/conftest.py`; leave the original conftest
    intact for the topology tests slated for M2 deletion.
  - Update import paths inside the moved tests
    (`from t0_agents.compose.orchestrator import` →
    `from t0_agents.delegation.orchestrator import`, etc.).

- **Trim `compose/__init__.py`:**
  - Remove the sub-agent re-exports (`agent_as_tool`, `install_sub_agents`,
    `bind_parent_episode_id`, `get_parent_episode_id`,
    `update_parent_episode_id`, `default_tool_name`,
    `assert_unique_names`).
  - Leave the topology re-exports (the 9 factories + their
    `*Composer` classes + `Composable`/`Composer`/`ComposerError`/
    `CompositeRun`/`DEFAULT_CANCEL_TIMEOUT`) for M2.

### Out

- No deletion of `compose/builtins.py`, `compose/_runtime.py`,
  `compose/topology.py` (that's M2).
- No deletion of `glow/compose.py` (M2).
- No deletion of topology tests (M2).
- No new examples (M2).
- No doc rewrites (M2).
- No CHANGELOG entry (M2 owns the `### Removed` block).
- No changes to `src/t0_agents/mcp/`.
- No changes to `examples/08_multi_agent/` beyond verifying its imports
  still resolve (it should, if `from t0_agents import agent_as_tool`
  is the entry path).

## Checklist

- [ ] **Commit 1: `feat(compose-removal): create delegation/ package skeleton`**
  - [ ] `mkdir src/t0_agents/delegation/`.
  - [ ] Write `delegation/__init__.py` with an empty `__all__` initially
        (populated commit-by-commit as files arrive).
  - [ ] `mkdir tests/test_delegation/`.
  - [ ] Write `tests/test_delegation/__init__.py` (empty).
  - [ ] If `tests/test_compose/conftest.py` exists with fixtures
        relevant to sub-agent tests: copy those parts to
        `tests/test_delegation/conftest.py`.

- [ ] **Commit 2: `refactor(compose-removal): move orchestrator from compose to delegation`**
  - [ ] `git mv src/t0_agents/compose/orchestrator.py src/t0_agents/delegation/orchestrator.py`.
  - [ ] Verify the file's internal imports don't reference its old
        path (`from t0_agents.compose...`); if any do, update.
  - [ ] Update `delegation/__init__.py` to re-export
        `install_sub_agents`, `bind_parent_episode_id`,
        `get_parent_episode_id`, `update_parent_episode_id`.
  - [ ] `uv run mypy src/` should still pass (other consumers still
        import the old path; this commit alone may temporarily fail
        mypy if it doesn't — that's OK provided commit 5 fixes it).

- [ ] **Commit 3: `refactor(compose-removal): move agent_as_tool from compose to delegation`**
  - [ ] `git mv src/t0_agents/compose/delegation.py src/t0_agents/delegation/agent_as_tool.py`.
  - [ ] Update internal imports in the moved file if needed.
  - [ ] Update `delegation/__init__.py` to re-export `agent_as_tool`.

- [ ] **Commit 4: `refactor(compose-removal): move routing helpers from compose to delegation`**
  - [ ] `git mv src/t0_agents/compose/routing.py src/t0_agents/delegation/routing.py`.
  - [ ] Update `delegation/__init__.py` to re-export
        `default_tool_name`, `assert_unique_names`.

- [ ] **Commit 5: `refactor(core): update SubAgentMiddleware + Agent imports to delegation`**
  - [ ] `src/t0_agents/core/middleware/builtins.py` — change all
        `from t0_agents.compose.orchestrator import ...` to
        `from t0_agents.delegation.orchestrator import ...`.
  - [ ] `src/t0_agents/core/agent.py` — change
        `from t0_agents.compose.orchestrator import install_sub_agents`
        to `from t0_agents.delegation import install_sub_agents`.
  - [ ] Verify `uv run mypy src/` is now green.
  - [ ] Verify `uv run pytest -m "not integration"` is green.

- [ ] **Commit 6: `refactor(t0_agents): re-export sub-agent symbols from delegation/`**
  - [ ] `src/t0_agents/__init__.py` — change the import of
        `agent_as_tool` to come from `t0_agents.delegation`. Confirm
        `SubAgentMiddleware` and `SubAgentError` import paths are
        unchanged (they live in `core/middleware` and `core/exceptions`
        respectively).
  - [ ] Verify
        `from t0_agents import agent_as_tool, SubAgentMiddleware, SubAgentError`
        works in a REPL.

- [ ] **Commit 7: `test(compose-removal): move sub-agent tests to test_delegation/`**
  - [ ] `git mv tests/test_compose/test_orchestrator.py tests/test_delegation/test_orchestrator.py`.
  - [ ] `git mv tests/test_compose/test_delegation.py tests/test_delegation/test_agent_as_tool.py`.
  - [ ] `git mv tests/test_compose/test_compat_sub_agents.py tests/test_delegation/test_compat.py`.
  - [ ] `git mv tests/test_compose/test_integration.py tests/test_delegation/test_integration.py`.
  - [ ] Update import paths inside each moved test file.
  - [ ] Verify `uv run pytest tests/test_delegation/ -m "not integration"` passes.

- [ ] **Commit 8: `refactor(compose): trim __init__.py to only re-export topology`**
  - [ ] `src/t0_agents/compose/__init__.py` — remove all references to
        `orchestrator`, `delegation`, `routing` modules (they no longer
        exist in `compose/`).
  - [ ] Keep only the topology re-exports: 9 factories + their
        `*Composer` classes + `Composable`, `Composer`, `ComposerError`,
        `CompositeRun`, `DEFAULT_CANCEL_TIMEOUT`.
  - [ ] Run final validation: `uv run ruff check src/ tests/`,
        `uv run mypy src/`, `uv run pytest -m "not integration"`.

## Acceptance criteria

- `git grep -n "compose\.orchestrator\|compose\.delegation\|compose\.routing" src/ tests/` returns no matches.
- `git grep -n "from t0_agents.compose" src/` only matches:
  - `src/t0_agents/compose/__init__.py`
  - `src/t0_agents/compose/builtins.py`
  - `src/t0_agents/compose/_runtime.py`
  - `src/t0_agents/compose/topology.py`
  - `src/t0_agents/glow/compose.py`
- `uv run pytest -m "not integration"` is green (count unchanged modulo
  the test moves).
- `uv run mypy src/` strict passes.
- `uv run ruff check src/` passes.
- `git diff --stat src/t0_agents/mcp/` is empty.
- `from t0_agents import agent_as_tool, SubAgentMiddleware, SubAgentError`
  works.
- `from t0_agents.delegation import agent_as_tool, install_sub_agents,
  bind_parent_episode_id, get_parent_episode_id, update_parent_episode_id,
  default_tool_name, assert_unique_names` works.
- `tests/test_compose/` still exists but contains only topology tests
  (the 11 files slated for M2 deletion). The 4 sub-agent test files
  are now in `tests/test_delegation/`.

## Verification commands

```bash
# On the topic branch compose-removal/m1-relocate off experimental/glow:

# 1. The relocation grep — must return zero matches:
git grep -n "compose\.orchestrator\|compose\.delegation\|compose\.routing" src/ tests/

# 2. The residual compose import grep — must only match the four
#    topology-internal files plus glow/compose.py:
git grep -n "from t0_agents.compose" src/

# 3. Lint / type / test:
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest -m "not integration"

# 4. MCP invariant:
git diff --stat src/t0_agents/mcp/   # must be empty

# 5. Public API smoke (in a REPL):
uv run python -c "
from t0_agents import agent_as_tool, SubAgentMiddleware, SubAgentError
from t0_agents.delegation import (
    agent_as_tool as at2,
    install_sub_agents,
    bind_parent_episode_id,
    get_parent_episode_id,
    update_parent_episode_id,
    default_tool_name,
    assert_unique_names,
)
assert agent_as_tool is at2, 'top-level re-export should be the same object'
print('OK — delegation/ public surface wired correctly')
"

# 6. Commit count for the milestone:
git log --oneline experimental/glow..HEAD | wc -l   # should be in [6,8]
```

## Cross-references

- **Plan file:** `~/.claude/plans/compose-removal-plan.md` §"What's
  being removed vs preserved" and §"Implementation plan — milestones".
- **Investigation report:**
  `~/.claude/plans/quiero-quitar-compose-la-drifting-nygaard.md` §3.B
  describes the sub-agent orchestration that's being preserved.
- **Source files moving:**
  - `src/t0_agents/compose/orchestrator.py` → `src/t0_agents/delegation/orchestrator.py`.
  - `src/t0_agents/compose/delegation.py` → `src/t0_agents/delegation/agent_as_tool.py`.
  - `src/t0_agents/compose/routing.py` → `src/t0_agents/delegation/routing.py`.
- **Source files updated (import-only):**
  - `src/t0_agents/core/agent.py:156-169`.
  - `src/t0_agents/core/middleware/builtins.py:454` and surrounding.
  - `src/t0_agents/__init__.py:7-21` (the compose import block).
- **Tests moving:**
  - `tests/test_compose/test_orchestrator.py` (~24 tests).
  - `tests/test_compose/test_delegation.py` (~7 tests).
  - `tests/test_compose/test_compat_sub_agents.py` (~4 tests).
  - `tests/test_compose/test_integration.py` (2 tests, integration-marked).

## Dev-log

When the milestone completes, drop
`docs/internals/compose-removal/logs/milestone-1-dev-log.md` with the
five-section template (Summary · Deviations · Open questions ·
Carry-forward · Verification).

The most important paragraph for M2's implementer is §4
Carry-forward: spell out the exact state of `delegation/` (commit SHA
range), confirm which `test_compose/*.py` files are left behind for M2
deletion, and note any unforeseen complications with imports,
fixtures, or `_OrchestrationToolProvider`'s underscore-prefixed export.
