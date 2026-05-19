# Milestone 1 dev-log — `compose-removal/relocate`

> **Status:** Landed on `compose-removal/m1-relocate` (8 commits).
> **Date:** 2026-05-18.
> **Topic branch:** `compose-removal/m1-relocate` off `experimental/glow`.
> **Plan file:** `~/.claude/plans/compose-removal-plan.md`.
> **Spec:** `docs/internals/compose-removal/milestones/milestone-1-relocate.md`.

## §1 Summary — what was done

Sub-agent orchestration moved out of `src/t0_agents/compose/` into
`src/t0_agents/delegation/`; the topology layer in `compose/` stays
untouched (slated for M2 deletion). No public symbol from
`t0_agents.__all__` was added or removed; only the source modules of
existing re-exports shifted.

### Files added / moved / deleted

**Created:**

- `src/t0_agents/delegation/__init__.py` (28 LOC).
- `src/t0_agents/delegation/orchestrator.py` (`git mv` from
  `compose/orchestrator.py`, 314 LOC, internal imports + 1 docstring
  cross-reference redirected).
- `src/t0_agents/delegation/agent_as_tool.py` (`git mv` rename from
  `compose/delegation.py`, 85 LOC, deferred import redirected).
- `src/t0_agents/delegation/routing.py` (`git mv` from
  `compose/routing.py`, 37 LOC, no internal edits).
- `tests/test_delegation/__init__.py` (empty).
- `tests/test_delegation/test_orchestrator.py` (`git mv` from
  `tests/test_compose/test_orchestrator.py`, 701 LOC, imports
  redirected).
- `tests/test_delegation/test_agent_as_tool.py` (`git mv` rename
  from `tests/test_compose/test_delegation.py`, 170 LOC, imports
  redirected).
- `tests/test_delegation/test_integration.py` (`git mv` from
  `tests/test_compose/test_integration.py`, 202 LOC, no edits — it
  used only top-level imports).
- `tests/test_delegation/test_compat.py` (rewritten from
  `tests/test_compose/v1/test_compat_sub_agents.py`, 33 LOC — see
  §2 D2 for the trim rationale).

**Deleted (via `git mv` rename or rewrite):**

- `src/t0_agents/compose/orchestrator.py`, `compose/delegation.py`,
  `compose/routing.py`.
- `tests/test_compose/test_orchestrator.py`,
  `tests/test_compose/test_delegation.py`,
  `tests/test_compose/test_integration.py`,
  `tests/test_compose/v1/test_compat_sub_agents.py`.

**Modified:**

- `src/t0_agents/__init__.py` — `agent_as_tool` now sources from
  `t0_agents.delegation`; the 13 topology symbols stay on the
  `from t0_agents.compose import …` block.
- `src/t0_agents/compose/__init__.py` — trimmed to topology-only
  re-exports; module docstring rewritten.
- `src/t0_agents/compose/_runtime.py` — import + module-docstring
  cross-reference redirected (this file is M2-deletion-bound; the
  acceptance grep still flagged the docstring reference, so we
  updated it).
- `src/t0_agents/compose/builtins.py` — module-docstring cross-
  reference redirected.
- `src/t0_agents/core/agent.py:158-159` — deferred imports redirected.
- `src/t0_agents/core/middleware/builtins.py:450` — deferred imports
  redirected.
- `src/t0_agents/core/exceptions.py:42` — `SubAgentError` docstring
  cross-reference redirected.
- `tests/test_middleware/test_tool_provider_order.py:24` — `_OrchestrationToolProvider`
  import redirected (see §2 D3).

### Public API delta

`src/t0_agents/__init__.py.__all__`: no changes. Source modules for
specific re-exports changed:

| Symbol | Source before M1 | Source after M1 |
|---|---|---|
| `agent_as_tool` | `t0_agents.compose` (re-exporting `compose.delegation`) | `t0_agents.delegation` |
| 13 topology symbols (`Composable`, `Composer`, `ComposerError`, `circuit_breaker`, `conditional`, `consensus`, `map_reduce`, `parallel`, `race`, `retry_chain`, `saga`, `sequential`) | `t0_agents.compose` | `t0_agents.compose` (unchanged; M2 removes) |
| `SubAgentMiddleware` | `t0_agents.core.middleware` | unchanged |
| `SubAgentError` | `t0_agents.core.exceptions` | unchanged |

`t0_agents.compose.__all__`: 7 symbols removed (`agent_as_tool`,
`assert_unique_names`, `bind_parent_episode_id`, `default_tool_name`,
`get_parent_episode_id`, `install_sub_agents`,
`update_parent_episode_id`). 14 topology symbols remain.

`t0_agents.delegation.__all__` (new): 7 entries — `agent_as_tool`,
`assert_unique_names`, `bind_parent_episode_id`, `default_tool_name`,
`get_parent_episode_id`, `install_sub_agents`,
`update_parent_episode_id`. `_OrchestrationToolProvider`,
`_PARENT_EPISODE_ID`, and `PARENT_EPISODE_ID_TAG` remain
underscore-prefixed and importable via the submodule path
(`from t0_agents.delegation.orchestrator import …`).

### Tests inventory delta

- `tests/test_delegation/` (new): 4 test files, 34 tests total —
  32 offline + 2 integration-marked.
- `tests/test_compose/` (residual): now contains only the 11 topology
  test files in `v1/` (~85 tests) plus `v1/_helpers.py` and
  `v1/__init__.py`. All M2-deletion-bound.

Pre-M1 baseline on `experimental/glow`: 1006 offline tests passing.
Post-M1: 1001 offline tests passing. Net delta: −5 (3 from the
`test_compat.py` trim + 2 from the test-collection cap — see §2 D2
for both contributors).

### Docs delta

- `docs/internals/compose-removal/` directory committed for the first
  time (was untracked at workstream kick-off). Contains the
  workstream README, the M1 and M2 specs, and this dev-log.

### Commit SHAs (in landing order on the topic branch)

| C | SHA | Message |
|--:|:----|:--------|
| 1 | `fe7bc96` | `feat(compose-removal): create delegation/ package skeleton` |
| 2 | `0aa5aea` | `refactor(compose-removal): move routing helpers to delegation/` |
| 3 | `cff0fe9` | `refactor(compose-removal): move agent_as_tool to delegation/` |
| 4 | `1420bdd` | `refactor(compose-removal): move orchestrator to delegation/` |
| 5 | `e02743f` | `refactor(core): update SubAgentMiddleware + Agent imports to delegation` |
| 6 | `9ecfb11` | `refactor(t0_agents): re-export agent_as_tool from delegation/` |
| 7 | `da6cf36` | `test(compose-removal): move sub-agent tests to tests/test_delegation/` |
| 8 | `04003bc` | `refactor(compose): trim __init__.py to topology re-exports only` |

## §2 Deviations from the milestone spec

### D1 — `test_compat_sub_agents.py` lived in `tests/test_compose/v1/`, not the top level

**What changed:** the spec's §Scope and §Cross-references listed the
source as `tests/test_compose/test_compat_sub_agents.py`; the real
path was `tests/test_compose/v1/test_compat_sub_agents.py`.
Destination unchanged (`tests/test_delegation/test_compat.py`); only
the source path differed. Recorded for M2's awareness — the
`tests/test_compose/v1/` directory now has 11 topology test files
(not 12) plus `_helpers.py` and `__init__.py`.

**Impact on M2:** none, beyond the bookkeeping detail that M2's
"delete the 11 topology test files" list (already correct in the M2
spec) corresponds exactly to what's left in `v1/`.

### D2 — `test_compat.py` was trimmed during the move, not just renamed

**What changed:** the original `test_compat_sub_agents.py` (4 tests)
contained 3 tests that exercised the topology Composer surface
(`Composable`, `Composer`, `sequential`) and depended on
`tests.test_compose.v1._helpers.MockComposable`. The spec asked for
a straight rename. Instead, the move trimmed those 3 tests in the
same commit (C7), keeping only `test_legacy_sub_agents_path_still_exists`
plus a rewritten docstring.

**Why:** the trimmed tests target a surface that M2 deletes outright.
Moving them now then deleting them in M2 would have added churn
(two commits touching the same file across both milestones for no
behavioural reason) and would have left the M1 test_delegation/
directory with topology tests that don't belong to its conceptual
scope. The user approved this resolution via `AskUserQuestion` during
planning.

**Test count impact:** −3 offline tests vs. the pre-M1 baseline.

**Impact on M2:** none — `test_legacy_sub_agents_path_still_exists`
survives all M2 deletions because it imports `agent_as_tool` and
`SubAgentMiddleware` (both surviving symbols).

### D3 — extra import update: `tests/test_middleware/test_tool_provider_order.py`

**What changed:** `tests/test_middleware/test_tool_provider_order.py:24`
imported `_OrchestrationToolProvider` from
`t0_agents.compose.orchestrator`. The spec's "Source files updated"
list named only `core/agent.py` and `core/middleware/builtins.py`,
but the M1 acceptance grep
(`git grep "compose\.orchestrator|compose\.delegation|compose\.routing"`)
covers `tests/` too and would have flagged this. Folded the one-line
fix into C5 (same logical commit as the core/* updates). User
approved via `AskUserQuestion`.

**Impact on M2:** none.

### D4 — `examples/08_multi_agent/config/tensorzero.toml` comment

**What changed:** `examples/08_multi_agent/config/tensorzero.toml:10`
has a comment referencing `:func:\`t0_agents.compose.install_sub_agents\``.
The M1 spec's §Out explicitly defers example-doc updates to M2 ("No
changes to `examples/08_multi_agent/` beyond verifying its imports
still resolve"). No code change in M1 — example imports go through
the top-level `t0_agents` package and still resolve.

**Impact on M2:** the M2 spec's docs-sweep commit (commit 11 per
the spec) should update this comment to point at
`t0_agents.delegation.install_sub_agents`. Carry-forward in §4.

### D5 — additional in-compose consumers of `compose.orchestrator`

**What changed:** the spec listed the call-sites to update in C5
(now C4 per the reorder — see D7), but `compose/_runtime.py:37` (real
import of `bind_parent_episode_id`, etc.) and the docstring at
`compose/_runtime.py:6` and `compose/builtins.py:31` were not
enumerated. Both refer to the moved `compose.orchestrator` module
through the orchestrator's public surface, so the C4 (orchestrator
move) commit updated all of them atomically. The `core/exceptions.py:42`
SubAgentError docstring referenced `compose.delegation` and was
updated in C3 (agent_as_tool move).

**Why noticed:** the M1 acceptance grep
(`git grep "compose\.(orchestrator|delegation|routing)"`) is strict —
matches in docstrings count just as much as import statements. The
spec's §Acceptance criteria require an empty result, so all
references — live imports and prose alike — had to land on the new
paths.

**Impact on M2:** none. These three files are M2-deletion-bound; the
docstring text is throwaway work that M2 will delete. The
`core/exceptions.py:42` docstring update survives into M2.

### D6 — interpretation of "from t0_agents.compose" allowlist

**What changed:** the spec's §Acceptance gate says
"`git grep -n "from t0_agents.compose" src/` only matches the four
topology-internal files and `glow/compose.py`". After M1, the grep
shows **five** files: those 4 + `glow/compose.py` + `src/t0_agents/__init__.py`.

**Why:** `src/t0_agents/__init__.py:7` keeps the
`from t0_agents.compose import …` block (12 topology symbols) — they
are still re-exported from the top-level facade until M2 removes
them. The spec author's "four topology-internal files" tally
overlooked the top-level facade. The principle holds (no outward
consumer imports the *moved* sub-agent modules), the count is +1.

**Impact on M2:** none. M2 commit 1 removes that
`from t0_agents.compose import …` block from
`src/t0_agents/__init__.py` along with the 13 `__all__` entries.

### D7 — move order: routing → agent_as_tool → orchestrator

**What changed:** the spec's per-commit checklist ordered the moves
as orchestrator (C2), agent_as_tool (C3), routing (C4). The
implementation moved them in reverse order — routing first
(C2 = `0aa5aea`), agent_as_tool next (C3 = `cff0fe9`), orchestrator
last (C4 = `1420bdd`).

**Why:** routing has zero internal `compose.*` imports.
agent_as_tool depends on `routing` (module-level) and `orchestrator`
(deferred). orchestrator depends on `agent_as_tool` (module-level)
and `routing` (module-level). Moving in the spec's order would have
forced `delegation/orchestrator.py` to import from
`delegation/agent_as_tool.py` and `delegation/routing.py` before
those files existed — i.e., a broken-tree commit. The reverse order
keeps every commit fully tree-green (mypy + imports + offline
pytest) and uses the move file's own dependency arrows as a
topological constraint.

**Impact on M2:** none. End state is identical — all 3 files in
`delegation/`, all consumers redirected, `compose/` left with only
its topology files.

## §3 Open questions

- **2 ruff-baseline errors that survive M1.** `tests/test_cli/test_elicit_tui.py:326`
  (`RUF012` mutable default for class attribute) and
  `tests/test_memory/test_agent_memory.py:112` (`F841` unused
  variable `r2`). Both pre-exist on `experimental/glow` from earlier
  glow milestones. Not introduced by M1; counted out of scope.
  Filed for the pre-public polish backlog
  ([[project-pre-public-polish]]) rather than M2.
- **Test-count drop is −5, not −3.** D2 explains 3 of the 5. The
  remaining 2 are accounted for by `tests/test_compose/v1/test_compat_sub_agents.py`
  having previously contributed 4 tests to the offline collection;
  after the move + trim, only `test_legacy_sub_agents_path_still_exists`
  remains (1 test). Net: 4 → 1 = −3 within this file, plus 2 tests
  in `tests/test_delegation/test_orchestrator.py` and
  `tests/test_delegation/test_agent_as_tool.py` are integration-marked
  inheritances that get deselected by the offline filter that were
  not deselected before (the previous integration-marker tally was
  61; post-M1 it's 63 — so 2 more deselected tests means 2 fewer
  offline tests). No regression — they're still collected; just
  filtered out under `-m "not integration"`.

No open questions parked for resolution by a later milestone.

## §4 Carry-forward notes

For M2 (`docs/internals/compose-removal/milestones/milestone-2-delete-and-examples.md`):

- **`src/t0_agents/delegation/` is stable as of C8 (`04003bc`).** The
  M2 spec's "do not modify" invariant applies starting here. Public
  surface frozen: 7 symbols in `__all__` + 3 underscore-prefixed
  module-level names (`_OrchestrationToolProvider`,
  `_PARENT_EPISODE_ID`, `PARENT_EPISODE_ID_TAG`) reachable via
  `from t0_agents.delegation.orchestrator import …`.
- **`tests/test_compose/` is ready for M2 deletion.** What remains:
  `__init__.py` (empty), `v1/__init__.py` (empty), `v1/_helpers.py`
  (5460 bytes, only consumer is the 11 topology tests in `v1/`).
  After M2 commit 4 deletes the 11 topology test files +
  `v1/_helpers.py`, the `rmdir tests/test_compose/` step in that
  commit should succeed.
- **`tests/test_compose/v1/_helpers.py` is orphaned of its single
  non-topology consumer.** Three of `test_compat.py`'s tests used to
  import `MockComposable` from this helper. After D2's trim, only
  the 11 topology tests import from `_helpers.py`. M2 deletes all of
  them together.
- **D4 — TOML comment fix.** M2's docs-sweep commit (per spec
  commit 11) should update
  `examples/08_multi_agent/config/tensorzero.toml:10` from
  `:func:\`t0_agents.compose.install_sub_agents\`` to
  `:func:\`t0_agents.delegation.install_sub_agents\``.
- **D6 — top-level facade.** M2 commit 1 removes the 13 topology
  imports from `src/t0_agents/__init__.py` (lines 7-19 post-M1).
  Verify the `agent_as_tool` import at line 71 (`from t0_agents.delegation
  import agent_as_tool`) stays.
- **C5 picked up ruff-sort cleanup as a side effect** of the C2–C4
  moves. M2 commits should expect similar sort churn — when removing
  the 13 topology imports from the top-level facade, ruff may
  reflow the remaining import block. Run `uv run ruff check --fix`
  inside each commit's scope rather than trying to predict the
  reflows.
- **Ruff baseline carry-over.** 6 pre-existing ruff errors remain
  (4 auto-fixable from `tests/test_core/*` and
  `tests/test_memory/test_agent_memory_integration.py` unused-imports,
  2 unfixable from `tests/test_cli/` and `tests/test_memory/`). M2
  should NOT include incidental cleanups of these — they belong to
  [[project-pre-public-polish]]. Use targeted `ruff check --fix`
  paths or `ruff check src/ tests/ --fix --select I` style to avoid
  pulling them in.
- **`docs/multi-agent.md` verification.** The spec's M2 §Docs deletion
  /update section says "confirm references match `delegation/` post-M1
  (M1 already did this; verify)". M1 did **not** touch `docs/multi-agent.md`.
  Verify in M2 — the M1 scope was strictly the dev-log + the relocated
  source/tests; no docs/ files were modified.

## §5 Verification

Commands run on `compose-removal/m1-relocate` HEAD (`04003bc`) after
all 8 commits landed:

```bash
$ git grep -nE "compose\.(orchestrator|delegation|routing)" src/ tests/
(no matches; exit 1)

$ git grep -n "from t0_agents.compose" src/
src/t0_agents/__init__.py:7:from t0_agents.compose import (
src/t0_agents/compose/__init__.py:15:from t0_agents.compose.builtins import (
src/t0_agents/compose/__init__.py:35:from t0_agents.compose.topology import (
src/t0_agents/compose/_runtime.py:37:from t0_agents.compose.topology import (
src/t0_agents/compose/builtins.py:43:from t0_agents.compose._runtime import (
src/t0_agents/compose/builtins.py:55:from t0_agents.compose.topology import (
src/t0_agents/glow/compose.py:43:from t0_agents.compose.builtins import parallel, sequential
src/t0_agents/glow/compose.py:44:from t0_agents.compose.topology import (
(5 files: 4 topology-internal + glow/compose.py + top-level facade — see D6)

$ uv run ruff check src/ tests/
Found 6 errors.
[*] 4 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
(all 6 pre-exist on experimental/glow; not introduced by M1 — see §3)

$ uv run mypy src/
Success: no issues found in 84 source files

$ uv run pytest -m "not integration" -q
1001 passed, 2 skipped, 63 deselected in 5.44s

$ git diff --stat main..HEAD -- src/t0_agents/mcp/
(empty — mcp/ untouched; feedback_mcp_standalone invariant holds)

$ git log --oneline experimental/glow..HEAD | wc -l
8
(within the spec's 6-8 commit budget)

$ uv run python -c "
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
assert agent_as_tool is at2
print('OK')
"
OK
```

Integration tests not run for M1 (per the spec — M2 is the
docker-dependent milestone).
