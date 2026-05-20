# Milestone 3 — Dev Log

**Date**: 2026-05-19
**Base branch**: main
**Branch**: `milestone/3-agent-drop-ephemeral-fallback`
**PR**: https://github.com/sproutfi/t0-agents/pull/53

## Implemented

- `src/t0_agents/core/agent.py` (commit `d72e029`)
  - Deleted `_resolve_chat_tool_provider` (old lines 41–59), the
    `@asynccontextmanager` that allocated
    `Client(FastMCP("agent-default"))` whenever `ctx.tool_provider`
    was `None`. With M1 + M2 in, both `run_agent` and
    `StreamingAgentLoop.stream` already treat `tool_provider=None`
    as "skip `list_tools()`; raise `T0AgentsFatalError` if the LLM
    still emits a tool call". The helper is no longer load-bearing.
  - Dropped `FastMCP` from the `from fastmcp import …` line and,
    transitively, the `AsyncIterator` + `asynccontextmanager`
    imports — both were used only by the deleted helper.
  - Restored `from fastmcp import Client` because the call-site
    `cast(Client[Any] | None, ctx.tool_provider)` (see Deviations)
    still needs the concrete type.
  - `_chat_terminal` (old line 381) and `_chat_streaming_terminal`
    (old line 434) no longer wrap their inner calls in
    `async with _resolve_chat_tool_provider(...)`. They now pass
    `cast(Client[Any] | None, ctx.tool_provider)` straight to
    `run_agent` / `streamer.stream`. Indentation level drops one
    rung — the bodies are flatter and the lifecycles converge on
    "the caller owns the `Client`".
  - Agent class main docstring (lines 51–62) and `_chat_terminal`
    docstring (lines 333–343) reworded to describe the direct
    pass-through instead of the deleted fallback.
  - Sweep `grep -nE 'ephemeral|agent-default|_resolve_chat_tool_provider'
    src/t0_agents/core/agent.py` returns **zero hits**.
- `src/t0_agents/delegation/orchestrator.py:139-143` (commit `7b85a18`)
  - Sub-agent `ConfigurationError` rewritten verbatim per PLAN.md:
    > `AgentConfig.sub_agents requires the parent's tool_provider to be a fastmcp.Client (got {type(client).__name__}). Pass tool_provider=Client(FastMCP(...)) when constructing the parent Agent.`
  - Verified that `tests/test_delegation/test_orchestrator.py::TestInstallSubAgentsErrors::test_non_fastmcp_client_provider_raises` still matches the message: the existing
    `pytest.raises(ConfigurationError, match=r"fastmcp\.Client")`
    regex is intentionally loose; the new wording preserves the
    `"fastmcp.Client"` substring verbatim. No test change required.
- `tests/test_core/test_agent.py:165` (commit `1ead96e`)
  - Extended `test_toolless_chat_invoke_returns_text` (not a new
    sibling — kept the original name + scope) with a
    `monkeypatch.setattr(fastmcp.FastMCP, "__init__", …)` spy that
    raises if `FastMCP()` is ever constructed during the toolless
    invoke. This anchors the M3 invariant: toolless chat allocates
    **no MCP server**. Any reintroduction of the deleted ephemeral
    fallback would fail this assertion loudly.
  - Refreshed the stale comment at line 161 ("the fallback is
    allocated lazily per invocation by
    `_resolve_chat_tool_provider`") and the docstring (lines 168–
    173) to describe the direct pass-through semantics.
- `src/t0_agents/glow/app.py:101-103` (commit `5c4e740`)
  - Refreshed the M0-era comment that named the deleted
    `core/agent.py:_resolve_chat_tool_provider` helper as the
    reason glow always owns its `Client`. New wording is generic
    ("glow always runs with a concrete in-process FastMCP server …
    toolless-chat semantics are handled by the v1.0 core directly")
    so the comment no longer dangles on a deleted symbol.

## Deviations from plan

- **Restored `from fastmcp import Client`** (the plan said to drop
  `FastMCP` from the import line, keeping `Client`; I initially
  dropped the whole line because the deletion of
  `_resolve_chat_tool_provider` removed the only runtime use of
  `Client` — but mypy then flagged the two `run_agent` /
  `streamer.stream` calls because `RunContext.tool_provider` is
  typed as `ToolProvider | None` (the protocol) while
  `run_agent` / `streamer.stream` advertise `Client[Any] | None`
  (the concrete type)).
  - **What changed**: re-added `from fastmcp import Client` and a
    `cast(Client[Any] | None, ctx.tool_provider)` at both call
    sites — the exact pattern the deleted helper used at its `yield
    cast(Client[Any], tool_provider)` line, moved inline.
  - **Why** (decision at implementation time): the alternative
    fixes are worse: (a) loosening `run_agent` / `stream` to
    accept `ToolProvider | None` would force `mcp/loop.py` to
    import the protocol from `core/types.py`, violating the
    `mcp/` standalone rule (`feedback_mcp_standalone.md`); (b)
    rewriting `RunContext.tool_provider` to `Client[Any] | None`
    would propagate through every middleware. The cast at the
    boundary is the same trick the helper used and is structurally
    consistent with how `_OrchestrationToolProvider` already
    bridges the two types.
  - **Impact on next milestones**: none. M4 (docs/CHANGELOG)
    describes the boundary precisely once; M5 (showcase example)
    is unaffected because the example uses `Agent` not `run_agent`.

- **2 pre-existing F401s in `tests/test_core/test_agent_*.py`
  remain parked.** The M3 plan's lint surface lists
  `tests/test_core/`, which means `ruff check tests/test_core/`
  reports 2 errors:
  - `tests/test_core/test_agent_integration.py:16` — unused
    `tests.server.mcp` import (F401)
  - `tests/test_core/test_agent_parity.py:15` — unused
    `fastmcp.Client` import (F401)
  - **What changed**: these are 2 of the 6 pre-existing ruff
    errors M1 and M2 already documented as out-of-scope. M1's
    dev-log enumerates them ("two `tests/test_core/test_agent_*.py`
    F401s") and parked them for M4 or a housekeeping PR; this PR
    inherits that decision rather than silently folding the fixes.
  - **Why** (decision at implementation time): per
    `feedback_pre_public_polish_log.md` ("when spotting out-of-
    scope issues, append to `project_pre_public_polish.md`; don't
    silently fix inside unrelated PRs"). The M3-introduced code
    in `tests/test_core/test_agent.py` (the FastMCP spy) is itself
    ruff-clean; only the unrelated `_integration.py` and
    `_parity.py` files carry the residue.
  - **Impact on next milestones**: still parked for M4. The full
    6-error set (`test_cli/test_elicit_tui.py:326` RUF012,
    `tests/test_core/test_agent_*.py` F401s ×2,
    `tests/test_memory/test_agent_memory.py:112` F841,
    `tests/test_memory/test_agent_memory_integration.py`
    F401s ×2) is unchanged from M1+M2.

## Tests & lint

- `uv run pytest -m "not integration"` → **928 passed**, 2 skipped,
  61 deselected (5.68s). Identical totals to the M2 baseline; the
  M3 spy extension lives inside the existing
  `test_toolless_chat_invoke_returns_text` test, so the count
  doesn't move. No regressions across `core/`, `delegation/`,
  `glow/`, `guardrails/`, `memory/`, `middleware/`, `output/`,
  `run/`, `serve/`, `stream/`, `testing/`.
- `uv run pytest -m "not integration" tests/test_core/test_agent.py
  -k "toolless"` → **1 passed**, 33 deselected (0.19s). FastMCP
  spy assertion is exercised on the toolless path.
- `uv run pytest -m "not integration"
  "tests/test_delegation/test_orchestrator.py::TestInstallSubAgentsErrors::test_non_fastmcp_client_provider_raises"`
  → **1 passed** (0.16s). The loose regex
  `match=r"fastmcp\.Client"` matches the new error message verbatim.
- `uv run mypy src/t0_agents/core/ src/t0_agents/delegation/` →
  **Success: no issues found in 23 source files**.
- `uv run ruff check src/t0_agents/core/ src/t0_agents/delegation/
  tests/test_delegation/` → **All checks passed!**
- `uv run ruff check tests/test_core/` → 2 pre-existing F401s
  (see Deviations); the M3-introduced spy code is itself clean.
- `uv run ruff check src/ tests/` → same **6 pre-existing errors**
  as M1+M2 reported (4 F401s, 1 F841, 1 RUF012); none introduced
  by M3.
- `grep -rn '_resolve_chat_tool_provider\|FastMCP("agent-default")'
  src/t0_agents/` → **zero hits**.
- `grep -nE 'ephemeral|agent-default|_resolve_chat_tool_provider'
  src/t0_agents/core/agent.py` → **zero hits**.
- `grep -rn '"mcp_client"\|mcp_client=' src/ tests/` → 2 hits, both
  **assertion strings** in M1+M2's toolless tests checking that the
  legacy `"mcp_client"` token is **absent** from error messages
  (`tests/test_mcp/test_loop_toolless.py:108`,
  `tests/test_stream/test_runner_toolless.py:128`). No real
  callers remain.
- Integration suite (`-m integration`): **not run** in M3 by
  design. Same reasoning as M1+M2 — M3 deletes a code path that
  is strictly an in-process allocation detail; no gateway-
  observable behaviour changes.

## Notes for next milestone

M4 (`docs-changelog-optional-client`) inherits the following:

- **CHANGELOG Breaking entry covers two facets, not one.**
  The Unreleased breaking bullet should mention both (a) the
  `mcp_client → tool_provider` rename on `run_agent` and
  `StreamingAgentLoop.stream` (already accurate from M1+M2),
  **and** (b) the deletion of the ephemeral
  `Client(FastMCP("agent-default"))` fallback. A toolless chat
  invoke now performs **zero MCP allocations** — that is
  observable in profiling and should be called out as a perf
  win, not just a refactor.
- **README chat-only snippet must match reality.** The
  `Agent(config, t0=gateway)` snippet without `tool_provider`
  now does what the README docstring already claims: no FastMCP
  import, no client lifecycle. M4 can drop any disclaimer about
  "an in-memory server is allocated for you" if such a phrasing
  exists.
- **Sub-agent error message is now the canonical wording.** The
  rewrite in `delegation/orchestrator.py:139-143` is the
  reference UX for "you passed something that isn't a
  `fastmcp.Client`". If M4's docs cover sub-agent setup, the
  doc copy should reuse the constructor hint verbatim:
  `Client(FastMCP(...))`.
- **The 6 pre-existing ruff errors are now the only ruff debt.**
  M1, M2, and M3 have all parked them. M4 is the natural place
  to either (a) bundle them into the docs PR as a "housekeeping"
  commit (acceptable because docs PRs touch tests/docs anyway),
  or (b) cut a separate `chore(lint)` PR. The full list is
  enumerated in M1's dev-log under "Deviations".
- **The `cast(Client[Any] | None, ctx.tool_provider)` boundary
  pattern is now in two places in `core/agent.py` (lines 360 and
  413).** If M4 touches these lines for docstring polish, leave
  the cast — it's load-bearing for mypy until the
  `RunContext.tool_provider` type is revisited (a separate
  decision, out of scope).
- **The auto-memory note `feedback_mcp_standalone.md`** says the
  `run_agent` signature is "frozen" — strictly that's no longer
  true (M1 renamed the parameter and added a default). M4
  should update the memory note to reflect the new baseline.
  Per `CLAUDE.md`, memory updates happen in the dev-log/PR,
  not via diff.

## Open questions

None. M3 closes cleanly against the M1+M2 contract. M4 may
proceed.
