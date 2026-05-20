# Milestone 4 — Dev Log

**Date**: 2026-05-20
**Base branch**: main
**Branch**: `milestone/4-docs-changelog-optional-client`
**PR**: https://github.com/sproutfi/t0-agents/pull/54

## Implemented

User-facing prose (commit `2a35474`, `docs: surface toolless-chat
pattern across public prose`):

- `README.md` — inserted a 4-line "Chat-only" sub-snippet between
  the existing tool-using `Agent` example and the "Glow compiles
  down to exactly this `Agent`" paragraph.  Calls out "no `Client`,
  no FastMCP server, no `list_tools()` roundtrip per turn".
- `docs/getting-started.md:87` — replaced the "Both the
  `AsyncTensorZeroGateway` and `Client` must be entered" framing
  with a description of the optional `tool_provider` contract
  (pass `None`, skip `list_tools()`, no FastMCP allocated).
- `docs/agent.md` — Composition table row updated from
  "required when `AgentConfig.output` is a `ChatOutput`" to
  "optional in every output mode" with the `T0AgentsFatalError`
  guard called out.  JSON-mode paragraph rewritten to also cover
  chat-mode parity ("chat-mode agents *also* accept
  `tool_provider=None`"); links added to
  `getting-started.md` alongside the existing
  `output-modes.md` / `structured-outputs.md` cross-references.
- `docs/glow/getting-started.md:49-57` — rewrote the "Toolless
  chat — no MCP plumbing needed" section to describe the
  M3-shipped behaviour (the v1.0 core accepts `tool_provider=None`
  directly; no ephemeral FastMCP allocation; the unlock lives in
  the core, glow inherits it).
- `CLAUDE.md:29` — extended the "Agent loop flow" line with
  "Tools and the MCP `Client` are optional; single-shot output
  modes and chat-only agents work without them
  (`run_agent(..., tool_provider=None)` skips `list_tools()` and
  never allocates a FastMCP server)."

API reference (commit `7310a2c`, `docs(api): refresh API reference
for tool_provider rename`):

- `docs/api-reference.md` — `run_agent` signature block updated to
  `tool_provider: Client[Any] | None = None`; `user_message` and
  `function_name` shown with `= ""` defaults (the Python-syntax
  workaround M1 documented).
- `run_agent` parameter table — `tool_provider` row describes the
  optional contract (when `None`, no tools advertised, single
  inference + return, hallucinated tool call raises
  `T0AgentsFatalError`).  `user_message` / `function_name` rows
  document the `""` defaults and the empty-string `ValueError`
  guard.
- `run_agent` Raises section — `T0AgentsFatalError` added with a
  one-paragraph explanation of the "hallucinated tool call while
  `tool_provider is None`" path; `ValueError` description expanded
  to cover the empty-string case.
- `execute_tool_calls` and `execute_tool_calls_parallel`
  signatures + parameter tables — renamed `mcp_client →
  tool_provider`; documentation explicitly notes that these
  helpers stay non-optional (they only run inside the
  tool-dispatch branch).

CHANGELOG (commit `b4fa07b`, `docs(changelog): document optional
tool_provider in v1.0`):

- `CHANGELOG.md` intro paragraph — replaced "`mcp/` is untouched
  across the entire refactor" with a description of the
  `mcp_client → tool_provider` rename + optional default.  Points
  the reader at the new `Changed` section for the full contract.
- Glow "App-owned FastMCP server + Client lifecycle." Added
  entry — dropped the dangling
  `core/agent.py:_resolve_chat_tool_provider` reference (symbol
  deleted in M3); rephrased in terms of the shipped contract
  ("glow agents never run the v1.0 core's toolless-chat
  short-circuit — they always have a real `Client` to call into").
- New first `Changed` bullet — **Breaking** entry covering the
  `mcp_client → tool_provider` rename on `run_agent` and
  `StreamingAgentLoop.stream`, the `Client[Any] | None = None`
  type, the `T0AgentsFatalError` guard for hallucinated tool
  calls, and the lock-step (non-optional) rename of the
  `execute_tool_calls` / `execute_tool_calls_parallel` helpers.
- Existing `Agent.__init__ accepts tool_provider=None` entry —
  rewrote to describe the final v1.0 behaviour (chat terminal
  forwards `None` straight through; no FastMCP server, no
  `Client` lifecycle implied; `T0AgentsFatalError` surfaces
  policy violations).  Restated the `mcp/` standalone invariant
  in terms of the new shared `Client[Any] | None` typed parameter.

Test docstring (commit `5c224ae`, `docs(test): refresh stale
toolless-chat docstring`):

- `tests/test_core/test_agent.py:144-153` —
  `test_chat_agent_without_tool_provider_constructs` docstring
  rewritten: removed the obsolete `FastMCP("agent-default")`
  ephemeral-allocation language; described the no-allocation
  pass-through behaviour; cross-referenced the sibling
  `test_toolless_chat_invoke_returns_text` that owns the
  `FastMCP.__init__` spy assertion.

Lint debt (commit `9a296be`, `chore(lint): clear pre-existing ruff
debt`) — closing the 6 errors M1+M2+M3 dev-logs parked:

- `tests/test_cli/test_elicit_tui.py:326` — RUF012: annotated
  `_TrackingRenderer.pauses` with `ClassVar[list[str]]`.  The
  list is intentionally class-shared (one tracking sink across
  the test); `ClassVar` is the right rune for that intent.
- `tests/test_core/test_agent_integration.py:16` — F401: dropped
  the unused `tests.server.mcp` import.
- `tests/test_core/test_agent_parity.py:15` — F401: dropped the
  unused `fastmcp.Client` import.
- `tests/test_memory/test_agent_memory.py:112` — F841: dropped
  the unused `r2 =` assignment (the assertion uses `t0.calls[1]`
  directly).
- `tests/test_memory/test_agent_memory_integration.py:16-17` —
  two F401s: dropped unused `InMemoryMemory` and `tests.server.mcp`
  imports; re-spaced the blank line between the import block and
  `pytestmark`.

Auto-memory update (this dev-log per CLAUDE.md memory rules,
not via diff): the note `feedback_mcp_standalone.md` previously
asserted that `run_agent`'s signature was "frozen".  After M1+M2
that is no longer literally true — the second parameter was
renamed and given an optional default.  The *standalone-submodule
rule* it codifies (no imports from higher layers like
`core/types.py:ToolProvider`) is still strictly enforced and
exactly the reason M3 had to add `cast(Client[Any] | None, ...)`
at the two boundary call sites in `core/agent.py`.  The memory
note should be reworded the next time it is touched so it
captures the structural rule rather than the literal frozen-signature
claim.

## Deviations from plan

- **M4 scope expanded by 4 items beyond the original PLAN.md
  checklist.**
  - **What changed**: in addition to the 5 user-facing files
    PLAN.md M4 listed (README, getting-started.md, agent.md,
    CLAUDE.md, CHANGELOG.md), the M4 implementation also
    refreshed `docs/api-reference.md` (6 stale `mcp_client`
    signature documentations), `docs/glow/getting-started.md`
    (a paragraph that described the M3-deleted ephemeral
    fallback as glow's "unlock"), `CHANGELOG.md` line 278 (a
    dangling `_resolve_chat_tool_provider` symbol reference
    inside the glow App-owned Client entry), and the
    `test_chat_agent_without_tool_provider_constructs` docstring
    in `tests/test_core/test_agent.py`.  PLAN.md M4's
    acceptance criterion ("no stale `mcp_client` references
    survive") would have been impossible to satisfy without
    the api-reference fix; the other three are M3 follow-ups
    PLAN.md missed.
  - **Why** (decision at implementation time): all four were
    discovered during M4's Phase-1 exploration of the public
    surface and confirmed by the user during the planning
    `AskUserQuestion` round.  Folding them into M4 keeps the
    "optional `tool_provider`" workstream a clean single PR
    chain rather than spawning a separate `docs(polish)` PR
    for stragglers.
  - **Impact on next milestones**: M5 (`examples/agent/00_chat_only/`
    showcase, OPTIONAL) is unaffected — the example's contract
    is independent of the API reference.  If M5 is greenlit,
    the README chat-only snippet from this milestone can be
    linked from the example's README.

- **Local-variable name `mcp_client` retained in 13 user-code
  examples across `docs/` and `src/t0_agents/core/agent.py`'s
  class docstring.**
  - **What changed**: the broad acceptance grep
    `grep -rn "mcp_client" docs/ README.md CLAUDE.md` returns
    13 hits (vs. the "zero hits" PLAN.md M4 target).  All 13
    are user-code local-variable names (e.g. `mcp_client =
    Client(mcp)`; `tool_provider=mcp_client` as a kwarg) and
    one is the actual `conftest.py` fixture name documented
    in CLAUDE.md.  No remaining hit names the legacy parameter
    `mcp_client` in any framework API.
  - **Why** (decision at implementation time): user-confirmed
    during the M4 planning round (`AskUserQuestion`: "Leave
    them (Recommended) — local variable names are user's
    choice and don't conflict with the API rename").  Renaming
    them mechanically would touch ~8 unrelated files with no
    pedagogical or API improvement; the broad grep was over-
    specified.  The narrow check
    `grep -rn 'mcp_client[^=]*[:,=]' src/t0_agents/` (looking
    for stale parameter-style usages) returns zero hits — the
    *parameter* `mcp_client` has been fully eliminated; only
    user-code identifier name reuse remains.
  - **Impact on next milestones**: none.  Document the
    carve-out semantics in the public-quality-polish backlog
    if a future polish pass wants to address it.

- **Test assertion strings carve-out (2 intentional `"mcp_client"`
  occurrences in `tests/`).**
  - **What changed**:
    `grep -rn '"mcp_client"\|mcp_client=' tests/` returns
    2 hits, both inside `assert "mcp_client" not in message`
    assertions in `tests/test_mcp/test_loop_toolless.py:108`
    and `tests/test_stream/test_runner_toolless.py:128`.
  - **Why** (decision at implementation time): these are
    load-bearing assertions inherited from M1+M2 that prove
    the legacy `"mcp_client"` token is **absent** from the
    `T0AgentsFatalError` messages the loop raises on
    hallucinated tool calls.  Renaming them would defeat the
    test's purpose; the M3 dev-log already explicitly
    documented this carve-out.  M4 simply inherits it.
  - **Impact on next milestones**: none.  These are
    permanent assertion anchors; they should never be
    rewritten until either runner's message wording itself
    is intentionally changed.

## Tests & lint

- `uv run pytest -m "not integration"` → **928 passed**, 2
  skipped, 61 deselected (3.26s).  Identical totals to the
  M3 baseline.  No behavioural changes in M4, so the count
  doesn't move — docstring/CHANGELOG/lint-only.
- `uv run pytest -m "not integration" tests/test_core/test_agent.py
  -k "toolless"` → **1 passed**, 33 deselected (M3 baseline; the
  M4 docstring rewrite kept assertions intact).
- `uv run mypy src/` → **Success: no issues found in 79 source
  files**.  M4 didn't touch `src/` outside the `core/agent.py`
  class docstring (no source-level impact).
- `uv run ruff check src/ tests/` → **All checks passed!** The
  M3 baseline was 6 errors; commit `9a296be` cleared every one
  documented under M1+M2+M3.
- `grep -rn "agent-default\|_resolve_chat_tool_provider"
  docs/ README.md CLAUDE.md CHANGELOG.md src/ tests/ examples/`
  → **zero hits**.
- `grep -rn "mcp_client" docs/ README.md CLAUDE.md` → 13 hits,
  all user-code local-variable names or the conftest fixture
  name (see Deviations).  No remaining parameter-name reference.
- `grep -rn "mcp_client" src/` → 1 hit in
  `src/t0_agents/core/agent.py:76`, inside the `Agent` class
  docstring example showing user-code; same carve-out.
- `grep -rn '"mcp_client"\|mcp_client=' tests/` → **2 hits**
  matching the documented M1+M2 carve-out (load-bearing
  assertion-of-absence strings).
- Signature inspector: `tool_provider: Client[Any] | None = None`
  confirmed on both `run_agent` and `StreamingAgentLoop.stream`.
- Integration suite (`-m integration`): **not run** by design.
  M4 is docs/changelog/docstring/lint-only with no
  gateway-observable behavioural change.  Same rationale as
  M1+M2+M3.

## Notes for next milestone

M5 (`examples/agent/00_chat_only/` showcase) is **OPTIONAL** per
PLAN.md.  If the user opts in, the following M4-established
invariants make it a straightforward landing:

- **README chat-only snippet is the canonical 4-line shape.**
  The M5 example's `main.py` can structurally mirror it
  (`Agent(AgentConfig(name=..., function_name=...), t0=t0)` +
  `await agent.invoke(...)` + `print(result.response)`); the
  M5 README can link both directions ("see README.md for the
  4-line embedded snippet").
- **`docs/glow/getting-started.md` now describes the same
  unlock from the glow angle.**  If M5 wants to mention glow
  as an even shorter alternative, it can link this section
  rather than restating.
- **`docs/api-reference.md` accurately documents the optional
  signature.**  M5's `main.py` can be type-checked against the
  documented contract without surprises.
- **The compose-pin selection rule still holds.**  Per memory
  `feedback_example_compose_images.md`, M5's `docker-compose.yml`
  should reuse the same compose pin as `examples/14_multi_agent`
  or `examples/17_structured`; avoid `examples/02..07,18`.
- **Auto-memory note pending**: the next time a memory pass
  touches `feedback_mcp_standalone.md`, reword it so it
  captures the *standalone-submodule rule* explicitly (no
  imports from higher layers; `Client[Any] | None` is the only
  allowed type on the `tool_provider` parameter) instead of
  claiming the signature is "frozen".  The literal frozen
  claim is no longer accurate after M1+M2's rename; the
  underlying structural rule is unchanged.

## Open questions

None.  M4 closes the optional-`tool_provider` workstream's
docs/changelog/lint surface cleanly.  M5 may proceed if the
user opts in; otherwise the workstream is complete with
M1+M2+M3+M4 merged.
