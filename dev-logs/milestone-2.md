# Milestone 2 — Dev Log

**Date**: 2026-05-19
**Base branch**: main
**Branch**: `milestone/2-stream-runner-optional-tool-provider`
**PR**: _filled after Phase 5_

## Implemented

- `src/t0_agents/stream/runner.py`
  - `StreamingAgentLoop.stream` signature: `tool_provider: Client[Any]` →
    `tool_provider: Client[Any] | None = None`. `user_message` and
    `function_name` carry `= ""` defaults purely as a Python-syntax
    workaround so `tool_provider` can default at position 2 — same
    trick M1 documented for `run_agent`. Existing positional callers
    (the `Agent._chat_streaming_terminal` site at
    `core/agent.py:434`, the 6 call sites in `tests/test_stream/
    test_runner.py`, the two `mock.patch` references in
    `tests/test_delegation/test_orchestrator.py`) keep working
    unchanged.
  - Empty-input guard inserted right after the existing
    `system`/`system_arguments` mutex check: leaving `user_message`
    or `function_name` at `""` raises `ValueError("...requires
    non-empty `user_message` and `function_name`...")`. The
    generator function defers the raise to the first `__anext__`,
    which is fine — tests drive the iterator to surface it.
  - `mcp_tools = await tool_provider.list_tools() if tool_provider
    is not None else []` — `convert_tools([])` already returned
    `[]`, no change downstream. The existing
    `"function=%s, max_turns=%d, tools=%d"` log line at the loop
    entry prints `tools=0` for the toolless path.
  - Defensive guard injected right before the tool-dispatch
    `executor(...)` call (`runner.py:268-275`): if
    `tool_provider is None`, raise `T0AgentsFatalError` with the
    operator-facing fragment kept identical to `loop.py:475` —
    only the action hint swaps `run_agent(..., tool_provider=...)`
    for `StreamingAgentLoop.stream(..., tool_provider=...)`. Keeps
    log greps for `"tool calls from the LLM but no `tool_provider`
    was configured"` stable across both runners.
  - `T0AgentsFatalError` added to the existing
    `from t0_agents.core.exceptions import …` line. No other
    helper in `runner.py` takes `tool_provider` (`_open_stream`,
    `_tap_final_chunk`, `_build_assistant_blocks`,
    `_SyntheticChatResponse`, `_final_text`) — nothing else to
    rename.
  - Method docstring grows a `tool_provider` paragraph mirroring
    `run_agent`'s, and notes the `ValueError` / `T0AgentsFatalError`
    contract.
- `tests/test_stream/test_runner_toolless.py` — new file, **5
  async unit tests**:
  1. `test_streaming_loop_toolless_yields_done_event` — happy path,
     asserts `result.response`, episode/inference id propagation,
     `result.turns == 1`, and `fake_t0.calls[0]["additional_tools"]
     == []` plus `fake_t0.calls[0]["stream"] is True`. A
     `_ListToolsAssertingClient` spy stays in scope as a sentinel
     but is **not** passed in.
  2. `test_streaming_loop_toolless_raises_on_hallucinated_tool_call`
     — asserts `T0AgentsFatalError`, checks the message contains
     `"tool calls from the LLM"` and `` "`tool_provider`" ``, and
     guarantees the legacy `"mcp_client"` token is absent.
  3. `test_streaming_loop_empty_user_message_raises_value_error` —
     surfaces the `ValueError("requires non-empty …")` from the
     dummy-default guard.
  4. `test_streaming_loop_empty_function_name_raises_value_error` —
     same shape, swapped guard target.
  5. `test_streaming_loop_with_spy_client_passed_in` — sanity
     inverse: when the spy is actually passed as `tool_provider`,
     `list_tools` is reached and the spy raises. Anchors the
     negative assertion in test #1.

## Deviations from plan

- **`_ListToolsAssertingClient` stays local, not shared**
  - **What changed**: the spy class is duplicated verbatim in
    `tests/test_stream/test_runner_toolless.py` instead of being
    promoted to `tests/_fake_mcp.py`.
  - **Why** (decision at implementation time): the M1 dev-log
    explicitly deferred this extraction ("Reuse the
    `_ListToolsAssertingClient` spy pattern (consider extracting it
    into `tests/_fake_mcp.py` if M2 wants to share; not done in M1
    to keep the new module self-contained)"). Promoting it now
    would change the M1 file as a side-effect of an M2 PR — not
    worth the cross-module coupling for two use-sites. Revisit if
    M3 testing introduces a third use-site.
  - **Impact on next milestones**: none functional. M3 may revisit
    if its `test_agent.py` `monkeypatch` of `FastMCP.__init__`
    feels like it would benefit from the same spy idiom; the
    decision is local.

- **Pre-existing ruff issues outside M2 scope (carried over from M1)**
  - **What changed**: `uv run ruff check src/ tests/` still
    reports the same 6 errors in test files that M1's dev-log
    flagged (`test_cli/test_elicit_tui.py:326` RUF012,
    `tests/test_core/test_agent_*.py` F401s, `tests/test_memory/
    test_agent_memory.py:112` F841, `tests/test_memory/
    test_agent_memory_integration.py` F401s). M2 didn't touch any
    of those files.
  - **Why** (decision at implementation time): per the polish-
    backlog feedback rule, M2 doesn't silently fold unrelated
    fixes into the PR. The M2 acceptance criteria scope is
    `src/t0_agents/stream/ tests/test_stream/` (both clean).
  - **Impact on next milestones**: still parked for M4 (docs
    milestone) or a separate housekeeping PR.

## Tests & lint

- `uv run pytest -m "not integration" tests/test_stream/` →
  **30 passed** (existing 25 + 5 new), 0.33s.
- `uv run pytest -m "not integration"` → **928 passed**, 2
  skipped, 61 deselected (5.69s). M1's baseline was 921 passed +
  2 skipped + 61 deselected — the +7 reflects the 5 new
  streaming-toolless tests plus 2 other tests that were added
  upstream in unrelated PRs between M1 and M2 (`test_agent_unit`
  variants surfaced in `git log`). No regressions; the deselect
  count and skip count match M1.
- `uv run mypy src/t0_agents/stream/` → **clean** (4 source files).
- `uv run ruff check src/t0_agents/stream/ tests/test_stream/`
  → **All checks passed!**
- `uv run ruff check src/ tests/` → same 6 pre-existing errors
  as in M1's dev log (see Deviations); none caused by M2.
- `grep -rn "mcp_client" src/t0_agents/mcp/ src/t0_agents/stream/`
  → zero hits.
- Signature parity script:
  ```
  parity OK: run_agent[tool_provider]=Client[Any] | None;
             stream[tool_provider]=Client[Any] | None
  ```
- `grep -rn 'tool calls from the LLM but no' src/t0_agents/` →
  **2 hits** (`mcp/loop.py:475`, `stream/runner.py:269`) — the
  operator-facing fragment is greppable in both runners.
- Integration suite (`-m integration`): **not run** in M2 by
  design — same reasoning as M1 (M2 doesn't touch behaviour the
  gateway-dependent tests cover; the parity tests are unit-level
  by construction).

## Notes for next milestone

These are the contracts M3 (`agent-drop-ephemeral-fallback`)
inherits:

- **Both terminals now accept `None`.** `run_agent` (M1) and
  `StreamingAgentLoop.stream` (M2) both treat
  `tool_provider=None` as "skip `list_tools`; raise
  `T0AgentsFatalError` if the LLM still hallucinates a tool call".
  This unblocks M3's central deletion: `_resolve_chat_tool_provider`
  at `src/t0_agents/core/agent.py:41-59` no longer protects any
  meaningful behavior — both `_chat_terminal` (line 381) and
  `_chat_streaming_terminal` (line 434) can replace the
  `async with _resolve_chat_tool_provider(ctx.tool_provider) as
  tool_provider:` wrapper with a direct pass-through of
  `ctx.tool_provider`. The `streamer.stream(...)` call site at
  `core/agent.py:434-451` already forwards `tool_provider`
  positionally, so the call shape doesn't change.
- **Guard message wording is the load-bearing fragment to keep.**
  If M3 ends up touching either runner, preserve the substring
  `"tool calls from the LLM but no `tool_provider` was configured"`
  verbatim — both `tests/test_mcp/test_loop_toolless.py` and
  `tests/test_stream/test_runner_toolless.py` assert on it.
- **The dummy-default `""` convention is now established in two
  public-API surfaces.** If M3 adds new entry points, follow the
  same precedent (defaulted `tool_provider` + `""` defaults on
  required-string positionals + a `ValueError` guard at the top
  of the body). M3 itself doesn't add new entry points, but the
  pattern is now consistent and M4's docs / CHANGELOG entry can
  describe it as a single API rule.
- **Spy duplication is intentional.** Don't promote
  `_ListToolsAssertingClient` to `tests/_fake_mcp.py` opportunistically
  — see Deviations above. If a third use-site appears (e.g. in
  `tests/test_core/test_agent.py` for the
  `test_*_does_not_allocate_fastmcp` M3 spy work), revisit then.
- **Dev-log directory** stays `dev-logs/` at repo root.

## Open questions

None. M3 picks up cleanly against the two-terminal optional
contract this milestone established.
