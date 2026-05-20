# Milestone 1 — Dev Log

**Date**: 2026-05-19
**Base branch**: main
**Branch**: `milestone/1-run-agent-optional-tool-provider`
**PR**: _(filled in after `gh pr create`)_

## Implemented

- `src/t0_agents/mcp/loop.py`
  - `run_agent` signature: `mcp_client: Client[Any]` → `tool_provider: Client[Any] | None = None`. Docstring `Args:` block rewritten to describe the new contract (when `None`, no tools advertised, single inference + return; when provided, must already be inside `async with`, forwarded into the helpers on tool-dispatch turns).
  - `user_message` and `function_name` now carry `= ""` defaults because Python forbids a required positional after a defaulted one. Every existing positional caller (≈ 80 sites across `src/` and `tests/`) keeps working unchanged.
  - `list_tools()` call gated: `mcp_tools = await tool_provider.list_tools() if tool_provider is not None else []`. The downstream `convert_tools([])` already returned `[]`; no change there. The existing `"function=%s, max_turns=%d, tools=%d"` log line at the loop entry prints `tools=0` for the toolless path.
  - Defensive guard injected right before the tool-dispatch `executor(...)` call: if `tool_provider is None`, raise `T0AgentsFatalError` with the message
    > `Agent loop received tool calls from the LLM but no \`tool_provider\` was configured. Pass a connected \`fastmcp.Client\` to \`run_agent(..., tool_provider=...)\` if your agent is expected to dispatch tools.`

    Wording chosen so `T0AgentsFatalError`'s pass-through behaviour in `_execute_single_tool_call` surfaces the policy violation verbatim instead of getting stringified into a `ToolResult`.
  - Helpers renamed in lock-step (shipped as a separate commit, `4958a45`):
    - `_execute_single_tool_call(tool_provider: Client[Any], ...)`
    - `execute_tool_calls(tool_provider: Client[Any], ...)`
    - `execute_tool_calls_parallel(tool_provider: Client[Any], ...)`
    All three stay required (no default) because they are only reached inside the tool-dispatch branch, after the M1 guard has already proven `tool_provider is not None`.
- `tests/test_mcp/test_loop_toolless.py` — new file with three async unit tests:
  1. `test_run_agent_toolless_returns_text` — happy path, asserts `result.response`, episode/inference id propagation, `result.turns == 1`, and that `t0.calls[0]["additional_tools"] == []`.
  2. `test_run_agent_toolless_raises_on_hallucinated_tool_call` — asserts `T0AgentsFatalError`, checks the message fragments `"tool calls from the LLM"` and `` "`tool_provider`" ``, and guarantees the legacy `"mcp_client"` token is absent.
  3. `test_run_agent_toolless_with_spy_client_passed_in` — sanity inverse: when a `_ListToolsAssertingClient` spy is actually passed in, `list_tools` is reached and the spy raises. Anchors the negative assertion in test 1.
- `PLAN.md` — committed at the top of the branch so the dev-log and PR can link back to the canonical 5-milestone plan.

## Deviations from plan

- **Dummy defaults on `user_message` / `function_name`**
  - **What changed**: gave them `= ""` defaults instead of leaving them as required positionals.
  - **Why** (decision at implementation time): the plan explicitly mandates `tool_provider: Client[Any] | None = None` at position 2 **and** "Positional callers stay unaffected." Python's syntax rule (non-default cannot follow default) makes those two clauses jointly satisfiable only by giving #3 and #4 defaults too. Keyword-only would have broken ~80 positional call sites for no functional gain.
  - **Impact on next milestones**: M2 will need the same trick on `StreamingAgentLoop.stream` — its current signature is `stream(self, t0, tool_provider, function_name, user_message, ...)`, so check whether the same dummy-default workaround is required (likely yes). Note the cross-module style consistency.

- **Pre-existing ruff issues outside M1 scope**
  - **What changed**: `uv run ruff check src/ tests/` reports 6 errors in test files unrelated to M1 (`test_cli/test_elicit_tui.py:326` RUF012, two `tests/test_core/test_agent_*.py` F401s, one `tests/test_memory/test_agent_memory.py:112` F841, two `tests/test_memory/test_agent_memory_integration.py` F401s).
  - **Why** (decision at implementation time): all six predate M1; the plan acceptance criterion was scoped to `src/t0_agents/mcp/ tests/test_mcp/` (both clean). They were already logged in `project_pre_public_polish.md` (memory) on 2026-05-16. Not silently fixed inside this PR per the polish-backlog feedback rule.
  - **Impact on next milestones**: none. M4 (docs/CHANGELOG) is the natural place to bundle them — flag the option during M4 planning.

## Tests & lint

- `uv run pytest -m "not integration" tests/test_mcp/` → **139 passed**, 25 deselected (1.04s). Includes the 3 new toolless tests.
- `uv run pytest -m "not integration"` → **921 passed**, 2 skipped, 61 deselected (5.47s). No regressions across `core/`, `delegation/`, `glow/`, `guardrails/`, `memory/`, `middleware/`, `output/`, `run/`, `serve/`, `stream/`, `testing/`.
- `uv run mypy src/t0_agents/` → **clean** (79 source files).
- `uv run ruff check src/t0_agents/mcp/ tests/test_mcp/` → **clean**.
- `uv run ruff check src/ tests/` → 6 pre-existing errors (see Deviations); none caused by M1.
- `grep -rn "mcp_client" src/t0_agents/mcp/` → zero hits.
- `python -c "...; assert sig.parameters['tool_provider'].default is None"` → passes; annotation reports `Client[Any] | None`.
- Integration suite (`-m integration`): **not run** in M1 by design; M1 doesn't touch behaviour the gateway-dependent tests cover.

## Notes for next milestone

These are the contracts M2 (`stream-runner-optional-tool-provider`) must inherit verbatim:

- **Guard message (copy verbatim into `stream/runner.py`)**:
  ```
  Agent loop received tool calls from the LLM but no `tool_provider`
  was configured. Pass a connected `fastmcp.Client` to
  `run_agent(..., tool_provider=...)` if your agent is expected to
  dispatch tools.
  ```
  If M2 wants the message to read `StreamingAgentLoop.stream(..., tool_provider=...)` instead of `run_agent(...)`, that is a localised rewording — keep the rest identical so the operator-facing fragment `"tool calls from the LLM but no `tool_provider` was configured"` remains a stable greppable string.
- **Helper naming convention**: parameter named `tool_provider` everywhere. Helpers stay non-optional; only the public top-level entry-point grows the `| None = None`.
- **Test-file naming**: new file `tests/test_stream/test_runner_toolless.py` mirroring `tests/test_mcp/test_loop_toolless.py`. Reuse the `_ListToolsAssertingClient` spy pattern (consider extracting it into `tests/_fake_mcp.py` if M2 wants to share; not done in M1 to keep the new module self-contained).
- **`Client[Any] | None` not `ToolProvider`** — confirmed during M1 exploration; the `mcp/` standalone rule (`feedback_mcp_standalone.md`) forbids importing protocols from `core/types.py`. M2's runner already lives under `stream/`, which is allowed to depend on `core/`, so `ToolProvider` *would* be reachable there. **Do not use it.** Use `Client[Any] | None` so the contract stays uniform across the public surface.
- **Dummy-defaults precedent**: if M2's `stream` signature has required positionals after `tool_provider`, apply the same `= ""` workaround documented under "Deviations" above rather than reordering kwargs.
- **Dev-log directory**: `dev-logs/` at repo root is the established location for this PLAN.md's milestones.

## Open questions

None for M2. The above contract notes resolve every M1↔M2 interface concern.
