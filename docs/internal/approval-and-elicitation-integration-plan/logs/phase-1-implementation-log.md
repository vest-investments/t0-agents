# Phase 1 Implementation Log

> **Date:** 2026-04-14
> **Status:** Complete -- 16 offline tests pass, 2 integration tests ready for docker compose

## What was done

### Server tools (`tests/server.py`)
- Added `multi_step(ctx)` -- two sequential `ctx.elicit()` calls: `response_type=str` then `response_type=None`
- Added `pick_priority(ctx)` -- constrained choice `ctx.elicit("Priority?", response_type=["low","medium","high"])`
- Added `ask_age(ctx)` -- scalar int `ctx.elicit("Age?", response_type=int)`

### Fixtures (`tests/conftest.py`)
- Added `auto_approve_handler` -- returns `ElicitResult(action="accept", content={})`
- Added `auto_decline_handler` -- returns `ElicitResult(action="decline")`
- Added `elicitation_client` -- `Client(mcp, elicitation_handler=auto_approve_handler)`

### Pattern tests (`tests/test_mcp/test_elicitation.py` -- `TestElicitationPatterns`)
- P1.3 `test_approval_accept` -- fixture handler approves
- P1.4 `test_approval_decline` -- fixture handler declines
- P1.5 `test_approval_cancel` -- handler returns `action="cancel"`, same else branch as decline
- P1.6 `test_scalar_str` -- stateful handler, `content={"value": "Alice"}`, multi_step tool
- P1.7 `test_scalar_int` -- `content={"value": 42}`, ask_age tool
- P1.8 `test_structured_input` -- `content={"name":"Bob","age":25}`, collect_info tool
- P1.9 `test_constrained_choice` -- `content={"value": "medium"}`, pick_priority tool
- P1.10 `test_multi_turn_accept` -- both elicitations accepted, asserts handler called twice
- P1.11 `test_multi_turn_cancel` -- first accepted, second declined

### Error path tests (`tests/test_mcp/test_elicitation.py` -- `TestElicitationErrors`)
- P1.12 `test_no_handler_raises_tool_error` -- ToolError (not McpError)
- P1.13 `test_no_handler_through_execute` -- real Client + `execute_tool_calls` -> ToolResult with error
- P1.14 `test_timeout_covers_elicitation` -- slow handler + short timeout -> McpError/TimeoutError

### Integration tests (`tests/test_mcp/test_integration.py`)
- P1.15 `test_run_agent_elicitation_approve` -- full loop with auto-approve handler
- P1.16 `test_run_agent_elicitation_decline` -- full loop with decline handler

## Divergences from the plan

### 1. P1.12/P1.13: ToolError not McpError (carried from Phase 0)

The implementation plan says P1.12 should expect `McpError(-32600)` and P1.13 should produce `"ERROR: MCP protocol error"`. Per Phase 0's finding, the actual behavior is `ToolError` (caught by `except ToolError` at loop.py line 114), producing `"ERROR: tool execution failed: ..."`.

### 2. P1.7: Added `ask_age` tool (not in original plan)

The implementation plan didn't specify a tool for scalar int testing. Added `ask_age(ctx)` to `tests/server.py` since testing `response_type=int` requires a tool that actually uses it server-side.

### 3. P1.14: Accepts both McpError and TimeoutError

The timeout test accepts `pytest.raises((McpError, TimeoutError))` because the exact exception type depends on how the in-memory transport propagates `anyio.fail_after`. In practice, `McpError` is raised.

### 4. Integration tests in test_integration.py

The implementation plan placed P1.15-P1.16 in `test_elicitation.py`. They were moved to `test_integration.py` to keep all docker-dependent tests together (user preference).

### 5. Dropped `tool_choice={"specific": "approve_action"}` from integration tests

`run_agent` forwards `tool_choice` to **every** inference turn. With `{"specific": "approve_action"}`, the second turn (where the LLM produces final text) fails with Groq error `"Tool choice is required, but model did not call a tool"`. Fixed by omitting `tool_choice` and relying on the prompt to direct the LLM, matching the pattern used by all other `run_agent` integration tests.

### 5. Content format for scalar/constrained-choice handlers

Scalar types (`str`, `int`) and constrained choices (`["low","medium","high"]`) require `content={"value": <scalar>}` because FastMCP wraps them in `ScalarElicitationType[T]`. This is documented in the integration contract (section 8, constraint #5) but easy to miss when writing handlers.

## Verification

```
uv run pytest tests/test_mcp/test_elicitation.py -v    # 16 passed in 0.42s
uv run pytest -m "not integration"                      # 118 passed, 26 deselected in 1.02s
uv run ruff check tests/                               # All checks passed
uv run mypy src/                                        # no issues found in 7 source files
```

Integration tests (P1.15-P1.16) verified with `docker compose up -d`.

## Files modified

- `tests/server.py` -- Added `multi_step`, `pick_priority`, `ask_age` tools
- `tests/conftest.py` -- Added `auto_approve_handler`, `auto_decline_handler`, `elicitation_client` fixtures
- `tests/test_mcp/test_elicitation.py` -- Added `TestElicitationPatterns` (9 tests) and `TestElicitationErrors` (3 tests)
- `tests/test_mcp/test_integration.py` -- Added `test_run_agent_elicitation_approve` and `test_run_agent_elicitation_decline`
