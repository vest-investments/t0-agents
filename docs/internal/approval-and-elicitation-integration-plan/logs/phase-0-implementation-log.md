# Phase 0 Implementation Log

> **Date:** 2026-04-14
> **Status:** Complete — all 4 smoke tests pass

## What was done

- Added `approve_action` (binary approval) and `collect_info` (structured input via `UserInfo` dataclass) tools to `tests/server.py`
- Created `tests/test_mcp/test_elicitation.py` with 4 smoke tests:
  - `test_smoke_approve` — handler accepts, tool returns "Approved: deploy"
  - `test_smoke_structured_input` — handler returns `{name, age}`, tool returns greeting
  - `test_smoke_decline` — handler declines, tool returns "Rejected: deploy"
  - `test_smoke_no_handler` — no handler, ToolError raised with "Elicitation not supported"

## Divergences from the plan

### 1. No-handler error type: ToolError, not McpError

**Contract prediction (section 4.1):** `call_tool()` raises `McpError(-32600, "Elicitation not supported")`.

**Actual behavior:** The server-side `McpError` is caught by FastMCP's tool dispatch, wrapped into `CallToolResult(isError=True)`, and sent back as a normal `tools/call` response. The client's `call_tool()` then raises `ToolError` (because `raise_on_error=True` by default), not `McpError`.

**Impact:** None for production. `_execute_single_tool_call` in `loop.py` catches `ToolError` at line 114 (before the `McpError` catch at line 123), so the error path is handled correctly. The error message "Elicitation not supported" propagates through both paths.

**Action:** Updated the smoke test to expect `ToolError` with `match="Elicitation not supported"`. Contract section 4.1 should be annotated in Phase 2 docs.

### 2. Four smoke tests instead of two

The implementation plan (P0.3, P0.4) showed two test functions, but the exit criteria said "All 4 smoke tests pass." Added two additional tests (`test_smoke_structured_input`, `test_smoke_decline`) to cover the structured input tool and the decline path, matching the exit criteria count.

## Verification

```
uv run pytest tests/test_mcp/test_elicitation.py -v    # 4 passed in 0.06s
uv run ruff check tests/                               # All checks passed
uv run mypy src/                                        # no issues found in 7 source files
uv run pytest -m "not integration"                      # 106 passed, 24 deselected in 0.48s
```

## Files modified

- `tests/server.py` — Added `Context` import, `dataclass` import, `approve_action` tool, `UserInfo` dataclass, `collect_info` tool
- `tests/test_mcp/test_elicitation.py` — New file (4 smoke tests)
