# 05 Error Handling

How the t0-agents loop recovers from tool failures without crashing.

## Error contract

Every tool error is caught inside `_execute_single_tool_call` and returned to the LLM as a `ToolResult` with an `"ERROR: ..."` string. The LLM sees the error on the next turn and can retry with different tools or arguments. The agent loop never crashes from a single tool failure.

### Error types

| Cause | ToolResult payload | Example |
|-------|-------------------|---------|
| Hallucinated tool name | `ERROR: tool name '<name>' is not in the schema` | LLM invents `fetch_users` when only `get_data` exists |
| Invalid JSON arguments | `ERROR: invalid JSON arguments from LLM: <raw>` | LLM emits malformed JSON in the arguments field |
| `ToolError` (tool raises) | `ERROR: tool execution failed: <message>` | `unreliable_api` raises `ValueError` |
| `McpError` (protocol/timeout) | `ERROR: MCP protocol error (possibly timeout): <message>` | `slow_process` exceeds `tool_timeout` |
| Unexpected `Exception` | `ERROR: unexpected <ExceptionType>: <message>` | Catch-all for anything else |

In all cases the error string lands on `ToolResult.result`, which the LLM reads as the tool's output on the next inference turn.

## MaxTurnsExceededError

If the LLM keeps making tool calls without producing a text-only response, the loop aborts after `max_turns` iterations by raising `MaxTurnsExceededError`. This prevents infinite retry loops (e.g., the LLM repeatedly calling a permanently broken tool).

Catch it to handle the situation gracefully:

```python
from t0_agents import MaxTurnsExceededError

try:
    result = await run_agent(t0, client, prompt, "agent", max_turns=3)
except MaxTurnsExceededError as e:
    print(f"Loop hit the safety valve at {e.max_turns} turns")
```

## Scenarios in this example

1. **Error recovery** -- The LLM calls `unreliable_api`, sees the `ERROR:` response, and falls back to `get_data`.
2. **Timeout** -- `slow_process` with a 1-second delay completes within the 2-second `tool_timeout`. (Set `tool_timeout=0.5` to observe the timeout error path.)
3. **Max turns safety** -- The LLM is told to keep retrying a broken tool. After 3 turns, `MaxTurnsExceededError` fires.

## Prerequisites

- Python 3.11+
- `uv sync` (install dependencies)
- `GROQ_API_KEY` in your environment

## Run

```bash
docker compose -f examples/05_error_handling/docker-compose.yml up -d
uv run python examples/05_error_handling/main.py
docker compose -f examples/05_error_handling/docker-compose.yml down
```
