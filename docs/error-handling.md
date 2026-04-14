# Error Handling

How `t0-agents` handles errors during tool execution and inference, and how
to handle errors in your own code.

---

## Philosophy

Tool errors never crash the agent loop. When a tool call fails for any
reason, the error is serialized as an `"ERROR: ..."` string and sent back to
the LLM as the tool result. The LLM sees the error message on the next turn
and can retry, choose a different tool, or respond to the user explaining the
failure. This keeps the loop running and lets the model self-correct.

---

## Error taxonomy

Every tool call is executed through `_execute_single_tool_call`, which
catches all exceptions and converts them to `ToolResult` objects with an
error string in the `result` field. The five cases:

| Error | Cause | `ToolResult.result` message | When |
|-------|-------|-----------------------------|------|
| Hallucinated tool name | LLM produced a tool name not in the schema | `"ERROR: tool name '<raw_name>' is not in the schema"` | `ToolCall.name` is `None` (TensorZero validated it against `additional_tools` and zeroed the field) |
| Invalid JSON arguments | LLM generated malformed JSON for the tool arguments | `"ERROR: invalid JSON arguments from LLM: <raw_arguments>"` | `ToolCall.arguments` is `None` (TensorZero parsed and rejected the JSON, but `raw_arguments` carries the original string) |
| `ToolError` | The tool function raised an exception | `"ERROR: tool execution failed: <e>"` | FastMCP catches the exception inside the tool and surfaces it through `ToolError` |
| `McpError` | MCP protocol error or transport timeout (code `-32000`) | `"ERROR: MCP protocol error (possibly timeout): <e>"` | MCP transport layer failure, including per-call timeout expiry |
| Unexpected exception | Any other exception type | `"ERROR: unexpected <ExceptionType>: <e>"` | Catch-all safety net so a single misbehaving tool cannot crash the loop |

In all five cases, the tool call is not retried automatically at the
execution level. The error string becomes the `ToolResult.result` that the
LLM reads on its next turn. The LLM may then choose to retry the tool call,
try a different approach, or report the error to the user.

---

## Recoverable inference errors

Some `TensorZeroError` exceptions indicate a problem with the LLM's output
that can be corrected by re-prompting. The loop detects these by matching
`e.text` against `RECOVERABLE_INFERENCE_PATTERNS`:

```python
RECOVERABLE_INFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"not in request\.tools"),
]
```

When a match is found (status code 400 and text matches a pattern):

1. A corrective user message is injected listing the available tool names and
   asking the model to retry with a valid name.
2. An empty assistant message is appended to maintain the
   assistant/user alternation.
3. The inference ID for that turn is recorded as
   `"error-recoverable-turn-N"` (a placeholder, not a real TensorZero ID).
4. The loop continues to the next turn.

Non-matching `TensorZeroError` exceptions propagate to the caller unchanged.

---

## `MaxTurnsExceededError`

Raised when the loop completes `max_turns` inference calls without the LLM
producing a response that contains no tool calls. The default `max_turns` is
`20`.

```python
class MaxTurnsExceededError(Exception):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
```

The `max_turns` attribute carries the limit that was exceeded. The loop does
not return a partial `AgentResult` -- it raises unconditionally. Callers that
need the intermediate state (episode ID, inference IDs) should track those
externally or wrap the call in a try/except.

---

## Examples

### Easy -- basic try/except

```python
from t0_agents import run_agent, MaxTurnsExceededError

try:
    result = await run_agent(t0, client, "What is 2 + 2?", "my_chat")
    print(result.response)
except MaxTurnsExceededError as e:
    print(f"Agent did not finish within {e.max_turns} turns")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Intermediate -- custom max_turns with logging

Enable the `t0_agents` logger to see error flow in real time. Tool errors
appear at `ERROR` level, turn-by-turn progress at `INFO`, and tool
arguments/results at `DEBUG`.

```python
import logging
from t0_agents import run_agent, MaxTurnsExceededError

logging.basicConfig(level=logging.INFO)
logging.getLogger("t0_agents").setLevel(logging.DEBUG)

try:
    result = await run_agent(
        t0, client, "Summarize the document", "my_chat",
        max_turns=5,
        tool_timeout=60.0,
    )
    print(f"Done in {result.turns} turns: {result.response[:200]}")
except MaxTurnsExceededError:
    print("Agent stuck in a tool-calling loop, check DEBUG logs above")
```

### Advanced -- error budget monitoring

Track how many tool errors occur per episode and alert when a threshold is
exceeded. This pattern is useful in production to detect misbehaving tools or
models that repeatedly hallucinate tool names.

```python
import logging
from t0_agents import run_agent, MaxTurnsExceededError

class ErrorBudgetHandler(logging.Handler):
    """Count tool errors emitted by the agent loop."""

    def __init__(self, threshold: int = 5) -> None:
        super().__init__(level=logging.ERROR)
        self.threshold = threshold
        self.error_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.error_count += 1
        if self.error_count >= self.threshold:
            raise RuntimeError(
                f"Error budget exhausted: {self.error_count} tool errors"
            )

budget = ErrorBudgetHandler(threshold=3)
logging.getLogger("t0_agents").addHandler(budget)

try:
    result = await run_agent(t0, client, "Complex research task", "my_chat")
except MaxTurnsExceededError as e:
    print(f"Max turns ({e.max_turns}) exceeded")
except RuntimeError as e:
    print(f"Aborted: {e}")
finally:
    logging.getLogger("t0_agents").removeHandler(budget)
    print(f"Total tool errors: {budget.error_count}")
```
