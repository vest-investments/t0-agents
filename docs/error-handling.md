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

**One taxonomy, both loops.** The turn-level logic around dispatch —
input validation, the corrective retry, the assistant append, final-text
mining, `AgentResult`, the max-turns guard — is shared by `run_agent`
and `StreamingAgentLoop` via `core/turn_engine.py`, and both dispatch
through `_execute_single_tool_call`. The first two rows apply
identically on the streaming path: TensorZero only zeroes
`name`/`arguments` on non-streaming responses, so the streaming rebuild
mirrors that zeroing against the advertised tool list before dispatch —
a hallucinated name or unparseable arguments never reach the tool on
either loop.

**Elicitation not supported.** When a tool calls `ctx.elicit()` but the
Client was created without an `elicitation_handler`, the server raises
`McpError(-32600, "Elicitation not supported")`. FastMCP's tool dispatch
catches this, wraps it into `CallToolResult(isError=True)`, and returns it
as a normal `tools/call` response. The client raises `ToolError` (because
`raise_on_error=True` by default), which hits the `ToolError` branch in the
table above. The resulting `ToolResult` message is:
`"ERROR: tool execution failed: Elicitation not supported"`.

This means the existing error contract handles the no-handler case
automatically -- no new error types or catch branches are needed. The LLM
sees the error on its next turn and can explain the failure or try a
different approach. To enable elicitation, pass an `elicitation_handler`
when creating the Client. See [Elicitation](elicitation.md) for the handler
interface.

In all five cases, the tool call is not retried automatically at the
execution level. The error string becomes the `ToolResult.result` that the
LLM reads on its next turn. The LLM may then choose to retry the tool call,
try a different approach, or report the error to the user.

---

## Non-text tool content

A tool can return images, audio, or embedded resources instead of (or
alongside) text. These cannot be forwarded to the LLM as text, but — in the
same spirit as the error taxonomy — they are never dropped silently.
`serialize_tool_result` summarises them into a loud placeholder and logs a
`WARNING`:

- **Non-text only** → the placeholder is the whole result, e.g. `"[tool
  returned 1 image (image/png); non-text content is not forwarded to the
  model]"`.
- **Mixed text + non-text** → the text is preserved verbatim and the omitted
  content is appended as a trailing note.
- **Genuinely empty** (a tool that returned nothing) → `""`, unchanged.

Structured results (a tool returning an object) serialize as JSON through the
normal `.data` path; the raw structured dict is additionally exposed on
[`ToolResultEvent.structured`](streaming.md) for streaming consumers.

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

## The timeout stack

A tool call crosses up to four independent timeout layers. Knowing which one
fired (and which one *didn't*) is the difference between a five-minute fix and
a day of confusion:

| Layer | Knob | Semantics | Default |
|-------|------|-----------|---------|
| 1. Per-call deadline | `tool_timeout` (AgentConfig / `@app.agent` / `run_agent`) → `call_tool(timeout=)` | **Wall-clock** limit on receiving the tool's response. Transport-independent (an anyio deadline on the response future). **Not** reset by server activity — progress notifications, log messages, and elicitation requests do not extend it. `None` disables it. | `30.0` s |
| 2. Session default | `fastmcp.Client(timeout=)` | Same mechanism as layer 1; a per-call `timeout=` beats it. Unset means *wait forever*. | unset |
| 3. httpx read window | `MCPServer(timeout=)` on a glow `url` mount | Maximum **silent gap between bytes** on the remote HTTP hop. **Any** SSE frame the server emits — progress, log message, elicitation request — resets it, so a heartbeating server can hold a small window open indefinitely. | 300 s read (SDK `httpx.Timeout(30.0, read=300.0)`) |
| 4. Proxy inner hop | — (pre-runtime; see note) | A glow `url` mount proxies through a fresh inner client per call. The inner call has **no per-call deadline**; only layer 3 governs the wire. | — |

The **tool-inventory fan-out** has a fifth, separate layer: none of the above
covers `list_tools()` (layer 1 binds `call_tool` only, and a swallowed read
timeout leaves nothing to resolve the request), so each handle's share of the
fan-out is bounded by `MCPRuntime(tool_list_timeout=)`. A server that cannot
answer within it degrades — its last-known inventory is served, or it is
excluded if it never answered — instead of hanging the merge for every
conversation in the process.

Two ground truths worth committing to memory (mcp SDK 1.27.0, source-verified):

- **A dead request stream is a hang, not an error.** When layer 3 fires
  mid-request, the SDK swallows the read-timeout on the request's SSE stream
  (`"SSE stream ended"` at DEBUG) and — unless the server attached `id:`s to
  its SSE events (requires a server-side event store) — the request future
  simply never resolves. With `tool_timeout=None` and no session default,
  that is an infinite hang. Pair a small read window with a per-call deadline,
  or make the server heartbeat. The inventory fan-out is the one path with
  its own backstop against this (`tool_list_timeout` above); a tool *call*
  still needs the pairing.
- **Stream resumption exists but needs server support.** The SDK client
  auto-reconnects a dropped request stream via GET + `Last-Event-ID` when it
  has seen at least one `id:`-bearing event (independent of the
  `resumption_token` machinery), and the standalone GET stream retries up to
  2 consecutive failures. Without a server event store there are no event
  ids, so neither path engages.

Rules of thumb:

- Elicitation-capable tools: `tool_timeout=None` (or a value comfortably above
  the server's elicitation wait) **and** the client's read window above the
  server's elicit deadline — the client-side window must always exceed the
  server's wait, with margin.
- Servers that hold calls open for minutes should emit a heartbeat
  (`ctx.info(...)` or progress with a token) every few seconds; it keeps every
  read window on the path alive and turns "mystery hang" into "bounded wait".
- Fast tools: keep the 30 s default. It converts a wedged server into a
  readable `McpError` (code `-32000`-style timeout) the loop folds into an
  `ERROR:` tool result the LLM can react to.

---

## Session loss

A server that rolls invalidates the warm session behind every handle
pointing at it — and a server that simply *goes away* for a while breaks
the same handle a third way. All three reach the runtime, and on the
call path they recover differently:

| Shape | Cause | Reaches the runtime as | Recovery (call path) |
|-------|-------|------------------------|----------------------|
| HTTP 404 — the spec mapping | server rolled | `McpError`, code `32600` | rebuild the client, re-send the call once |
| bare HTTP 400 — older server SDKs | server rolled | `httpx.HTTPStatusError` | rebuild the client, **propagate** |
| client lost its session | server was *absent* | `RuntimeError("Client is not connected…")` | rebuild the client, re-send the call once |

The asymmetry is a safety property, not an oversight, and it follows
entirely from *where* each failure happens relative to dispatch. The SDK
synthesises the `32600` error *per rejected POST*, so it carries that
request's own id and the call provably never dispatched — re-sending is
safe even for a non-idempotent order write. A bare 400 instead tears the
whole transport down, failing every call in flight on that handle,
including one the server already dispatched and ran. The two are
indistinguishable from the client, so the call is never re-sent; only
the handle is rebuilt, which is what stops the *next* call from
inheriting a dead client.

The third shape is the safest of all: fastmcp raises it from the
`Client.session` property on *attribute access*, before any network I/O,
so the request could not have left the process. It appears when a server
is absent long enough for the transport to die — scaled to zero, evicted,
crash-looping — after which fastmcp drops its session while the handle
still believes it is connected. Nothing was ever rejected, so neither
session predicate matches it; without its own arm the handle stays dead
for the lifetime of the process, which is exactly what was observed in
production before this shape was handled.

The inventory fan-out (`list_tools`) carries none of that caution:
listing is read-only, so on *any* of the three shapes the handle is
rebuilt and the list re-fetched exactly once, inside the same per-server
`tool_list_timeout` budget. Only if the rebuilt client also fails does
the [bounded fan-out](multi-server.md)'s degradation apply — last-known
inventory, else exclusion. This is what lets a process heal from a
server rollout *or* a server outage on the next turn instead of serving
a stale inventory until it restarts.

A propagated rejection lands in the taxonomy's catch-all row above:
`"ERROR: unexpected HTTPStatusError: …"`.

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
