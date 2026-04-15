# API Reference

Complete API reference for the `t0-agents` library. All public symbols are
importable from the top-level package:

```python
from t0_agents import run_agent, AgentResult, MaxTurnsExceededError, ToolChoice
```

Lower-level helpers are available from `t0_agents.mcp`:

```python
from t0_agents.mcp import (
    convert_tools,
    serialize_tool_result,
    build_tool_result,
    execute_tool_calls,
    execute_tool_calls_parallel,
    has_tool_calls,
)
```

---

## `run_agent`

Main entry point. Runs the full agent loop: discover tools, call TensorZero
inference, execute tool calls, feed results back, repeat until the LLM
responds without tool calls or `max_turns` is reached.

### Signature

```python
async def run_agent(
    t0: AsyncTensorZeroGateway,
    mcp_client: Client[Any],
    user_message: str,
    function_name: str,
    *,
    episode_id: str | None = None,
    max_turns: int = 20,
    tool_timeout: float = 30.0,
    parallel_tools: bool = False,
    dryrun: bool = False,
    tool_choice: ToolChoice | None = None,
) -> AgentResult:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `t0` | `AsyncTensorZeroGateway` | *(required)* | Connected TensorZero gateway instance (already inside `async with`). |
| `mcp_client` | `Client[Any]` | *(required)* | Connected FastMCP client (already inside `async with`). Must remain connected for the full loop. |
| `user_message` | `str` | *(required)* | The user's query as a plain string. |
| `function_name` | `str` | *(required)* | Name of the chat function defined in `tensorzero.toml`. |
| `episode_id` | `str \| None` | `None` | Existing episode ID to continue. `None` starts a fresh episode; TensorZero allocates one and the loop captures it. |
| `max_turns` | `int` | `20` | Maximum inference turns before aborting with `MaxTurnsExceededError`. |
| `tool_timeout` | `float` | `30.0` | Per-tool-call timeout in seconds, forwarded to `Client.call_tool(timeout=...)`. Covers the **entire** `call_tool()` duration including any [elicitation](elicitation.md) wait time. When using elicitation-enabled tools, set to `None` (no timeout) or a generous value (e.g., `300.0`). Only applies to foreground execution. |
| `parallel_tools` | `bool` | `False` | If `True`, dispatches tool calls concurrently via `asyncio.gather`. Otherwise sequential. |
| `dryrun` | `bool` | `False` | Forwarded to every `t0.inference(dryrun=True)` call. Prevents writes to ClickHouse. |
| `tool_choice` | `ToolChoice \| None` | `None` | Tool-choice directive forwarded to every inference call. See `ToolChoice` below. |

### Returns

`AgentResult` with the final text response, episode ID, per-turn inference
IDs, turn count, and response source.

### Raises

- `MaxTurnsExceededError` -- Loop ran `max_turns` times without a
  tool-call-free response.
- `TensorZeroError` -- Non-recoverable inference error from TensorZero
  (propagated). Recoverable errors (matching `RECOVERABLE_INFERENCE_PATTERNS`)
  trigger an automatic retry with a corrective user message.

### Behavior notes

- `dryrun=True` makes inference IDs ephemeral. They are never stored in
  ClickHouse, so any subsequent `t0.feedback()` call targeting those IDs
  will fail with HTTP 400.
- When the LLM returns only `Thought` blocks (no `Text`), the thought
  content is used as a fallback and `response_source` is set to `"thought"`.
- Recoverable `TensorZeroError` (status 400, text matching `"not in
  request.tools"`) injects a corrective user message listing available tools
  and retries the loop turn.
- Tool results from all tool calls in a single turn are bundled into one user
  message (not one message per result).
- When elicitation-enabled tools are in use (tools that call `ctx.elicit()`
  server-side), `tool_timeout` must account for human response time. The
  default `30.0` seconds may not be sufficient. Use `tool_timeout=None` to
  disable or `tool_timeout=300.0` for a generous budget. See
  [Elicitation](elicitation.md) for details.

### Examples

**Easy** -- minimal call:

```python
async with AsyncTensorZeroGateway("http://localhost:3000") as t0:
    async with Client(mcp_server) as client:
        result = await run_agent(t0, client, "Add 2 + 3", "my_chat")
        print(result.response)
```

**Intermediate** -- with `max_turns` and `tool_timeout`:

```python
result = await run_agent(
    t0, client, "Search and summarize results", "my_chat",
    max_turns=5,
    tool_timeout=60.0,
)
print(f"Completed in {result.turns} turns")
```

**Advanced** -- with `parallel_tools`, `tool_choice`, and episode continuation:

```python
result1 = await run_agent(
    t0, client, "Look up the weather in Berlin", "my_chat",
    parallel_tools=True,
    tool_choice="required",
)
result2 = await run_agent(
    t0, client, "Now compare it to Munich", "my_chat",
    episode_id=result1.episode_id,
    tool_choice={"specific": "get_weather"},
)
```

---

## `convert_tools`

Converts FastMCP tool definitions to TensorZero's `additional_tools` format.

### Signature

```python
def convert_tools(mcp_tools: list[MCPTool]) -> list[dict[str, Any]]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mcp_tools` | `list[MCPTool]` | *(required)* | Tool definitions from `client.list_tools()`. |

### Returns

`list[dict[str, Any]]` -- each dict has keys `"name"` (str), `"description"`
(str), and `"parameters"` (JSON Schema dict).

### Behavior notes

- `description` is coerced: `None` and empty strings become `""`. This
  handles `@mcp.tool` functions that have no docstring and no explicit
  `description=` parameter.
- `inputSchema` from the MCP tool is passed through as-is to `"parameters"`.

### Example

```python
mcp_tools = await client.list_tools()
t0_tools = convert_tools(mcp_tools)
# [{"name": "add", "description": "Add two numbers", "parameters": {...}}]
```

---

## `serialize_tool_result`

Serializes a FastMCP `CallToolResult` to the `str` that TensorZero expects
for `ToolResult.result`.

### Signature

```python
def serialize_tool_result(result: CallToolResult) -> str:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `CallToolResult` | *(required)* | Result from `client.call_tool()`. |

### Returns

`str` -- serialized result.

### Behavior notes

Four branches, evaluated in order:

1. **`is_error=True`** -- returns `"ERROR: "` + joined `.content[*].text`.
   Falls back to `"ERROR: Unknown tool error"` if no textual block is present.
2. **`.data` is `str`** -- passthrough, returned as-is.
3. **`.data` is non-`None`, non-`str`** -- `json.dumps(data, default=str,
   ensure_ascii=False)`. Booleans become JSON-cased (`"true"` / `"false"`).
   `datetime` and `UUID` are stringified via `default=str`. If `json.dumps`
   fails, falls through to branch 4.
4. **`.data` is `None`** (or `json.dumps` failed) -- concatenates `.text`
   from every content block that has it, joined with `"\n"`. Non-textual
   blocks (`ImageContent`, `AudioContent`, `EmbeddedResource`) are skipped.

### Example

```python
mcp_result = await client.call_tool("add", {"a": 1, "b": 2})
text = serialize_tool_result(mcp_result)  # "3" or '{"sum": 3}'
```

---

## `build_tool_result`

Builds a TensorZero `ToolResult` from a `ToolCall` and a FastMCP
`CallToolResult`.

### Signature

```python
def build_tool_result(tool_call: ToolCall, mcp_result: CallToolResult) -> ToolResult:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_call` | `ToolCall` | *(required)* | The tool call from the inference response. |
| `mcp_result` | `CallToolResult` | *(required)* | The result from FastMCP `call_tool`. |

### Returns

`ToolResult` with `name`, `result` (serialized string), and `id`.

### Behavior notes

- `ToolCall.name` is `Optional[str]` (`None` when the LLM hallucinated a
  tool name not in the schema). `ToolResult.name` must be `str`, so the
  function falls back to `tool_call.raw_name` -- the raw string the model
  produced, which is always populated.

### Example

```python
tool_result = build_tool_result(tool_call, mcp_result)
# ToolResult(name="add", result="5", id="call_abc123")
```

---

## `execute_tool_calls`

Executes every `ToolCall` in a list of content blocks sequentially.

### Signature

```python
async def execute_tool_calls(
    mcp_client: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mcp_client` | `Client[Any]` | *(required)* | Connected FastMCP client. |
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response (mix of `Text`, `Thought`, `ToolCall`). |
| `timeout` | `float` | `30.0` | Per-tool-call timeout in seconds. |

### Returns

`list[ToolResult]` -- one per `ToolCall` block, in input order.

### Behavior notes

- Non-`ToolCall` blocks (`Text`, `Thought`) are skipped silently.
- Output order matches the input order of `ToolCall` blocks.
- Uses `_execute_single_tool_call` internally, so the full error contract
  (hallucinated names, invalid JSON, `ToolError`, `McpError`, catch-all)
  applies per call.

### Example

```python
results = await execute_tool_calls(client, response.content, timeout=10.0)
```

---

## `execute_tool_calls_parallel`

Executes `ToolCall` blocks concurrently via `asyncio.gather`.

### Signature

```python
async def execute_tool_calls_parallel(
    mcp_client: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mcp_client` | `Client[Any]` | *(required)* | Connected FastMCP client. |
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response. |
| `timeout` | `float` | `30.0` | Per-tool-call timeout in seconds. |

### Returns

`list[ToolResult]` -- one per `ToolCall` block, in input order (guaranteed by
`asyncio.gather`).

### Behavior notes

- Same contract as `execute_tool_calls`: skips non-`ToolCall` blocks,
  preserves order, same error handling per call.
- Returns an empty list if no `ToolCall` blocks are present.
- Best suited for I/O-bound tools where concurrent dispatch reduces
  wall-clock time.

### Example

```python
results = await execute_tool_calls_parallel(client, response.content)
```

---

## `has_tool_calls`

Checks whether any content block in a list is a `ToolCall`.

### Signature

```python
def has_tool_calls(content_blocks: list[Any]) -> bool:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response. |

### Returns

`bool` -- `True` if at least one block is a `ToolCall` instance.

### Behavior notes

- `Thought` blocks are not `ToolCall` instances and return `False`.
- Used by `run_agent` to decide whether to continue the loop (tool execution)
  or return the final response.

### Example

```python
if has_tool_calls(response.content):
    results = await execute_tool_calls(client, response.content)
```

---

## `AgentResult`

Dataclass returned by `run_agent`.

### Definition

```python
@dataclass
class AgentResult:
    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `response` | `str` | *(required)* | Joined text from the terminating turn's `Text` blocks (or `Thought` blocks as fallback). |
| `episode_id` | `str \| None` | `None` | TensorZero episode ID. Captured from the first inference response. |
| `inference_ids` | `list[str]` | `[]` | One inference ID per turn. Includes `"error-recoverable-turn-N"` placeholders for recovered errors. |
| `turns` | `int` | `0` | Number of inference turns actually used. |
| `response_source` | `Literal["text", "thought"]` | `"text"` | `"text"` if the response came from `Text` blocks, `"thought"` if from `Thought` blocks (fallback). |

### Example

```python
result = await run_agent(t0, client, "Hello", "my_chat")
print(result.response)         # "Hi there!"
print(result.turns)            # 1
print(result.response_source)  # "text"
```

---

## `MaxTurnsExceededError`

Exception raised when the agent loop reaches `max_turns` without the LLM
producing a tool-call-free response.

### Definition

```python
class MaxTurnsExceededError(Exception):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")
```

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `max_turns` | `int` | The limit that was exceeded. |

### Example

```python
try:
    result = await run_agent(t0, client, "Complex task", "my_chat", max_turns=3)
except MaxTurnsExceededError as e:
    print(f"Loop stopped after {e.max_turns} turns")
```

---

## `ToolChoice`

Re-exported from `tensorzero`. Controls how the LLM selects tools during
inference.

### Type

```python
ToolChoice = Literal["auto", "required", "off"] | dict[str, str]
```

### Accepted values

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to call a tool. |
| `"required"` | Model must call at least one tool. |
| `"off"` | Tool calling disabled for this inference. |
| `{"specific": "<name>"}` | Model must call the named tool. |

### Behavior notes

- Passed as `tool_choice` to `run_agent`. `None` (the default) is equivalent
  to letting the model decide freely (TensorZero's default behavior).
- The `tool_choice` directive applies to every inference call in the loop, not
  just the first.

### Example

```python
# Force a specific tool
result = await run_agent(
    t0, client, "Get weather", "my_chat",
    tool_choice={"specific": "get_weather"},
)
```

---

## Reference Elicitation Handlers

Pre-built handlers for use with `Client(mcp, elicitation_handler=...)`. Intended
for **testing and prototyping** — production applications should implement a
handler that presents prompts to the user via their UI.

Available from `t0_agents.mcp`:

```python
from t0_agents.mcp import auto_approve_handler, auto_decline_handler
```

### `auto_approve_handler`

Approves binary approval requests (`response_type is None`) and declines
structured input requests (where data would need to be provided).

#### Signature

```python
async def auto_approve_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
```

#### Behavior

| `response_type` | Action | Content |
|-----------------|--------|---------|
| `None` (approval) | `"accept"` | `{}` |
| Anything else | `"decline"` | — |

#### Example

```python
from fastmcp import Client
from t0_agents.mcp import auto_approve_handler

client = Client(mcp_server, elicitation_handler=auto_approve_handler)
async with client:
    result = await client.call_tool("delete_files", {"files": ["a.txt"]})
```

### `auto_decline_handler`

Declines all elicitation requests unconditionally. Useful for verifying that
tools handle rejection gracefully.

#### Signature

```python
async def auto_decline_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
```

#### Behavior

Always returns `ElicitResult(action="decline")` regardless of `response_type`.

#### Example

```python
from fastmcp import Client
from t0_agents.mcp import auto_decline_handler

client = Client(mcp_server, elicitation_handler=auto_decline_handler)
async with client:
    result = await client.call_tool("delete_files", {"files": ["a.txt"]})
    # Tool receives decline and handles gracefully
```

See [Elicitation](elicitation.md) for the full handler interface, content
format guide, and server-side patterns.
