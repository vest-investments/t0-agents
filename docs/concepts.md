# Concepts

## The three participants

```
+---------------------+       +-------------+       +--------------------+
| TensorZero Gateway  | <---> |  t0-agents  | <---> | FastMCP Server(s)  |
| (inference + logs)  |       |  (bridge)   |       | (tools)            |
+---------------------+       +-------------+       +--------------------+
```

**TensorZero Gateway** -- An LLM inference gateway that routes requests to model providers, enforces schemas, and logs every inference to ClickHouse for observability and feedback. See [TensorZero docs](https://www.tensorzero.com/docs/).

**t0-agents** -- The bridge layer. Converts MCP tool definitions into TensorZero's `additional_tools` format, orchestrates the inference-execute loop, and serializes tool results back to the string format TensorZero expects.

**FastMCP Server(s)** -- MCP servers that expose tools (functions) via the Model Context Protocol. Tools are discovered at runtime through `list_tools()` and executed via `call_tool()`. See [FastMCP docs](https://gofastmcp.com/).

## Agent loop lifecycle

The `run_agent` function implements a turn-based loop with 7 logical steps:

### 1. Discover

```python
mcp_tools = await mcp_client.list_tools()
t0_tools = convert_tools(mcp_tools)
```

Fetches all available tools from the MCP server and converts them to the `additional_tools` list of dicts. TensorZero requires tools as `additional_tools` dicts at inference time ([TensorZero inference API](https://www.tensorzero.com/docs/gateway/api-reference#post-inference)), each containing `name`, `description`, and `parameters` (JSON Schema).

### 2. Infer

```python
response = await t0.inference(
    function_name=function_name,
    input={"messages": messages},
    additional_tools=t0_tools,
)
```

Sends the conversation history and tool schemas to the gateway. The response contains `content` blocks: `Text`, `Thought`, and/or `ToolCall` instances.

### 3. Check

```python
if not has_tool_calls(response.content):
    # Terminal turn -- extract text and return
```

`has_tool_calls()` inspects content blocks for `ToolCall` instances. `Thought` and `Text` blocks are not tool calls and pass through as `False`.

### 4. Execute

```python
tool_results = await execute_tool_calls(mcp_client, response.content, tool_timeout)
# or: execute_tool_calls_parallel() for concurrent dispatch via asyncio.gather
```

Each `ToolCall` is dispatched to `client.call_tool()`, producing a `CallToolResult`. Errors at every level (hallucinated tool name, invalid JSON arguments, `ToolError`, `McpError`, unexpected exceptions) are caught and converted to error-string `ToolResult` payloads so the LLM can read and retry.

### 5. Serialize

```python
# Inside execute: build_tool_result(tool_call, mcp_result) calls serialize_tool_result()
```

`build_tool_result()` wraps the execution result into a `ToolResult(name, result, id)`. The `result` field is always `str` because TensorZero's `ToolResult.result` type is `str` ([TensorZero types](https://www.tensorzero.com/docs/gateway/api-reference)).

### 6. Loop

```python
messages.append(Message(role="assistant", content=response.content))
messages.append(Message(role="user", content=tool_results))
```

The assistant message (with tool calls) and a single user message (with all tool results) are appended to the conversation history. N tool results go into ONE user message, matching TensorZero's expected message format.

### 7. Return

```python
return AgentResult(
    response=text,
    episode_id=episode_id,
    inference_ids=inference_ids,
    turns=turn + 1,
    response_source=response_source,
)
```

When no tool calls remain, `Text` blocks are joined into the final response string. If the LLM emitted only `Thought` blocks (no `Text`), those are used as a fallback and `response_source` is set to `"thought"`.

## Data flow

```
FastMCP Tool  -->  convert_tools()  -->  dict(name, description, parameters)

ToolCall  -->  call_tool()  -->  CallToolResult  -->  serialize_tool_result()  -->  str

str  -->  ToolResult(name, result, id)  -->  Message(role="user", content=[...])
```

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Async-first | Both the TensorZero SDK (`AsyncTensorZeroGateway`) and FastMCP `Client` are async; synchronous wrappers would add complexity without benefit |
| Stateless loop | Continuity across calls is maintained only via `episode_id`; `run_agent` builds a fresh `messages` list each invocation |
| Error-as-result | All tool errors become `ToolResult` strings (e.g. `"ERROR: tool execution failed: ..."`) so the LLM sees the error and can retry on the next turn |
| Thought-only fallback | If the LLM emits only `Thought` blocks with no `Text`, the thought content is extracted as the response and `response_source` is set to `"thought"` |
| Caller owns context | `run_agent` does not wrap `AsyncTensorZeroGateway` or `Client` in `async with`; the caller manages both lifecycles |
| Description coercion | `None` descriptions are coerced to `""` because TensorZero treats the `description` field as required |
| Recoverable inference errors | `TensorZeroError` responses matching known patterns (e.g. hallucinated tool names rejected by the gateway) trigger a corrective user message and retry instead of propagating |

## AgentResult

| Field | Type | Description |
|-------|------|-------------|
| `response` | `str` | Final text from the terminating turn. Joined from `Text` blocks, or `Thought` blocks if no text was present |
| `episode_id` | `str \| None` | TensorZero episode ID. Captured from the first inference response; `None` only if no inference occurred |
| `inference_ids` | `list[str]` | One inference ID per turn. Useful for sending feedback via `t0.feedback()` targeting specific inferences |
| `turns` | `int` | Number of inference turns the loop executed (1 = single inference, no tool calls) |
| `response_source` | `Literal["text", "thought"]` | `"text"` when the response came from `Text` blocks; `"thought"` when it fell back to `Thought` blocks |
