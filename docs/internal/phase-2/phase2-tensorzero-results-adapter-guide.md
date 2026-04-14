# Phase 2: Results Adapter (FastMCP → TensorZero) — Implementation Guide

> Everything you need from TensorZero's documentation to implement the Results Adapter.
> FastMCP-specific details are left as placeholders (`[FastMCP: ...]`) for you to fill in.

---

## 1. The Complete Flow You're Building

```
User query
  → T0 inference (returns ToolCall content blocks)
    → Your adapter extracts name, arguments, id
      → FastMCP executes the tool (call_tool)
        → Your adapter serializes the result
          → T0 receives the ToolResult in the next inference turn
            → LLM produces a coherent response
```

This maps to checklist items **F2.1** through **F2.8**.

---

## 2. Understanding TensorZero's ToolCall Response (What You Receive)

When TensorZero's inference returns a tool call, the response `content` array contains blocks of type `tool_call` with this structure:

```json
{
  "type": "tool_call",
  "arguments": {
    "location": "Tokyo",
    "units": "celsius"
  },
  "id": "123456789",
  "name": "get_temperature",
  "raw_arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}",
  "raw_name": "get_temperature"
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"tool_call"` |
| `arguments` | `object \| null` | **Validated** arguments parsed by T0. `null` if the model produced invalid JSON. |
| `id` | `string` | Unique ID for this tool call. **You must pass this back in the `tool_result`.** |
| `name` | `string \| null` | **Validated** tool name. `null` if the model hallucinated a tool name not in the schema. |
| `raw_arguments` | `string` | The raw JSON string the model generated (may be malformed). |
| `raw_name` | `string` | The raw tool name string the model generated (may not match any tool). |

> **Critical for F2.1**: Use `arguments` (the validated `object`) to pass to FastMCP's `call_tool()`, NOT `raw_arguments`. If `arguments` is `null`, the model produced invalid tool call arguments — handle this as an error. Similarly, if `name` is `null`, the model hallucinated a tool name.

*Source: gateway-api-reference.md > "Chat Function" Response section*

### Python SDK Types

When using the TensorZero Python SDK, the response content blocks are typed:

```python
from tensorzero import Text, ToolCall, ToolResult, Message
```

- `content_block.name` → validated tool name
- `content_block.arguments` → validated arguments (dict)
- `content_block.id` → tool call ID
- `content_block.raw_name` → raw model output name
- `content_block.raw_arguments` → raw model output arguments (string)

*Source: examples/mcp-model-context-protocol/main.py*

---

## 3. Executing the Tool via FastMCP (F2.1)

### What T0 Gives You → What FastMCP Needs

```python
# From the T0 response:
for content_block in response.content:
    if isinstance(content_block, ToolCall):
        tool_name = content_block.name        # validated name (string)
        tool_args = content_block.arguments   # validated args (dict)
        tool_id   = content_block.id          # unique ID (string)

        # [FastMCP: call_tool(tool_name, tool_args)]
        # mcp_result = await session.call_tool(tool_name, tool_args)
```

> **Key detail from the MCP example**: The `arguments` field from T0's `ToolCall` is already a `dict` (parsed JSON), which is the format MCP's `call_tool` expects. No additional parsing is needed.

*Source: examples/mcp-model-context-protocol/main.py*

---

## 4. Serializing the Tool Result (F2.2 – F2.6)

### What TensorZero Expects as a `tool_result`

For the **native T0 API** (`POST /inference`), a `tool_result` content block has this shape:

```json
{
  "type": "tool_result",
  "id": "123456789",
  "name": "get_temperature",
  "result": "70"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `string` | yes | Must be `"tool_result"` |
| `id` | `string` | yes | Must match the `id` from the original `tool_call` |
| `name` | `string` | yes | Must match the `name` from the original `tool_call` |
| `result` | `string` | yes | The tool's output as a **string** |

> **The `result` field is always a string.** This is the most important detail for your serializer.

*Source: gateway-api-reference.md > "Content Block" types for `tool_result`*

### For the OpenAI-Compatible API (`POST /openai/v1/chat/completions`)

The format is different — tool results go as `role: "tool"` messages:

```json
{
  "role": "tool",
  "tool_call_id": "123456789",
  "content": "70"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `role` | `string` | Must be `"tool"` |
| `tool_call_id` | `string` | Must match the `id` from the tool call |
| `content` | `string` | The tool result as a string |

*Source: gateway-api-reference.md > OpenAI-compatible API `messages` section*

### Serialization Rules for `serialize_tool_result()`

Since T0 always expects the result as a **string**, your serializer must handle these cases:

| Case | Input | Output | Checklist |
|------|-------|--------|-----------|
| **str** | `"70"` | `"70"` (pass through) | F2.2 |
| **dict** | `{"temp": 70}` | `json.dumps({"temp": 70})` | F2.3 |
| **None / empty** | `None` | Fallback: concatenate `.content[*].text` blocks | F2.4 |
| **complex types** | `datetime`, `UUID` | `json.dumps(data, default=str)` | F2.5 |
| **error** | `.is_error=True` | [FastMCP: how you surface errors] | F2.6 |

### Reference Implementation from TensorZero's MCP Example

The official T0 + MCP example handles serialization like this:

```python
# MCP servers technically could return multiple blocks, so we concatenate them
t0_tool_result_text = "\n".join([block.text for block in mcp_tool_result.content])
```

> **This is the F2.4 fallback pattern**: when the MCP result contains content blocks (not a single data value), iterate over `.content` and join the `.text` fields.

Then construct the `ToolResult`:

```python
t0_tool_result = ToolResult(
    name=content_block.name,
    result=t0_tool_result_text,
    id=content_block.id,
)
```

*Source: examples/mcp-model-context-protocol/main.py*

### Complete Serializer Skeleton

```python
import json

def serialize_tool_result(mcp_result) -> str:
    """
    Convert a FastMCP tool result into a string suitable for T0's `result` field.
    """
    # [FastMCP: adapt these attribute names to your FastMCP result object]

    # F2.6 — Handle errors
    # [FastMCP: check if mcp_result.is_error is True]
    # if mcp_result.is_error:
    #     return f"ERROR: {extract_error_text(mcp_result)}"

    # F2.4 — Handle None / missing data → fallback to content blocks
    # [FastMCP: check mcp_result.data or equivalent]
    data = getattr(mcp_result, 'data', None)

    if data is None:
        # Fallback: concatenate text from content blocks
        # [FastMCP: adjust attribute path as needed]
        return "\n".join([block.text for block in mcp_result.content])

    # F2.2 — str passthrough
    if isinstance(data, str):
        return data

    # F2.3 + F2.5 — dict / complex types
    # default=str handles datetime, UUID, etc.
    return json.dumps(data, default=str)
```

---

## 5. Sending the Tool Result Back to TensorZero (F2.7)

### Native T0 API Pattern

Tool results go inside a **user** message as content blocks:

```python
# Collect all tool results from this turn
tool_results = []
for content_block in response.content:
    if isinstance(content_block, ToolCall):
        # ... execute tool, serialize result ...
        t0_tool_result = ToolResult(
            name=content_block.name,
            result=serialized_result,  # always a string
            id=content_block.id,
        )
        tool_results.append(t0_tool_result)

# Add as a user message with tool_result content blocks
if tool_results:
    user_message = Message(
        role="user",
        content=tool_results,
    )
    messages.append(user_message)
```

Then call inference again with the updated messages:

```python
response = await t0.inference(
    function_name="your_function",
    episode_id=episode_id,  # IMPORTANT: reuse the same episode
    input={
        "messages": messages,
    },
    additional_tools=t0_tools,
)
```

*Source: examples/mcp-model-context-protocol/main.py*

### Raw HTTP (Native API) Pattern

```json
{
  "function_name": "your_function",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Tokyo?"
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "tool_call",
            "id": "123456789",
            "name": "get_temperature",
            "arguments": "{\"location\": \"Tokyo\"}"
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "tool_result",
            "id": "123456789",
            "name": "get_temperature",
            "result": "70"
          }
        ]
      }
    ]
  }
}
```

*Source: gateway-api-reference.md > "Chat Function with Multi-Turn Tool Use" example*

### Raw HTTP (OpenAI-Compatible API) Pattern

```json
{
  "model": "tensorzero::function_name::weather_chatbot",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Tokyo (°F)?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "123",
          "type": "function",
          "function": {
            "name": "get_temperature",
            "arguments": "{\"location\": \"Tokyo\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "123",
      "content": "70"
    }
  ]
}
```

*Source: gateway-guides.md > "Tool Use (Function Calling)" multi-turn example*

---

## 6. Episode Management for Multi-Turn Tool Use

TensorZero uses **episodes** to track sequences of related inferences.

- **First inference**: Don't provide `episode_id`. T0 creates one and returns it in `response.episode_id`.
- **Subsequent inferences** (including the one with the tool result): Pass back the same `episode_id`.

```python
# First call — no episode_id
response = await t0.inference(
    function_name="your_function",
    input={"messages": messages},
    additional_tools=t0_tools,
)
episode_id = response.episode_id  # save this

# Second call — with tool results, same episode
response2 = await t0.inference(
    function_name="your_function",
    episode_id=episode_id,       # reuse
    input={"messages": messages},
    additional_tools=t0_tools,
)
```

*Source: gateway-guides.md > "Episodes"; examples/mcp-model-context-protocol/main.py*

---

## 7. Tool Configuration Recap

For completeness, here's how tools are defined in TensorZero. This affects what `name` values appear in `ToolCall` responses.

### Static (in config file)

```toml
[tools.get_temperature]
description = "Get the current temperature for a given location."
parameters = "tools/get_temperature.json"
strict = true  # optional, defaults to false

[functions.weather_chatbot]
type = "chat"
tools = ["get_temperature"]
```

The JSON Schema file:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for"
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

### Dynamic (at inference time via `additional_tools`)

```python
t0_tools = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"],  # JSON Schema object, NOT a file path
    }
    for tool in mcp_tools
]

response = await t0.inference(
    function_name="your_function",
    input={"messages": messages},
    additional_tools=t0_tools,
)
```

> **Important**: When using `additional_tools`, the `parameters` field is the JSON Schema **object** itself, not a file path (unlike the config file).

*Source: gateway-guides.md > "Defining tools dynamically at inference time"; gateway-api-reference.md > `additional_tools`*

### Tool Choice Strategy

You can control whether tool use is optional or mandatory:

| Strategy | Behavior |
|----------|----------|
| `none` | No tools used |
| `auto` (default) | Model decides |
| `required` | Model must use a tool |
| `{ specific = "tool_name" }` | Model must use that specific tool |

Set in config (`tool_choice = "auto"`) or at inference time.

*Source: gateway-configuration-reference.md > `tool_choice`; gateway-guides.md > "Customizing the tool calling strategy"*

---

## 8. Handling Edge Cases

### Multiple Tool Calls in One Response

If `parallel_tool_calls = true`, the model may return multiple `tool_call` blocks in a single response. You must execute all of them and return all results in the same user message.

```python
tool_results = []
for content_block in response.content:
    if isinstance(content_block, ToolCall):
        # Execute each tool
        mcp_result = await session.call_tool(content_block.name, content_block.arguments)
        result_text = serialize_tool_result(mcp_result)
        tool_results.append(ToolResult(
            name=content_block.name,
            result=result_text,
            id=content_block.id,
        ))

# All results go in ONE user message
messages.append(Message(role="user", content=tool_results))
```

*Source: gateway-guides.md > "Calling multiple tools in parallel"; examples/mcp-model-context-protocol/main.py*

### Mixed Content Blocks (Text + Tool Calls)

The assistant response can contain **both** `text` and `tool_call` blocks:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Sure, I can help you with that."
    },
    {
      "type": "tool_call",
      "id": "0",
      "name": "get_temperature",
      "arguments": "{\"location\": \"New York\"}"
    }
  ]
}
```

When replaying this as an assistant message, include **all** content blocks.

*Source: gateway-api-reference.md > "Example" accordion under `input.messages`*

### Invalid Tool Calls (`name: null` or `arguments: null`)

If `name` is `null`, the model hallucinated a tool name. If `arguments` is `null`, the model produced invalid JSON for the arguments. In both cases, you should NOT attempt to execute the tool.

*Source: gateway-api-reference.md > ToolCall response fields*

### Streaming Responses with Tool Calls

In streaming mode, `tool_call` chunks arrive incrementally:

```json
{
  "type": "tool_call",
  "id": "123456789",
  "raw_name": "get_",
  "raw_arguments": "{\"loc"
}
```

You must accumulate `raw_name` and `raw_arguments` deltas across chunks before parsing.

*Source: gateway-api-reference.md > Streaming response section*

---

## 9. Checklist Verification Map

| Checklist Item | What to Verify | Key Documentation |
|---------------|----------------|-------------------|
| **F2.1** | `call_tool(name, arguments)` receives correct values from T0's `ToolCall` | Section 3 above |
| **F2.2** | `str` data passes through unchanged | Section 4, serialization table |
| **F2.3** | `dict` data becomes `json.dumps()` | Section 4, serialization table |
| **F2.4** | `None` data falls back to `.content[*].text` concatenation | Section 4, MCP example pattern |
| **F2.5** | `datetime`, `UUID` handled via `default=str` | Section 4, serialization table |
| **F2.6** | `.is_error=True` is handled | Section 4, `[FastMCP: ...]` placeholder |
| **F2.7** | `ToolResult` accepted by T0 in next inference | Section 5, message construction |
| **F2.8** | LLM produces coherent response | End-to-end test: user → T0 → tool → T0 → response |

---

## 10. Full Reference: Source Files

| Topic | File | Section |
|-------|------|---------|
| Tool call response format | `gateway-api-reference.md` | "Chat Function" response, `tool_call` content block |
| `tool_result` content block spec | `gateway-api-reference.md` | Content block types |
| OpenAI-compatible tool messages | `gateway-api-reference.md` | OpenAI-compatible `messages` |
| Tool use guide | `gateway-guides.md` | "Tool Use (Function Calling)" |
| Tool configuration | `gateway-configuration-reference.md` | `[tools.tool_name]` |
| MCP integration example | `examples/mcp-model-context-protocol/main.py` | Full file |
| Episodes | `gateway-guides.md` | "Episodes" |
| Dynamic tools at inference | `gateway-api-reference.md` | `additional_tools` |
| `tool_choice` strategies | `gateway-configuration-reference.md` | `tool_choice` |
| Parallel tool calls | `gateway-guides.md` | "Calling multiple tools in parallel" |

---

*Generated from TensorZero documentation. FastMCP-specific sections marked with `[FastMCP: ...]` are placeholders for you to complete.*
