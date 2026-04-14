# Phase 3: Agent Loop — TensorZero Implementation Guide

> All information in this guide is sourced exclusively from the TensorZero documentation.
> Sections marked **[fast-mcp]** are placeholders for you to fill in.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [TensorZero Configuration (Prerequisites)](#2-tensorzero-configuration-prerequisites)
3. [Content Block Types (Request & Response)](#3-content-block-types-request--response)
4. [The Agent Loop — Step by Step](#4-the-agent-loop--step-by-step)
5. [Episode Management](#5-episode-management)
6. [Parallel Tool Calls](#6-parallel-tool-calls)
7. [Message History Format](#7-message-history-format)
8. [Continuous Conversation](#8-continuous-conversation)
9. [Complete Reference Example (MCP)](#9-complete-reference-example-mcp)
10. [Checklist Mapping](#10-checklist-mapping)

---

## 1. Architecture Overview

The agent loop is a cycle:

```
User query
  → POST /inference (with messages + episode_id)
    → LLM returns content blocks
      → If tool_call blocks exist:
          Execute tools → collect tool_result blocks
          → Append assistant message + user(tool_results) to history
          → Loop back to POST /inference
      → If only text blocks:
          Done. Present final response to user.
```

TensorZero does **not** run tools for you. The gateway returns `tool_call` content blocks; your application executes the tools and sends results back in the next inference call.

---

## 2. TensorZero Configuration (Prerequisites)

### 2.1 Define tools in `tensorzero.toml`

Each tool needs a name, description, and a JSON Schema file for its parameters:

```toml
# tensorzero.toml

[tools.get_temperature]
description = "Get the current temperature for a given location."
parameters = "tools/get_temperature.json"
strict = true  # optional, defaults to false
```

The JSON Schema file:

```json
// tools/get_temperature.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "Get the current temperature for a given location.",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for (e.g. \"New York\")"
    },
    "units": {
      "type": "string",
      "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\"). Defaults to \"fahrenheit\".",
      "enum": ["fahrenheit", "celsius"]
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

> (source: `gateway-guides.md` > "Tool Use (Function Calling)", URL: https://www.tensorzero.com/docs/gateway/guides/tool-use.md)

### 2.2 Attach tools to a chat function

Only functions of type `chat` can call tools:

```toml
[functions.weather_chatbot]
type = "chat"
tools = ["get_temperature"]
```

> (source: `gateway-configuration-reference.md` > `[functions.function_name]` > `tools`)

### 2.3 Dynamic tools at inference time (for MCP)

When integrating with MCP servers, tools are discovered at runtime. Use `additional_tools` instead of configuring them in TOML:

```python
t0_tools = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"],
    }
    for tool in mcp_tools
]
```

Pass them in each inference call via `additional_tools`. The `parameters` field must contain the JSON Schema object directly (not a file path).

> (source: `gateway-guides.md` > "Defining tools dynamically at inference time"; `examples/mcp-model-context-protocol/main.py`)

### 2.4 Tool choice strategy

You can control tool calling behavior via `tool_choice` (in config or per-request):

| Strategy | Behavior |
|----------|----------|
| `none` | The function should not use any tools |
| `auto` | The model decides whether to use a tool (default when tools are present) |
| `required` | The model must use a tool |
| `{ specific = "tool_name" }` | Force a specific tool |

> (source: `gateway-api-reference.md` > "API Reference: Inference" > `tool_choice`)

### 2.5 Parallel tool calls

```toml
[functions.my_function]
type = "chat"
parallel_tool_calls = true
tools = ["tool_a", "tool_b"]
```

Or at inference time: `parallel_tool_calls=True`.

**Important:** Most model providers do **not** support parallel tool calls. At the moment, only **Fireworks AI** and **OpenAI** support them.

> (source: `gateway-api-reference.md` > "API Reference: Inference" > `parallel_tool_calls`)

---

## 3. Content Block Types (Request & Response)

Understanding content blocks is critical for building the agent loop correctly.

### 3.1 In responses (from TensorZero)

The `content` field in the response is a list of content blocks:

**`text` block:**
```json
{
  "type": "text",
  "text": "The weather in Tokyo is 25°C."
}
```

**`tool_call` block:**
```json
{
  "type": "tool_call",
  "id": "123456789",
  "name": "get_temperature",
  "raw_name": "get_temperature",
  "arguments": { "location": "Tokyo", "units": "celsius" },
  "raw_arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"
}
```

Key fields in `tool_call` response blocks:
- `arguments` (object): The **validated** arguments (`null` if invalid)
- `name` (string): The **validated** tool name (`null` if invalid)
- `raw_arguments` (string): The raw arguments string from the model
- `raw_name` (string): The raw tool name from the model
- `id` (string): The ID you must reference in the `tool_result`

> (source: `gateway-api-reference.md` > "API Reference: Inference" > Response > Chat Function > `content`)

### 3.2 In requests (to TensorZero)

When sending messages back to the gateway, use these content block formats:

**`tool_call` block (in assistant messages):**
```json
{
  "type": "tool_call",
  "id": "0",
  "name": "get_temperature",
  "arguments": "{\"location\": \"Tokyo\"}"
}
```

Note: In request `tool_call` blocks, `arguments` is a **string** (JSON-serialized).

**`tool_result` block (in user messages):**
```json
{
  "type": "tool_result",
  "id": "0",
  "name": "get_temperature",
  "result": "70"
}
```

**Critical:** The `result` field **must be a string**.

> (source: `gateway-api-reference.md` > "API Reference: Inference" > `input.messages` > content blocks)

---

## 4. The Agent Loop — Step by Step

### 4.1 Initialize state

```python
messages = []
episode_id = None
```

### 4.2 Get user input

```python
if not messages or messages[-1]["role"] == "assistant":
    user_input = input("[User] ")
    messages.append(Message(role="user", content=user_input))
```

### 4.3 Call inference

Using the native TensorZero Python SDK:

```python
response = await t0.inference(
    function_name="my_function",
    episode_id=episode_id,
    input={
        "messages": messages,
    },
    additional_tools=t0_tools,  # if using dynamic tools (MCP)
)
```

Using HTTP directly:

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "weather_bot",
    "episode_id": "...",
    "input": {
      "messages": [...]
    }
  }'
```

> (source: `gateway-api-reference.md` > "API Reference: Inference" > `POST /inference`)

### 4.4 Capture `episode_id` from first response

```python
episode_id = response.episode_id
```

The first call returns a new `episode_id` (since you passed `None`). **You must propagate this to all subsequent calls.**

> (source: `gateway-guides.md` > "Episodes": "When you make the first inference request, you don't have to provide an `episode_id`. The gateway will create a new episode for you and return the `episode_id` in the response.")

### 4.5 Append assistant message to history

```python
assistant_message = Message(role="assistant", content=response.content)
messages.append(assistant_message)
```

### 4.6 Process content blocks — determine if loop continues

```python
tool_results = []
for content_block in response.content:
    if isinstance(content_block, Text):
        print(content_block.text)
    elif isinstance(content_block, ToolCall):
        # Execute the tool (this is YOUR code / fast-mcp)
        result = await execute_tool(content_block.name, content_block.arguments)

        tool_results.append(ToolResult(
            name=content_block.name,
            result=str(result),  # must be a string
            id=content_block.id,
        ))
```

### 4.7 Termination condition

**The loop terminates when the LLM responds with only `text` blocks (no `tool_call` blocks).**

```python
if tool_results:
    # More work to do — append tool results and loop
    user_message = Message(role="user", content=tool_results)
    messages.append(user_message)
    # → go back to step 4.3
else:
    # Done — the assistant's text response is the final answer
    break
```

### 4.8 Tool execution with fast-mcp

**[fast-mcp]** — Fill in your fast-mcp tool execution logic here. The MCP example from TensorZero uses:

```python
mcp_tool_result = await session.call_tool(content_block.name, content_block.arguments)
t0_tool_result_text = "\n".join([block.text for block in mcp_tool_result.content])
```

> (source: `examples/mcp-model-context-protocol/main.py`)

---

## 5. Episode Management

### 5.1 What is an episode?

An episode is a sequence of inferences associated with a common downstream outcome. It can span multiple functions and multiple calls to the same function.

### 5.2 Rules

1. **First inference:** Do **not** provide `episode_id` (or pass `None`). The gateway generates one.
2. **Subsequent inferences:** **Must** provide the `episode_id` from the first response.
3. **Do not generate your own episode IDs** unless you use UUIDv7. If using Python, use `from tensorzero.util import uuid7` — do **not** use `pip install uuid7` (it's broken and causes `"Invalid Episode ID: Timestamp is in the future"` errors).
4. Episode IDs are returned in both streaming and non-streaming responses.

### 5.3 Feedback on episodes

You can provide feedback at the episode level using the `/feedback` endpoint:

```python
await t0.feedback(
    episode_id=episode_id,
    metric_name="task_success",
    value=True,
)
```

The metric must be configured with `level = "episode"` in your TOML.

> (source: `gateway-guides.md` > "Episodes", URL: https://www.tensorzero.com/docs/gateway/guides/episodes.md; `gateway-api-reference.md` > "API Reference: Feedback")

---

## 6. Parallel Tool Calls

### F3.5: N tool calls → N results in a SINGLE `Message(role="user")`

When the model returns multiple `tool_call` blocks in one response, you must:

1. Execute all N tools
2. Collect all N `ToolResult` objects
3. Send them as a **single** user message with a list of `tool_result` content blocks

```python
# Response has multiple tool_call blocks
tool_results = []
for block in response.content:
    if isinstance(block, ToolCall):
        result = await execute_tool(block.name, block.arguments)
        tool_results.append(ToolResult(
            name=block.name,
            result=str(result),
            id=block.id,
        ))

# ALL results go in ONE user message
if tool_results:
    messages.append(Message(role="user", content=tool_results))
```

The assistant message can also contain **mixed** content blocks (text + tool_calls):

```json
{
  "role": "assistant",
  "content": [
    { "type": "text", "text": "Sure, I can help you with that." },
    {
      "type": "tool_call",
      "id": "0",
      "name": "get_temperature",
      "arguments": "{\"location\": \"New York\"}"
    }
  ]
}
```

> (source: `gateway-api-reference.md` > "API Reference: Inference" > `input.messages` example)

### F3.8: Parallel execution with `asyncio.gather`

**[fast-mcp]** — If your MCP client supports concurrent tool calls, you can use:

```python
results = await asyncio.gather(*[
    execute_tool(block.name, block.arguments)
    for block in tool_call_blocks
])
```

The TensorZero MCP example notes:

> "NB: depending on the use case, you might want to await multiple tool calls in parallel"

> (source: `examples/mcp-model-context-protocol/main.py`)

---

## 7. Message History Format

### 7.1 Correct message sequence

The message history must follow this pattern:

```
user → assistant → user(tool_results) → assistant → user(tool_results) → ... → assistant(text only)
```

Rules from the documentation:
- Messages have `role`: `"user"` or `"assistant"`
- `tool_call` blocks go in **assistant** messages
- `tool_result` blocks go in **user** messages
- The `id` in `tool_result` must match the `id` from the corresponding `tool_call`
- The `name` in `tool_result` must match the tool name
- The `result` in `tool_result` **must be a string**

### 7.2 Full example of a multi-turn message history

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like in Tokyo?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_call",
          "id": "123456789",
          "name": "get_temperature",
          "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"
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
          "result": "25"
        }
      ]
    }
  ]
}
```

> (source: `gateway-api-reference.md` > "API Reference: Inference" > "Chat Function with Multi-Turn Tool Use" example, URL: https://www.tensorzero.com/docs/gateway/api-reference/inference.md)

---

## 8. Continuous Conversation

### F3.7: After a final response, the user can submit another query

The loop naturally supports this. After the LLM returns only text (no tool calls), you prompt the user for new input and continue with the same `messages` list and `episode_id`:

```python
while True:
    # If no tool results pending, get user input
    if not messages or messages[-1]["role"] == "assistant":
        message = input("\n[User]\n")
        messages.append(Message(role="user", content=message))

    response = await t0.inference(
        function_name="my_function",
        episode_id=episode_id,
        input={"messages": messages},
        additional_tools=t0_tools,
    )

    episode_id = response.episode_id
    messages.append(Message(role="assistant", content=response.content))

    tool_results = []
    for block in response.content:
        if isinstance(block, Text):
            print(block.text)
        elif isinstance(block, ToolCall):
            result = await execute_tool(block.name, block.arguments)
            tool_results.append(ToolResult(
                name=block.name,
                result=str(result),
                id=block.id,
            ))

    if tool_results:
        messages.append(Message(role="user", content=tool_results))
    # else: loop continues, user will be prompted for input
```

> (source: `examples/mcp-model-context-protocol/main.py` — this is the exact pattern used in the official MCP example)

---

## 9. Complete Reference Example (MCP)

This is the full agent loop from the official TensorZero MCP example:

```python
import asyncio
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult

async def main():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as t0:

        # [fast-mcp] Initialize your MCP session here
        # mcp_tools = await session.list_tools()

        # Convert MCP tools to TensorZero format
        t0_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            }
            for tool in mcp_tools
        ]

        messages = []
        episode_id = None

        while True:
            # Prompt user if no pending tool results
            if not messages or messages[-1]["role"] == "assistant":
                message = input("\n[User]\n")
                messages.append(Message(role="user", content=message))

            # Call TensorZero
            response = await t0.inference(
                function_name="my_copilot",
                episode_id=episode_id,
                input={"messages": messages},
                additional_tools=t0_tools,
            )

            # Capture episode_id from first response
            episode_id = response.episode_id

            # Append assistant response to history
            messages.append(Message(role="assistant", content=response.content))

            # Process content blocks
            tool_results = []
            for content_block in response.content:
                if isinstance(content_block, Text):
                    print(f"\n[Assistant]\n{content_block.text}")
                elif isinstance(content_block, ToolCall):
                    print(f"\n[Tool Call: {content_block.raw_name}]")
                    print(content_block.raw_arguments)

                    # [fast-mcp] Execute the tool via your MCP session
                    # mcp_result = await session.call_tool(
                    #     content_block.name, content_block.arguments
                    # )
                    # result_text = "\n".join(
                    #     [block.text for block in mcp_result.content]
                    # )

                    tool_results.append(ToolResult(
                        name=content_block.name,
                        result=result_text,
                        id=content_block.id,
                    ))

            # If there are tool results, add them as a single user message
            if tool_results:
                messages.append(Message(role="user", content=tool_results))
```

> (source: `examples/mcp-model-context-protocol/main.py`, URL: https://github.com/tensorzero/tensorzero/tree/main/examples/mcp-model-context-protocol)

### Required Python dependencies

```toml
# pyproject.toml
[project]
requires-python = ">=3.10"
dependencies = ["mcp>=1.6.0", "tensorzero"]
```

> (source: `examples/mcp-model-context-protocol/pyproject.toml`)

### Minimal TensorZero config for MCP

```toml
# config/tensorzero.toml
[functions.my_copilot]
type = "chat"

[functions.my_copilot.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini-2025-04-14"
system_template = "functions/my_copilot/baseline/system.txt"
```

Note: No `tools = [...]` in the function config — tools are provided dynamically via `additional_tools`.

> (source: `examples/mcp-model-context-protocol/config/tensorzero.toml`)

---

## 10. Checklist Mapping

| Item | Key TensorZero Detail | Where to Verify |
|------|----------------------|-----------------|
| **F3.1** Single-turn | Response has `tool_call` → execute → send `tool_result` → get text response | Section 4 |
| **F3.2** Multi-turn | Loop: check for `tool_call` blocks, execute, append, call again | Section 4, 8 |
| **F3.3** `episode_id` propagation | `episode_id = response.episode_id` on first call, pass to all subsequent | Section 5 |
| **F3.4** Termination | Loop ends when `response.content` has no `ToolCall` instances (only `Text`) | Section 4.7 |
| **F3.5** Parallel tool calls | All N `ToolResult` objects in a SINGLE `Message(role="user")` | Section 6 |
| **F3.6** Message history | `user → assistant(tool_calls) → user(tool_results) → ...` | Section 7 |
| **F3.7** Continuous conversation | Same `messages` list + `episode_id`, prompt user again after final text | Section 8 |
| **F3.8** `asyncio.gather` | Gather tool executions in parallel, collect results, same flow | Section 6 |

---

## Gotchas & Easy-to-Miss Details

1. **`tool_result.result` must be a string.** Not JSON, not an object — a string. If your tool returns structured data, serialize it first.

2. **`tool_call.arguments` in requests is a string** (JSON-serialized), but in responses it's an **object** (validated). When building the message history, the SDK handles this, but if building HTTP requests manually, be careful.

3. **Response `tool_call` has both `name`/`arguments` AND `raw_name`/`raw_arguments`.** Use `name` and `arguments` (validated) for execution. Use `raw_name` and `raw_arguments` for display/logging. If validation fails, `name` and `arguments` will be `null`.

4. **Don't generate your own episode IDs** unless using `from tensorzero.util import uuid7`. The external `uuid7` pip package is broken.

5. **Parallel tool calls are limited.** Only OpenAI and Fireworks AI support them. Other providers will ignore the `parallel_tool_calls` flag.

6. **The gateway does NOT store conversation history.** You must send the full message history with every `/inference` call. The gateway is stateless with respect to message context.

7. **When using `additional_tools`, `parameters` is the schema object itself**, not a file path. This differs from the TOML config where `parameters` is a path to a JSON file.

---

## Source Citations

| Section | File | Original URL |
|---------|------|-------------|
| Tool configuration | `gateway-guides.md` > "Tool Use (Function Calling)" | https://www.tensorzero.com/docs/gateway/guides/tool-use.md |
| Inference API | `gateway-api-reference.md` > "API Reference: Inference" | https://www.tensorzero.com/docs/gateway/api-reference/inference.md |
| Episodes | `gateway-guides.md` > "Episodes" | https://www.tensorzero.com/docs/gateway/guides/episodes.md |
| Config reference | `gateway-configuration-reference.md` | https://www.tensorzero.com/docs/gateway/configuration-reference/ |
| MCP example | `examples/mcp-model-context-protocol/main.py` | https://github.com/tensorzero/tensorzero/tree/main/examples/mcp-model-context-protocol |
| Feedback API | `gateway-api-reference.md` > "API Reference: Feedback" | https://www.tensorzero.com/docs/gateway/api-reference/feedback.md |
| Data model | `gateway-data-model.md` | https://www.tensorzero.com/docs/gateway/data-model/ |
