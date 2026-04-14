# Meta-Plan v3 (FINAL): Dependency Tree

> **Changes vs v2**: Added M0.5 (pytest config), corrected M5.1 to capture
> McpError in addition to ToolError, added M5.3b for McpError import, clarified
> M3.7 for optional parallel execution, adjusted estimates.

---

## Milestones

### M0: Base Infrastructure

```
M0.1  Configure T0 Gateway + ClickHouse (Docker Compose)
      Deliverable: docker-compose.yml + functional tensorzero.toml
      Verification: curl http://localhost:3000/health → 200

M0.2  Create test FastMCP server with 2+ tools
      Deliverable: server.py with @mcp.tool decorators
      Verification: instantiate FastMCP("TestServer") without error

M0.3  Verify in-memory Client → list_tools + call_tool
      Depends on: M0.2
      Deliverable: test_server.py that validates discovery and execution
      Verification: pytest passes

M0.4  Verify basic inference (without tools) against T0
      Depends on: M0.1
      Deliverable: test_t0.py with AsyncTensorZeroGateway.build_http()
      Verification: response.content[0] is Text with valid text

M0.5  Configure pytest with asyncio_mode = "auto"
      Deliverable: pyproject.toml with [tool.pytest.ini_options] asyncio_mode = "auto"
      Verification: async tests run without @pytest.mark.asyncio
```

**Dependencies**:
```
M0.1 (independent)
M0.2 (independent)
M0.5 (independent)
M0.2 → M0.3
M0.1 → M0.4
```

---

### M1: Tool Adapter

```
M1.1  Implement convert_tools(list[Tool]) → list[dict]
      Depends on: M0.3
      Deliverable: tool_adapter.py::convert_tools()
      Logic:
        [{"name": t.name, "description": t.description or "", "parameters": t.inputSchema}
         for t in tools]
      Note: description can be None in FastMCP if there is no docstring.
            The "or ''" is defensive.

M1.2  Unit test — tool conversion
      Depends on: M1.1, M0.5
      Deliverable: tests/test_tool_adapter.py
      Cases:
        - simple tool (str params)
        - tool with Pydantic model params
        - tool without description (None → "")
        - tool with complex schema (nested)

M1.3  Integration test — T0 accepts the tools
      Depends on: M0.4, M1.1
      Deliverable: tests/test_integration.py
      Verification: t0.inference(additional_tools=convert_tools(tools))
                    returns response without error
```

**Dependencies**:
```
M0.3 → M1.1
M1.1 + M0.5 → M1.2
M0.4 + M1.1 → M1.3
```

---

### M2: Result Adapter

```
M2.1  Implement serialize_tool_result(CallToolResult) → str
      Depends on: M0.3
      Deliverable: tool_adapter.py::serialize_tool_result()
      Logic:
        1. if is_error → extract error from content blocks
        2. if data is not None:
             if isinstance(data, str) → data
             else → json.dumps(data, default=str, ensure_ascii=False)
        3. fallback → concat content[*].text

M2.2  Implement build_tool_result(ToolCall, CallToolResult) → ToolResult
      Depends on: M2.1
      Deliverable: tool_adapter.py::build_tool_result()
      Logic:
        ToolResult(name=tc.name, result=serialize_tool_result(r), id=tc.id)

M2.3  Unit test — result serialization
      Depends on: M2.1, M0.5
      Cases:
        - data is str ("hello")
        - data is dict ({"count": 42})
        - data is list ([1, 2, 3])
        - data is int / float (auto-wrapped by FastMCP)
        - data is None (fallback to content)
        - data is datetime / UUID (via default=str)
        - is_error=True (error message)

M2.4  Integration test — ToolResult accepted by T0
      Depends on: M1.3, M2.2
      Verification: Complete single-turn flow with tool call + result
                    → T0 produces final response
```

**Dependencies**:
```
M0.3 → M2.1
M2.1 → M2.2
M2.1 + M0.5 → M2.3
M1.3 + M2.2 → M2.4
```

---

### M3: Agent Loop

```
M3.1  Implement execute_tool_calls(client, content_blocks) → list[ToolResult]
      Depends on: M2.2
      Deliverable: agent.py::execute_tool_calls()
      Logic:
        for each ToolCall in content:
          if arguments is None → ToolResult with error message (EC-2)
          else → try call_tool + build_tool_result
                 except (ToolError, McpError) → ToolResult with error (EC-5)
                 except Exception → ToolResult with unexpected error
        return list of ToolResult

M3.2  Implement single-turn agent loop
      Depends on: M3.1, M2.4
      Deliverable: agent.py::run_agent() (first version)
      Logic:
        1. list_tools → convert_tools → additional_tools
        2. inference with additional_tools
        3. if there are ToolCalls → execute → add results → inference again
        4. if only Text → return response

M3.3  Extend to multi-turn loop
      Depends on: M3.2
      Logic:
        while turn < max_turns:
          response = inference(messages, additional_tools, episode_id)
          episode_id = response.episode_id
          if there are ToolCalls → execute → push results → continue
          if only Text → break

M3.4  Implement episode_id management
      Depends on: M3.2
      Logic:
        - First call: episode_id=None
        - Capture response.episode_id
        - Propagate in all subsequent calls

M3.5  Implement correct message history
      Depends on: M3.3
      Logic:
        messages = []
        messages.append(Message(role="user", content=user_input))
        # After each response:
        messages.append(Message(role="assistant", content=response.content))
        if tool_results:
            messages.append(Message(role="user", content=tool_results))

M3.6  End-to-end test
      Depends on: M3.3, M3.4, M3.5
      Verification:
        - Complete conversation with 2+ tool calls
        - Consistent episode_id
        - Loop ends with Text response

M3.7  (Optional) Parallel execution of tool calls
      Depends on: M3.1
      Logic:
        import asyncio
        results = await asyncio.gather(*[
            execute_single_tool_call(client, block, timeout)
            for block in content_blocks
            if isinstance(block, ToolCall)
        ])
      Note: Only useful if there are multiple tool calls AND the tools are I/O-bound.
            Start with sequential execution (M3.1), parallelize later if needed.
```

**Dependencies**:
```
M2.2 → M3.1
M3.1 + M2.4 → M3.2
M3.2 → M3.3
M3.2 → M3.4
M3.3 → M3.5
M3.3 + M3.4 + M3.5 → M3.6
M3.1 → M3.7
```

---

### M4: Multi-Server (native feature validation)

```
M4.1  Create 2 test FastMCP servers with different tools
      Depends on: M0.2
      Deliverable: server_a.py, server_b.py

M4.2  Compose with mount() + namespacing
      Depends on: M4.1
      Deliverable: composed_server.py
      Logic:
        main = FastMCP("Main")
        main.mount(server_a, namespace="a")
        main.mount(server_b, namespace="b")

M4.3  Verify that list_tools returns all tools with prefixes
      Depends on: M4.2
      Verification: tool names = ["a_tool1", "a_tool2", "b_tool3", "b_tool4"]

M4.4  End-to-end test with agent loop
      Depends on: M3.6, M4.3
      Verification: The LLM can invoke tools from both servers
                    in the same conversation
```

**Dependencies**:
```
M0.2 → M4.1
M4.1 → M4.2
M4.2 → M4.3
M3.6 + M4.3 → M4.4
```

---

### M5: Robustness

```
M5.1  Handle ToolError AND McpError
      Depends on: M3.1
      Logic:
        from fastmcp.exceptions import ToolError
        from mcp import McpError

        try:
            result = await client.call_tool(name, args, timeout=timeout)
        except ToolError as e:
            result_str = f"Error: Tool execution failed: {e}"
        except McpError as e:
            result_str = f"Error: MCP protocol error (possibly timeout): {e}"
        except Exception as e:
            result_str = f"Unexpected error: {type(e).__name__}: {e}"
        → ToolResult(name=..., result=result_str, id=...)

      CORRECTION vs v2: McpError is raised on timeout and protocol errors,
      NOT ToolError. Capture both explicitly.

M5.2  Handle arguments=null (invalid JSON from LLM)
      Depends on: M3.1
      Logic:
        if block.arguments is None:
            result_str = f"Error: invalid JSON arguments: {block.raw_arguments}"
        → ToolResult with error message

M5.3  Implement timeout in call_tool
      Depends on: M5.1
      Logic:
        result = await client.call_tool(name, args, timeout=30.0)
      Note: timeout accepts float | int | datetime.timedelta
      Note: timeout in FastMCP only applies to foreground execution.
            Does not apply to background tasks.
      (source: servers-tools.md > "Timeouts")

M5.4  Implement turn limit (anti-infinite-loop)
      Depends on: M3.3
      Logic:
        MAX_TURNS = 20
        for turn in range(MAX_TURNS):
            response = ...
            if only Text → break
        else:
            raise MaxTurnsExceeded

M5.5  Robustness test
      Depends on: M5.1, M5.2, M5.3, M5.4
      Cases:
        - tool that always fails → ToolError captured
        - slow tool (simulate timeout) → McpError captured
        - LLM that generates invalid JSON → arguments=null handled
        - LLM that never stops calling tools → max turns respected
        - non-existent tool → ToolError captured
```

**Dependencies**:
```
M3.1 → M5.1
M3.1 → M5.2
M5.1 → M5.3
M3.3 → M5.4
M5.1 + M5.2 + M5.3 + M5.4 → M5.5
```

---

### M6: Observability

```
M6.1  Capture inference_id + episode_id on each loop turn
      Depends on: M3.4
      Deliverable: agent.py returns AgentResult with traceable ids

M6.2  Implement feedback loop
      Depends on: M6.1
      Logic:
        await t0.feedback(
            episode_id=episode_id,
            metric_name="task_success",
            value=True,
        )

M6.3  Structured logging
      Depends on: M3.6
      Deliverable: Logs with: [TOOL_CALL name args] [TOOL_RESULT data] [ERROR msg]

M6.4  Verify traceability in ClickHouse
      Depends on: M6.1, M6.2, M6.3
      Verification:
        - ChatInference has dynamic_tools with the sent tools
        - ModelInference has input/output for each turn
        - Feedback is linked to the correct episode_id
```

**Dependencies**:
```
M3.4 → M6.1
M6.1 → M6.2
M3.6 → M6.3
M6.1 + M6.2 + M6.3 → M6.4
```

---

## Critical Path

```
M0.1 + M0.2 → M0.3 → M1.1 → M1.3 → M2.4 → M3.1 → M3.2 → M3.3 → M3.6
```

**Estimate**: If M0 is ready, the critical path consists of ~8 implementation
tasks to have the agent loop functional (M3.6).

## Possible Parallelism

```
Track A (main):       M0 → M1 → M2 → M3 → M6
Track B (parallel):   M0.2 → M4.1 → M4.2 → M4.3 (waiting for M3.6 for M4.4)
Track C (parallel):   M3.1 → M5.1 → M5.2 → M5.3 (waiting for M3.3 for M5.4)
Track D (early):      M0.5 (pytest config, independent)
```

---

## Resolved Questions

| # | Question | Answer | Source |
|---|---|---|---|
| 1 | Does JSON Schema with `$defs` inlined cause problems in T0? | No. T0 passes the schema as-is to the provider. FastMCP dereferences `$ref` by default. | gateway-api-reference.md, servers-tools.md |
| 2 | Must `ToolResult.result` be a string? | Yes. Use `json.dumps(data, default=str)`. | gateway-api-reference.md |
| 3 | Do parallel tool results go in one message or several? | ONE single `Message(role="user", content=[...all_results])`. | examples/mcp-model-context-protocol/main.py |
| 4 | What exception does a timeout raise? | `McpError` (code -32000), NOT `ToolError`. | python-sdk-fastmcp-client-mixins-tools.md, servers-tools.md |
| 5 | Where is McpError imported from? | `from mcp import McpError` | servers-middleware.md (usage example) |

## Remaining Open Questions

None blocking. One scenario to validate empirically:

- **EC-1**: Do highly complex schemas (3+ levels of Pydantic nesting) cause
  problems in providers like Anthropic or OpenAI? → Validate in F1.5.
