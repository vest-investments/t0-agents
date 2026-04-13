# Validation Checklist v3 (FINAL): TensorZero + FastMCP

> **Changes vs v2**: Added F5.5b for McpError (timeout), adjusted F5.7 for
> description None, added F0.6 to verify pytest config, added F3.8
> for optional parallel execution.

Each item is binary (✅ passes / ❌ does not pass). Complete all items in
a phase before moving on to the next.

---

## Phase 0: Infrastructure

- [ ] **P0.1** — T0 Gateway starts with Docker Compose (`docker compose up`)
- [ ] **P0.2** — Basic inference without tools works against T0 (`t0.inference(function_name=..., input={...})`)
- [ ] **P0.3** — Test FastMCP server is instantiated with at least 2 tools using `@mcp.tool`
- [ ] **P0.4** — FastMCP `Client(mcp_instance)` in-memory connects and `list_tools()` returns the tools
- [ ] **P0.5** — `call_tool()` in-memory executes a tool and returns `CallToolResult` with `.data`
- [ ] **P0.6** — pytest configured with `asyncio_mode = "auto"` in `pyproject.toml`

### Success criterion
You can run inference against T0 AND execute FastMCP tools independently.
Both systems work separately.

---

## Phase 1: Tools Adapter (FastMCP → T0)

- [ ] **F1.1** — `list_tools()` returns objects with valid `.name`, `.description`, `.inputSchema`
- [ ] **F1.2** — The adapter converts `list[mcp.types.Tool]` → `list[dict]` with keys `name`, `description`, `parameters`
- [ ] **F1.3** — T0 accepts the converted `additional_tools` without error (successful inference with tools available)
- [ ] **F1.4** — The LLM "sees" the tools: with `tool_choice="required"`, it always returns a `ToolCall`
- [ ] **F1.5** — Tools with Pydantic models as parameters generate schemas accepted by T0
- [ ] **F1.6** — Tools with `description=None` are converted with `""` without error in T0

### Success criterion
T0 receives the FastMCP tools and the LLM can invoke them.

---

## Phase 2: Results Adapter (FastMCP → T0)

- [x] **F2.1** — `call_tool(name, arguments)` executes correctly with the `arguments` from T0's ToolCall
- [x] **F2.2** — `serialize_tool_result()` converts `.data: str` → `str` without modification
- [x] **F2.3** — `serialize_tool_result()` converts `.data: dict` → valid `json.dumps()`
- [x] **F2.4** — `serialize_tool_result()` converts `.data: None` → fallback to `.content[*].text`
- [x] **F2.5** — `serialize_tool_result()` handles complex types (datetime, UUID) via `default=str`
- [x] **F2.6** — `serialize_tool_result()` handles `.is_error=True` (result with `raise_on_error=False`)
- [x] **F2.7** — The constructed `ToolResult` is accepted by T0 in the next inference turn
- [x] **F2.8** — The LLM produces a coherent response that incorporates the tool result

### Success criterion
A complete flow: T0 requests tool → FastMCP executes → result returns to T0 →
LLM responds to the user.

---

## Phase 3: Agent Loop

- [x] **F3.1** — Single-turn: user → LLM → 1 tool call → 1 result → final response
- [x] **F3.2** — Multi-turn: user → LLM → tool → result → LLM → tool → result → final response
- [x] **F3.3** — `episode_id` is captured from the first response and propagated to all subsequent turns
- [x] **F3.4** — The loop terminates correctly when the LLM responds with only `Text` (no tool calls)
- [x] **F3.5** — Parallel tool calls: N tool calls in one response → N results in a SINGLE `Message(role="user")`
- [x] **F3.6** — The message history is built correctly (user → assistant → user(tools) → ...)
- [x] **F3.7** — Continuous conversation: after a final response, the user can submit another query
- [x] **F3.8** — (Optional) Parallel tool execution with `asyncio.gather` does not break the flow

### Success criterion
A conversational agent that can use tools across multiple turns,
maintaining the episode context.

---

## Phase 4: Multi-Server

- [x] **F4.1** — Option A (mount): `main.mount(server_a, namespace="a")` + `main.mount(server_b, namespace="b")` works — `TestMountedNamespacing::test_list_tools_returns_prefixed_names`
- [x] **F4.2** — The combined tools have a namespace prefix (`a_tool1`, `b_tool2`) — same test + `test_convert_tools_passes_namespaced_names_through`
- [x] **F4.3** — T0 receives all combined tools and the LLM can invoke any of them — `test_run_agent_mounted_servers_forced_add` / `_forced_subtract` / `_natural_selection`
- [x] **F4.4** — ToolCalls use the namespaced name and FastMCP routes them to the correct server — integration tests above + `TestRunAgentOverComposedServer` unit tests + `test_call_tool_routes_to_correct_server_via_prefix`
- [x] **F4.5** — Option B (Client config dict): `Client({"mcpServers": {...}})` discovers tools from N servers — **validated manually** via the snippet in `docs/phase-4/phase-4-implementation-log.md` (interview decision, Phase 1–3 precedent)
- [x] **F4.6** — There are no name collisions between tools from different servers — `test_call_tool_routes_to_correct_server_via_prefix` (pins `a_greet` vs `b_greet`) + `test_run_agent_surfaces_colliding_greet_from_correct_namespace`

### Success criterion
The agent can use tools from multiple sources in the same conversation.

### Note
FastMCP natively resolves namespacing and routing. Phase 4 is primarily
validation that the native behavior works end-to-end with T0.

---

## Phase 5: Robustness and Error Handling

- [x] **F5.1** — `ToolError` from FastMCP is caught and sent as a ToolResult with an error message to the LLM — pinned by `TestRobustness::test_f5_1_tool_error_caught` (unit) + `test_f5_1_run_agent_survives_failing_tool` (integration, via the `fail_always` fixture in `tests/server.py`)
- [x] **F5.2** — `McpError` from mcp (timeout, protocol) is caught and sent as a ToolResult with error to the LLM — pinned by `TestRobustness::test_f5_2_mcp_error_caught` (unit, non-timeout code `-32603`)
- [x] **F5.3** — `block.arguments is None` (invalid JSON from the LLM) is detected and reported as an error to the LLM — pinned by `TestRobustness::test_f5_3_invalid_json_detected` (unit)
- [x] **F5.4** — Timeout in `call_tool(..., timeout=N)` generates `McpError` that is caught correctly — pinned by `TestRobustness::test_f5_4_timeout_surfaces_as_mcperror` (unit, code `-32000`) + `test_f5_4_tool_timeout_does_not_crash_run_agent` (integration, via the `slow_tool` fixture + `tool_timeout=0.5`)
- [x] **F5.5** — The LLM can retry a tool after receiving an error — pinned by `TestRobustness::test_f5_5_loop_continues_after_error_and_llm_retries` (unit: strict 3-turn fail→retry→text scenario) + `test_f5_5_run_agent_allows_llm_retry` (integration, loose per interview decision)
- [x] **F5.6** — `ValidationError` from FastMCP (invalid schema parameters) is handled without a crash — pinned by `TestRobustness::test_f5_6_validation_error_surfaces_through_tool_error` (unit: FastMCP pre-converts pydantic `ValidationError` to `ToolError` before the client, so our existing F5.1 catch handles it)
- [x] **F5.7** — Maximum turn limit for the loop (prevents infinite loop if the LLM always requests tools) — pinned by `TestRobustness::test_f5_7_max_turns_raises_with_correct_limit` (unit) + `test_f5_7_max_turns_raises_against_live_gateway` (integration, `max_turns=1` forces the raise)
- [x] **F5.8** — Tools with empty or `None` `description` do not cause an error in T0 — pinned by `TestConvertTools::test_convert_tools_none_description_on_synthesized_tool` + `::test_convert_tools_empty_description_stays_empty` (unit, adapter-level) + `TestRobustness::test_f5_8_none_description_round_trip` (unit, end-to-end through `run_agent`) + `test_f5_8_none_description_tool_is_usable_end_to_end` (integration, live call on `echo_no_docs`)

### Success criterion
The agent does not crash under adverse conditions and communicates errors to the LLM
so it can recover.

---

## Phase 6: Observability and Feedback

- [ ] **F6.1** — `inference_id` and `episode_id` are captured from each T0 response
- [ ] **F6.2** — Boolean feedback can be sent to the `episode_id` (metric defined in tensorzero.toml)
- [ ] **F6.3** — Float feedback can be sent to the `inference_id`
- [ ] **F6.4** — A `comment` can be sent as feedback on an inference
- [ ] **F6.5** — A `demonstration` can be sent as feedback (ideal output)
- [ ] **F6.6** — Dynamic tools are recorded in ClickHouse (`dynamic_tools` column)
- [ ] **F6.7** — The agent loop log shows: tool calls, results, errors, timestamps

### Success criterion
You can trace a complete conversation in ClickHouse and send feedback
for future optimization.

---

## Progress summary

| Phase | Items | Completed | Status |
|---|---|---|---|
| 0 — Infrastructure | 6 | _ / 6 | ⬜ |
| 1 — Tools Adapter | 6 | _ / 6 | ⬜ |
| 2 — Results Adapter | 8 | 8 / 8 | ✅ |
| 3 — Agent Loop | 8 | 8 / 8 | ✅ |
| 4 — Multi-Server | 6 | 6 / 6 | ✅ |
| 5 — Robustness | 8 | 8 / 8 | ✅ |
| 6 — Observability | 7 | _ / 7 | ⬜ |
| **Total** | **49** | **_ / 49** | ⬜ |
