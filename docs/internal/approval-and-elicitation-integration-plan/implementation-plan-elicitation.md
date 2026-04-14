# Implementation Plan: Elicitation Support in t0-agents

> **Date:** 2026-04-14
> **Companion:** `integration-contract-elicitation-final.md`
> **Goal:** Enable human-in-the-loop approval and structured input via MCP Elicitation.

---

## Overview

t0-agents needs **no production code changes** to support elicitation. The FastMCP Client already handles elicitation transparently — the `call_tool()` await simply takes longer when a tool pauses for human input. The existing error contract in `_execute_single_tool_call` already catches the `McpError` raised when no handler is present.

The implementation work is:
1. **Validate** the integration empirically (smoke tests)
2. **Build a test suite** proving all elicitation patterns work through the agent loop
3. **Document** the patterns for library consumers
4. **Provide examples** showing real usage

---

## Phase 0: Smoke Test & Validation

**Goal:** Confirm empirically that elicitation works end-to-end with the in-memory test infrastructure.

**Milestone:** A passing smoke test that proves `Client(mcp, elicitation_handler=handler)` + `call_tool()` works with a tool that calls `ctx.elicit()`.

### Checklist

- [ ] **P0.1** Add an elicitation-enabled tool to `tests/server.py`:
  ```python
  @mcp.tool
  async def approve_action(action: str, ctx: Context) -> str:
      """Request approval before performing an action."""
      result = await ctx.elicit(
          f"Approve: {action}?", response_type=None
      )
      if result.action == "accept":
          return f"Approved: {action}"
      return f"Rejected: {action}"
  ```

- [ ] **P0.2** Add a structured-input elicitation tool to `tests/server.py`:
  ```python
  @dataclass
  class UserInfo:
      name: str
      age: int

  @mcp.tool
  async def collect_info(ctx: Context) -> str:
      """Collect user information via elicitation."""
      result = await ctx.elicit(
          "Please provide your information",
          response_type=UserInfo,
      )
      if result.action == "accept":
          return f"Hello {result.data.name}, age {result.data.age}"
      return "Info not provided"
  ```

- [ ] **P0.3** Write a smoke test in `tests/test_mcp/test_elicitation.py`:
  ```python
  async def test_smoke_elicitation_approve():
      async def handler(message, response_type, params, context):
          return ElicitResult(action="accept", content={})

      async with Client(mcp, elicitation_handler=handler) as client:
          result = await client.call_tool("approve_action", {"action": "deploy"})
          assert result.data == "Approved: deploy"
  ```

- [ ] **P0.4** Write a smoke test for the no-handler error path:
  ```python
  async def test_smoke_no_handler_raises_mcp_error():
      async with Client(mcp) as client:  # no handler
          with pytest.raises(McpError):
              await client.call_tool("approve_action", {"action": "deploy"})
  ```

- [ ] **P0.5** Run smoke tests: `uv run pytest tests/test_mcp/test_elicitation.py -v`

### Files Modified
- `tests/server.py` — Add elicitation tools
- `tests/test_mcp/test_elicitation.py` — New file

### Exit Criteria
All 4 smoke tests pass. If any fail, investigate before proceeding — the assumptions from the contract need revision.

---

## Phase 1: Comprehensive Test Suite

**Goal:** Full coverage of elicitation patterns through the agent loop.

**Milestone:** A test suite covering all elicitation patterns (approval, scalar, structured, multi-turn, decline, cancel, error paths) that passes with `uv run pytest -m "not integration"`.

### Checklist

#### Server-side test tools (add to `tests/server.py`)

- [ ] **P1.1** Multi-turn elicitation tool:
  ```python
  @mcp.tool
  async def multi_step(ctx: Context) -> str:
      r1 = await ctx.elicit("Step 1: name?", response_type=str)
      if r1.action != "accept":
          return "Cancelled at step 1"
      r2 = await ctx.elicit("Step 2: confirm?", response_type=None)
      if r2.action != "accept":
          return "Rejected at step 2"
      return f"Completed: {r1.data}"
  ```

- [ ] **P1.2** Constrained-choice elicitation tool:
  ```python
  @mcp.tool
  async def pick_priority(ctx: Context) -> str:
      result = await ctx.elicit(
          "Priority?", response_type=["low", "medium", "high"]
      )
      if result.action == "accept":
          return f"Priority: {result.data}"
      return "No priority set"
  ```

#### Unit tests — elicitation patterns (`tests/test_mcp/test_elicitation.py`)

- [ ] **P1.3** Approval — accept: handler returns `accept` with `content={}`, tool proceeds.
- [ ] **P1.4** Approval — decline: handler returns `decline`, tool handles gracefully.
- [ ] **P1.5** Approval — cancel: handler returns `cancel`, tool handles gracefully.
- [ ] **P1.6** Scalar input (str): handler returns a string value, tool receives it.
- [ ] **P1.7** Scalar input (int): handler returns an integer, tool receives it.
- [ ] **P1.8** Structured input (dataclass): handler returns populated dataclass, tool receives it.
- [ ] **P1.9** Constrained choice: handler returns one of the allowed options.
- [ ] **P1.10** Multi-turn: handler responds to two sequential elicitation requests.
- [ ] **P1.11** Multi-turn with cancellation: first elicitation accepted, second declined.

#### Error path tests

- [ ] **P1.12** No handler — `McpError(-32600)` raised, message contains "Elicitation not supported".
- [ ] **P1.13** No handler through `_execute_single_tool_call` — returns `ToolResult` with `"ERROR: MCP protocol error"`.
- [ ] **P1.14** Timeout with elicitation — tool with elicitation + short `call_tool(timeout=0.1)` → `McpError` timeout, to confirm the timeout-covers-elicitation behavior.

#### Integration with `run_agent` loop

- [ ] **P1.15** Full loop: TensorZero inference → LLM calls elicitation-enabled tool → handler approves → tool returns → LLM produces final text.
  - This requires `@pytest.mark.integration` (needs gateway running).
  - Uses a tool that the LLM can be directed to call via `tool_choice`.

- [ ] **P1.16** Full loop with decline: same as above but handler declines → LLM receives rejection string → LLM produces final text explaining the rejection.

#### Fixture for reusable handlers (`tests/conftest.py`)

- [ ] **P1.17** Add reusable handler fixtures:
  ```python
  @pytest.fixture
  def auto_approve_handler():
      async def handler(message, response_type, params, context):
          if response_type is None:
              return ElicitResult(action="accept", content={})
          return ElicitResult(action="decline")
      return handler

  @pytest.fixture
  def auto_decline_handler():
      async def handler(message, response_type, params, context):
          return ElicitResult(action="decline")
      return handler

  @pytest.fixture
  async def elicitation_client(auto_approve_handler):
      return Client(mcp, elicitation_handler=auto_approve_handler)
  ```

### Files Modified
- `tests/server.py` — Add multi-turn and constrained-choice tools
- `tests/conftest.py` — Add handler fixtures
- `tests/test_mcp/test_elicitation.py` — All new tests

### Exit Criteria
`uv run pytest tests/test_mcp/test_elicitation.py -v` — all tests pass. Integration tests (P1.15, P1.16) pass with `docker compose up`.

---

## Phase 2: Documentation

**Goal:** Library consumers know how to use elicitation with t0-agents.

**Milestone:** A complete elicitation guide and updated API reference.

### Checklist

- [ ] **P2.1** Create `docs/elicitation.md` covering:
  - What elicitation is and why it matters (human-in-the-loop)
  - How it works (transparent to the agent loop — the flow diagram)
  - The `elicitation_handler` interface (signature, parameters, response contract)
  - How to distinguish approval from input (`response_type is None`)
  - Timeout guidance (`tool_timeout=None` or generous value)
  - Server-side patterns (approval gate, structured input, multi-turn)
  - Constraints (flat schemas, stateful session required)

- [ ] **P2.2** Update `docs/api-reference.md`:
  - Add timeout guidance to `run_agent`'s `tool_timeout` parameter docs
  - Note that `tool_timeout` covers elicitation wait time
  - Recommend `tool_timeout=None` when elicitation-enabled tools are in use

- [ ] **P2.3** Update `docs/production.md`:
  - Add a section on elicitation under "Transport Selection" — note that stateless HTTP mode breaks elicitation
  - Add timeout advice for elicitation scenarios

- [ ] **P2.4** Update `docs/error-handling.md`:
  - Document the `McpError(-32600, "Elicitation not supported")` error
  - Explain when it occurs (tool calls `ctx.elicit()` but Client has no handler)
  - Note that the existing error contract handles it automatically

### Files Modified
- `docs/elicitation.md` — New file
- `docs/api-reference.md` — Update `tool_timeout` docs
- `docs/production.md` — Add elicitation transport guidance
- `docs/error-handling.md` — Add no-handler error docs

### Exit Criteria
Documentation is self-contained — a consumer can read `docs/elicitation.md` and implement elicitation without reading FastMCP docs.

---

## Phase 3: Example

**Goal:** A runnable example demonstrating elicitation with the full agent loop.

**Milestone:** `examples/07_elicitation/` with a working example and README.

### Checklist

- [ ] **P3.1** Create `examples/07_elicitation/server.py`:
  - A FastMCP server with 2-3 tools, at least one using `ctx.elicit()` for approval and one for structured input.
  - Example: a "file manager" server where `delete_files` requires approval and `create_task` collects structured input.

- [ ] **P3.2** Create `examples/07_elicitation/main.py`:
  - Shows the full pattern: create Client with `elicitation_handler`, connect to TensorZero, run the agent loop.
  - Implements a simple CLI-based handler (reads from stdin).

- [ ] **P3.3** Create `examples/07_elicitation/README.md`:
  - What it demonstrates
  - Prerequisites (docker-compose for gateway)
  - How to run
  - What to observe (the handler prompts, the approval flow)

### Files Created
- `examples/07_elicitation/server.py`
- `examples/07_elicitation/main.py`
- `examples/07_elicitation/README.md`

### Exit Criteria
Running the example with `docker compose up` + `uv run python examples/07_elicitation/main.py` produces an interactive session where the user approves/rejects tool actions.

---

## Phase 4: Optional Enhancements

**Goal:** Quality-of-life utilities for common patterns.

**Milestone:** Reference handlers available for import. This phase is optional and should only be implemented if the patterns prove useful during Phase 1-3.

### Checklist

- [ ] **P4.1** Evaluate whether reference handlers belong in the library or in examples:
  - If multiple examples/tests repeat the same handler pattern → put in library
  - If each use case is different → keep in examples only

- [ ] **P4.2** If warranted, create `src/t0_agents/mcp/elicitation.py`:
  ```python
  async def auto_approve_handler(message, response_type, params, context):
      """Auto-approve all elicitation requests. For testing only."""
      if response_type is None:
          return ElicitResult(action="accept", content={})
      return ElicitResult(action="decline")

  async def auto_decline_handler(message, response_type, params, context):
      """Auto-decline all elicitation requests. For testing only."""
      return ElicitResult(action="decline")
  ```

- [ ] **P4.3** If P4.2 is implemented, export from `t0_agents.mcp`:
  - Add to `__init__.py`
  - Add to `docs/api-reference.md`
  - Add mypy types

- [ ] **P4.4** Consider a `require_approval()` server-side helper:
  - This is a server-side pattern, not a t0-agents library concern
  - Document it in `docs/elicitation.md` as a recommended pattern
  - Do NOT add it to the library — it belongs in the user's MCP server code

### Files Modified (if P4.2 is implemented)
- `src/t0_agents/mcp/elicitation.py` — New file
- `src/t0_agents/mcp/__init__.py` — Export new functions
- `docs/api-reference.md` — Document new functions

### Exit Criteria
`uv run mypy src/` and `uv run ruff check src/` pass. If P4.2 is implemented, tests cover the handlers.

---

## Dependency Tree

```
Phase 0: Smoke Test
|-- P0.1: Add approval tool to tests/server.py
|-- P0.2: Add structured-input tool to tests/server.py
|-- P0.3: Smoke test (approve)                         [depends: P0.1]
|-- P0.4: Smoke test (no handler)                      [depends: P0.1]
+-- P0.5: Run smoke tests                              [depends: P0.3, P0.4]
     |
     v
Phase 1: Test Suite                                     [depends: P0.5]
|-- P1.1-P1.2: Add more test tools                     [depends: P0.5]
|-- P1.3-P1.11: Pattern unit tests                     [depends: P1.1, P1.2]
|-- P1.12-P1.14: Error path tests                      [depends: P0.5]
|-- P1.15-P1.16: Integration tests                     [depends: P1.3]
+-- P1.17: Handler fixtures                            [parallel with P1.3]
     |
     v
Phase 2: Documentation                                 [depends: P1 complete]
|-- P2.1: docs/elicitation.md
|-- P2.2: Update api-reference.md
|-- P2.3: Update production.md
+-- P2.4: Update error-handling.md
     |
     v
Phase 3: Example                                       [depends: P2.1]
|-- P3.1: Server with elicitation tools
|-- P3.2: Main script with CLI handler
+-- P3.3: README
     |
     v
Phase 4: Optional Enhancements                         [depends: P1, P3]
|-- P4.1: Evaluate library vs examples
|-- P4.2: Reference handlers (if warranted)
|-- P4.3: Exports and types
+-- P4.4: Document server-side patterns
```

### Critical Path

```
P0.1 --> P0.3 --> P0.5 --> P1.3 --> P1.15 --> P2.1 --> P3.2
```

Estimated effort: Phases 0-3 are the core deliverable. Phase 4 is optional.

---

## Verification Checklist (End-to-End)

After all phases:

- [ ] `uv run pytest -m "not integration"` — all tests pass (including new elicitation tests)
- [ ] `uv run pytest -m integration` — integration tests pass (with `docker compose up`)
- [ ] `uv run mypy src/` — strict mode passes
- [ ] `uv run ruff check src/` — no lint errors
- [ ] `docs/elicitation.md` is self-contained and accurate
- [ ] `examples/07_elicitation/` runs successfully with an interactive CLI session
- [ ] No changes to `run_agent` signature, adapter, or core types
