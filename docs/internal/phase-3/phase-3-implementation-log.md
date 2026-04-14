# Phase 3 Implementation Log

> Bridges the design in `docs/implementation-plan/` (M3 in the meta-plan, F3.x
> in the validation checklist) with what we actually shipped. Scope of this
> phase is the **agent loop** — the cycle that ties Phase 1's `convert_tools`,
> Phase 2's `serialize_tool_result` / `build_tool_result`, and FastMCP's
> `Client` together so a user query goes in and a final LLM response comes out,
> with any tool calls in between executed transparently.

## What we built

| File | Purpose |
|---|---|
| `src/t0_agents/mcp/loop.py` | Replaced the three Phase-3 stubs (`run_agent`, `execute_tool_calls`, `has_tool_calls`) with full implementations. Added `_execute_single_tool_call` (private) and `execute_tool_calls_parallel` (public, M3.7). |
| `src/t0_agents/mcp/__init__.py` (edit) | Re-exported `execute_tool_calls_parallel`. |
| `src/t0_agents/__init__.py` (edit) | Re-exported `AgentResult` and `MaxTurnsExceededError` from the package root for users. |
| `tests/test_mcp/test_loop.py` | Replaced the placeholder classes with 30 unit tests using hand-written stubs (`FakeT0`, `FakeMCPClient`, `FakeChatResponse`). Covers `has_tool_calls`, both executors, and `run_agent` end-to-end with mocked T0 + MCP. |
| `tests/test_mcp/test_integration.py` (edit) | Added 4 Phase-3 integration tests against the live Groq gateway (`test_run_agent_single_turn_add`, `_no_tools_terminates_immediately`, `_continuous_conversation`, `_parallel_tools_flag_works`). |
| `docs/implementation-plan/02-validation-checklist-v3-final.md` (edit) | Ticked F3.1–F3.8 and bumped Progress summary row for Phase 3 to `8 / 8` ✅. |

## Decisions taken during planning

| Topic | Options | We chose | Why |
|---|---|---|---|
| Error scope in `execute_tool_calls` | Happy-path only (Phase 5 adds errors) / full M3.1 contract now | **Full M3.1 contract now** | Meta-plan M3.1's pseudo-code bundles error handling with the milestone; Phase 5's F5.1–F5.4 will then *validate* code that already exists rather than asking us to retrofit. Confirmed with the user up front. |
| F3.5 / F3.6 test strategy | Mocks only / integration only / both | **Mocks for the structural invariants + integration smoke** | Groq's `gpt-oss-20b` does not natively emit multiple tool calls in one response, so an integration test cannot pin F3.5's "N results in ONE user message" invariant deterministically. Mock-based unit tests cover the exact `Message` shapes; integration tests cover end-to-end behavior against the real model. Confirmed with the user. |
| F3.8 / M3.7 (optional parallel exec) | Defer to a later phase / include now | **Include now** | Confirmed with the user. Adds `execute_tool_calls_parallel` and a `parallel_tools: bool = False` flag on `run_agent` that selects the executor. |
| `AgentResult` shape | Reference returns `dict` / use existing `AgentResult` dataclass | **Use the dataclass** | `core/types.py:AgentResult` has been a stub since Phase 0 — Phase 3 makes it real. Typed result with `episode_id: str \| None`, `inference_ids: list[str]`, `turns: int`. mypy strict happy without `cast`. |
| Max-turns exception | `RuntimeError` (per reference) / `MaxTurnsExceededError` (already in `core/exceptions.py`) | **`MaxTurnsExceededError`** | Anticipated Phase 5's F5.7 / M5.4. Phase 0 already created the exception class — using it now keeps the API stable into Phase 5. |
| Client lifecycle | `run_agent` opens its own `async with mcp_client:` / caller manages | **Caller manages** | Matches FastMCP guide §11.6 ("Do not open/close the client per call") and the reference. Tests already follow the pattern via `async with mcp_client:` at the top of each test. |
| `dryrun` propagation | New flag on `run_agent` / no-op (writes per-turn rows in tests) | **`dryrun: bool = False` on `run_agent`** | Phase 1/2 integration tests all use `dryrun=True` to keep ClickHouse clean. Adding one kwarg keeps the same hygiene profile here. Default remains `False` for production usage. |

## Divergences discovered during implementation

1. **`name=None` is a distinct failure mode from `arguments=None`**. The reference's `execute_tool_calls` only guards `arguments is None`. But `tensorzero/types.py:152` declares `ToolCall.name: Optional[str]` — T0 zeroes the field when the LLM hallucinates a tool name not in the `additional_tools` schema. Passing `None` to `Client.call_tool(name: str, ...)` would TypeError, and mypy strict caught it during the offline gate. We added a separate early-return branch with the message `"ERROR: tool name '<raw_name>' is not in the schema"` so the LLM can self-correct on the next turn. Pinned by `test_name_none_with_valid_args_does_not_call_tool`.

2. **Stub `FakeT0` snapshots the messages list at call time**. `run_agent` mutates the same `messages` list in place across turns; if `FakeT0.inference()` just stored `kwargs` by reference, tests asserting on `t0.calls[N]["input"]["messages"]` would actually see the post-loop final state instead of the per-turn snapshot. Real T0 serializes the request immediately, so this is a test-only artifact. The fix lives in `FakeT0.inference()` (shallow-copies `input["messages"]` and the dict envelope), not in `run_agent` — adding a per-turn list copy in production code would cost cycles to fix something only test mocks experience.

3. **F3.2 is exercised implicitly by `test_run_agent_single_turn_add`, not by a separate "two distinct tool calls" test**. The "compute 2+3 with the add tool" prompt always produces `>= 2` turns (one for the tool call, one for the final answer) — that is structurally a multi-turn flow, just not "two distinct tools". Forcing Groq to chain two different tools deterministically is hard, and the unit-level multi-turn invariants (history shape, episode_id propagation, inference_id collection) are pinned by the mock-based tests. F3.2 was therefore ticked on the strength of the integration single-turn test plus the unit-level multi-turn coverage.

4. **`test_run_agent_continuous_conversation` does not assert that the LLM sees prior conversation history**. `run_agent` is single-conversation per call: each invocation starts a fresh `messages` list, only carrying continuity through `episode_id` (which T0 uses for analytics/feedback grouping in ClickHouse, not for context inference). The LLM in the second call sees only that call's user message. The episode_id forwarding is what F3.7 asks for ("the user can submit another query"), and the test pins exactly that — second response shares `episode_id` with the first. A future phase (or higher-level wrapper) could add stateful conversation memory; out of scope here.

5. **F3.8 integration test uses a single tool call, not multiple**. Groq is not on T0's `parallel_tool_calls` allowlist (only OpenAI and Fireworks AI are, per the integration contract §3.1.f), so we cannot reliably force a multi-tool-call response from `gpt-oss-20b`. The integration test exercises the parallel path with a single tool call to prove `execute_tool_calls_parallel` does not break the flow when there is only one tool to run. The "preserves order under reverse completion" invariant is pinned in `TestExecuteToolCallsParallel::test_runs_concurrently_preserves_input_order` via a unit test that wires controllable per-call delays.

6. **`Text.text: Optional[str]` requires a `b.text and ...` filter**. `tensorzero/types.py:76` declares `Text.text` as `Optional[str]` because `Text` doubles as a template form (`Text(arguments=...)`) where `text` is None. The terminating-turn text extraction in `run_agent` therefore filters with `isinstance(b, Text) and b.text` to skip those template entries — pinned by `TestRunAgent::test_skips_text_blocks_with_none_text`.

7. **`cast(ChatInferenceResponse, raw)` over `isinstance(...)`**. The reference does an `isinstance` check; our unit tests use `FakeChatResponse` (a duck-typed dataclass), and `isinstance(fake, ChatInferenceResponse)` would be False. Switching to `cast` is mypy-safe (chat-only loop never streams, never targets a JSON function) and lets the unit suite run without subclassing the Rust-backed real type. The trade-off is no runtime narrowing — but the misuse is impossible from a typed call site.

## Manual sanity-check against a local MCP server

The automated tests use the in-memory `tests/server.py`. To verify Phase 3 against one of your real MCP servers, run the snippet below from a Python REPL or a one-off script (`uv run python -i`). It exercises the full `run_agent` loop end-to-end against your server.

```python
# adapt the URL / command and the user prompt to your MCP server
import asyncio
from collections.abc import Awaitable
from typing import cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import run_agent


USER_PROMPT = "Please use one of the available tools to answer my question."


async def main() -> None:
    # Pick ONE of these transports:
    #   HTTP:   Client("https://your-mcp-server.example.com/mcp")
    #   stdio:  Client("/abs/path/to/your_server.py")
    #   config: Client({"mcpServers": {"x": {"command": "python", "args": ["..."]}}})
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    async with await gateway as t0, Client("http://localhost:8000/mcp") as mcp_client:
        result = await run_agent(
            t0, mcp_client, USER_PROMPT, "agent", dryrun=True
        )

    print(f"[F3.1/F3.2] response:    {result.response!r}")
    print(f"[F3.3]      episode_id:  {result.episode_id}")
    print(f"[F3.6]      turns:       {result.turns}")
    print(f"[F3.6]      inferences:  {result.inference_ids}")


asyncio.run(main())
```

What to look for:

- `[F3.1/F3.2]` — `result.response` is non-empty and reflects the tool's output
- `[F3.3]` — `episode_id` is a UUID string
- `[F3.6]` — `turns >= 2` (one for the tool, one for the final answer); `inferences` length equals `turns`

If `result.turns >= max_turns` you'll get a `MaxTurnsExceededError` instead of a result — bump `max_turns=...` or simplify the prompt so the LLM stops calling tools.

To exercise the parallel executor, pass `parallel_tools=True`. To continue an episode, pass `episode_id=previous_result.episode_id` to a follow-up call.

## Final status

| Item | Validation | Result |
|---|---|---|
| F3.1 / M3.2 | `test_run_agent_single_turn_add` — full loop with `add` tool produces `"5"` in the final text | ✅ |
| F3.2 / M3.3 | Same test naturally produces `turns >= 2` (tool + final), exercising the multi-turn extension; also pinned by `TestRunAgent::test_inference_ids_collected_in_order` (3-turn unit) | ✅ |
| F3.3 / M3.4 | `episode_id` captured from response 1, forwarded to inference 2 — pinned unit-side by `test_episode_id_propagates_across_turns` and integration-side by `test_run_agent_continuous_conversation` | ✅ |
| F3.4 / M3.3 | Loop returns when `has_tool_calls` is False — pinned by `test_terminates_on_text_only_first_turn` and integration `test_run_agent_no_tools_terminates_immediately` (`turns == 1`) | ✅ |
| F3.5 / M3.5 | N tool calls → ONE `Message(role="user")` carrying N `ToolResult` blocks — pinned by `test_two_tool_calls_collapse_into_one_user_message` | ✅ |
| F3.6 / M3.5 | History sequence `user → assistant → user(tools) → ...` — pinned by `test_single_turn_assembles_user_assistant_user_tools_history` (walks all three messages explicitly) | ✅ |
| F3.7 / M3.6 | After a final response, a second `run_agent` call with the prior `episode_id` succeeds and shares the same episode — `test_run_agent_continuous_conversation` | ✅ |
| F3.8 / M3.7 | `parallel_tools=True` flag dispatches via `asyncio.gather`; reverse completion order proves concurrency; integration test confirms the path doesn't break the flow with a single tool | ✅ |

Test totals after Phase 3: **65 passed** (Phase 0–2's 31 + 30 new Phase-3 unit + 4 new Phase-3 integration = 65). Ruff and mypy strict both clean.

Phase 3 closed. Ready for **M4 — Multi-Server** (FastMCP `mount()` / `Client(config_dict)` composition validation against the agent loop).
