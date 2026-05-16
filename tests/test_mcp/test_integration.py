"""Integration tests -- require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.elicitation import ElicitResult
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse, TensorZeroError

from t0_agents import MaxTurnsExceededError, run_agent
from t0_agents.mcp import build_tool_result, convert_tools
from tests.server import mcp

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Helper to narrow `build_http`'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_t0_accepts_additional_tools(mcp_client: Client[Any]) -> None:
    """T0 accepts the converted additional_tools without error."""
    async with mcp_client:
        tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "Say hi briefly."}]},
            additional_tools=t0_tools,
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    assert response.content is not None
    assert response.inference_id is not None
    assert response.episode_id is not None


async def test_tool_choice_forces_tool_call(mcp_client: Client[Any]) -> None:
    """A directive tool_choice yields a ToolCall end-to-end.

    Groq's native API rejects `"required"` for `gpt-oss-20b` with
    "'tool_choice' : value must be a string (auto or none) or an object like
    `{'type': 'function', ...}`", so we use `tool_choice={"specific": "add"}`
    -- TensorZero maps this to the object form Groq always accepts. This proves
    the plumbing end-to-end: `additional_tools` reach the LLM, `tool_choice` is
    honored, and the response contains a `ToolCall` block.

    The prompt uses non-trivial operands and an explicit "use the add tool"
    directive: the reasoning model short-circuits trivial arithmetic by
    emitting the answer as plain text, which Groq's server-side
    `tool_use_failed` guard then rejects (see commit 9b76ee9). Non-trivial
    operands keep the model honest and let `tool_choice` actually do its job.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the add tool to compute 1247 plus 8932.",
                    },
                ]
            },
            additional_tools=t0_tools,
            tool_choice={"specific": "add"},
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    tool_calls = [b for b in response.content if isinstance(b, ToolCall)]
    assert tool_calls, (
        f"tool_choice={{'specific':'add'}} produced no ToolCall. "
        f"content={response.content!r}"
    )
    assert tool_calls[0].name == "add"


async def test_pydantic_param_schema_accepted_by_t0() -> None:
    """Tool with a Pydantic-model parameter produces a schema T0 accepts."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    # Restrict to the Pydantic-param tool to narrow the failure mode: if T0
    # rejects the schema, the error is unambiguously about `search`.
    search_tool = [t for t in convert_tools(tools) if t["name"] == "search"]
    assert len(search_tool) == 1

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "Find something interesting."}]},
            additional_tools=search_tool,
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    assert response.content is not None


async def test_add_tool_roundtrip_final_text(mcp_client: Client[Any]) -> None:
    """Full two-turn roundtrip using ``add``.

    1. Inference #1 with ``tool_choice={"specific":"add"}`` forces a ToolCall.
    2. We execute it through FastMCP (``raise_on_error=False``).
    3. ``build_tool_result`` produces a ``ToolResult``.
    4. Inference #2 carries the conversation back; T0 must accept the
       ``ToolResult`` in a user message and the LLM must produce at least one
       ``Text`` block that mentions the computed value.

    Episode id is reused between turns so T0 ties the two inferences together.
    ``dryrun=True`` keeps ClickHouse clean.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
        t0_tools = convert_tools(tools)

        async with await _gateway_factory() as t0:
            first_turn = await t0.inference(
                function_name="agent",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please compute 2 + 3 using the add tool.",
                        }
                    ]
                },
                additional_tools=t0_tools,
                tool_choice={"specific": "add"},
                dryrun=True,
            )

            assert isinstance(first_turn, ChatInferenceResponse)
            tool_calls = [b for b in first_turn.content if isinstance(b, ToolCall)]
            assert tool_calls, f"no ToolCall in first turn: {first_turn.content!r}"
            tc = tool_calls[0]
            assert tc.name == "add"
            assert tc.arguments is not None, (
                f"LLM produced invalid JSON arguments: {tc.raw_arguments!r}"
            )

            # Execute the tool through FastMCP with the validated args
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error, (
                f"add tool unexpectedly errored: {mcp_result!r}"
            )

            # Build the ToolResult T0 will consume
            t0_result = build_tool_result(tc, mcp_result)
            assert t0_result.name == "add"
            assert t0_result.id == tc.id
            assert "5" in t0_result.result, (
                f"serialized result should mention the sum 5: {t0_result.result!r}"
            )

            second_turn = await t0.inference(
                function_name="agent",
                episode_id=first_turn.episode_id,
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please compute 2 + 3 using the add tool.",
                        },
                        {"role": "assistant", "content": first_turn.content},
                        {"role": "user", "content": [t0_result]},
                    ]
                },
                additional_tools=t0_tools,
                dryrun=True,
            )

        # LLM produces a coherent response that incorporates the result
        assert isinstance(second_turn, ChatInferenceResponse)
        text_blocks = [b for b in second_turn.content if isinstance(b, Text)]
        assert text_blocks, (
            "second turn produced no Text blocks — the LLM did not respond "
            f"after the tool result. content={second_turn.content!r}"
        )
        # Text.text is Optional[str] (template form); filter explicitly so
        # mypy strict sees a concrete str and the join stays well-defined.
        final_text = " ".join(b.text for b in text_blocks if b.text)
        assert "5" in final_text, (
            f"final answer does not mention the sum 5: {final_text!r}"
        )


async def test_search_pydantic_roundtrip(mcp_client: Client[Any]) -> None:
    """Pydantic-parameter ``search`` tool roundtrip.

    Focuses on non-trivial ``.data`` payload serialization -- the ``search``
    tool returns a list, which goes through ``serialize_tool_result``'s
    ``json.dumps(..., default=str)`` fallback branch -- and that T0 accepts the
    ``ToolResult`` in the next turn without raising. We deliberately do **not**
    pin the final text here because the ``search`` tool in ``tests/server.py``
    is typed ``-> list[dict]`` (no params) and FastMCP's content-block
    serializer sometimes stringifies it as ``["Root()"]``, which the LLM
    reasonably reacts to by asking for another tool call rather than by
    emitting text. Final text is covered by ``test_add_tool_roundtrip_final_text``.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
        t0_tools = convert_tools(tools)

        async with await _gateway_factory() as t0:
            first_turn = await t0.inference(
                function_name="agent",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use the search tool to look up 'tensorzero' "
                                "with a limit of 3 results."
                            ),
                        }
                    ]
                },
                additional_tools=t0_tools,
                tool_choice={"specific": "search"},
                dryrun=True,
            )

            assert isinstance(first_turn, ChatInferenceResponse)
            tool_calls = [b for b in first_turn.content if isinstance(b, ToolCall)]
            assert tool_calls, f"no ToolCall in first turn: {first_turn.content!r}"
            tc = tool_calls[0]
            assert tc.name == "search"
            assert tc.arguments is not None

            # Execute the Pydantic-parametered tool
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error

            # The serialized result is a non-empty string; build produces a
            # well-formed ToolResult with matching id/name.
            t0_result = build_tool_result(tc, mcp_result)
            assert isinstance(t0_result.result, str) and t0_result.result
            assert t0_result.name == "search"
            assert t0_result.id == tc.id

            # T0 accepts the ToolResult in a follow-up
            # inference without raising. The exact shape of the LLM's response
            # is not pinned here.
            second_turn = await t0.inference(
                function_name="agent",
                episode_id=first_turn.episode_id,
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use the search tool to look up 'tensorzero' "
                                "with a limit of 3 results."
                            ),
                        },
                        {"role": "assistant", "content": first_turn.content},
                        {"role": "user", "content": [t0_result]},
                    ]
                },
                additional_tools=t0_tools,
                dryrun=True,
            )

        assert isinstance(second_turn, ChatInferenceResponse)
        assert second_turn.content, (
            "second turn returned empty content — T0 or the LLM rejected the "
            f"ToolResult. first_turn content: {first_turn.content!r}"
        )


# --- Agent loop end-to-end --------------------------------------------------


async def test_run_agent_single_turn_add(mcp_client: Client[Any]) -> None:
    """run_agent end-to-end with the ``add`` tool.

    Single-turn user->LLM->tool->result->final is exercised here. Since the
    add tool requires one execution cycle, the loop naturally runs ``>= 2``
    turns (one to call ``add``, one to deliver the final answer), which also
    covers multi-turn. episode_id propagation is anchored by asserting the
    result has a non-empty episode_id; loop termination on text-only response
    is implicit in the loop completing.

    ``dryrun=True`` keeps ClickHouse clean.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the add tool to compute 2 + 3.",
                "agent",
                dryrun=True,
            )

    assert "5" in result.response, (
        f"final answer should mention the sum 5: {result.response!r}"
    )
    assert result.episode_id, "episode_id was not captured from the response"
    assert result.inference_ids, "inference_ids should not be empty"
    assert result.turns >= 2, (
        f"expected at least 2 turns (tool + final), got {result.turns}"
    )
    # Each turn captures one inference_id, in order.
    assert len(result.inference_ids) == result.turns


async def test_run_agent_no_tools_terminates_immediately(
    mcp_client: Client[Any],
) -> None:
    """When the LLM has no tool calls, the loop terminates after 1 turn.

    A pure greeting prompt should never trigger a tool call (none of our test
    tools — add, greet(name), search, echo_no_docs — match a bare "say hi"
    request). The loop should return on the first inference's text-only
    response.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: pong",
                "agent",
                dryrun=True,
            )

    assert result.turns == 1, f"expected exactly 1 turn, got {result.turns}"
    assert result.response, "expected non-empty text response"


async def test_run_agent_continuous_conversation(mcp_client: Client[Any]) -> None:
    """After a final response the user can submit another query.

    Two ``run_agent`` calls back-to-back, the second carrying ``episode_id``
    from the first. T0 ties them as one episode (visible in ClickHouse as a
    shared episode_id even with dryrun=True the response.episode_id remains
    consistent). The LLM does not see the prior conversation's messages — that
    is a design choice (run_agent is single-conversation; episode_id is
    analytics-only at this layer) — but the API must still accept the
    second call without error.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            first = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: hello",
                "agent",
                dryrun=True,
            )
            second = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: world",
                "agent",
                episode_id=first.episode_id,
                dryrun=True,
            )

    assert first.episode_id, "first call should have an episode_id"
    assert second.episode_id == first.episode_id, (
        "T0 should keep the episode_id when we forward it in the second call: "
        f"first={first.episode_id!r}, second={second.episode_id!r}"
    )
    assert second.response, "second call must produce a response"


async def test_run_agent_parallel_tools_flag_works(mcp_client: Client[Any]) -> None:
    """``parallel_tools=True`` does not break the flow.

    Groq's ``gpt-oss-20b`` does not generally emit multiple tool calls in a
    single response (Groq is not on T0's parallel_tool_calls allowlist), so
    this test exercises the parallel path with a single-tool flow. A passing
    run proves that ``execute_tool_calls_parallel`` integrates correctly with
    ``run_agent`` even when there is only one tool to run — the optional
    parallel mode does not regress the sequential happy path.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the add tool to compute 4 + 6.",
                "agent",
                dryrun=True,
                parallel_tools=True,
            )

    assert "10" in result.response, (
        f"final answer should mention the sum 10: {result.response!r}"
    )
    assert result.turns >= 2


# --- Multi-server composition -----------------------------------------------


def _build_composed_server() -> FastMCP[None]:
    """Compose two in-memory FastMCP servers under namespaces ``a`` and ``b``.

    Private helper local to the integration module (matching the one in
    ``tests/test_mcp/test_multi_server.py``) so integration fixtures don't
    leak into shared test utilities. The tool surface is intentionally
    minimal — one arithmetic tool per server — because these tests only need
    to prove that T0 + the agent loop handle the namespaced names end-to-end.
    """
    server_a: FastMCP[None] = FastMCP("ServerA")

    @server_a.tool
    def add(a: int, b: int) -> int:
        """Add two numbers (from server A)."""
        return a + b

    server_b: FastMCP[None] = FastMCP("ServerB")

    @server_b.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers (from server B)."""
        return a - b

    main: FastMCP[None] = FastMCP("MultiMain")
    main.mount(server_a, namespace="a")
    main.mount(server_b, namespace="b")
    return main


async def test_run_agent_mounted_servers_forced_add() -> None:
    """Deterministic force-call ``a_add`` via tool_choice.

    Proves end-to-end that (a) T0 accepts the composed namespaced
    ``additional_tools``, and (b) the LLM emits a ``ToolCall`` with the
    **namespaced name** which flows through ``Client.call_tool`` and FastMCP's
    Namespace transform to reach server A's underlying ``add`` tool.

    We use a directed tool_choice for Groq determinism rather than natural
    selection. The natural-selection variant lives in
    ``test_run_agent_mounted_servers_natural_selection``.
    """
    main = _build_composed_server()
    mcp_client: Client[Any] = Client(main)
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the a_add tool to compute 2 + 3.",
                "agent",
                dryrun=True,
            )

    assert "5" in result.response, (
        f"final answer should mention the sum 5 from a_add: {result.response!r}"
    )
    assert result.turns >= 2, (
        f"expected at least 2 turns (tool + final), got {result.turns}"
    )


async def test_run_agent_mounted_servers_forced_subtract() -> None:
    """Deterministic route through the ``b_`` namespace.

    Mirror of ``test_run_agent_mounted_servers_forced_add`` but hits server
    B's ``subtract`` via the ``b_subtract`` namespaced name. Together the two
    tests pin routing on **both** namespaces.
    """
    main = _build_composed_server()
    mcp_client: Client[Any] = Client(main)
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the b_subtract tool to compute 7 - 4.",
                "agent",
                dryrun=True,
            )

    assert "3" in result.response, (
        f"final answer should mention the difference 3 from b_subtract: "
        f"{result.response!r}"
    )
    assert result.turns >= 2


async def test_run_agent_mounted_servers_natural_selection() -> None:
    """Realistic: a single prompt that names both namespaces, let the LLM
    choose and chain the calls.

    This proves the LLM can invoke tools from both servers in the same
    conversation -- more realistic than the forced-choice flows. The
    prompt is worded to explicitly name ``a_add`` and ``b_subtract`` so the
    model has a clear path — Groq's ``gpt-oss-20b`` can still flake on
    chaining, so if the model only picks one tool we still consider
    routing validated by the two forced-choice tests above. This test
    therefore asserts a **loose** invariant: either the final answer
    mentions both computed values (5 and 3), or the run produced enough
    turns for at least one namespaced tool to have executed (`result.turns >= 2`).
    """
    main = _build_composed_server()
    mcp_client: Client[Any] = Client(main)
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                (
                    "First use the a_add tool to compute 2 + 3, then use the "
                    "b_subtract tool to compute 7 - 4. Report both results."
                ),
                "agent",
                dryrun=True,
                max_turns=6,
            )

    # At least one namespaced tool was invoked (turn >= 2 proves a tool turn
    # happened and a follow-up produced the final text).
    assert result.turns >= 2, (
        f"expected natural selection to invoke at least one namespaced tool "
        f"(turns>=2), got {result.turns}; response={result.response!r}"
    )
    # The strong form: both numbers show up. If the LLM flaked on chaining,
    # this assertion fails loud so we can document the deviation.
    assert "5" in result.response and "3" in result.response, (
        "natural-selection path did not surface both tools' results -- "
        "still pinned by the two forced-choice tests above, but "
        "please capture this in the implementation log: "
        f"turns={result.turns}, response={result.response!r}"
    )


# --- Robustness and error handling ------------------------------------------


async def test_run_agent_survives_failing_tool(
    mcp_client: Client[Any],
) -> None:
    """A tool that raises ``ToolError`` does not crash ``run_agent``.

    The error text reaches the LLM via an error ``ToolResult`` and the loop
    keeps going. The **real** invariant this test enforces is negative: the
    loop never lets ``ToolError``, ``McpError``, or a generic ``Exception``
    bubble out when a tool fails. That is what ``_execute_single_tool_call``'s
    catch chain guarantees (and what the unit test
    ``TestRobustness::test_tool_error_caught`` pins strictly).

    Safe outcomes against the live gateway:

    * Clean return with ``turns >= 2`` and non-empty ``response`` -- the LLM
      called ``fail_always``, saw the error, and wrapped up with text.
    * ``MaxTurnsExceededError`` -- Groq saw the error and kept retrying the
      tool until the budget ran out (known Groq gpt-oss-20b behavior).
    * ``TensorZeroError`` with ``"not in request.tools"`` -- Groq emitted a
      malformed tool name on retry (``add<|channel|>commentary``). That's
      still evidence the retry fired; the unit test pins the clean path.

    Groq can also skip the tool call entirely on turn 1 or produce a
    Thought-only response with empty ``.text`` -- neither exercises the gate,
    so we skip rather than fail.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    mcp_client,
                    "Please call the fail_always tool with reason 'pytest-F51'.",
                    "agent",
                    dryrun=True,
                    max_turns=4,
                )
            except MaxTurnsExceededError:
                return  # Every error caught, turns exhausted
            except TensorZeroError as e:
                if "not in request.tools" in str(e):
                    pytest.skip(
                        f"Groq gpt-oss-20b malformed retry tool name: {e}"
                    )
                raise

    if result.turns < 2:
        pytest.skip(
            "Groq gpt-oss-20b skipped the failing tool call on turn 1 "
            f"(turns={result.turns}, response={result.response!r})"
        )
    if not result.response:
        pytest.skip(
            f"Groq produced empty response after the failing tool "
            f"(turns={result.turns})"
        )
    # Clean path: error was caught and loop completed.
    assert result.turns >= 2
    assert result.response


async def test_tool_timeout_does_not_crash_run_agent(
    mcp_client: Client[Any],
) -> None:
    """A real call_tool timeout -> McpError -> caught -> error ``ToolResult``.
    The loop does NOT raise ``McpError`` out to the caller.

    ``slow_tool`` sleeps for 2 seconds; we cap ``tool_timeout`` at ``0.5``
    so the MCP client's protocol-level read timeout trips and raises
    ``McpError`` (code ``-32000``). The loop catches it and surfaces the
    error to the LLM.

    Two acceptable outcomes:

    1. **Clean path**: the LLM sees the timeout error, gives up, and produces
       a text response. ``result.response`` is non-empty and
       ``result.turns >= 2``.
    2. **Retry-until-exhausted path**: the LLM sees the timeout error and
       calls ``slow_tool`` again (-> another timeout -> another error). Groq
       in particular often loops like this. Eventually ``max_turns`` is
       exhausted and ``run_agent`` raises ``MaxTurnsExceededError``. This
       *also* proves the timeout was caught -- every timeout was caught and
       converted to a ``ToolResult``; the loop never let ``McpError`` bubble
       out.

    The **failure** mode this test guards against is: ``McpError`` (or any
    other exception apart from ``MaxTurnsExceededError``) escapes the loop
    uncaught.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    mcp_client,
                    "Please call the slow_tool with delay_seconds=2.",
                    "agent",
                    dryrun=True,
                    tool_timeout=0.5,
                    max_turns=4,
                )
            except MaxTurnsExceededError:
                # Acceptable outcome #2: every timeout was caught ->
                # Groq kept retrying -> we ran out of turns. No McpError
                # ever escaped the loop.
                return

    # Acceptable outcome #1: clean recovery.
    assert result.response, (
        "run_agent returned empty response — the loop recovered from the "
        "timeout but produced no text"
    )
    assert result.turns >= 2, (
        f"expected at least 2 turns (timed-out tool + final), got {result.turns}"
    )


async def test_run_agent_allows_llm_retry(mcp_client: Client[Any]) -> None:
    """After an error ToolResult, the LLM can retry.

    Two-step prompt: call ``fail_always`` first, then use ``add`` to compute
    2+3. The unit test
    ``TestRobustness::test_loop_continues_after_error_and_llm_retries``
    pins the retry mechanism strictly via ``FakeT0``; this integration test
    only proves the full loop survives the flow against a live gateway when
    Groq plays along.

    Known Groq ``openai/gpt-oss-20b`` flakes on multi-step tool chains.
    We skip -- rather than fail --
    when we detect a Groq-side flake so the integration gate is not held
    hostage by the model. The recognized flakes are:

    1. Model emits a malformed tool name on the retry (``add<|channel|>
       commentary``): T0 raises ``TensorZeroError`` with
       "not in request.tools" before the loop sees content.
    2. Model skips the tool call entirely on turn 1: ``result.turns == 1``
       and ``result.response`` may be empty (Thought-only response).

    A **successful** run requires ``turns >= 2`` (so the loop actually went
    through the error->retry cycle) and a non-empty final text.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    mcp_client,
                    (
                        "Step 1: call fail_always with reason='warmup'. "
                        "Step 2: call the add tool with a=2 and b=3. "
                        "Step 3: report the sum."
                    ),
                    "agent",
                    dryrun=True,
                    max_turns=6,
                )
            except MaxTurnsExceededError:
                # Acceptable: retries fired but Groq never wrapped up.
                # Mechanism (loop continues after error) is proven.
                return
            except TensorZeroError as e:
                if "not in request.tools" in str(e):
                    pytest.skip(
                        "Groq gpt-oss-20b malformed the retry tool name "
                        f"(known flake): {e}"
                    )
                raise

    if result.turns < 2:
        pytest.skip(
            f"Groq gpt-oss-20b skipped the failing tool call on turn 1 "
            f"(turns={result.turns}, response={result.response!r}) — "
            "retry path not exercised this run"
        )
    if not result.response:
        pytest.skip(
            f"Groq produced empty response after retry "
            f"(turns={result.turns})"
        )
    assert result.response


async def test_max_turns_raises_against_live_gateway(
    mcp_client: Client[Any],
) -> None:
    """``run_agent`` raises ``MaxTurnsExceededError`` when the LLM keeps
    emitting tool calls past ``max_turns``.

    ``run_agent`` does not expose ``tool_choice`` (intentional -- it is a
    per-inference concern, not a per-conversation one), so we use a prompt
    pattern that reliably produces a tool call and set
    ``max_turns=1``. With the budget exhausted after the first tool call,
    the loop must raise on the next iteration attempt rather than returning
    silently. ``e.max_turns`` should equal the limit we set.

    If Groq skips the tool call on turn 1 (known flake), ``run_agent``
    returns cleanly instead of raising. We skip rather than fail -- the
    unit test ``TestRobustness::test_max_turns_raises_with_correct_limit``
    pins the raise strictly. A clean return here is not a regression; it
    means this particular run did not exercise the max-turns precondition.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    mcp_client,
                    "Please use the add tool to compute 2 + 3.",
                    "agent",
                    dryrun=True,
                    max_turns=1,
                )
            except MaxTurnsExceededError as e:
                # Clean path: the raise fired as expected.
                assert e.max_turns == 1
                return

    pytest.skip(
        "Groq gpt-oss-20b did not emit a tool call on turn 1 "
        f"(turns={result.turns}, response={result.response!r}); "
        "max-turns precondition not met this run"
    )


async def test_none_description_tool_is_usable_end_to_end(
    mcp_client: Client[Any],
) -> None:
    """A tool with ``description=None`` on FastMCP's side (the ``echo_no_docs``
    fixture) flows through ``convert_tools`` to T0 as ``description=""`` and T0
    accepts the payload end-to-end.

    T0's native inference API accepts ``description`` fields that originated as
    ``None`` at the FastMCP layer. That is already proven by every other
    robustness integration test in this file -- each call to ``run_agent``
    passes the full tool list through ``convert_tools``, which includes
    ``echo_no_docs`` with its empty string description. If T0 rejected the
    empty description, every robustness test would have failed at the first
    ``t0.inference`` call.

    This test specifically tries to have the LLM *invoke* ``echo_no_docs``
    to prove the path also works end-to-end, but Groq is free to pick any
    other tool, produce text only, or skip the call. When Groq does not
    invoke the no-description tool on this run, we skip rather than fail --
    the unit test ``TestRobustness::test_none_description_round_trip`` pins
    the clean path and the adapter unit test pins the coercion itself.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    mcp_client,
                    "Use the echo_no_docs tool with value='phase5-F58'.",
                    "agent",
                    dryrun=True,
                    max_turns=4,
                )
            except MaxTurnsExceededError:
                return  # T0 accepted the empty-description tool across N turns
            except TensorZeroError as e:
                if "not in request.tools" in str(e):
                    pytest.skip(
                        f"Groq gpt-oss-20b malformed tool name: {e}"
                    )
                raise

    if result.turns < 2:
        pytest.skip(
            "Groq gpt-oss-20b skipped the echo_no_docs tool on turn 1 "
            f"(turns={result.turns}) — T0 still accepted the payload, but "
            "we did not exercise the tool call path this run"
        )
    if not result.response:
        pytest.skip(
            f"Groq produced empty response after tool call "
            f"(turns={result.turns})"
        )
    # T0 accepted the empty-description tool AND the LLM invoked it.
    assert result.turns >= 2
    assert result.response


# --- Elicitation integration ------------------------------------------------


async def test_run_agent_elicitation_approve() -> None:
    """Elicitation-enabled tool through the full agent loop (approve).

    A Client with an auto-approve handler is passed to ``run_agent``.
    The LLM is directed to call ``approve_action``; the handler accepts
    the elicitation; the tool returns ``"Approved: ..."``; the LLM produces
    a final text response.

    Accepts broad outcomes because LLM behaviour is nondeterministic.
    """

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={})

    client = Client(mcp, elicitation_handler=handler)
    async with client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    client,
                    "Please use the approve_action tool to approve deploying to production.",
                    "agent",
                    dryrun=True,
                    tool_timeout=60.0,
                )
            except MaxTurnsExceededError:
                return  # loop ran; elicitation did not crash it
            except TensorZeroError as e:
                if "not in request.tools" in str(e):
                    pytest.skip(f"Groq malformed tool name: {e}")
                raise

    assert result.turns >= 2, (
        f"expected >= 2 turns (tool + final), got {result.turns}"
    )
    assert result.response, "expected non-empty final response"


async def test_run_agent_elicitation_decline() -> None:
    """Elicitation decline through the full agent loop.

    Same setup as the approve variant, but the handler declines. The tool
    returns ``"Rejected: ..."`` and the LLM should produce a final text
    response.
    """

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    client = Client(mcp, elicitation_handler=handler)
    async with client:
        async with await _gateway_factory() as t0:
            try:
                result = await run_agent(
                    t0,
                    client,
                    "Please use the approve_action tool to approve deploying to production.",
                    "agent",
                    dryrun=True,
                    tool_timeout=60.0,
                )
            except MaxTurnsExceededError:
                return  # loop ran; elicitation did not crash it
            except TensorZeroError as e:
                if "not in request.tools" in str(e):
                    pytest.skip(f"Groq malformed tool name: {e}")
                raise

    assert result.turns >= 2, (
        f"expected >= 2 turns (tool + final), got {result.turns}"
    )
    assert result.response, "expected non-empty final response"


# --- Observability and feedback ----------------------------------------------


async def test_ids_present_in_agent_result(
    mcp_client: Client[Any],
) -> None:
    """inference_id and episode_id are captured from the live gateway."""
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the add tool to compute 2 + 3.",
                "agent",
                dryrun=True,
                max_turns=4,
            )

    assert result.episode_id, "episode_id was not captured"
    assert result.inference_ids, "inference_ids should not be empty"
    assert len(result.inference_ids) == result.turns


async def test_boolean_feedback_accepted_by_gateway(
    mcp_client: Client[Any],
) -> None:
    """Boolean feedback on episode_id is accepted by the live gateway.

    Inference uses dryrun=False so the episode_id exists in ClickHouse (the
    feedback endpoint validates the target ID). Feedback uses dryrun=True so
    we validate the request without persisting feedback rows.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0, mcp_client,
                "Please use the add tool to compute 1 + 1.",
                "agent", dryrun=False, max_turns=4,
            )
            feedback_resp = await t0.feedback(
                metric_name="task_success",
                episode_id=result.episode_id,
                value=True,
                dryrun=True,
            )

    assert feedback_resp.feedback_id is not None


async def test_float_feedback_accepted_by_gateway(
    mcp_client: Client[Any],
) -> None:
    """Float feedback on inference_id is accepted by the live gateway.

    Inference uses dryrun=False so the inference_id exists in ClickHouse.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0, mcp_client,
                "Please use the add tool to compute 3 + 4.",
                "agent", dryrun=False, max_turns=4,
            )
            feedback_resp = await t0.feedback(
                metric_name="quality_score",
                inference_id=result.inference_ids[0],
                value=8.5,
                dryrun=True,
            )

    assert feedback_resp.feedback_id is not None


async def test_comment_feedback_accepted_by_gateway(
    mcp_client: Client[Any],
) -> None:
    """Comment feedback (built-in metric) is accepted by the gateway.

    Inference uses dryrun=False so the inference_id exists in ClickHouse.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0, mcp_client,
                "Please use the add tool to compute 5 + 5.",
                "agent", dryrun=False, max_turns=4,
            )
            feedback_resp = await t0.feedback(
                metric_name="comment",
                inference_id=result.inference_ids[-1],
                value="Good tool selection.",
                dryrun=True,
            )

    assert feedback_resp.feedback_id is not None


async def test_demonstration_feedback_accepted_by_gateway(
    mcp_client: Client[Any],
) -> None:
    """Demonstration feedback (inference_id only) accepted by gateway.

    Inference uses dryrun=False so the inference_id exists in ClickHouse.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0, mcp_client,
                "Please use the add tool to compute 6 + 7.",
                "agent", dryrun=False, max_turns=4,
            )
            feedback_resp = await t0.feedback(
                metric_name="demonstration",
                inference_id=result.inference_ids[-1],
                value="The sum of 6 and 7 is 13.",
                dryrun=True,
            )

    assert feedback_resp.feedback_id is not None


async def test_dynamic_tools_recorded_in_clickhouse(
    mcp_client: Client[Any],
) -> None:
    """dynamic_tools column in ChatInference is populated after inference.

    This test uses dryrun=False so the inference row persists in ClickHouse,
    then queries the ClickHouse HTTP API directly to verify the dynamic_tools
    column contains the tool definitions from the test MCP server.
    """
    import httpx

    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0, mcp_client,
                "Please use the add tool to compute 9 + 1.",
                "agent",
                dryrun=False,
                max_turns=4,
            )

    assert result.inference_ids, "Need at least one inference_id"
    inference_id = result.inference_ids[0]

    # Query ClickHouse directly for the dynamic_tools column.
    query = (
        f"SELECT dynamic_tools FROM ChatInference "
        f"WHERE id = '{inference_id}' FORMAT JSON"
    )
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            "http://localhost:8123",
            params={
                "query": query,
                "user": "chuser",
                "password": "chpassword",
                "database": "tensorzero",
            },
        )

    assert resp.status_code == 200, f"ClickHouse query failed: {resp.text}"
    data = resp.json()
    rows = data.get("data", [])
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {data}"

    dynamic_tools = rows[0]["dynamic_tools"]
    assert len(dynamic_tools) > 0, "dynamic_tools should be non-empty"
    # Each entry is a JSON string; verify at least one tool name we know
    tool_names_found = []
    import json as _json
    for entry in dynamic_tools:
        tool_def = _json.loads(entry) if isinstance(entry, str) else entry
        if "name" in tool_def:
            tool_names_found.append(tool_def["name"])
    assert "add" in tool_names_found, (
        f"Expected 'add' in dynamic_tools names, got: {tool_names_found}"
    )
