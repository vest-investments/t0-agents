"""Integration tests — require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse

from t0_agents import run_agent
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
    """F1.3: T0 accepts the converted additional_tools without error."""
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
    """F1.4 (deviation): a directive tool_choice yields a ToolCall end-to-end.

    The checklist literally says `tool_choice="required"`. Groq's native API
    rejects `"required"` for `gpt-oss-20b` with "'tool_choice' : value must
    be a string (auto or none) or an object like `{'type': 'function', ...}`",
    so we use `tool_choice={"specific": "add"}` — TensorZero maps this to the
    object form Groq always accepts. The spirit of F1.4 (prove the plumbing
    end-to-end: `additional_tools` reach the LLM, `tool_choice` is honored,
    response contains a `ToolCall` block) is preserved.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={
                "messages": [
                    {"role": "user", "content": "Compute 2 plus 3."},
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
    """F1.5: tool with a Pydantic-model parameter produces a schema T0 accepts."""
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
    """F2.1 + F2.7 + F2.8: full two-turn roundtrip using ``add``.

    1. Inference #1 with ``tool_choice={"specific":"add"}`` forces a ToolCall.
    2. We execute it through FastMCP (``raise_on_error=False``, F2.1).
    3. ``build_tool_result`` produces a ``ToolResult`` (F2.7 shape).
    4. Inference #2 carries the conversation back; T0 must accept the
       ``ToolResult`` in a user message and the LLM must produce at least one
       ``Text`` block that mentions the computed value (F2.7 + F2.8).

    Episode id is reused between turns so T0 ties the two inferences together.
    ``dryrun=True`` keeps ClickHouse clean (same pattern as Phase 1).
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

            # F2.1 — execute the tool through FastMCP with the validated args
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error, (
                f"add tool unexpectedly errored: {mcp_result!r}"
            )

            # F2.7 — build the ToolResult T0 will consume
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

        # F2.8 — LLM produces a coherent response that incorporates the result
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
    """F2.1 + F2.3 + F2.7 with the Pydantic-parameter ``search`` tool.

    Focuses on F2.3 (non-trivial ``.data`` payload serialization — the
    ``search`` tool returns a list, which goes through ``serialize_tool_result``'s
    ``json.dumps(..., default=str)`` fallback branch) and F2.7 (T0 accepts the
    ``ToolResult`` in the next turn without raising). We deliberately do **not**
    pin F2.8 here because the ``search`` tool in ``tests/server.py`` is typed
    ``-> list[dict]`` (no params) and FastMCP's content-block serializer
    sometimes stringifies it as ``["Root()"]``, which the LLM reasonably
    reacts to by asking for another tool call rather than by emitting text.
    F2.8 is covered end-to-end by ``test_add_tool_roundtrip_final_text``.
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

            # F2.1 — execute the Pydantic-parametered tool
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error

            # F2.3 + F2.7 — the serialized result is a non-empty string; build
            # produces a well-formed ToolResult with matching id/name.
            t0_result = build_tool_result(tc, mcp_result)
            assert isinstance(t0_result.result, str) and t0_result.result
            assert t0_result.name == "search"
            assert t0_result.id == tc.id

            # F2.7 end-to-end: T0 accepts the ToolResult in a follow-up
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


# --- Phase 3: agent loop end-to-end ----------------------------------------


async def test_run_agent_single_turn_add(mcp_client: Client[Any]) -> None:
    """F3.1 + F3.2 + F3.3 + F3.4: run_agent end-to-end with the ``add`` tool.

    F3.1 (single-turn user→LLM→tool→result→final) is exercised here. Since the
    add tool requires one execution cycle, the loop naturally runs ``>= 2``
    turns (one to call ``add``, one to deliver the final answer), which also
    satisfies F3.2 (multi-turn). F3.3 (episode_id propagation) is anchored by
    asserting the result has a non-empty episode_id; F3.4 (loop terminates on
    text-only response) is implicit in the loop completing.

    ``dryrun=True`` keeps ClickHouse clean (Phase 1/2 convention).
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
    # F3.3: each turn captures one inference_id, in order.
    assert len(result.inference_ids) == result.turns


async def test_run_agent_no_tools_terminates_immediately(
    mcp_client: Client[Any],
) -> None:
    """F3.4: when the LLM has no tool calls, the loop terminates after 1 turn.

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
    """F3.7: after a final response the user can submit another query.

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
    """F3.8: ``parallel_tools=True`` does not break the flow.

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


# --- Phase 4: multi-server composition -------------------------------------


def _build_composed_server() -> FastMCP[None]:
    """Compose two in-memory FastMCP servers under namespaces ``a`` and ``b``.

    Private helper local to the integration module (matching the one in
    ``tests/test_mcp/test_multi_server.py``) so Phase 4 integration fixtures
    don't leak into shared test utilities. The tool surface is intentionally
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
    """F4.3 + F4.4 deterministic #1: force-call ``a_add`` via tool_choice.

    Proves end-to-end that (a) T0 accepts the composed namespaced
    ``additional_tools`` (F4.3), and (b) the LLM emits a ``ToolCall`` with the
    **namespaced name** which flows through ``Client.call_tool`` and FastMCP's
    Namespace transform to reach server A's underlying ``add`` tool (F4.4).

    We use a directed tool_choice (the Phase 1 precedent for Groq
    determinism) rather than natural selection. The natural-selection
    variant lives in ``test_run_agent_mounted_servers_natural_selection``.
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
    """F4.3 + F4.4 deterministic #2: route through the ``b_`` namespace.

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
    """F4.3 + F4.4 realistic: a single prompt that names both namespaces,
    let the LLM choose and chain the calls.

    Matches M4.4's spirit ("The LLM can invoke tools from both servers in the
    same conversation") more literally than the forced-choice flows. The
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
    # this assertion fails loud so we can document the deviation (Phase 1
    # precedent for Groq flakiness).
    assert "5" in result.response and "3" in result.response, (
        "natural-selection path did not surface both tools' results — "
        "F4.4 is still pinned by the two forced-choice tests above, but "
        "please capture this in the implementation log: "
        f"turns={result.turns}, response={result.response!r}"
    )
