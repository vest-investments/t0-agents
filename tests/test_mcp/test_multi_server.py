"""Multi-server composition (unit tests).

FastMCP natively supports multi-server composition via ``mount()`` +
``namespace=...`` -- the adapter and agent loop work transparently over a
composed server, because both ``convert_tools`` and ``execute_tool_calls``
treat ``Tool.name`` / ``ToolCall.name`` as opaque strings.

These tests validate tool listing, conversion, routing, and namespace
collision handling without touching T0 at all. The live-gateway end-to-end
tests live in ``tests/test_mcp/test_integration.py``.

Fixture shape (collision test):

* ``server_a`` has tools ``add(a, b) -> int`` and ``greet(name) -> str``
  (returns ``"hello from A: …"``).
* ``server_b`` has tools ``subtract(a, b) -> int`` and ``greet(name) -> str``
  (returns ``"hello from B: …"`` — **different** implementation so routing
  can be distinguished).
* Composed under ``FastMCP("MultiMain")`` with ``namespace="a"`` and
  ``namespace="b"``. ``list_tools()`` should therefore expose four distinct
  entries: ``a_add``, ``a_greet``, ``b_subtract``, ``b_greet``.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents import run_agent
from t0_agents.mcp import convert_tools
from tests._fake_t0 import FakeChatResponse, FakeT0


def _build_composed_server() -> FastMCP[None]:
    """Assemble two sub-servers into a mounted composite with distinct namespaces.

    The second ``greet`` tool is registered via ``@server_b.tool(name="greet")``
    with a distinct Python function name (``greet_from_b``) so that (a) the
    two server-side implementations coexist in module scope without triggering
    ruff's F811 "redefinition of unused name", and (b) each server reports a
    ``greet`` tool that, after namespacing, becomes ``a_greet`` / ``b_greet``.
    """
    server_a: FastMCP[None] = FastMCP("ServerA")

    @server_a.tool
    def add(a: int, b: int) -> int:
        """Add two numbers (from server A)."""
        return a + b

    @server_a.tool
    def greet(name: str) -> str:
        """Greet someone (from server A)."""
        return f"hello from A: {name}"

    server_b: FastMCP[None] = FastMCP("ServerB")

    @server_b.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers (from server B)."""
        return a - b

    @server_b.tool(name="greet")
    def greet_from_b(name: str) -> str:
        """Greet someone (from server B)."""
        return f"hello from B: {name}"

    main: FastMCP[None] = FastMCP("MultiMain")
    main.mount(server_a, namespace="a")
    main.mount(server_b, namespace="b")
    return main


# --- TestMountedNamespacing -------------------------------------------------


class TestMountedNamespacing:
    """Namespaced tool listing, conversion, and local routing via ``mount()``."""

    async def test_list_tools_returns_prefixed_names(self) -> None:
        """list_tools returns four distinct entries, each namespaced with
        ``{a,b}_{original}``."""
        main = _build_composed_server()
        client: Client[Any] = Client(main)
        async with client:
            tools = await client.list_tools()

        names = {t.name for t in tools}
        assert names == {"a_add", "a_greet", "b_subtract", "b_greet"}, (
            f"expected four namespaced tools, got {names!r}"
        )
        # Every listed tool name carries a `{ns}_` prefix.
        for t in tools:
            assert t.name.startswith(("a_", "b_")), (
                f"tool {t.name!r} missing namespace prefix"
            )

    async def test_convert_tools_passes_namespaced_names_through(self) -> None:
        """convert_tools preserves the prefixed names verbatim in the T0
        payload shape -- no renaming, no stripping."""
        main = _build_composed_server()
        client: Client[Any] = Client(main)
        async with client:
            tools = await client.list_tools()

        t0_tools = convert_tools(tools)
        names = {t["name"] for t in t0_tools}
        assert names == {"a_add", "a_greet", "b_subtract", "b_greet"}
        # Shape invariant: each entry has exactly these three keys.
        for t0_tool in t0_tools:
            assert set(t0_tool.keys()) == {"name", "description", "parameters"}
            assert t0_tool["parameters"].get("type") == "object"

    async def test_call_tool_routes_to_correct_server_via_prefix(self) -> None:
        """Each namespace routes to its own underlying server.

        ``a_add`` and ``b_subtract`` prove cross-namespace routing at the arith
        level; ``a_greet`` vs ``b_greet`` prove that even when both servers
        register a tool with the *same unqualified name*, the prefix resolves
        the collision -- the two implementations return different strings.
        """
        main = _build_composed_server()
        client: Client[Any] = Client(main)
        async with client:
            # Cross-namespace arithmetic
            add_result = await client.call_tool("a_add", {"a": 2, "b": 3})
            assert add_result.data == 5

            subtract_result = await client.call_tool(
                "b_subtract", {"a": 7, "b": 4}
            )
            assert subtract_result.data == 3

            # Collision pin: same unqualified name, different implementations.
            greet_a = await client.call_tool("a_greet", {"name": "x"})
            greet_b = await client.call_tool("b_greet", {"name": "x"})
            assert greet_a.data == "hello from A: x"
            assert greet_b.data == "hello from B: x"
            assert greet_a.data != greet_b.data, (
                "namespace must resolve the collision -- a_greet and "
                "b_greet must return the two distinct implementations"
            )


# --- TestRunAgentOverComposedServer ----------------------------------------


class TestRunAgentOverComposedServer:
    """run_agent routes a namespaced ToolCall transparently (mock-level).

    We stub out T0 with ``FakeT0`` (emits a forced ``ToolCall(name="a_add", …)``
    on turn 1, then a terminating ``Text`` on turn 2) and pass a **real**
    in-memory FastMCP ``Client`` over the composed server. If the namespaced
    name flows through ``execute_tool_calls → Client.call_tool`` untouched,
    FastMCP's Namespace transform strips the prefix and dispatches to the
    correct sub-server -- and ``run_agent`` never needs to know that
    namespaces exist.
    """

    async def test_run_agent_executes_namespaced_tool_call(self) -> None:
        tc = ToolCall(
            id="call-a-add",
            raw_arguments='{"a": 2, "b": 3}',
            raw_name="a_add",
            arguments={"a": 2, "b": 3},
            name="a_add",
        )
        episode = uuid4()
        fake_t0 = FakeT0(
            [
                FakeChatResponse(content=[tc], episode_id=episode),
                FakeChatResponse(
                    content=[Text(text="the answer is 5")], episode_id=episode
                ),
            ]
        )

        main = _build_composed_server()
        mcp_client: Client[Any] = Client(main)
        async with mcp_client:
            result = await run_agent(
                cast(AsyncTensorZeroGateway, fake_t0),
                mcp_client,
                "compute 2+3 using server A",
                "agent",
            )

        # The loop terminated on turn 2 with the canned text.
        assert result.response == "the answer is 5"
        assert result.turns == 2
        assert result.episode_id == str(episode)

        # Second inference saw the tool-call result as a single user message.
        second_messages = fake_t0.calls[1]["input"]["messages"]
        tool_result_block = second_messages[-1]["content"][0]
        # The namespaced name must be preserved on the ToolResult so T0 can
        # tie it back to the original ToolCall by id/name.
        assert tool_result_block.name == "a_add"
        assert tool_result_block.id == "call-a-add"
        # Serialized via serialize_tool_result -> json.dumps(5) -> "5"
        assert tool_result_block.result == "5"

    async def test_run_agent_executes_namespaced_tool_call_on_other_namespace(
        self,
    ) -> None:
        """Mirror of the above but routes through ``b_subtract`` so the mock
        covers **both** namespaces — complementing the integration test's
        forced-choice flows."""
        tc = ToolCall(
            id="call-b-sub",
            raw_arguments='{"a": 7, "b": 4}',
            raw_name="b_subtract",
            arguments={"a": 7, "b": 4},
            name="b_subtract",
        )
        fake_t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="the difference is 3")]),
            ]
        )

        main = _build_composed_server()
        mcp_client: Client[Any] = Client(main)
        async with mcp_client:
            result = await run_agent(
                cast(AsyncTensorZeroGateway, fake_t0),
                mcp_client,
                "compute 7-4 using server B",
                "agent",
            )

        assert result.response == "the difference is 3"
        tool_result_block = fake_t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert tool_result_block.name == "b_subtract"
        assert tool_result_block.result == "3"

    async def test_run_agent_surfaces_colliding_greet_from_correct_namespace(
        self,
    ) -> None:
        """When the ToolCall names ``a_greet``, the result reflects server A's
        implementation, not server B's."""
        tc = ToolCall(
            id="call-greet-a",
            raw_arguments='{"name": "world"}',
            raw_name="a_greet",
            arguments={"name": "world"},
            name="a_greet",
        )
        fake_t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="greeting delivered")]),
            ]
        )

        main = _build_composed_server()
        mcp_client: Client[Any] = Client(main)
        async with mcp_client:
            await run_agent(
                cast(AsyncTensorZeroGateway, fake_t0),
                mcp_client,
                "say hi via server A",
                "agent",
            )

        tool_result_block = fake_t0.calls[1]["input"]["messages"][-1]["content"][0]
        # ``.data`` is a str, so serialize_tool_result returns it unchanged
        # (the str passthrough branch, not json.dumps).
        assert tool_result_block.result == "hello from A: world"
        assert "hello from B" not in tool_result_block.result, (
            "a_greet must not leak B's implementation"
        )
