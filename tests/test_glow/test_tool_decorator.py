"""Unit tests for :meth:`t0_agents.glow.App.tool` (M3).

Six cases (no TZ gateway; uses the App-owned in-process FastMCP
client):

* Tool with no ``Depends``: registers cleanly; the FastMCP-generated
  JSON Schema lists every parameter from the function signature.
* Tool with ``Depends(get_db)``: the auto-generated schema does NOT
  include the ``db`` parameter (only the public params).
* Calling the tool via the MCP client populates ``db`` from the
  resolver.
* ``app.dependency_overrides[get_db] = mock_db`` makes the tool see
  the mock (looked up at call time, not snapshotted).
* Tool docstring → MCP tool description.
* ``Depends(glow_agent_wrapper)`` inside ``@app.tool`` raises
  :class:`ConfigurationError` at decoration time with a pointer at
  ``@app.agent``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, Depends


def _fake_t0() -> AsyncTensorZeroGateway:
    """Return an opaque sentinel cast to :class:`AsyncTensorZeroGateway`.

    Tool-decorator tests never call into TensorZero — the MCP client
    talks to the App-owned in-process server directly.
    """
    return cast(AsyncTensorZeroGateway, object())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _input_schema_for(app: App, tool_name: str) -> dict[str, Any]:
    """Return the JSON-Schema for ``tool_name`` as advertised by MCP.

    Walks the MCP-side ``list_tools`` result rather than the FastMCP
    internal ``_components`` dict so the test verifies what the LLM
    would actually see.
    """
    advertised = await app._owned_client.list_tools()
    for tool in advertised:
        if tool.name == tool_name:
            return tool.inputSchema
    raise AssertionError(
        f"Tool {tool_name!r} not found in advertised tools: "
        f"{[t.name for t in advertised]!r}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_tool_without_depends_advertises_full_schema() -> None:
    """A plain ``@app.tool`` exposes every parameter in its schema."""
    app = App(t0=_fake_t0())

    @app.tool
    async def lookup_order(order_id: str) -> str:
        """Look up an order by ID."""
        return f"order:{order_id}"

    async with app.lifespan():
        schema = await _input_schema_for(app, "lookup_order")

    assert "order_id" in schema["properties"]
    # Type: string in JSON Schema.
    assert schema["properties"]["order_id"]["type"] == "string"
    assert "order_id" in schema.get("required", [])


async def test_tool_with_depends_strips_param_from_schema() -> None:
    """``Depends`` defaults are removed from the MCP JSON Schema.

    The LLM should never see the ``db`` parameter — it is purely an
    internal glow concern, populated by the resolver before the
    user's function is invoked.
    """
    app = App(t0=_fake_t0())

    def get_db() -> dict[str, str]:
        return {"name": "real_db"}

    @app.tool
    async def lookup_order(
        order_id: str,
        db: dict[str, str] = Depends(get_db),
    ) -> str:
        """Look up an order by ID using ``db``."""
        return f"{db['name']}:{order_id}"

    async with app.lifespan():
        schema = await _input_schema_for(app, "lookup_order")

    assert "order_id" in schema["properties"]
    # The Depends parameter is stripped — it is not advertised to the LLM.
    assert "db" not in schema["properties"]
    assert "db" not in schema.get("required", [])


async def test_tool_call_populates_depends_at_call_time() -> None:
    """Calling the tool via the MCP client resolves ``Depends`` values.

    Exercises the full path: FastMCP receives the tool call with
    only the public arguments, the glow wrapper resolves ``Depends``
    via the App's :class:`DependencyResolver`, merges them in, and
    invokes the user function with both LLM-supplied and injected
    kwargs.
    """
    app = App(t0=_fake_t0())

    def get_db() -> dict[str, str]:
        return {"name": "real_db"}

    @app.tool
    async def lookup_order(
        order_id: str,
        db: dict[str, str] = Depends(get_db),
    ) -> str:
        """Look up an order by ID using ``db``."""
        return f"{db['name']}:{order_id}"

    async with app.lifespan():
        result = await app._owned_client.call_tool(
            "lookup_order", {"order_id": "42"}
        )

    # FastMCP returns a CallToolResult; the textual payload lives in
    # ``result.content[0].text`` for string-returning tools.
    text = result.content[0].text  # type: ignore[union-attr]
    assert text == "real_db:42"


async def test_dependency_overrides_replaces_resolved_value() -> None:
    """``app.dependency_overrides[get_db] = mock_db`` redirects the tool's view.

    The override is installed AFTER ``@app.tool`` fires, so the
    resolver cannot have snapshotted the provider at registration —
    it must look up overrides on every call.
    """
    app = App(t0=_fake_t0())

    def get_db() -> dict[str, str]:
        return {"name": "real_db"}

    def mock_db() -> dict[str, str]:
        return {"name": "mock_db"}

    @app.tool
    async def lookup_order(
        order_id: str,
        db: dict[str, str] = Depends(get_db),
    ) -> str:
        """Look up an order by ID using ``db``."""
        return f"{db['name']}:{order_id}"

    # Install the override BEFORE entering the lifespan.  The resolver
    # consults ``dependency_overrides`` on every call, so order does
    # not matter — but doing it before catches any accidental "snapshot
    # at lifespan-entry" regression.
    app.dependency_overrides[get_db] = mock_db

    async with app.lifespan():
        result = await app._owned_client.call_tool(
            "lookup_order", {"order_id": "7"}
        )

    text = result.content[0].text  # type: ignore[union-attr]
    assert text == "mock_db:7"


async def test_tool_docstring_becomes_mcp_description() -> None:
    """FastMCP picks up ``fn.__doc__`` as the tool description by default.

    Verifies the wrapper preserves the docstring (which it does via
    ``wrapper.__doc__ = fn.__doc__``).
    """
    app = App(t0=_fake_t0())

    @app.tool
    async def search(query: str) -> str:
        """Search the corpus for matches."""
        return f"results-for:{query}"

    async with app.lifespan():
        advertised = await app._owned_client.list_tools()

    matches = [t for t in advertised if t.name == "search"]
    assert len(matches) == 1
    assert matches[0].description == "Search the corpus for matches."


def test_tool_rejects_depends_on_glow_agent_wrapper() -> None:
    """``Depends(agent_wrapper)`` inside ``@app.tool`` raises.

    The agent-wrapper case is reserved for ``@app.agent`` (sub-agent
    special case).  A tool that re-entered the agent loop via
    ``Depends(agent)`` would surface non-trivial re-entrance
    semantics; M3 rejects it eagerly with a pointer to the correct
    surface.
    """
    app = App(t0=_fake_t0())

    @app.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    with pytest.raises(ConfigurationError, match=r"Depends.*glow agent wrapper"):

        @app.tool
        async def bogus_tool(q: str, t: Any = Depends(tutor)) -> str:  # pragma: no cover
            return "never reached"
