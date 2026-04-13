"""Unit tests for the MCP tool adapter."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import ContentBlock, TextContent
from tensorzero import ToolCall

from t0_agents.mcp import build_tool_result, convert_tools, serialize_tool_result


class TestConvertTools:
    async def test_list_tools_returns_valid_structure(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.1: list_tools() returns objects with .name, .description, .inputSchema."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        assert len(tools) >= 2
        for tool in tools:
            assert isinstance(tool.name, str) and tool.name
            assert tool.description is None or isinstance(tool.description, str)
            assert isinstance(tool.inputSchema, dict)
            assert tool.inputSchema.get("type") == "object"

    async def test_convert_tools_produces_expected_shape(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.2: adapter maps list[Tool] → list[dict] with name/description/parameters."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        t0_tools = convert_tools(tools)
        assert len(t0_tools) == len(tools)

        for t0_tool in t0_tools:
            # Exactly these keys — no leaks of FastMCP-specific fields.
            assert set(t0_tool.keys()) == {"name", "description", "parameters"}
            assert isinstance(t0_tool["name"], str) and t0_tool["name"]
            # After the adapter, description is always a string (F1.6).
            assert isinstance(t0_tool["description"], str)
            assert isinstance(t0_tool["parameters"], dict)
            assert t0_tool["parameters"].get("type") == "object"

        # Sanity-check names carry through unchanged.
        names = {t["name"] for t in t0_tools}
        assert {"add", "greet", "search", "echo_no_docs"}.issubset(names)

    async def test_convert_tools_handles_none_description(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.6: tool with description=None at the FastMCP layer becomes ""."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        # FastMCP leaves description as None when there is no docstring and no
        # explicit description= on the decorator. This is the precondition for
        # F1.6 to be meaningful — if it changes upstream, the adapter is still
        # correct but this assertion anchors the scenario we're testing.
        echo = next(t for t in tools if t.name == "echo_no_docs")
        assert echo.description is None

        t0_tools = convert_tools(tools)
        echo_t0 = next(t for t in t0_tools if t["name"] == "echo_no_docs")
        assert echo_t0["description"] == ""


def _text(value: str) -> TextContent:
    """Build a TextContent block for fixture construction."""
    return TextContent(type="text", text=value)


def _make_result(
    *,
    data: Any = None,
    content: list[ContentBlock] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """Build a synthetic FastMCP CallToolResult for unit tests.

    FastMCP's ``CallToolResult`` is a ``@dataclass`` with three required fields
    (``content``, ``structured_content``, ``meta``) and two optional
    (``data``, ``is_error``). We default ``structured_content`` and ``meta`` to
    ``None`` because the serializer never looks at them — it relies on
    ``.data``, ``.content``, and ``.is_error``.
    """
    return CallToolResult(
        content=content or [],
        structured_content=None,
        meta=None,
        data=data,
        is_error=is_error,
    )


class TestSerializeToolResult:
    """Phase 2 — F2.2 through F2.6."""

    def test_serialize_str_passthrough(self) -> None:
        """F2.2: .data: str → the same str, untouched."""
        assert serialize_tool_result(_make_result(data="hello")) == "hello"

    def test_serialize_int_as_json(self) -> None:
        """F2.2: .data: int → JSON number. FastMCP unwraps primitives into .data;
        ``json.dumps(42) == "42"`` matches both the meta-plan and ``str(int)``."""
        assert serialize_tool_result(_make_result(data=42)) == "42"

    def test_serialize_float_as_json(self) -> None:
        """F2.2: .data: float → JSON number. ``json.dumps(3.14) == "3.14"``."""
        assert serialize_tool_result(_make_result(data=3.14)) == "3.14"

    def test_serialize_bool_as_json(self) -> None:
        """F2.2: .data: bool → JSON ``"true"`` / ``"false"``.

        This matches meta-plan v3 M2.1 and integration contract §4.3, which
        prescribe ``json.dumps(data, default=str)`` for any non-string ``.data``.
        ``json.dumps(True) == "true"`` (lowercase), unlike ``str(True) == "True"``.
        """
        assert serialize_tool_result(_make_result(data=True)) == "true"
        assert serialize_tool_result(_make_result(data=False)) == "false"

    def test_serialize_dict_as_json(self) -> None:
        """F2.3: .data: dict → json.dumps(data, default=str, ensure_ascii=False)."""
        out = serialize_tool_result(_make_result(data={"count": 42, "label": "ok"}))
        assert json.loads(out) == {"count": 42, "label": "ok"}

    def test_serialize_list_as_json(self) -> None:
        """F2.3 (extended): .data: list → json.dumps(...)."""
        out = serialize_tool_result(_make_result(data=[1, 2, 3]))
        assert json.loads(out) == [1, 2, 3]

    def test_serialize_dict_preserves_unicode(self) -> None:
        """F2.3: ensure_ascii=False preserves non-ASCII characters raw."""
        out = serialize_tool_result(_make_result(data={"saludo": "hóla ñ"}))
        assert "hóla ñ" in out
        assert json.loads(out) == {"saludo": "hóla ñ"}

    def test_serialize_complex_datetime_via_default_str(self) -> None:
        """F2.5: datetime nested in a dict is serialized via default=str."""
        out = serialize_tool_result(
            _make_result(data={"when": datetime(2024, 1, 15, 10, 30, 0)})
        )
        decoded = json.loads(out)
        # default=str produces the repr of the datetime (isoformat-like),
        # which the LLM can still read. We do not pin the exact format, just
        # that the date is there as a string.
        assert isinstance(decoded["when"], str)
        assert "2024-01-15" in decoded["when"]

    def test_serialize_complex_uuid_via_default_str(self) -> None:
        """F2.5: UUID nested in a dict is serialized via default=str."""
        uid = UUID("550e8400-e29b-41d4-a716-446655440000")
        out = serialize_tool_result(_make_result(data={"id": uid}))
        decoded = json.loads(out)
        assert decoded["id"] == str(uid)

    def test_serialize_none_fallback_to_content_text(self) -> None:
        """F2.4: .data is None → concatenate .content[*].text."""
        out = serialize_tool_result(
            _make_result(data=None, content=[_text("fallback result")])
        )
        assert out == "fallback result"

    def test_serialize_none_multiple_content_blocks_joined(self) -> None:
        """F2.4: multiple TextContent blocks are joined with a newline."""
        out = serialize_tool_result(
            _make_result(data=None, content=[_text("line A"), _text("line B")])
        )
        assert out == "line A\nline B"

    def test_serialize_none_no_content_returns_empty(self) -> None:
        """F2.4 edge: .data is None and no content → empty string."""
        assert serialize_tool_result(_make_result(data=None, content=[])) == ""

    def test_serialize_is_error_prefix(self) -> None:
        """F2.6: .is_error=True → 'ERROR: ' + extracted text."""
        out = serialize_tool_result(
            _make_result(is_error=True, content=[_text("division by zero")])
        )
        assert out.startswith("ERROR: ")
        assert "division by zero" in out

    def test_serialize_is_error_multiple_blocks(self) -> None:
        """F2.6: multiple error blocks are joined with a newline under the prefix."""
        out = serialize_tool_result(
            _make_result(
                is_error=True,
                content=[_text("step 1 failed"), _text("step 2 failed")],
            )
        )
        assert out == "ERROR: step 1 failed\nstep 2 failed"

    def test_serialize_is_error_empty_content(self) -> None:
        """F2.6 edge: .is_error=True with no textual content → generic message."""
        assert (
            serialize_tool_result(_make_result(is_error=True, content=[]))
            == "ERROR: Unknown tool error"
        )

    def test_serialize_is_error_wins_over_data(self) -> None:
        """F2.6: when is_error=True, .data is ignored (errors come from content)."""
        out = serialize_tool_result(
            _make_result(
                is_error=True,
                data={"leaked": "should not appear"},
                content=[_text("boom")],
            )
        )
        assert out == "ERROR: boom"
        assert "leaked" not in out


class TestBuildToolResult:
    """Phase 2 — F2.7 (shape of the constructed ToolResult)."""

    def _tool_call(
        self,
        *,
        name: str | None = "add",
        raw_name: str = "add",
        call_id: str = "call-123",
        arguments: dict[str, Any] | None = None,
    ) -> ToolCall:
        return ToolCall(
            name=name,
            raw_name=raw_name,
            arguments=arguments if arguments is not None else {"a": 2, "b": 3},
            raw_arguments='{"a": 2, "b": 3}',
            id=call_id,
        )

    def test_build_preserves_id_and_name(self) -> None:
        """F2.7 shape: id and name round-trip from ToolCall to ToolResult."""
        tc = self._tool_call(name="add", call_id="call-xyz")
        tr = build_tool_result(tc, _make_result(data=5))
        assert tr.id == "call-xyz"
        assert tr.name == "add"

    def test_build_result_is_serialized_string(self) -> None:
        """F2.7: the result field is always a string (serialized through
        serialize_tool_result)."""
        tc = self._tool_call()
        tr = build_tool_result(tc, _make_result(data={"sum": 5}))
        assert isinstance(tr.result, str)
        assert json.loads(tr.result) == {"sum": 5}

    def test_build_falls_back_to_raw_name_when_name_is_none(self) -> None:
        """F2.7 defensive: if the model hallucinated a name (name=None),
        ToolResult.name falls back to raw_name so T0 still receives a str."""
        tc = self._tool_call(name=None, raw_name="hallucinated_tool")
        tr = build_tool_result(tc, _make_result(data="ignored"))
        assert tr.name == "hallucinated_tool"
