"""Unit tests for the MCP tool adapter."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from mcp.types import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel
from tensorzero import ToolCall

from t0_agents.mcp import build_tool_result, convert_tools, serialize_tool_result


class TestConvertTools:
    async def test_list_tools_returns_valid_structure(
        self, mcp_client: Client[Any]
    ) -> None:
        """list_tools() returns objects with .name, .description, .inputSchema."""
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
        """Adapter maps list[Tool] → list[dict] with name/description/parameters."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        t0_tools = convert_tools(tools)
        assert len(t0_tools) == len(tools)

        for t0_tool in t0_tools:
            # Exactly these keys — no leaks of FastMCP-specific fields.
            assert set(t0_tool.keys()) == {"name", "description", "parameters"}
            assert isinstance(t0_tool["name"], str) and t0_tool["name"]
            # After the adapter, description is always a string.
            assert isinstance(t0_tool["description"], str)
            assert isinstance(t0_tool["parameters"], dict)
            assert t0_tool["parameters"].get("type") == "object"

        # Sanity-check names carry through unchanged.
        names = {t["name"] for t in t0_tools}
        assert {"add", "greet", "search", "echo_no_docs"}.issubset(names)

    async def test_convert_tools_handles_none_description(
        self, mcp_client: Client[Any]
    ) -> None:
        """Tool with description=None at the FastMCP layer becomes ""."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        # FastMCP leaves description as None when there is no docstring and no
        # explicit description= on the decorator. This is the precondition for
        # this test to be meaningful — if it changes upstream, the adapter is still
        # correct but this assertion anchors the scenario we're testing.
        echo = next(t for t in tools if t.name == "echo_no_docs")
        assert echo.description is None

        t0_tools = convert_tools(tools)
        echo_t0 = next(t for t in t0_tools if t["name"] == "echo_no_docs")
        assert echo_t0["description"] == ""

    def test_convert_tools_none_description_on_synthesized_tool(self) -> None:
        """description=None on a synthesized ``MCPTool`` coerces to ``""``
        without depending on live FastMCP behavior.

        Complements ``test_convert_tools_handles_none_description`` above by
        pinning the adapter's ``or ""`` coercion directly — future FastMCP
        versions could change how they surface missing docstrings, but the
        adapter contract (never send ``None`` to T0) stays.
        """
        tool = MCPTool(
            name="no_desc",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )
        t0_tools = convert_tools([tool])
        assert len(t0_tools) == 1
        assert t0_tools[0]["description"] == ""

    def test_convert_tools_empty_description_stays_empty(self) -> None:
        """description="" stays "" after conversion.

        The adapter uses ``""`` rather than a placeholder like
        ``"No description available."`` for empty descriptions.
        This test locks that decision so a future refactor cannot silently
        inject a placeholder string the LLM would see.
        """
        tool = MCPTool(
            name="blank_desc",
            description="",
            inputSchema={"type": "object", "properties": {}},
        )
        t0_tools = convert_tools([tool])
        assert t0_tools[0]["description"] == ""


def _text(value: str) -> TextContent:
    """Build a TextContent block for fixture construction."""
    return TextContent(type="text", text=value)


def _image(mime: str = "image/png") -> ImageContent:
    return ImageContent(type="image", data="aGVsbG8=", mimeType=mime)


def _audio(mime: str = "audio/mpeg") -> AudioContent:
    return AudioContent(type="audio", data="aGVsbG8=", mimeType=mime)


def _resource() -> EmbeddedResource:
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri="file:///x.txt", text="body", mimeType="text/plain"
        ),
    )


def _make_result(
    *,
    data: Any = None,
    content: list[ContentBlock] | None = None,
    structured_content: dict[str, Any] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """Build a synthetic FastMCP CallToolResult for unit tests.

    FastMCP's ``CallToolResult`` is a ``@dataclass`` with fields ``content``,
    ``structured_content``, ``meta``, ``data``, ``is_error``. ``meta`` is
    always ``None`` here (the serializer ignores it); pass ``structured_content``
    to exercise the ``.structured_content`` step of the serializer's mapping
    policy (a schema-less object result carries it with ``data=None``).
    """
    return CallToolResult(
        content=content or [],
        structured_content=structured_content,
        meta=None,
        data=data,
        is_error=is_error,
    )


class TestSerializeToolResult:
    """Serialization of FastMCP CallToolResult to string."""

    def test_serialize_str_passthrough(self) -> None:
        """.data: str → the same str, untouched."""
        assert serialize_tool_result(_make_result(data="hello")) == "hello"

    def test_serialize_int_as_json(self) -> None:
        """.data: int → JSON number. FastMCP unwraps primitives into .data;
        ``json.dumps(42) == "42"`` matches ``str(int)``."""
        assert serialize_tool_result(_make_result(data=42)) == "42"

    def test_serialize_float_as_json(self) -> None:
        """.data: float → JSON number. ``json.dumps(3.14) == "3.14"``."""
        assert serialize_tool_result(_make_result(data=3.14)) == "3.14"

    def test_serialize_bool_as_json(self) -> None:
        """.data: bool → JSON ``"true"`` / ``"false"``.

        ``json.dumps(data, default=str)`` is used for any non-string ``.data``.
        ``json.dumps(True) == "true"`` (lowercase), unlike ``str(True) == "True"``.
        """
        assert serialize_tool_result(_make_result(data=True)) == "true"
        assert serialize_tool_result(_make_result(data=False)) == "false"

    def test_serialize_dict_as_json(self) -> None:
        """.data: dict → json.dumps(data, default=str, ensure_ascii=False)."""
        out = serialize_tool_result(_make_result(data={"count": 42, "label": "ok"}))
        assert json.loads(out) == {"count": 42, "label": "ok"}

    def test_serialize_list_as_json(self) -> None:
        """.data: list → json.dumps(...)."""
        out = serialize_tool_result(_make_result(data=[1, 2, 3]))
        assert json.loads(out) == [1, 2, 3]

    def test_serialize_dict_preserves_unicode(self) -> None:
        """ensure_ascii=False preserves non-ASCII characters raw."""
        out = serialize_tool_result(_make_result(data={"saludo": "hóla ñ"}))
        assert "hóla ñ" in out
        assert json.loads(out) == {"saludo": "hóla ñ"}

    def test_serialize_complex_datetime_via_default_str(self) -> None:
        """datetime nested in a dict is serialized via default=str."""
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
        """UUID nested in a dict is serialized via default=str."""
        uid = UUID("550e8400-e29b-41d4-a716-446655440000")
        out = serialize_tool_result(_make_result(data={"id": uid}))
        decoded = json.loads(out)
        assert decoded["id"] == str(uid)

    def test_serialize_none_fallback_to_content_text(self) -> None:
        """.data is None → concatenate .content[*].text."""
        out = serialize_tool_result(
            _make_result(data=None, content=[_text("fallback result")])
        )
        assert out == "fallback result"

    def test_serialize_none_multiple_content_blocks_joined(self) -> None:
        """Multiple TextContent blocks are joined with a newline."""
        out = serialize_tool_result(
            _make_result(data=None, content=[_text("line A"), _text("line B")])
        )
        assert out == "line A\nline B"

    def test_serialize_none_no_content_returns_empty(self) -> None:
        """.data is None and no content → empty string."""
        assert serialize_tool_result(_make_result(data=None, content=[])) == ""

    def test_serialize_is_error_prefix(self) -> None:
        """.is_error=True → 'ERROR: ' + extracted text."""
        out = serialize_tool_result(
            _make_result(is_error=True, content=[_text("division by zero")])
        )
        assert out.startswith("ERROR: ")
        assert "division by zero" in out

    def test_serialize_is_error_multiple_blocks(self) -> None:
        """Multiple error blocks are joined with a newline under the prefix."""
        out = serialize_tool_result(
            _make_result(
                is_error=True,
                content=[_text("step 1 failed"), _text("step 2 failed")],
            )
        )
        assert out == "ERROR: step 1 failed\nstep 2 failed"

    def test_serialize_is_error_empty_content(self) -> None:
        """.is_error=True with no textual content → generic message."""
        assert (
            serialize_tool_result(_make_result(is_error=True, content=[]))
            == "ERROR: Unknown tool error"
        )

    def test_serialize_is_error_wins_over_data(self) -> None:
        """When is_error=True, .data is ignored (errors come from content)."""
        out = serialize_tool_result(
            _make_result(
                is_error=True,
                data={"leaked": "should not appear"},
                content=[_text("boom")],
            )
        )
        assert out == "ERROR: boom"
        assert "leaked" not in out

    def test_serialize_image_only_returns_placeholder(self) -> None:
        """A result whose only content is an image is no longer dropped to "";
        it becomes a loud placeholder naming the omitted content."""
        out = serialize_tool_result(_make_result(data=None, content=[_image()]))
        assert out == (
            "[tool returned 1 image (image/png); "
            "non-text content is not forwarded to the model]"
        )

    def test_serialize_audio_only_returns_placeholder(self) -> None:
        out = serialize_tool_result(_make_result(data=None, content=[_audio()]))
        assert out.startswith("[tool returned ")
        assert "1 audio clip (audio/mpeg)" in out

    def test_serialize_resource_only_returns_placeholder(self) -> None:
        out = serialize_tool_result(_make_result(data=None, content=[_resource()]))
        assert out.startswith("[tool returned ")
        assert "1 embedded resource" in out

    def test_serialize_multiple_non_text_are_counted(self) -> None:
        out = serialize_tool_result(
            _make_result(data=None, content=[_image(), _image()])
        )
        assert "2 images" in out

    def test_serialize_mixed_keeps_text_and_notes_the_dropped_rest(self) -> None:
        """Mixed text + non-text: the text is preserved verbatim as the prefix
        and the omitted non-text is appended as a note — never silently lost."""
        out = serialize_tool_result(
            _make_result(data=None, content=[_text("here is the chart"), _image()])
        )
        lines = out.split("\n")
        assert lines[0] == "here is the chart"
        assert lines[1] == (
            "[tool returned 1 image (image/png); "
            "non-text content is not forwarded to the model]"
        )

    def test_serialize_non_text_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            serialize_tool_result(_make_result(data=None, content=[_image()]))
        assert any("non-text content" in record.message for record in caplog.records)


class TestBuildToolResult:
    """Shape of the constructed ToolResult."""

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
        """id and name round-trip from ToolCall to ToolResult."""
        tc = self._tool_call(name="add", call_id="call-xyz")
        tr = build_tool_result(tc, _make_result(data=5))
        assert tr.id == "call-xyz"
        assert tr.name == "add"

    def test_build_result_is_serialized_string(self) -> None:
        """The result field is always a string (serialized through
        serialize_tool_result)."""
        tc = self._tool_call()
        tr = build_tool_result(tc, _make_result(data={"sum": 5}))
        assert isinstance(tr.result, str)
        assert json.loads(tr.result) == {"sum": 5}

    def test_build_falls_back_to_raw_name_when_name_is_none(self) -> None:
        """Defensive: if the model hallucinated a name (name=None),
        ToolResult.name falls back to raw_name so T0 still receives a str."""
        tc = self._tool_call(name=None, raw_name="hallucinated_tool")
        tr = build_tool_result(tc, _make_result(data="ignored"))
        assert tr.name == "hallucinated_tool"


# The exact model-visible string for every result shape that exists BEFORE the
# structured-content work. This is the prompt-stability gate for the v2.2.0
# upgrade: the string a tool result serializes to is part of the LLM prompt, so
# these bytes must not drift across the release. A change that moves any entry
# is a prompt change — justify it in the PR, never edit the frozen value to make
# the test pass.
_PROMPT_STABILITY_GOLDEN: list[tuple[str, CallToolResult, str]] = [
    ("data_str", _make_result(data="hello"), "hello"),
    ("data_int", _make_result(data=42), "42"),
    ("data_float", _make_result(data=3.14), "3.14"),
    ("data_bool_true", _make_result(data=True), "true"),
    ("data_bool_false", _make_result(data=False), "false"),
    (
        "data_dict",
        _make_result(data={"count": 42, "label": "ok"}),
        '{"count": 42, "label": "ok"}',
    ),
    ("data_list", _make_result(data=[1, 2, 3]), "[1, 2, 3]"),
    ("data_unicode", _make_result(data={"saludo": "hóla ñ"}), '{"saludo": "hóla ñ"}'),
    ("text_single", _make_result(content=[_text("fallback result")]), "fallback result"),
    (
        "text_multi",
        _make_result(content=[_text("line A"), _text("line B")]),
        "line A\nline B",
    ),
    ("empty_no_content", _make_result(content=[]), ""),
    (
        "is_error_single",
        _make_result(is_error=True, content=[_text("division by zero")]),
        "ERROR: division by zero",
    ),
    (
        "is_error_multi",
        _make_result(
            is_error=True, content=[_text("step 1 failed"), _text("step 2 failed")]
        ),
        "ERROR: step 1 failed\nstep 2 failed",
    ),
    ("is_error_empty", _make_result(is_error=True, content=[]), "ERROR: Unknown tool error"),
]


class TestPromptStabilityGolden:
    """Freeze the model-visible string for every pre-structured-content shape."""

    @pytest.mark.parametrize(
        ("result", "expected"),
        [pytest.param(r, s, id=name) for name, r, s in _PROMPT_STABILITY_GOLDEN],
    )
    def test_serialized_string_is_byte_stable(
        self, result: CallToolResult, expected: str
    ) -> None:
        assert serialize_tool_result(result) == expected


class _BoundaryWeather(BaseModel):
    """Module-level so FastMCP can resolve the tool's return annotation
    (pydantic evaluates it against the function's module globals)."""

    temp: int
    city: str


class TestSerializeThroughFastMCPBoundary:
    """Real FastMCP results (not hand-built) pin the serializer against the
    schema hydration the client actually performs — the framework-boundary
    gate. Object results carry ``structured_content`` (the raw dict the event
    exposes); the model-visible string still comes from ``.data``, which
    FastMCP hydrates from ``structured_content`` even without an output schema,
    so a schema-less dict and a schema'd one serialise identically."""

    async def test_object_results_are_stable_and_carry_structured_content(
        self,
    ) -> None:
        server = FastMCP("adapter-boundary")

        @server.tool
        def as_dict() -> dict[str, Any]:
            return {"count": 42, "label": "ok"}

        @server.tool
        def as_scalar() -> int:
            return 42

        @server.tool
        def as_list() -> list[int]:
            return [1, 2, 3]

        @server.tool
        def as_model() -> _BoundaryWeather:
            return _BoundaryWeather(temp=22, city="Bogotá")

        async with Client(server) as client:
            dict_r = await client.call_tool("as_dict", {})
            scalar_r = await client.call_tool("as_scalar", {})
            list_r = await client.call_tool("as_list", {})
            model_r = await client.call_tool("as_model", {})

        # The model-visible string: byte-identical to the pre-upgrade behavior.
        assert serialize_tool_result(dict_r) == '{"count": 42, "label": "ok"}'
        assert serialize_tool_result(scalar_r) == "42"
        assert serialize_tool_result(list_r) == "[1, 2, 3]"
        # Known, prompt-stable wart: a pydantic-model .data stringifies via
        # default=str (its repr), NOT clean JSON. The clean dict lives on
        # structured_content (→ ToolResultEvent.structured), not in this string.
        model_str = serialize_tool_result(model_r)
        assert model_str != '{"temp": 22, "city": "Bogotá"}'
        assert "22" in model_str and "Bogotá" in model_str

        # structured_content is the clean dict for every object result — the raw
        # payload ToolResultEvent.structured surfaces.
        assert dict_r.structured_content == {"count": 42, "label": "ok"}
        assert model_r.structured_content == {"temp": 22, "city": "Bogotá"}
        assert scalar_r.structured_content == {"result": 42}
