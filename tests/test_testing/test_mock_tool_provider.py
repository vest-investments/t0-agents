"""Unit tests for ``MockToolProvider``."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.types import Tool as MCPTool

from t0_agents.testing import MockToolProvider


class TestListTools:
    async def test_yields_real_mcp_tool_instances(self) -> None:
        provider = MockToolProvider(
            {"lookup": lambda args: "42", "echo": lambda args: args["text"]}
        )

        tools = await provider.list_tools()

        assert len(tools) == 2
        assert all(isinstance(t, MCPTool) for t in tools)
        assert {t.name for t in tools} == {"lookup", "echo"}

    async def test_default_schema_when_none_provided(self) -> None:
        provider = MockToolProvider({"t": lambda args: "x"})
        [tool] = await provider.list_tools()
        assert tool.inputSchema == {"type": "object", "properties": {}}

    async def test_custom_schema_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }
        provider = MockToolProvider(
            {"square": lambda args: args["n"] ** 2},
            schemas={"square": schema},
        )
        [tool] = await provider.list_tools()
        assert tool.inputSchema == schema


class TestCallTool:
    async def test_sync_handler(self) -> None:
        provider = MockToolProvider({"add": lambda args: args["a"] + args["b"]})

        result = await provider.call_tool("add", {"a": 2, "b": 3})

        assert result.is_error is False
        assert result.data == 5

    async def test_async_handler(self) -> None:
        async def handler(args: dict[str, Any]) -> str:
            return f"hello {args['name']}"

        provider = MockToolProvider({"greet": handler})
        result = await provider.call_tool("greet", {"name": "world"})

        assert result.is_error is False
        assert result.data == "hello world"

    async def test_unknown_tool_raises_keyerror(self) -> None:
        provider = MockToolProvider({"known": lambda args: "ok"})

        with pytest.raises(KeyError) as exc_info:
            await provider.call_tool("unknown", {})

        assert "unknown" in str(exc_info.value)
        assert "known" in str(exc_info.value)

    async def test_calls_log_records_name_and_arguments(self) -> None:
        provider = MockToolProvider({"greet": lambda args: args["name"]})

        await provider.call_tool("greet", {"name": "a"})
        await provider.call_tool("greet", {"name": "b"})

        assert provider.calls == [
            {"name": "greet", "arguments": {"name": "a"}},
            {"name": "greet", "arguments": {"name": "b"}},
        ]

    async def test_timeout_is_accepted_but_ignored(self) -> None:
        provider = MockToolProvider({"fast": lambda args: "done"})

        result = await provider.call_tool("fast", {}, timeout=0.001)

        assert result.data == "done"

    async def test_handler_raising_propagates(self) -> None:
        """A raising handler exercises the loop's error-translation path."""

        def failing(_args: dict[str, Any]) -> str:
            raise RuntimeError("tool is offline")

        provider = MockToolProvider({"flaky": failing})

        with pytest.raises(RuntimeError, match="tool is offline"):
            await provider.call_tool("flaky", {})


class TestSerializerCompatibility:
    """Exercise the adapter's serialize_tool_result on mock output."""

    async def test_string_data_serializes_verbatim(self) -> None:
        from t0_agents.mcp.adapter import serialize_tool_result

        provider = MockToolProvider({"t": lambda args: "hello"})
        result = await provider.call_tool("t", {})

        assert serialize_tool_result(result) == "hello"

    async def test_dict_data_serializes_as_json(self) -> None:
        from t0_agents.mcp.adapter import serialize_tool_result

        provider = MockToolProvider({"t": lambda args: {"n": 42, "ok": True}})
        result = await provider.call_tool("t", {})

        parsed = json.loads(serialize_tool_result(result))
        assert parsed == {"n": 42, "ok": True}
