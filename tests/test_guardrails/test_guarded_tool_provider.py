"""Unit tests for ``GuardedToolProvider`` semantics."""

from __future__ import annotations

from typing import Any, cast

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import Guardrail, GuardrailLogEntry, ToolProvider
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.runtime import GuardedToolProvider
from t0_agents.mcp.adapter import serialize_tool_result
from tests._fake_mcp import FakeMCPClient


def _ctx() -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message="go",
    )


def _ok_result(text: str = "done") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        meta=None,
        data=text,
        is_error=False,
    )


class _PassGuardrail:
    name = "pass"

    async def before_run(self, ctx: Any) -> None: ...

    async def before_tool(
        self, ctx: Any, tool_call: Any
    ) -> str | None:
        return None

    async def after_tool(
        self, ctx: Any, tool_call: Any, result_str: str
    ) -> None: ...

    async def after_run(self, ctx: Any, result: Any) -> None: ...


class _NonFatalError(_PassGuardrail):
    name = "raises_nonfatal"

    async def before_tool(
        self, ctx: Any, tool_call: Any
    ) -> str | None:
        raise ValueError("bug in guardrail")


class _FatalError(_PassGuardrail):
    name = "raises_fatal"

    async def before_tool(
        self, ctx: Any, tool_call: Any
    ) -> str | None:
        raise GuardrailBlockedError("hard stop")


class _AfterToolBoom(_PassGuardrail):
    name = "after_tool_boom"

    async def after_tool(
        self, ctx: Any, tool_call: Any, result_str: str
    ) -> None:
        raise RuntimeError("diagnostic blew up")


class TestGuardedToolProvider:
    async def test_list_tools_proxied_unchanged(self) -> None:
        tools = [
            MCPTool(
                name="x",
                description="d",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        inner = FakeMCPClient(tools=tools)
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _PassGuardrail())],
            _ctx(),
            [],
        )
        assert await wrapped.list_tools() == tools

    async def test_pass_path_calls_inner(self) -> None:
        ok = _ok_result()
        inner = FakeMCPClient(call_results={"x": ok})
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _PassGuardrail())],
            _ctx(),
            [],
        )
        assert await wrapped.call_tool("x", {}) is ok
        assert inner.call_tool_invocations == [("x", {}, None)]

    async def test_nonfatal_hook_exception_does_not_crash(self) -> None:
        """A buggy guardrail must not corrupt a tool call."""
        ok = _ok_result()
        inner = FakeMCPClient(call_results={"x": ok})
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _NonFatalError())],
            _ctx(),
            [],
        )
        result = await wrapped.call_tool("x", {})
        assert result is ok
        # Inner call still happened.
        assert inner.call_tool_invocations == [("x", {}, None)]

    async def test_fatal_hook_exception_propagates(self) -> None:
        inner = FakeMCPClient(call_results={"x": _ok_result()})
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _FatalError())],
            _ctx(),
            [],
        )
        import pytest

        with pytest.raises(GuardrailBlockedError):
            await wrapped.call_tool("x", {})
        # The inner call MUST NOT happen.
        assert inner.call_tool_invocations == []

    async def test_after_tool_exception_does_not_poison_result(self) -> None:
        ok = _ok_result()
        inner = FakeMCPClient(call_results={"x": ok})
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _AfterToolBoom())],
            _ctx(),
            [],
        )
        result = await wrapped.call_tool("x", {})
        assert result is ok

    async def test_short_circuit_serializes_with_single_error_prefix(self) -> None:
        class _Reject:
            name = "reject"

            async def before_run(self, ctx: Any) -> None: ...

            async def before_tool(
                self, ctx: Any, tool_call: Any
            ) -> str | None:
                # Caller-supplied prefix must not double up after serialisation.
                return "ERROR: policy violation"

            async def after_tool(
                self, ctx: Any, tool_call: Any, result_str: str
            ) -> None: ...

            async def after_run(self, ctx: Any, result: Any) -> None: ...

        inner = FakeMCPClient(call_results={"x": _ok_result()})
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _Reject())],
            _ctx(),
            [],
        )

        result = await wrapped.call_tool("x", {})
        serialised = serialize_tool_result(result)
        assert serialised.startswith("ERROR: ")
        assert serialised.count("ERROR: ") == 1
        assert "policy violation" in serialised

    async def test_log_sink_receives_entries(self) -> None:
        inner = FakeMCPClient(call_results={"x": _ok_result()})
        log: list[GuardrailLogEntry] = []
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _PassGuardrail())],
            _ctx(),
            log,
        )
        await wrapped.call_tool("x", {})
        # At least one "pass" entry recorded by the wrapper.
        assert any(e.kind == "pass" for e in log)

    async def test_after_tool_activations_logged(self) -> None:
        """Each successful ``after_tool`` activation appends a pass entry."""
        inner = FakeMCPClient(call_results={"x": _ok_result()})
        log: list[GuardrailLogEntry] = []
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _PassGuardrail())],
            _ctx(),
            log,
        )
        await wrapped.call_tool("x", {})
        after_tool_entries = [e for e in log if e.hook == "after_tool"]
        assert len(after_tool_entries) == 1
        entry = after_tool_entries[0]
        assert entry.kind == "pass"
        assert entry.guardrail_name == "pass"
        assert entry.tool_name == "x"

    async def test_after_tool_not_logged_when_hook_raises(self) -> None:
        """``after_tool`` audit entry only appears when the hook succeeds."""
        inner = FakeMCPClient(call_results={"x": _ok_result()})
        log: list[GuardrailLogEntry] = []
        wrapped = GuardedToolProvider(
            cast(ToolProvider, inner),
            [cast(Guardrail, _AfterToolBoom())],
            _ctx(),
            log,
        )
        await wrapped.call_tool("x", {})
        assert [e for e in log if e.hook == "after_tool"] == []
