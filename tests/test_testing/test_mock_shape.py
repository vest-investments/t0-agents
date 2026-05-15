"""Contract tests: mocks stay compatible with the real surfaces.

These tests are the line of defence against silent drift.  When a
future phase adds a kwarg to :meth:`AsyncTensorZeroGateway.inference`
or :meth:`fastmcp.Client.call_tool`, the mock might keep working for
old tests while silently ignoring the new kwarg — breaking tests that
rely on the new behaviour.  The shape tests inspect the real method
signatures and assert that every keyword the agent loop passes is
accepted by the mock.

The contract is narrow on purpose: we do not require the mock to
accept *every* kwarg the real method exposes, only the subset the
agent loop currently emits.  That subset lives in
:mod:`t0_agents.mcp.loop` (the canonical caller) and in
:mod:`t0_agents.core.types.ToolProvider` (the structural contract).
"""

from __future__ import annotations

import inspect
from typing import Any

from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import JsonInferenceResponse

from t0_agents.core.types import ToolProvider
from t0_agents.testing import MockGateway, MockToolProvider

# Kwargs the agent loop ( ``src/t0_agents/mcp/loop.py:310`` ) passes to
# ``t0.inference(...)``.  Update this list when the loop gains a new
# inference kwarg — the test will fail fast if the mock is behind.
_INFERENCE_KWARGS = [
    "function_name",
    "episode_id",
    "input",
    "additional_tools",
    "dryrun",
    "tool_choice",
    "tags",
    "namespace",
]

# Kwargs ``run_structured_inference`` (the JSON dispatch helper used by
# ``Agent._run_json``) passes to ``t0.inference(...)``
# (``src/t0_agents/core/structured.py``).  The JSON path drops the chat-
# only ``additional_tools`` / ``tool_choice`` and adds ``output_schema`` /
# ``params``.  Update when the helper gains a new kwarg.
_JSON_INFERENCE_KWARGS = [
    "function_name",
    "episode_id",
    "input",
    "output_schema",
    "params",
    "tags",
    "namespace",
    "dryrun",
]


class TestMockGatewayShape:
    async def test_inference_accepts_every_loop_kwarg(self) -> None:
        gateway = MockGateway.script([])
        kwargs: dict[str, Any] = {k: None for k in _INFERENCE_KWARGS}
        # MockGateway must return something usable — no raises.
        response = await gateway.inference(**kwargs)
        assert response is not None

    def test_real_inference_accepts_same_kwargs(self) -> None:
        """Sanity: every kwarg the loop passes is also a real inference kwarg.

        Guards against typos in the loop.  If this fails, the real
        method signature has drifted; treat it as a contract check, not
        a mock-drift check.
        """
        sig = inspect.signature(AsyncTensorZeroGateway.inference)
        real_params = set(sig.parameters)
        for kw in _INFERENCE_KWARGS:
            assert kw in real_params, (
                f"Loop passes {kw!r} to t0.inference, but the real "
                f"method signature no longer accepts it. Loop and "
                f"signature have drifted — update both together."
            )

    async def test_feedback_accepts_kwargs(self) -> None:
        gateway = MockGateway.script([])
        result = await gateway.feedback(metric_name="m", value=True)
        assert result.feedback_id is not None


class TestMockGatewayJsonShape:
    async def test_script_json_accepts_every_loop_kwarg(self) -> None:
        gateway = MockGateway.script_json([{"raw": '{"x": 1}'}])
        kwargs: dict[str, Any] = {k: None for k in _JSON_INFERENCE_KWARGS}
        # ``input`` cannot be None — the source ignores it but mypy /
        # the loop assume a dict; pass the minimal valid shape.
        kwargs["input"] = {"messages": []}
        kwargs["function_name"] = "extract"

        response = await gateway.inference(**kwargs)
        assert response is not None

    def test_real_inference_accepts_same_json_kwargs(self) -> None:
        """Sanity: every JSON kwarg the dispatch passes is also real.

        Guards against drift between ``run_structured_inference``'s
        forwarded kwargs and the real ``AsyncTensorZeroGateway.inference``
        signature — if either side changes, the test fails fast.
        """
        sig = inspect.signature(AsyncTensorZeroGateway.inference)
        real_params = set(sig.parameters)
        for kw in _JSON_INFERENCE_KWARGS:
            assert kw in real_params, (
                f"run_structured_inference passes {kw!r} to t0.inference, "
                f"but the real method signature no longer accepts it. "
                f"Helper and signature have drifted — update both together."
            )

    async def test_script_json_returns_required_response_fields(self) -> None:
        """``Agent._run_json`` reads five fields off the JSON response.

        The retry orchestration at ``src/t0_agents/core/agent.py`` reads
        ``output.raw``, ``output.parsed``, ``episode_id``, ``inference_id``,
        and ``usage`` (via the cost-tracker closure).  Pinning their
        presence on ``MockGateway.script_json``'s output prevents silent
        regression if the SDK reshapes ``JsonInferenceResponse``.
        """
        gateway = MockGateway.script_json(
            [{"raw": '{"x": 1}', "parsed": {"x": 1}}]
        )
        response = await gateway.inference(
            function_name="f", input={"messages": []}
        )
        assert isinstance(response, JsonInferenceResponse)
        assert response.output.raw == '{"x": 1}'
        assert response.output.parsed == {"x": 1}
        assert response.episode_id is not None
        assert response.inference_id is not None
        assert response.usage is not None


class TestMockToolProviderShape:
    async def test_satisfies_toolprovider_protocol(self) -> None:
        """MockToolProvider must match the runtime-checkable Protocol."""
        provider = MockToolProvider({"t": lambda args: "x"})
        assert isinstance(provider, ToolProvider)

    async def test_list_tools_returns_mcp_types_tool(self) -> None:
        provider = MockToolProvider({"t": lambda args: "x"})
        [tool] = await provider.list_tools()
        assert isinstance(tool, MCPTool)
        # Required downstream attributes (used by convert_tools).
        assert isinstance(tool.name, str)
        assert isinstance(tool.inputSchema, dict)

    async def test_call_tool_returns_real_calltoolresult(self) -> None:
        """serialize_tool_result requires a real CallToolResult shape."""
        provider = MockToolProvider({"t": lambda args: "hi"})
        result = await provider.call_tool("t", {})
        assert isinstance(result, CallToolResult)
        # Fields serialize_tool_result reads:
        assert hasattr(result, "is_error")
        assert hasattr(result, "data")
        assert hasattr(result, "content")
