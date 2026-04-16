"""Unit tests for ``require_approval``."""

from __future__ import annotations

from typing import Any

from fastmcp.client.elicitation import ElicitResult

from t0_agents.guardrails import require_approval
from t0_agents.guardrails.base import GuardrailContext


def _ctx() -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message="go",
    )


def _tool_call(name: str, arguments: dict[str, Any] | None = None) -> Any:
    return type(
        "_TC",
        (),
        {"name": name, "arguments": arguments or {"to": "a@b"}},
    )()


class TestRequireApproval:
    async def test_non_gated_tool_passes(self) -> None:
        g = require_approval(tools=["send_email"], handler=_always_accept)
        result = await g.before_tool(_ctx(), _tool_call("read_calendar"))
        assert result is None

    async def test_handler_accepts(self) -> None:
        g = require_approval(tools=["send_email"], handler=_always_accept)
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is None

    async def test_handler_declines(self) -> None:
        g = require_approval(tools=["send_email"], handler=_always_decline)
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is not None
        assert "declined" in result

    async def test_handler_cancels(self) -> None:
        g = require_approval(tools=["send_email"], handler=_always_cancel)
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is not None
        assert "declined" in result

    async def test_handler_is_none(self) -> None:
        g = require_approval(tools=["send_email"])
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is not None
        assert "no handler" in result.lower()

    async def test_handler_raises(self) -> None:
        async def _boom(*args: Any) -> Any:
            raise RuntimeError("connection lost")

        g = require_approval(tools=["send_email"], handler=_boom)
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is not None
        assert "handler failed" in result

    async def test_handler_returns_raw_dict_treated_as_accept(self) -> None:
        """``create_elicitation_callback`` parity: raw return → action='accept'."""
        async def _raw_dict(*args: Any) -> Any:
            return {"ok": True}

        g = require_approval(tools=["send_email"], handler=_raw_dict)
        result = await g.before_tool(_ctx(), _tool_call("send_email"))
        assert result is None

    async def test_prompt_template_is_rendered(self) -> None:
        seen: dict[str, str] = {}

        async def _capture(message: str, *args: Any) -> Any:
            seen["message"] = message
            return ElicitResult(action="accept", content={})

        g = require_approval(
            tools=["send_email"],
            handler=_capture,
            prompt_template="Send to {tool_name}? Args={arguments}",
        )
        await g.before_tool(_ctx(), _tool_call("send_email", {"to": "z"}))
        assert "Send to send_email" in seen["message"]
        assert "{'to': 'z'}" in seen["message"]


async def _always_accept(
    message: str, response_type: Any, params: Any, context: Any
) -> ElicitResult[Any]:
    return ElicitResult(action="accept", content={})


async def _always_decline(
    message: str, response_type: Any, params: Any, context: Any
) -> ElicitResult[Any]:
    return ElicitResult(action="decline")


async def _always_cancel(
    message: str, response_type: Any, params: Any, context: Any
) -> ElicitResult[Any]:
    return ElicitResult(action="cancel")
