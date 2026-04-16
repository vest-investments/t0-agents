"""Structural checks for the ``Guardrail`` Protocol.

Every built-in factory returns an object that satisfies the
``runtime_checkable`` Protocol; ad-hoc objects missing any of the four
hooks do not pass ``isinstance`` narrowing.
"""

from __future__ import annotations

from t0_agents.core.types import Guardrail
from t0_agents.guardrails import (
    block_tools,
    max_cost,
    max_tool_calls,
    require_approval,
)


class TestProtocolCompliance:
    def test_require_approval_is_guardrail(self) -> None:
        assert isinstance(require_approval(tools=["x"]), Guardrail)

    def test_max_cost_is_guardrail(self) -> None:
        assert isinstance(max_cost(per_episode=1.0), Guardrail)

    def test_max_tool_calls_is_guardrail(self) -> None:
        assert isinstance(max_tool_calls(per_run=5), Guardrail)

    def test_block_tools_is_guardrail(self) -> None:
        assert isinstance(block_tools(["danger"]), Guardrail)

    def test_builtins_expose_name_field(self) -> None:
        for g in (
            require_approval(tools=["x"]),
            max_cost(per_episode=1.0),
            max_tool_calls(per_run=5),
            block_tools(["danger"]),
        ):
            assert isinstance(g.name, str)
            assert g.name

    def test_missing_hook_fails_protocol(self) -> None:
        class Incomplete:
            name = "incomplete"

            async def before_run(self, ctx: object) -> None: ...
            async def before_tool(
                self, ctx: object, tool_call: object
            ) -> str | None:
                return None
            # after_tool and after_run intentionally missing.

        assert not isinstance(Incomplete(), Guardrail)
