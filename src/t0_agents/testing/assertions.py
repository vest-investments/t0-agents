"""Shared vocabulary for agent tests.

Every assertion helper takes keyword-only arguments so test readers
know exactly what the check is pinning down, and so a new parameter
(e.g. a tighter failure message) can land without breaking callers.

The tool-call assertions accept **either** a :class:`MockGateway`
(walks its :attr:`MockGateway.responses` for
:class:`~tensorzero.ToolCall` blocks) **or** a :class:`MockToolProvider`
(walks its per-call log).  Use the gateway path to assert what the
*model* emitted; use the tool-provider path to assert what actually
executed after guardrails + the loop got a say.
"""

from __future__ import annotations

from typing import Any

from tensorzero import ToolCall

from t0_agents.core.types import AgentResult
from t0_agents.testing.mocks import MockGateway, MockToolProvider


def _collect_gateway_tool_calls(gateway: MockGateway) -> list[ToolCall]:
    """Return every :class:`ToolCall` the gateway emitted across all turns."""
    calls: list[ToolCall] = []
    for response in gateway.responses:
        calls.extend(b for b in response.content if isinstance(b, ToolCall))
    return calls


def assert_tool_called(
    name: str,
    *,
    gateway: MockGateway | None = None,
    tool_provider: MockToolProvider | None = None,
) -> None:
    """Assert that *name* was called at least once.

    Exactly one of *gateway* or *tool_provider* must be supplied:

    * ``gateway=`` walks :attr:`MockGateway.responses` and looks for
      :class:`ToolCall` content blocks whose ``name`` equals *name* —
      i.e. what the model *emitted*, regardless of whether guardrails
      or the loop short-circuited it.
    * ``tool_provider=`` walks :attr:`MockToolProvider.calls` — i.e.
      what actually *executed* after the full pipeline.
    """
    if (gateway is None) == (tool_provider is None):
        raise ValueError(
            "assert_tool_called: pass exactly one of gateway= or tool_provider="
        )
    if tool_provider is not None:
        matched = [c for c in tool_provider.calls if c["name"] == name]
        assert matched, (
            f"Tool {name!r} was not called. "
            f"Observed calls: {[c['name'] for c in tool_provider.calls]!r}"
        )
        return
    assert gateway is not None  # narrow for mypy
    emitted = _collect_gateway_tool_calls(gateway)
    matched_names = [tc.name for tc in emitted if tc.name == name]
    assert matched_names, (
        f"Gateway never emitted a ToolCall for {name!r}. "
        f"Observed: {[tc.name for tc in emitted]!r}"
    )


def assert_tool_not_called(
    name: str,
    *,
    gateway: MockGateway | None = None,
    tool_provider: MockToolProvider | None = None,
) -> None:
    """Assert that *name* was never called.

    Accepts the same exclusive *gateway* / *tool_provider* pair as
    :func:`assert_tool_called`.
    """
    if (gateway is None) == (tool_provider is None):
        raise ValueError(
            "assert_tool_not_called: pass exactly one of "
            "gateway= or tool_provider="
        )
    if tool_provider is not None:
        matched = [c for c in tool_provider.calls if c["name"] == name]
        assert not matched, (
            f"Tool {name!r} should not have been called, "
            f"but was called {len(matched)} time(s) with arguments: "
            f"{[c['arguments'] for c in matched]!r}"
        )
        return
    assert gateway is not None
    emitted = _collect_gateway_tool_calls(gateway)
    matched_emitted = [tc for tc in emitted if tc.name == name]
    assert not matched_emitted, (
        f"Gateway should not have emitted ToolCall for {name!r}, "
        f"but emitted {len(matched_emitted)} time(s)"
    )


def assert_tool_args_equal(
    name: str,
    expected_args: dict[str, Any],
    *,
    tool_provider: MockToolProvider,
) -> None:
    """Assert that the most recent call to *name* used *expected_args*.

    Only supports the tool-provider path — the gateway path would
    compare against the *parsed* ``ToolCall.arguments`` which can be
    ``None`` when the model emits invalid JSON; the tool-provider path
    guarantees the dict the loop built before dispatch.
    """
    matching = [c for c in tool_provider.calls if c["name"] == name]
    assert matching, (
        f"Tool {name!r} was not called; cannot check arguments. "
        f"Observed calls: {[c['name'] for c in tool_provider.calls]!r}"
    )
    actual = matching[-1]["arguments"]
    assert actual == expected_args, (
        f"Tool {name!r} last-call arguments mismatch.\n"
        f"  expected: {expected_args!r}\n"
        f"  actual:   {actual!r}"
    )


def assert_max_turns(result: AgentResult, *, at_most: int) -> None:
    """Assert that *result* completed in at most *at_most* turns."""
    assert result.turns <= at_most, (
        f"Agent used {result.turns} turn(s); expected at most {at_most}"
    )


def assert_response_contains(result: AgentResult, text: str) -> None:
    """Case-insensitive substring check on ``result.response``."""
    assert text.lower() in result.response.lower(), (
        f"Expected response to contain {text!r}; "
        f"got: {result.response!r}"
    )


__all__ = [
    "assert_max_turns",
    "assert_response_contains",
    "assert_tool_args_equal",
    "assert_tool_called",
    "assert_tool_not_called",
]
