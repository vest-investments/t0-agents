"""Unit tests for the assertion helpers."""

from __future__ import annotations

import pytest
from tensorzero import Text, ToolCall

from t0_agents.core.types import AgentResult
from t0_agents.testing import (
    MockGateway,
    MockToolProvider,
    assert_max_turns,
    assert_response_contains,
    assert_tool_args_equal,
    assert_tool_called,
    assert_tool_not_called,
)


def _make_result(**overrides: object) -> AgentResult:
    defaults: dict[str, object] = {
        "response": "",
        "episode_id": None,
        "inference_ids": [],
        "turns": 1,
    }
    defaults.update(overrides)
    return AgentResult(**defaults)  # type: ignore[arg-type]


class TestAssertToolCalled:
    async def test_tool_provider_path_passes(self) -> None:
        provider = MockToolProvider({"add": lambda args: "ok"})
        await provider.call_tool("add", {})

        assert_tool_called("add", tool_provider=provider)

    async def test_tool_provider_path_fails_when_never_called(self) -> None:
        provider = MockToolProvider({"add": lambda args: "ok"})

        with pytest.raises(AssertionError, match="was not called"):
            assert_tool_called("add", tool_provider=provider)

    async def test_gateway_path_passes(self) -> None:
        gateway = MockGateway.script(
            [
                [
                    ToolCall(
                        id="c1",
                        raw_name="search",
                        raw_arguments="{}",
                        name="search",
                        arguments={},
                    )
                ],
                [Text(text="done")],
            ]
        )
        await gateway.inference(function_name="f")

        assert_tool_called("search", gateway=gateway)

    async def test_gateway_path_fails_when_never_emitted(self) -> None:
        gateway = MockGateway.script([[Text(text="no tools here")]])
        await gateway.inference(function_name="f")

        with pytest.raises(AssertionError, match="never emitted"):
            assert_tool_called("search", gateway=gateway)

    def test_requires_exactly_one_of_gateway_or_tool_provider(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            assert_tool_called("x")

        with pytest.raises(ValueError, match="exactly one"):
            assert_tool_called(
                "x",
                gateway=MockGateway.script([]),
                tool_provider=MockToolProvider({}),
            )


class TestAssertToolNotCalled:
    async def test_passes_when_not_called(self) -> None:
        provider = MockToolProvider({"other": lambda args: "ok"})
        await provider.call_tool("other", {})

        assert_tool_not_called("forbidden", tool_provider=provider)

    async def test_fails_when_called(self) -> None:
        provider = MockToolProvider({"forbidden": lambda args: "ok"})
        await provider.call_tool("forbidden", {})

        with pytest.raises(AssertionError, match="should not have been called"):
            assert_tool_not_called("forbidden", tool_provider=provider)

    async def test_gateway_path(self) -> None:
        gateway = MockGateway.script([[Text(text="text only")]])
        await gateway.inference(function_name="f")

        assert_tool_not_called("any", gateway=gateway)


class TestAssertToolArgsEqual:
    async def test_passes_on_last_call(self) -> None:
        provider = MockToolProvider({"add": lambda args: args["a"] + args["b"]})
        await provider.call_tool("add", {"a": 1, "b": 1})
        await provider.call_tool("add", {"a": 2, "b": 3})

        assert_tool_args_equal("add", {"a": 2, "b": 3}, tool_provider=provider)

    async def test_fails_when_never_called(self) -> None:
        provider = MockToolProvider({"add": lambda args: "ok"})

        with pytest.raises(AssertionError, match="was not called"):
            assert_tool_args_equal("add", {"a": 1}, tool_provider=provider)

    async def test_fails_when_args_differ(self) -> None:
        provider = MockToolProvider({"add": lambda args: "ok"})
        await provider.call_tool("add", {"a": 1})

        with pytest.raises(AssertionError, match="mismatch"):
            assert_tool_args_equal("add", {"a": 2}, tool_provider=provider)


class TestAssertMaxTurns:
    def test_passes_when_under_limit(self) -> None:
        assert_max_turns(_make_result(turns=3), at_most=5)

    def test_passes_at_exact_limit(self) -> None:
        assert_max_turns(_make_result(turns=5), at_most=5)

    def test_fails_when_over_limit(self) -> None:
        with pytest.raises(AssertionError, match=r"Agent used \d+ turn"):
            assert_max_turns(_make_result(turns=6), at_most=5)


class TestAssertResponseContains:
    def test_case_insensitive_match(self) -> None:
        assert_response_contains(
            _make_result(response="The Result IS 42"), "result is 42"
        )

    def test_fails_when_missing(self) -> None:
        with pytest.raises(AssertionError):
            assert_response_contains(_make_result(response="no match"), "answer")
