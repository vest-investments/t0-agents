"""Pipeline composition + ordering + short-circuit + double-call detection.

Direct unit tests of :func:`build_pipeline` and :class:`_NextGuard`.
No Agent / Run involved — pure callable composition tests.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from t0_agents.core.exceptions import T0AgentsError
from t0_agents.core.middleware.base import (
    Next,
    RunContext,
    build_pipeline,
)
from t0_agents.core.types import AgentResult


def _make_ctx() -> RunContext:
    """Minimal RunContext for pipeline tests — no gateway, no tools."""
    return RunContext(
        config=cast(Any, None),
        user_message="msg",
        episode_id=None,
        tags=None,
        system=None,
        system_arguments=None,
        extra_body=None,
        output=None,
        t0=cast(Any, None),
        tool_provider=None,
    )


class _LoggingMiddleware:
    """Append ``name`` to ``ctx.extras['log']`` before and after next_()."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult:
        ctx.extras.setdefault("log", []).append(f"{self.name}:before")
        result = await next_(ctx)
        ctx.extras["log"].append(f"{self.name}:after")
        return result


async def _terminal(ctx: RunContext) -> AgentResult:
    ctx.extras.setdefault("log", []).append("terminal")
    return AgentResult(response="ok", episode_id=None, inference_ids=[], turns=1)


class TestBuildPipelineOrder:
    async def test_outermost_first_three_middlewares(self) -> None:
        pipeline = build_pipeline(
            [_LoggingMiddleware("A"), _LoggingMiddleware("B"), _LoggingMiddleware("C")],
            _terminal,
        )
        ctx = _make_ctx()

        await pipeline(ctx)

        # Outermost-first: A enters first, exits last.  Reads like a
        # nested ``with``: A → B → C → terminal → C → B → A.
        assert ctx.extras["log"] == [
            "A:before",
            "B:before",
            "C:before",
            "terminal",
            "C:after",
            "B:after",
            "A:after",
        ]

    async def test_empty_middleware_list_returns_terminal_unchanged(self) -> None:
        # An empty pipeline reduces to the terminal — useful for
        # ``middleware=[]`` (the explicit "no middleware" override).
        pipeline = build_pipeline([], _terminal)
        ctx = _make_ctx()

        result = await pipeline(ctx)

        assert ctx.extras["log"] == ["terminal"]
        assert result.response == "ok"

    async def test_single_middleware_wraps_terminal(self) -> None:
        pipeline = build_pipeline([_LoggingMiddleware("solo")], _terminal)
        ctx = _make_ctx()

        await pipeline(ctx)

        assert ctx.extras["log"] == ["solo:before", "terminal", "solo:after"]


class TestShortCircuit:
    async def test_middleware_returning_without_next_skips_inner(self) -> None:
        terminal_called = False

        async def tracking_terminal(_ctx: RunContext) -> AgentResult:
            nonlocal terminal_called
            terminal_called = True
            return AgentResult(
                response="terminal", episode_id=None, inference_ids=[], turns=1,
            )

        class ShortCircuit:
            name = "short_circuit"

            async def __call__(
                self, ctx: RunContext, next_: Next,
            ) -> AgentResult:
                # Don't call next_(ctx).  Return a synthesised result.
                return AgentResult(
                    response="cached",
                    episode_id=None,
                    inference_ids=[],
                    turns=0,
                )

        pipeline = build_pipeline([ShortCircuit()], tracking_terminal)
        ctx = _make_ctx()

        result = await pipeline(ctx)

        assert result.response == "cached"
        assert terminal_called is False


class TestDoubleCallDetection:
    async def test_calling_next_twice_raises_t0agents_error(self) -> None:
        class DoubleCaller:
            name = "double_caller"

            async def __call__(
                self, ctx: RunContext, next_: Next,
            ) -> AgentResult:
                await next_(ctx)
                await next_(ctx)  # second call must raise
                return AgentResult(
                    response="never", episode_id=None, inference_ids=[], turns=0,
                )

        pipeline = build_pipeline([DoubleCaller()], _terminal)
        ctx = _make_ctx()

        with pytest.raises(T0AgentsError, match="more than once"):
            await pipeline(ctx)

    async def test_error_message_names_offending_middleware(self) -> None:
        class NamedDouble:
            name = "my_bad_middleware"

            async def __call__(
                self, ctx: RunContext, next_: Next,
            ) -> AgentResult:
                await next_(ctx)
                await next_(ctx)
                return AgentResult(
                    response="never", episode_id=None, inference_ids=[], turns=0,
                )

        pipeline = build_pipeline([NamedDouble()], _terminal)
        ctx = _make_ctx()

        with pytest.raises(T0AgentsError) as exc_info:
            await pipeline(ctx)

        # Error should name the offending middleware so debugging is fast.
        assert "my_bad_middleware" in str(exc_info.value)


class TestChainOnInference:
    async def test_callbacks_fire_in_registration_order(self) -> None:
        ctx = _make_ctx()
        log: list[str] = []

        async def first(_: Any) -> None:
            log.append("first")

        async def second(_: Any) -> None:
            log.append("second")

        async def third(_: Any) -> None:
            log.append("third")

        ctx.chain_on_inference(first)
        ctx.chain_on_inference(second)
        ctx.chain_on_inference(third)
        assert ctx.on_inference is not None

        await ctx.on_inference(object())

        # First registered fires first — composes "after the existing
        # callback" per the docstring contract.
        assert log == ["first", "second", "third"]

    async def test_first_callback_set_directly_no_compose(self) -> None:
        ctx = _make_ctx()
        log: list[str] = []

        async def only(_: Any) -> None:
            log.append("only")

        ctx.chain_on_inference(only)

        assert ctx.on_inference is only
        await ctx.on_inference(object())
        assert log == ["only"]


class TestReplaceToolProvider:
    async def test_replace_records_prior_provider_on_stack(self) -> None:
        ctx = _make_ctx()
        original = cast(Any, "original-provider")
        ctx.tool_provider = original

        first_wrap = cast(Any, "wrap-1")
        ctx.replace_tool_provider(first_wrap)
        assert ctx.tool_provider is first_wrap
        assert ctx.extras["_tool_provider_stack"] == [original]

        second_wrap = cast(Any, "wrap-2")
        ctx.replace_tool_provider(second_wrap)
        assert ctx.tool_provider is second_wrap
        # Stack grows outermost-first.
        assert ctx.extras["_tool_provider_stack"] == [original, first_wrap]
