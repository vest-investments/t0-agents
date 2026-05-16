"""Lifecycle tests for :class:`t0_agents.core.run.Run`.

Covers the state machine introduced in v1.0 milestone 1: eager vs lazy
start, the ``pending → running → succeeded/failed/cancelled``
transitions, ``__await__`` parity with ``result()``, and the new
:class:`RunCancelledError` exception.  Failed runs raise the original
captured exception (see :meth:`Run.result`); :class:`RunFailedError`
is the defensive fallback for the "failed without captured cause" edge
case.  Uses the shared ``FakeT0`` / ``FakeMCPClient`` fakes so the
assertions exercise the actual orchestration body without docker.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import (
    Agent,
    AgentConfig,
    Run,
    RunCancelledError,
)
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import (
    FakeChatResponse,
    FakeT0,
    chat_chunk,
    final_chunk,
    text_chunk,
)


def _terminal_response(text: str = "done") -> FakeChatResponse:
    return FakeChatResponse(content=[Text(type="text", text=text)])


def _terminal_stream(text: str = "done") -> list[Any]:
    """Stream script for a one-turn, text-only assistant response."""
    return [
        chat_chunk([text_chunk(text, block_id="t1")]),
        final_chunk(),
    ]


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    client: FakeMCPClient | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "test", "function_name": "agent"}
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_terminal_response()])),
        tool_provider=cast(
            ToolProvider, client or FakeMCPClient(tools=[]),
        ),
    )


class TestEagerStart:
    async def test_eager_run_is_running_after_construction(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        # Eager: the asyncio task is created in __init__ and the state
        # has already transitioned to "running".
        assert run.state == "running"
        assert isinstance(run, Run)
        # Drain so the test doesn't leave a pending task behind.
        await run.result()

    async def test_eager_run_completes_to_succeeded(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        result = await run.result()
        assert run.state == "succeeded"
        assert result.response == "done"


class TestLazyStart:
    async def test_lazy_run_is_pending_before_consumption(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi", eager=False)
        assert run.state == "pending"

    async def test_lazy_run_starts_on_first_result(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi", eager=False)
        assert run.state == "pending"
        result = await run.result()
        assert run.state == "succeeded"
        assert result.response == "done"

    async def test_lazy_run_starts_on_first_events(self) -> None:
        # events() drives via the streaming path — feed the stream
        # script instead of the non-streaming FakeChatResponse.
        agent = _make_agent(t0=FakeT0([_terminal_stream()]))
        run = agent.invoke("hi", eager=False)
        events = [ev async for ev in run.events(replay=True)]
        assert run.state == "succeeded"
        assert any(getattr(ev, "type", None) == "done" for ev in events)


class TestAwaitAlias:
    async def test_await_run_returns_same_result_as_result(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        result_via_await = await run
        # Result is now cached on the Run; a second await must return
        # the same object.
        result_via_method = await run.result()
        assert result_via_await is result_via_method
        assert result_via_await.response == "done"


class TestCancellation:
    async def test_cancel_before_consumption_transitions_to_cancelled(
        self,
    ) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        # Chat agents wait for first-consumer signal before driving,
        # so the task is parked on _mode_chosen here.
        await run.cancel()
        assert run.state == "cancelled"

    async def test_result_after_cancel_raises_run_cancelled_error(
        self,
    ) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        await run.cancel()
        with pytest.raises(RunCancelledError):
            await run.result()

    async def test_cancel_is_idempotent(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        await run.cancel()
        # Second cancel is a no-op — must not raise.
        await run.cancel()
        assert run.state == "cancelled"


class TestFailureRouting:
    async def test_failed_run_raises_original_exception(self) -> None:
        # Surfacing the original exception matches v0.12 idioms and
        # mirrors ``events()`` behaviour — callers can ``except
        # SpecificError`` with standard Python patterns.
        agent = _make_agent(t0=FakeT0([RuntimeError("boom")]))
        run = agent.invoke("hi")
        with pytest.raises(RuntimeError, match="boom"):
            await run.result()
        assert run.state == "failed"

    async def test_failed_run_keeps_state_for_repeated_inspection(
        self,
    ) -> None:
        agent = _make_agent(t0=FakeT0([RuntimeError("boom")]))
        run = agent.invoke("hi")
        with pytest.raises(RuntimeError, match="boom"):
            await run.result()
        # State stays "failed" so callers can branch on it; awaiting
        # a second time raises the same exception.
        assert run.state == "failed"
        with pytest.raises(RuntimeError, match="boom"):
            await run.result()


class TestInputProperty:
    async def test_input_snapshot_carries_call_args(self) -> None:
        agent = _make_agent()
        run = agent.invoke(
            "hi",
            tags={"experiment": "alpha"},
        )
        assert run.input.user_message == "hi"
        assert run.input.function_name == "agent"
        # ``function_kind`` is no longer a RunInput field in v1.0 —
        # dispatch routes via ``run.output.expects_tool_loop`` instead.
        assert run.output.expects_tool_loop is True
        assert dict(run.input.tags or {}) == {"experiment": "alpha"}
        await run.result()

    async def test_episode_id_captured_after_run(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        result = await run.result()
        assert run.episode_id == result.episode_id
        assert run.episode_id is not None
        assert run.inference_ids == tuple(result.inference_ids)


class TestEagerKwargForwarding:
    """The eager kwarg is forwarded through Agent.invoke().

    The same FakeT0 + FakeMCPClient produce a stable response no
    matter how the Run is consumed; what changes is the timing of
    when the task starts running.  We verify both ends of the kwarg
    are observable on the Run.
    """

    async def test_eager_default_is_true(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        assert run.state == "running"
        await run.result()

    async def test_eager_false_means_pending_until_consumption(
        self,
    ) -> None:
        agent = _make_agent()
        run = agent.invoke("hi", eager=False)
        assert run.state == "pending"
        # Give the event loop a tick to confirm the task really is
        # not scheduled.
        await asyncio.sleep(0)
        assert run.state == "pending"
        await run.result()
        assert run.state == "succeeded"
