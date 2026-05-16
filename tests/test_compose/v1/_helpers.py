"""Shared fakes for Composer / topology tests.

Each topology test under ``tests/test_compose/v1/`` consumes mock
Composables and mock Runs from this module so the tests don't need
a running TensorZero gateway or FastMCP server.  Built on top of
:class:`Run`'s ``agent=None`` mode for composite runs — the mock Run
overrides ``_drive`` to return a pre-baked :class:`AgentResult`
without touching any middleware chain.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.run import Run, RunInput
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import DoneEvent

_log = logging.getLogger(__name__)


class MockRun(Run[Any]):
    """Run that produces a pre-baked :class:`AgentResult` (or raises)."""

    def __init__(
        self,
        *,
        value: Any = "ok",
        episode_id: str | None = "ep_mock",
        inference_ids: list[str] | None = None,
        raises: BaseException | None = None,
        delay: float = 0.0,
        turns: int = 1,
        eager: bool = True,
    ) -> None:
        super().__init__(
            None,
            RunInput(user_message="mock", function_name="mock"),
            ChatOutput(),
            eager=False,
        )
        self._mock_value = value
        self._mock_episode_id = episode_id
        self._mock_inference_ids = inference_ids or [f"inf_{id(self)}"]
        self._mock_raises = raises
        self._mock_delay = delay
        self._mock_turns = turns
        if eager:
            self._start()

    async def _drive(self) -> None:
        try:
            if self._mock_delay > 0:
                await asyncio.sleep(self._mock_delay)
            if self._mock_raises is not None:
                raise self._mock_raises
            result = AgentResult(
                value=self._mock_value,
                episode_id=self._mock_episode_id,
                inference_ids=list(self._mock_inference_ids),
                turns=self._mock_turns,
            )
            self._capture_result(result)
            await self._buffer.put(DoneEvent(result=result))
            self._state = "succeeded"
        except asyncio.CancelledError:
            self._state = "cancelled"
            raise
        except BaseException as exc:
            self._exception = exc
            self._state = "failed"
        finally:
            try:
                await self._buffer.close()
            except Exception:
                _log.debug("EventBuffer.close raised", exc_info=True)


class MockComposable:
    """Composable that records every invocation and returns a :class:`MockRun`.

    Configure the produced Run via the constructor; per-call overrides
    flow through :attr:`override_value` / :attr:`override_raises` (set
    by the test before invoking).

    The ``transform`` hook lets tests parameterise the Run output on
    the input: e.g. ``MockComposable(transform=lambda x: x.upper())``
    produces a Run whose ``.value`` is the input transformed.
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        value: Any = None,
        episode_id: str | None = "ep_mock",
        inference_ids: list[str] | None = None,
        raises: BaseException | None = None,
        delay: float = 0.0,
        turns: int = 1,
        transform: Any = None,
    ) -> None:
        self.name = name
        self._value = value
        self._episode_id = episode_id
        self._inference_ids = inference_ids
        self._raises = raises
        self._delay = delay
        self._turns = turns
        self._transform = transform
        self.invocations: list[Any] = []
        self.received_kwargs: list[dict[str, Any]] = []

    def invoke(self, input_value: Any, /, **kwargs: Any) -> Run[Any]:
        self.invocations.append(input_value)
        self.received_kwargs.append(kwargs)
        if self._transform is not None:
            value = self._transform(input_value)
        elif self._value is not None:
            value = self._value
        else:
            value = input_value
        return MockRun(
            value=value,
            episode_id=self._episode_id,
            inference_ids=(
                list(self._inference_ids)
                if self._inference_ids
                else [f"inf_{self.name}_{len(self.invocations)}"]
            ),
            raises=self._raises,
            delay=self._delay,
            turns=self._turns,
        )


class CountingComposable(MockComposable):
    """MockComposable that fails the first N invocations, then succeeds."""

    def __init__(
        self,
        *,
        fail_count: int,
        failure: BaseException,
        success_value: Any = "ok",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fail_count = fail_count
        self._failure = failure
        self._success_value = success_value
        self._call_count = 0

    def invoke(self, input_value: Any, /, **kwargs: Any) -> Run[Any]:
        self._call_count += 1
        self.invocations.append(input_value)
        self.received_kwargs.append(kwargs)
        if self._call_count <= self._fail_count:
            return MockRun(raises=self._failure)
        return MockRun(
            value=self._success_value,
            episode_id=self._episode_id,
            inference_ids=[f"inf_{self.name}_{self._call_count}"],
        )
