"""A gateway-free :class:`Agent` fake for unit tests.

:class:`ScriptedAgent` duck-types the slice of :class:`Agent` that glow
and ``serve`` touch — ``invoke() -> Run`` — but replays a fixed script
instead of calling TensorZero, so tests need no gateway, no Docker, and
no environment variables.

    from t0_agents.testing import ScriptedAgent
    from t0_agents.stream.events import TextDeltaEvent

    agent = ScriptedAgent(
        result="It's 22C",
        events=[TextDeltaEvent(id="1", delta="It's 22C")],
    )
    run = agent.invoke("weather?")
    assert (await run.result()).response == "It's 22C"
    async for ev in run.events():
        ...

``events=`` are the **non-terminal** events to replay; the single
terminal :class:`DoneEvent` is synthesised from ``result=``.  ``raises=``
makes the run fail: the exception propagates out of ``result()`` and
``events()`` exactly as a real fatal error would (no ``ErrorEvent`` — that
event is reserved for recoverable, inline anomalies).

**Test-only.**  Never import this from a production code path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, cast

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.run import Run, RunInput, build_run_input
from t0_agents.core.types import (
    AgentResult,
    ExtraBodyPatch,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.stream.events import AgentEvent, DoneEvent

__all__ = ["ScriptedAgent", "ScriptedRun"]


def _coerce_result(result: Any) -> AgentResult[Any]:
    """Wrap ``result`` in an :class:`AgentResult`, passing one through as-is."""
    if isinstance(result, AgentResult):
        return result
    return AgentResult(value="" if result is None else result, turns=1)


class ScriptedRun(Run[Any]):
    """A :class:`Run` that replays a fixed script from its own buffer.

    Overrides :meth:`Run._drive` (not :meth:`Run.events`) and pushes into
    the inherited ``self._buffer``, so both ``result()`` and ``events()``
    work with ``agent=None`` — no gateway is ever consulted.  Always
    drives as :class:`ChatOutput` (streaming-capable) so ``events()`` is
    legal regardless of the mode the caller declared.
    """

    def __init__(
        self,
        input_: RunInput,
        *,
        events: Sequence[AgentEvent],
        result: Any,
        raises: BaseException | None,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._scripted_events: list[AgentEvent] = list(events)
        self._scripted_result = result
        self._scripted_raises = raises
        super().__init__(None, input_, ChatOutput(), eager=eager, context=context)

    async def _drive(self) -> None:
        try:
            await self._mode_chosen.wait()
            # Scripted events replay BEFORE ``raises=`` fires so a script
            # models a mid-stream abort — the shape real fatal errors have
            # (some events already delivered, then the drive loop dies).
            for event in self._scripted_events:
                await self._buffer.put(event)
            if self._scripted_raises is not None:
                raise self._scripted_raises
            result = _coerce_result(self._scripted_result)
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
            await self._buffer.close()


class ScriptedAgent(Agent):
    """An :class:`Agent` whose ``invoke()`` returns a :class:`ScriptedRun`.

    Constructs over ``t0=None``: :meth:`Agent.__init__` stores the
    gateway without dereferencing it, and :class:`ScriptedRun` never
    reaches for it, so a ``ScriptedAgent`` is usable with no gateway.
    """

    def __init__(
        self,
        *,
        result: Any = None,
        events: Sequence[AgentEvent] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        super().__init__(
            AgentConfig(name="scripted", function_name="scripted"),
            t0=cast(Any, None),
            tool_provider=None,
        )
        self._result = result
        self._events: list[AgentEvent] = list(events) if events is not None else []
        self._raises = raises

    def invoke(
        self,
        user_message: str,
        *,
        output: OutputMode[Any] | None = None,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        prior_messages: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Sequence[ExtraBodyPatch] | None = None,
        middleware: Sequence[Any] | None = None,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        tool_provider: ToolProvider | None = None,
        memory: MemoryProvider | None = None,
    ) -> Run[Any]:
        """Return a :class:`ScriptedRun` replaying the configured script.

        ``output``, ``middleware``, and the per-run ``tool_provider`` /
        ``memory`` overrides are accepted for drop-in compatibility with
        :meth:`Agent.invoke` and ignored — the fake replays fixed events
        and result regardless of mode.
        """
        input_ = build_run_input(
            user_message=user_message,
            function_name=self.config.function_name,
            episode_id=episode_id,
            tags=tags,
            system=system,
            system_arguments=system_arguments,
            prior_messages=prior_messages,
            extra_body=extra_body,
        )
        return ScriptedRun(
            input_,
            events=self._events,
            result=self._result,
            raises=self._raises,
            eager=eager,
            context=context,
        )
