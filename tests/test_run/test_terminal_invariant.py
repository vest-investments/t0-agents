"""Property: every ``events(on_error="event")`` stream ends with exactly
one ``DoneEvent | RunErrorEvent``.

Generative suite over :class:`ScriptedAgent` — a seeded stdlib-``random``
script generator (no new test dependency) crossed with the four named
production paths: happy text, tool turns, reasoning (thought-only), and
mid-stream abort (``raises=``).  200 generated scripts alternating
abort/success keep the space honest beyond the hand-picked cases; the
fixed seed keeps failures reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import pytest

from t0_agents import (
    AgentResult,
    DoneEvent,
    RunErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.stream.events import AgentEvent
from t0_agents.testing import ScriptedAgent

_SEED = 0x6105  # fixed: failures must reproduce

_EXCEPTIONS: list[BaseException] = [
    ValueError("boom"),
    RuntimeError("died"),
    TimeoutError(),
    MaxTurnsExceededError(20),
]


def _event_pool(rng: random.Random) -> list[AgentEvent]:
    block = f"b{rng.randrange(4)}"
    return [
        TextDeltaEvent(id=block, delta=rng.choice(["", " ", "word"])),
        TextEndEvent(id=block),
        ThoughtDeltaEvent(id=block, delta="th"),
        ThoughtEvent(id=block, text="thought"),
        ToolCallEvent(id=block, name="tool", arguments={}),
        ToolProgressEvent(id=block, progress=rng.random(), total=1.0),
        ToolResultEvent(id=block, name="tool", result="ok"),
        TurnEndEvent(turn=rng.randrange(3)),
    ]


def _generated_cases() -> Iterator[Any]:
    rng = random.Random(_SEED)
    for case in range(200):
        script = [rng.choice(_event_pool(rng)) for _ in range(rng.randrange(0, 12))]
        raises = rng.choice(_EXCEPTIONS) if case % 2 else None
        yield pytest.param(script, raises, id=f"gen-{case:03d}")


def _named_cases() -> Iterator[Any]:
    yield pytest.param(
        [TextDeltaEvent(id="t1", delta="Hello"), TextEndEvent(id="t1")],
        None,
        id="happy-text",
    )
    yield pytest.param(
        [
            ToolCallEvent(id="c1", name="get_weather", arguments={}),
            ToolProgressEvent(id="c1", progress=1.0, total=2.0),
            ToolResultEvent(id="c1", name="get_weather", result="22C"),
            TurnEndEvent(turn=1),
            TextDeltaEvent(id="t1", delta="22C"),
            TextEndEvent(id="t1"),
        ],
        None,
        id="tool-turns",
    )
    yield pytest.param(
        [ThoughtDeltaEvent(id="th1", delta="hmm"), ThoughtEvent(id="th1", text="hmm")],
        None,
        id="reasoning-thought-only",
    )
    yield pytest.param(
        [TextDeltaEvent(id="t1", delta="par")],
        ValueError("mid-stream abort"),
        id="raise-path",
    )


def _terminals(events: list[AgentEvent]) -> list[AgentEvent]:
    return [e for e in events if isinstance(e, DoneEvent | RunErrorEvent)]


@pytest.mark.parametrize(("script", "raises"), [*_named_cases(), *_generated_cases()])
async def test_every_stream_ends_with_exactly_one_terminal(
    script: list[AgentEvent], raises: BaseException | None
) -> None:
    agent = ScriptedAgent(
        result=AgentResult(value="done", turns=1),
        events=script,
        raises=raises,
    )
    collected: list[AgentEvent] = []
    async for event in agent.invoke("q").events(on_error="event"):
        collected.append(event)

    terminals = _terminals(collected)
    assert len(terminals) == 1, f"expected exactly one terminal, got {terminals!r}"
    assert collected[-1] is terminals[0], "the terminal must be the LAST event"
    expected_type = RunErrorEvent if raises is not None else DoneEvent
    assert isinstance(terminals[0], expected_type)
    assert collected[: len(script)] == script, "script events precede the terminal"
