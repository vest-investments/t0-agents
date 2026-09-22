"""``extra_body`` from a fluent run to the gateway, on every terminal.

The parameter is a pure passthrough, so the only assertion that proves
anything is the one made on the gateway's captured kwargs — a hop that
silently drops the patches still satisfies every helper-level check.
One test per terminal (chat loop, streaming loop, single-shot JSON),
plus the absent-key pins that keep an untouched run's request identical
to what it was before the parameter existed.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents.glow import App
from t0_agents.testing import MockGateway, MockStreamingGateway

_PATCH: dict[str, Any] = {
    "variant_name": "chat_v1",
    "pointer": "/reasoning_effort",
    "value": "max",
}
_OTHER_PATCH: dict[str, Any] = {
    "variant_name": "chat_v1",
    "pointer": "/reasoning_effort",
    "value": "low",
}


class _Answer(BaseModel):
    answer: str


def _chat(gateway: Any) -> Any:
    app = App(t0=cast(AsyncTensorZeroGateway, gateway))

    @app.agent("chat")
    async def chat(q: str) -> str:
        """Answer the question."""

    return chat


async def test_the_chat_loop_sends_the_patches() -> None:
    gateway = MockGateway.script([[Text(type="text", text="hi")]])

    await _chat(gateway)("q").with_(extra_body=[_PATCH])

    assert gateway.calls[0]["extra_body"] == [_PATCH]


async def test_the_streaming_loop_sends_the_patches() -> None:
    gateway = MockStreamingGateway([[Text(text="hi")]])

    async for _ in _chat(gateway)("q").with_(extra_body=[_PATCH]).events():
        pass

    assert gateway.calls[0]["extra_body"] == [_PATCH]


async def test_the_single_shot_terminal_sends_the_patches() -> None:
    gateway = MockGateway.script_json([{"raw": '{"answer": "hi"}'}])

    await _chat(gateway)("q").with_(extra_body=[_PATCH]).json(_Answer)

    assert gateway.calls[0]["extra_body"] == [_PATCH]


async def test_the_chat_loop_omits_the_key_without_patches() -> None:
    gateway = MockGateway.script([[Text(type="text", text="hi")]])

    await _chat(gateway)("q")

    assert "extra_body" not in gateway.calls[0]


async def test_the_single_shot_terminal_omits_the_key_without_patches() -> None:
    gateway = MockGateway.script_json([{"raw": '{"answer": "hi"}'}])

    await _chat(gateway)("q").json(_Answer)

    assert "extra_body" not in gateway.calls[0]


async def test_repeated_with_is_last_wins_for_extra_body() -> None:
    gateway = MockGateway.script([[Text(type="text", text="hi")]])

    await _chat(gateway)("q").with_(extra_body=[_PATCH]).with_(
        extra_body=[_OTHER_PATCH]
    )

    assert gateway.calls[0]["extra_body"] == [_OTHER_PATCH]
