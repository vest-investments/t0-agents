"""Shared fakes for ``AsyncTensorZeroGateway`` in unit tests.

Hand-written stubs rather than ``unittest.mock`` so the suite stays
mypy-strict-clean. ``run_agent`` does ``cast(ChatInferenceResponse, raw)``
instead of ``isinstance``-checking, so the duck-typed ``FakeChatResponse``
satisfies the runtime contract without subclassing the real Rust-backed type.

Extracted from ``tests/test_mcp/test_loop.py`` so that multiple test modules
can share a single source of truth for the T0 mock contract.

Phase E adds streaming fakes: ``stream=True`` on ``FakeT0.inference``
returns an async iterator over a scripted ``list[ChatChunk]``.  Build
those scripts with :func:`chat_chunk` / :func:`text_chunk` /
:func:`tool_call_chunk` / :func:`thought_chunk` / :func:`final_chunk`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from tensorzero.types import (
    ChatChunk,
    TextChunk,
    ThoughtChunk,
    ToolCallChunk,
    Usage,
)


@dataclass
class FakeUsage:
    """Duck-typed stand-in for ``tensorzero.Usage``.

    Only ``cost`` is read by the Phase C guardrail cost-tracking path.
    ``input_tokens`` / ``output_tokens`` stay here so future tests that
    exercise token-budget guardrails can populate them without touching
    the fake contract again.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


@dataclass
class FakeChatResponse:
    """Duck-typed stand-in for ``tensorzero.types.ChatInferenceResponse``.

    Carries ``content``, ``episode_id``, ``inference_id`` (read by
    ``run_agent``'s main loop) and ``usage`` (read by the Phase C
    ``on_inference`` cost-tracker).  ``usage`` defaults to an empty
    :class:`FakeUsage` so existing tests that never mention cost keep
    working unchanged.
    """

    content: list[Any]
    episode_id: UUID = field(default_factory=uuid4)
    inference_id: UUID = field(default_factory=uuid4)
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeFeedbackResponse:
    """Duck-typed stand-in for ``tensorzero.types.FeedbackResponse``."""

    feedback_id: UUID = field(default_factory=uuid4)


# ---------------------------------------------------------------------------
# Streaming helpers (Phase E)
# ---------------------------------------------------------------------------


def text_chunk(text: str, *, block_id: str = "text-1") -> TextChunk:
    """Build a ``TextChunk`` with a stable ``id`` for assembler tests."""
    return TextChunk(id=block_id, text=text, type="text")


def tool_call_chunk(
    raw_name: str = "",
    raw_arguments: str = "",
    *,
    block_id: str = "tc-1",
) -> ToolCallChunk:
    """Build a ``ToolCallChunk`` — pass partial name/args to simulate deltas."""
    return ToolCallChunk(
        id=block_id,
        raw_name=raw_name,
        raw_arguments=raw_arguments,
        type="tool_call",
    )


def thought_chunk(text: str, *, block_id: str = "th-1") -> ThoughtChunk:
    """Build a ``ThoughtChunk`` with the given reasoning-text delta."""
    return ThoughtChunk(id=block_id, text=text, type="thought")


def chat_chunk(
    content: list[Any],
    *,
    usage: Usage | None = None,
    episode_id: UUID | None = None,
    inference_id: UUID | None = None,
    variant_name: str = "test-variant",
) -> ChatChunk:
    """Build a ``ChatChunk`` carrying one or more content-block deltas."""
    return ChatChunk(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name=variant_name,
        content=content,
        usage=usage,
    )


def final_chunk(
    *,
    cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    episode_id: UUID | None = None,
    inference_id: UUID | None = None,
    variant_name: str = "test-variant",
) -> ChatChunk:
    """Build the terminal ``ChatChunk`` — empty content, populated ``usage``."""
    return ChatChunk(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name=variant_name,
        content=[],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),  # type: ignore[call-arg]
    )


StreamScript = list[ChatChunk]
"""One turn's worth of streaming chunks — feed to ``FakeT0`` response queue."""


class FakeT0:
    """Records every ``inference()`` and ``feedback()`` call.

    The response queue accepts ``FakeChatResponse``, ``StreamScript``
    (``list[ChatChunk]`` for ``stream=True`` calls), or ``Exception``
    entries.  When the next item is an ``Exception``, ``inference()``
    raises it — letting tests simulate ``TensorZeroError`` and other
    failures at the inference layer.
    """

    def __init__(
        self,
        responses: list[FakeChatResponse | StreamScript | Exception],
    ) -> None:
        self._queue: list[FakeChatResponse | StreamScript | Exception] = list(
            responses
        )
        self.calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []

    async def inference(
        self, **kwargs: Any
    ) -> FakeChatResponse | AsyncIterator[ChatChunk]:
        # Snapshot mutable args at call time. Real T0 serializes the request
        # immediately; without snapshotting, ``run_agent`` mutating ``messages``
        # later in the loop would retroactively change what tests see in
        # ``self.calls[N]``.
        captured: dict[str, Any] = dict(kwargs)
        if isinstance(captured.get("input"), dict):
            captured["input"] = dict(captured["input"])
            if isinstance(captured["input"].get("messages"), list):
                captured["input"]["messages"] = [
                    dict(m) if isinstance(m, dict) else m
                    for m in captured["input"]["messages"]
                ]
        self.calls.append(captured)
        if not self._queue:
            raise AssertionError("FakeT0 ran out of canned responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if kwargs.get("stream") is True:
            if not isinstance(item, list):
                raise AssertionError(
                    "FakeT0: stream=True call expected a StreamScript "
                    f"(list[ChatChunk]), got {type(item).__name__}",
                )
            return _script_iterator(item)
        if isinstance(item, list):
            raise AssertionError(
                "FakeT0: non-streaming call received a StreamScript; "
                "pass stream=True or script FakeChatResponse instead",
            )
        return item

    async def feedback(self, **kwargs: Any) -> FakeFeedbackResponse:
        self.feedback_calls.append(dict(kwargs))
        return FakeFeedbackResponse()


async def _script_iterator(
    chunks: list[ChatChunk],
) -> AsyncIterator[ChatChunk]:
    for chunk in chunks:
        yield chunk
