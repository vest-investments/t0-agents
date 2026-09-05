"""Shared fakes for ``AsyncTensorZeroGateway`` in unit tests.

Hand-written stubs rather than ``unittest.mock`` so the suite stays
mypy-strict-clean. ``run_agent`` does ``cast(ChatInferenceResponse, raw)``
instead of ``isinstance``-checking, so the duck-typed ``FakeChatResponse``
satisfies the runtime contract without subclassing the real Rust-backed type.

Extracted from ``tests/test_mcp/test_loop.py`` so that multiple test modules
can share a single source of truth for the T0 mock contract.

Supports streaming fakes: ``stream=True`` on ``FakeT0.inference``
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

    Mirrors the SDK field set so the ``on_inference`` cost / footprint
    tracker reads it exactly like the real type.  The two
    ``provider_cache_*`` fields default to ``None`` (as on the SDK type),
    so call sites that do not exercise caching stay unchanged.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    provider_cache_read_input_tokens: int | None = None
    provider_cache_write_input_tokens: int | None = None


@dataclass
class FakeChatResponse:
    """Duck-typed stand-in for ``tensorzero.types.ChatInferenceResponse``.

    Carries ``content``, ``episode_id``, ``inference_id`` (read by
    ``run_agent``'s main loop) and ``usage`` (read by the
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


@dataclass
class FakeListInferencesResponse:
    """Duck-typed stand-in for ``tensorzero.GetInferencesResponse``.

    ``run_agent`` does not call ``list_inferences`` itself — this is for
    ``TensorZeroMemory`` and any other consumer that inspects the
    observability API.  ``inferences`` is intentionally typed as a free-
    form list so tests can drop in real ``StoredInferenceChat`` instances
    or hand-rolled namespaces with the same shape.
    """

    inferences: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming helpers
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


def thought_chunk(
    text: str | None = None,
    *,
    summary_text: str | None = None,
    block_id: str = "th-1",
    signature: str | None = None,
    provider_type: str | None = None,
) -> ThoughtChunk:
    """Build a ``ThoughtChunk`` carrying reasoning content.

    Reasoning models surface content via either ``text`` (raw thought, the
    common case) or ``summary_text`` (provider-side summary — e.g. gpt-oss
    harmony's analysis channel, OpenAI o-series reasoning summaries).  Tests
    pass either or both; at least one must be non-empty.
    """
    return ThoughtChunk(
        id=block_id,
        text=text,
        summary_text=summary_text,
        type="thought",
        signature=signature,
        provider_type=provider_type,
    )


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
        *,
        list_inferences_responses: (
            list[FakeListInferencesResponse | Exception] | None
        ) = None,
    ) -> None:
        self._queue: list[FakeChatResponse | StreamScript | Exception] = list(
            responses
        )
        self._list_queue: list[FakeListInferencesResponse | Exception] = list(
            list_inferences_responses or []
        )
        self.calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []
        self.list_inferences_calls: list[dict[str, Any]] = []

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

    async def list_inferences(
        self, *, request: Any,
    ) -> FakeListInferencesResponse:
        """Pop the next ``list_inferences_responses`` entry.

        Captures the inbound :class:`ListInferencesRequest` on
        ``self.list_inferences_calls`` so tests can assert request
        shape (filters, episode_id, order_by, limit) in addition to
        the records the consumer derives from the response.

        Raises :class:`AssertionError` when the queue is empty so
        tests fail loudly rather than dispatching unscripted reads.
        """
        self.list_inferences_calls.append({"request": request})
        if not self._list_queue:
            raise AssertionError(
                "FakeT0: ran out of canned list_inferences responses"
            )
        item = self._list_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def _script_iterator(
    chunks: list[ChatChunk],
) -> AsyncIterator[ChatChunk]:
    for chunk in chunks:
        yield chunk
