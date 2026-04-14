"""Shared fakes for ``AsyncTensorZeroGateway`` in unit tests.

Hand-written stubs rather than ``unittest.mock`` so the suite stays
mypy-strict-clean. ``run_agent`` does ``cast(ChatInferenceResponse, raw)``
instead of ``isinstance``-checking, so the duck-typed ``FakeChatResponse``
satisfies the runtime contract without subclassing the real Rust-backed type.

Extracted from ``tests/test_mcp/test_loop.py`` so that multiple test modules
can share a single source of truth for the T0 mock contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass
class FakeChatResponse:
    """Duck-typed stand-in for ``tensorzero.types.ChatInferenceResponse``.

    Only carries the three attributes ``run_agent`` reads (``content``,
    ``episode_id``, ``inference_id``); everything else on the real dataclass
    (``usage``, ``raw_usage``, etc.) is unused.
    """

    content: list[Any]
    episode_id: UUID = field(default_factory=uuid4)
    inference_id: UUID = field(default_factory=uuid4)


@dataclass
class FakeFeedbackResponse:
    """Duck-typed stand-in for ``tensorzero.types.FeedbackResponse``."""

    feedback_id: UUID = field(default_factory=uuid4)


class FakeT0:
    """Records every ``inference()`` and ``feedback()`` call.

    The response queue accepts both ``FakeChatResponse`` and ``Exception``
    items.  When the next item is an ``Exception``, ``inference()`` raises
    it — this lets tests simulate ``TensorZeroError`` and other failures
    at the inference layer.
    """

    def __init__(self, responses: list[FakeChatResponse | Exception]) -> None:
        self._queue: list[FakeChatResponse | Exception] = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> FakeChatResponse:
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
        return item

    async def feedback(self, **kwargs: Any) -> FakeFeedbackResponse:
        self.feedback_calls.append(dict(kwargs))
        return FakeFeedbackResponse()
