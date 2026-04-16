"""Unit tests for ``POST /feedback``.

Routes through ``httpx.AsyncClient`` + ``ASGITransport`` so the suite
stays offline; ``FakeT0.feedback`` records every call into
``feedback_calls`` for inspection.
"""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.serve import create_app
from tests.test_serve.conftest import FakeFastMCPClient


async def _post_feedback(app: Any, payload: dict[str, Any]) -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        return await http.post("/feedback", json=payload)


async def test_feedback_minimal_kwargs(base_agent: Agent) -> None:
    """Only ``metric_name`` + ``value`` → passthrough with those two kwargs."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_feedback(app, {"metric_name": "m", "value": True})

    assert resp.status_code == 200
    assert "feedback_id" in resp.json()
    # FakeT0 wraps kwargs into feedback_calls
    calls = base_agent._t0.feedback_calls  # type: ignore[attr-defined]
    assert calls == [{"metric_name": "m", "value": True}]


async def test_feedback_all_six_kwargs(base_agent: Agent) -> None:
    """Supplying every optional field forwards all six kwargs."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    payload = {
        "metric_name": "action_approved",
        "value": False,
        "inference_id": "inf-123",
        "episode_id": "ep-456",
        "tags": {"extra": "yes"},
        "dryrun": True,
    }
    resp = await _post_feedback(app, payload)
    assert resp.status_code == 200

    calls = base_agent._t0.feedback_calls  # type: ignore[attr-defined]
    assert calls == [payload]


async def test_feedback_skips_none_kwargs(base_agent: Agent) -> None:
    """Unset optionals are dropped before the passthrough."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_feedback(
        app,
        {
            "metric_name": "m",
            "value": 3.14,
            "episode_id": "only-episode",
        },
    )
    assert resp.status_code == 200

    calls = base_agent._t0.feedback_calls  # type: ignore[attr-defined]
    assert calls == [
        {"metric_name": "m", "value": 3.14, "episode_id": "only-episode"},
    ]
    assert "inference_id" not in calls[0]
    assert "tags" not in calls[0]
    assert "dryrun" not in calls[0]


async def test_feedback_response_id_is_string(base_agent: Agent) -> None:
    """``FeedbackResponse.feedback_id`` is always a string."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_feedback(app, {"metric_name": "m", "value": 1})

    body = resp.json()
    assert isinstance(body["feedback_id"], str)
    assert len(body["feedback_id"]) > 0


async def test_feedback_forwards_gateway_4xx(base_agent: Agent) -> None:
    """A 4xx ``TensorZeroError`` from the gateway passes through as that status."""
    from tensorzero.types import TensorZeroError

    async def failing_feedback(**_: Any) -> Any:
        raise TensorZeroError(status_code=404, text="Unknown metric: missing")

    base_agent._t0.feedback = failing_feedback  # type: ignore[method-assign,attr-defined]
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_feedback(app, {"metric_name": "missing", "value": True})
    assert resp.status_code == 404
    assert "Unknown metric" in resp.json()["detail"]


async def test_feedback_wraps_gateway_5xx_as_502(base_agent: Agent) -> None:
    """A 5xx ``TensorZeroError`` still maps to HTTP 502 (Bad Gateway)."""
    from tensorzero.types import TensorZeroError

    async def failing_feedback(**_: Any) -> Any:
        raise TensorZeroError(status_code=500, text="internal boom")

    base_agent._t0.feedback = failing_feedback  # type: ignore[method-assign,attr-defined]
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_feedback(app, {"metric_name": "m", "value": True})
    assert resp.status_code == 502
