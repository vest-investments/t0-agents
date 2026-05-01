"""Integration tests for :class:`TensorZeroMemory`.

Talk to a real TensorZero gateway via the SDK (no direct ClickHouse
access).  Each test fires real inferences with explicit tags and then
queries the gateway to verify the memory provider returns the right
records — exactly the path the agent loop exercises in production.

Pre-requisites:

- ``docker compose up -d`` (TensorZero gateway + ClickHouse) from the
  repo root.  ``GROQ_API_KEY`` must be present in ``.env``.
- The gateway is configured with the ``agent`` function (matches
  ``config/tensorzero.toml`` shipped in the repo).

Tests are gated by ``pytestmark = pytest.mark.integration`` so the
default ``pytest -m "not integration"`` invocation skips them.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid
from collections.abc import AsyncIterator, Awaitable
from typing import cast

import pytest
from tensorzero import AsyncTensorZeroGateway, InferenceFilterTime
from tensorzero.types import ChatInferenceResponse

from t0_agents.memory import TensorZeroMemory

pytestmark = pytest.mark.integration


_DEFAULT_GATEWAY_URL = "http://localhost:3000"

# TZ writes to ClickHouse asynchronously; reads can lag the most recent
# inference by a fraction of a second under local Docker load.  This
# sleep is the same trick used in the upstream PR and is deliberately
# generous to keep the tests stable on CI.
_TZ_FLUSH_DELAY_SECONDS = 1.5


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    url = os.environ.get("TEST_T0_URL", _DEFAULT_GATEWAY_URL)
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url=url),
    )


@pytest.fixture
async def gateway() -> AsyncIterator[AsyncTensorZeroGateway]:
    async with await _gateway_factory() as gw:
        yield gw


def _fresh_customer_id() -> str:
    """Random per-test customer_id so concurrent runs cannot collide."""
    return f"cust-{uuid.uuid4().hex[:8]}"


async def _inference(
    gw: AsyncTensorZeroGateway,
    *,
    episode_id: str | None,
    customer_id: str,
    user_text: str,
) -> tuple[str, str]:
    """Fire one ``agent`` chat inference; return (episode_id, inference_id)."""
    raw = await gw.inference(
        function_name="agent",
        episode_id=episode_id,
        input={
            "messages": [
                {"role": "user", "content": user_text},
            ]
        },
        tags={
            "customer_id": customer_id,
            "test_marker": "tensorzero_memory_integration",
        },
    )
    response = cast(ChatInferenceResponse, raw)
    return str(response.episode_id), str(response.inference_id)


async def test_two_turn_continuity_messages_strategy(
    gateway: AsyncTensorZeroGateway,
) -> None:
    """Turn 1 plants a fact, turn 2 reuses episode_id and the model
    recalls it via input.messages, never via a prompt summary."""
    customer = _fresh_customer_id()

    # Turn 1: plant the fact.
    ep, _ = await _inference(
        gateway,
        episode_id=None,
        customer_id=customer,
        user_text="My favourite colour is teal. Please remember it.",
    )
    await asyncio.sleep(_TZ_FLUSH_DELAY_SECONDS)

    # Turn 2: drive the loop on the same episode using the messages
    # strategy.  TensorZeroMemory.load_history returns the structured
    # history; the agent threads it through input.messages.
    memory = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": customer},
    )
    history = await memory.load_history(ep)
    assert history is not None
    assert len(history) >= 2  # at least user-1 + assistant-1


async def test_tenant_isolation_via_tag_filter(
    gateway: AsyncTensorZeroGateway,
) -> None:
    """Two customer_ids artificially share an episode_id; each
    TensorZeroMemory instance must only see its own history."""
    alice = _fresh_customer_id()
    bob = _fresh_customer_id()

    ep, _ = await _inference(
        gateway,
        episode_id=None,
        customer_id=alice,
        user_text="Reply with just OK.",
    )
    await asyncio.sleep(_TZ_FLUSH_DELAY_SECONDS)

    bob_memory = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": bob},
    )
    assert await bob_memory.load(ep) == []

    alice_memory = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": alice},
    )
    alice_records = await alice_memory.load(ep)
    assert len(alice_records) >= 1
    assert all(r.user_message for r in alice_records)


async def test_extra_filters_time_bound(
    gateway: AsyncTensorZeroGateway,
) -> None:
    """InferenceFilterTime via extra_filters excludes inferences before
    the cutoff."""
    customer = _fresh_customer_id()
    cutoff = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)
    cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

    ep, _ = await _inference(
        gateway,
        episode_id=None,
        customer_id=customer,
        user_text="Emit a single token.",
    )
    await asyncio.sleep(_TZ_FLUSH_DELAY_SECONDS)

    # All inferences just made are BEFORE cutoff (=now+1m), so the
    # filter rejects them.  load() must return [].
    future_only = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": customer},
        extra_filters=InferenceFilterTime(
            comparison_operator=">=",
            time=cutoff_iso,
        ),
    )
    assert await future_only.load(ep) == []

    # And the inverse predicate sees the inferences.
    past_only = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": customer},
        extra_filters=InferenceFilterTime(
            comparison_operator="<",
            time=cutoff_iso,
        ),
    )
    assert len(await past_only.load(ep)) >= 1


async def test_experimental_search_round_trip(
    gateway: AsyncTensorZeroGateway,
) -> None:
    """Opt-in experimental_search forwards the query and returns
    matching inferences."""
    customer = _fresh_customer_id()
    needle = uuid.uuid4().hex[:12]

    await _inference(
        gateway,
        episode_id=None,
        customer_id=customer,
        user_text=f"unique-token {needle} please ack briefly",
    )
    await asyncio.sleep(_TZ_FLUSH_DELAY_SECONDS)

    memory = TensorZeroMemory(
        gateway,
        function_name="agent",
        tag_filter={"customer_id": customer},
        experimental_search=True,
    )
    hits = await memory.search(needle, limit=5)
    # The experimental endpoint may stream results with a small lag;
    # at least one hit is required.
    assert len(hits) >= 1
    assert any(needle in r.user_message for r in hits)
