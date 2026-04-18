"""Async lifespan wiring for FastAPI.

The lifespan context manager invokes two pairs of hooks on startup /
shutdown: :meth:`SessionFactory.startup` /
:meth:`SessionFactory.shutdown` and :meth:`Agent.warmup`.  Both default
to no-ops today but exist on the call path so extensions (embedded
gateway warm-up, per-pod cache hydration) do not widen the public API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from t0_agents.core.agent import Agent
from t0_agents.serve.session import SessionFactory


def build_lifespan(
    *,
    session_factory: SessionFactory,
    agent: Agent,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Return the FastAPI ``lifespan`` callable.

    Startup runs :meth:`SessionFactory.startup` followed by
    :meth:`Agent.warmup`; shutdown runs :meth:`SessionFactory.shutdown`
    inside a ``finally`` so that a warm-up exception still tears the
    factory down cleanly.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await session_factory.startup()
        try:
            await agent.warmup()
            yield
        finally:
            await session_factory.shutdown()

    return lifespan
