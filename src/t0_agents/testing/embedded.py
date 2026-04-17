"""Embedded-gateway convenience helpers for integration tests without Docker.

Thin async-context-manager wrappers around
:meth:`tensorzero.AsyncTensorZeroGateway.build_embedded` (source-validated
in ``.venv/.../tensorzero/tensorzero.pyi:1017-1035``).  All backend-DB URLs
are optional there — running with none gives degraded-but-useful mode
(feedback IDs are ephemeral, mirroring ``dryrun=True``) which is the
sweet spot for tests that want real LLM inference but no ClickHouse or
Postgres overhead.

**Test-only.**  ``EmbeddedGatewayHelper`` is documented as a test helper,
not a production deployment path — production deployments should use
``build_http`` against a managed gateway.

Usage::

    from t0_agents.testing import EmbeddedGatewayHelper

    async with EmbeddedGatewayHelper.from_config("tests/configs/minimal.toml") as t0:
        agent = Agent(AgentConfig(...), t0=t0, tool_provider=client)
        result = await agent.run("hi")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import cast

from tensorzero import AsyncTensorZeroGateway


class EmbeddedGatewayHelper:
    """Async-context wrappers around :meth:`AsyncTensorZeroGateway.build_embedded`.

    Three entry points for the three observability configurations a test
    might want:

    * :meth:`from_config` — config only, no backend DBs.  Real LLM
      inference; feedback IDs are ephemeral.
    * :meth:`with_clickhouse` — plus ClickHouse for tests that assert on
      inference rows or episode continuity.
    * :meth:`with_postgres` — plus Postgres for tests that assert on
      configuration / experimentation rows.

    Each method returns an ``async with`` context manager whose value is
    the real :class:`AsyncTensorZeroGateway` — no shim, no adapter.  The
    test owns the gateway's lifecycle; teardown happens on context exit.
    """

    @staticmethod
    @asynccontextmanager
    async def from_config(
        config_file: str,
    ) -> AsyncIterator[AsyncTensorZeroGateway]:
        """Build an embedded gateway with only a config file.

        ``config_file`` is forwarded verbatim to ``build_embedded``; it
        must point at a ``tensorzero.toml`` whose functions are
        backed by providers whose credentials are in the environment
        (e.g. ``GROQ_API_KEY``).
        """
        # ``build_embedded(async_setup=True)`` — the default — returns an
        # ``Awaitable[AsyncTensorZeroGateway]``.  The return annotation in
        # tensorzero.pyi is a union with the sync-setup path, so we cast
        # before awaiting to satisfy mypy strict.
        awaitable = cast(
            "Awaitable[AsyncTensorZeroGateway]",
            AsyncTensorZeroGateway.build_embedded(config_file=config_file),
        )
        gw = await awaitable
        async with gw:
            yield gw

    @staticmethod
    @asynccontextmanager
    async def with_clickhouse(
        config_file: str,
        clickhouse_url: str,
    ) -> AsyncIterator[AsyncTensorZeroGateway]:
        """Build an embedded gateway with ClickHouse observability."""
        awaitable = cast(
            "Awaitable[AsyncTensorZeroGateway]",
            AsyncTensorZeroGateway.build_embedded(
                config_file=config_file,
                clickhouse_url=clickhouse_url,
            ),
        )
        gw = await awaitable
        async with gw:
            yield gw

    @staticmethod
    @asynccontextmanager
    async def with_postgres(
        config_file: str,
        postgres_url: str,
    ) -> AsyncIterator[AsyncTensorZeroGateway]:
        """Build an embedded gateway with a Postgres backend."""
        awaitable = cast(
            "Awaitable[AsyncTensorZeroGateway]",
            AsyncTensorZeroGateway.build_embedded(
                config_file=config_file,
                postgres_url=postgres_url,
            ),
        )
        gw = await awaitable
        async with gw:
            yield gw


__all__ = ["EmbeddedGatewayHelper"]
