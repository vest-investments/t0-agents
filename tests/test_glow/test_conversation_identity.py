"""Conversation identity legs — auth overlays, memory swap, elicitation
handler resolution.

Unit-level coverage of the conversation surface added for the long-lived
multi-tenant App; the cross-conversation isolation battery (concurrency,
wire witnesses, event-id unification) lives in ``test_identity.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core._runtime import EventBuffer
from t0_agents.core.elicitation_forwarder import ElicitationForwarder
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, MCPServer, ServerCredentials
from t0_agents.glow.conversation import (
    _normalize_credential_overlays,
    _ObservedHandler,
)
from t0_agents.memory.base import MemoryRecord
from t0_agents.testing import MockGateway


class _HeaderAuth(httpx.Auth):
    def auth_flow(self, request: httpx.Request) -> Any:
        request.headers["X-Auth"] = "on"
        yield request


class RecordingMemory:
    def __init__(self) -> None:
        self.loads: list[str] = []
        self.saves: list[MemoryRecord] = []

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        self.loads.append(episode_id)
        return []

    async def save(self, record: MemoryRecord) -> None:
        self.saves.append(record)

    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        return []


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


class TestCredentialNormalization:
    def test_headers_mapping_sugar(self) -> None:
        resolved = _normalize_credential_overlays({"svc": {"X-User": "u1"}})
        assert resolved["svc"] == ServerCredentials(headers={"X-User": "u1"})

    def test_httpx_auth_sugar(self) -> None:
        auth = _HeaderAuth()
        resolved = _normalize_credential_overlays({"svc": auth})
        assert resolved["svc"].auth is auth
        assert resolved["svc"].headers is None

    def test_server_credentials_pass_through(self) -> None:
        cred = ServerCredentials(headers={"X-User": "u1"}, key="u1")
        assert _normalize_credential_overlays({"svc": cred})["svc"] is cred

    def test_invalid_value_fails_loud(self) -> None:
        with pytest.raises(ConfigurationError, match="svc"):
            _normalize_credential_overlays({"svc": cast(Any, 42)})


class TestConversationProviderResolution:
    def test_degenerate_conversation_overrides_nothing(self) -> None:
        """No auth / no handler → no provider override: the run rides the
        agent's own provider, byte-identical to the pre-identity path."""
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        conv = chat.conversation()
        assert conv._conversation_tool_provider() is None

    def test_auth_without_mcp_surface_fails_loud(self) -> None:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        conv = chat.conversation(auth={"svc": {"X-User": "u1"}})
        with pytest.raises(ConfigurationError, match="mounts no MCP servers"):
            conv._conversation_tool_provider()

    async def test_identity_conversation_builds_a_runtime_view(self) -> None:
        from fastmcp import FastMCP

        server: FastMCP[Any] = FastMCP("svc")

        @server.tool
        async def ping() -> str:
            """Return a marker."""
            return "pong"

        app = App(
            t0=_fake_t0(), servers=[MCPServer(server=server, namespace="svc")]
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        async with app.lifespan():
            conv = chat.conversation(
                elicitation_handler=ElicitationForwarder()
            )
            provider = conv._conversation_tool_provider()
            assert provider is not None
            names = {tool.name for tool in await provider.list_tools()}
            assert names == {"svc_ping"}

    async def test_agent_tools_filter_reapplied_on_the_view(self) -> None:
        """The identity view must never widen the agent's tool surface —
        the agent-level allow-list wraps the view."""
        from fastmcp import FastMCP

        server: FastMCP[Any] = FastMCP("svc")

        @server.tool
        async def visible() -> str:
            """Allowed."""
            return "v"

        @server.tool
        async def hidden() -> str:
            """Filtered out."""
            return "h"

        app = App(
            t0=_fake_t0(), servers=[MCPServer(server=server, namespace="svc")]
        )

        @app.agent("chat", tools=["svc_visible"])
        async def chat(q: str) -> str:
            """A filtered agent."""

        async with app.lifespan():
            conv = chat.conversation(
                elicitation_handler=ElicitationForwarder()
            )
            provider = conv._conversation_tool_provider()
            assert provider is not None
            names = {tool.name for tool in await provider.list_tools()}
            assert names == {"svc_visible"}


class TestConversationMemory:
    def test_memory_requires_memory_enabled_agent(self) -> None:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A memory-less agent."""

        with pytest.raises(ConfigurationError, match="not memory-enabled"):
            chat.conversation(memory=RecordingMemory())

    async def test_conversation_memory_swaps_the_provider_per_send(self) -> None:
        base = RecordingMemory()
        scoped = RecordingMemory()
        gateway = MockGateway.script([[Text(type="text", text="hola")]])
        app = App(t0=cast(AsyncTensorZeroGateway, gateway))

        @app.agent("chat", memory=base)
        async def chat(q: str) -> str:
            """A memory-backed agent."""

        conv = chat.conversation(memory=scoped)
        reply = await conv.send("hey")
        assert reply == "hola"
        assert scoped.loads and scoped.saves
        assert not base.loads and not base.saves

    async def test_hydrate_uses_the_conversation_memory(self) -> None:
        class HistoryMemory(RecordingMemory):
            def __init__(self) -> None:
                super().__init__()
                self.history_loads: list[str] = []

            async def load_history(self, episode_id: str) -> Any:
                self.history_loads.append(episode_id)
                return None

        scoped = HistoryMemory()
        app = App(t0=_fake_t0())

        @app.agent("chat", memory=RecordingMemory())
        async def chat(q: str) -> str:
            """A memory-backed agent."""

        conv = chat.conversation(
            id="00000000-0000-0000-0000-000000000009",
            hydrate=True,
            memory=scoped,
        )
        await conv.hydrate()
        assert scoped.history_loads == ["00000000-0000-0000-0000-000000000009"]

    async def test_hydrate_rejects_history_less_conversation_memory(self) -> None:
        app = App(t0=_fake_t0())

        @app.agent("chat", memory=RecordingMemory())
        async def chat(q: str) -> str:
            """A memory-backed agent."""

        conv = chat.conversation(
            id="00000000-0000-0000-0000-000000000009",
            hydrate=True,
            memory=RecordingMemory(),
        )
        with pytest.raises(ConfigurationError, match="load_history"):
            await conv.hydrate()


class TestElicitationHandlerResolution:
    def _conversation(self, **kwargs: Any) -> Any:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        return chat.conversation(**kwargs)

    def test_no_handler_anywhere_resolves_none(self) -> None:
        conv = self._conversation()
        assert conv._current_elicitation_handler() is None

    def test_forwarder_resolves_to_its_handler_and_binds_observer(self) -> None:
        forwarder = ElicitationForwarder()
        conv = self._conversation(elicitation_handler=forwarder)
        resolved = conv._current_elicitation_handler()
        assert resolved == forwarder.handler
        assert forwarder.observer is not None

    def test_bare_handler_resolves_to_observed_wrapper(self) -> None:
        async def handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> Any:
            return {"ok": True}

        conv = self._conversation(elicitation_handler=handler)
        resolved = conv._current_elicitation_handler()
        assert isinstance(resolved, _ObservedHandler)

    def test_app_level_handler_is_the_fallback(self) -> None:
        async def app_handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> Any:
            return {"ok": True}

        app = App(t0=_fake_t0(), elicitation_handler=app_handler)

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        conv = chat.conversation()
        resolved = conv._current_elicitation_handler()
        assert isinstance(resolved, _ObservedHandler)

    def test_handler_is_rebindable_between_turns(self) -> None:
        first = ElicitationForwarder()
        second = ElicitationForwarder()
        conv = self._conversation(elicitation_handler=first)
        assert conv._current_elicitation_handler() == first.handler
        conv.elicitation_handler = second
        assert conv._current_elicitation_handler() == second.handler
        assert second.observer is not None


class TestObservedHandlerEvents:
    async def _emitting_conversation(self) -> tuple[Any, EventBuffer]:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        conv = chat.conversation()
        buffer = EventBuffer()
        conv._pending = cast(
            Any, SimpleNamespace(_run=SimpleNamespace(_buffer=buffer))
        )
        return conv, buffer

    async def _drain(self, buffer: EventBuffer, count: int) -> list[Any]:
        events: list[Any] = []
        async for event in buffer.iter_from(start_cursor=0):
            events.append(event)
            if len(events) == count:
                break
        return events

    async def test_plain_return_emits_accept_pair(self) -> None:
        conv, buffer = await self._emitting_conversation()

        async def handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> Any:
            return {"value": True}

        observed = _ObservedHandler(handler, conv)
        result = await observed("Confirm?", None, None, None)
        assert result == {"value": True}

        requested, resolved = await self._drain(buffer, 2)
        assert requested.type == "elicitation_requested"
        assert requested.message == "Confirm?"
        assert resolved.type == "elicitation_resolved"
        assert resolved.action == "accept"
        assert resolved.id == requested.id

    async def test_raising_handler_emits_decline_and_propagates(self) -> None:
        conv, buffer = await self._emitting_conversation()

        async def handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> Any:
            raise RuntimeError("no channel")

        observed = _ObservedHandler(handler, conv)
        with pytest.raises(RuntimeError, match="no channel"):
            await observed("Confirm?", None, None, None)

        _, resolved = await self._drain(buffer, 2)
        assert resolved.action == "decline"

    async def test_emit_without_active_run_is_a_noop(self) -> None:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        conv = chat.conversation()

        async def handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> Any:
            return {"ok": True}

        observed = _ObservedHandler(handler, conv)
        result = await asyncio.wait_for(
            observed("Confirm?", None, None, None), timeout=1.0
        )
        assert result == {"ok": True}


class TestConversationToolScoping:
    def _app_with_tools(self) -> tuple[App, Any]:
        from fastmcp import FastMCP

        server: FastMCP[Any] = FastMCP("svc")

        @server.tool
        async def read_thing() -> str:
            """Read."""
            return "read"

        @server.tool
        async def write_thing() -> str:
            """Write."""
            return "write"

        app = App(
            t0=_fake_t0(), servers=[MCPServer(server=server, namespace="svc")]
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        return app, chat

    async def test_tools_narrow_visibility_and_block_forged_calls(self) -> None:
        from t0_agents.core.exceptions import T0AgentsError

        app, chat = self._app_with_tools()
        async with app.lifespan():
            conv = chat.conversation(tools=["svc_read_*"])
            await conv._prepare_send()
            provider = conv._conversation_tool_provider()
            assert provider is not None
            names = {tool.name for tool in await provider.list_tools()}
            assert names == {"svc_read_thing"}
            with pytest.raises(T0AgentsError, match="not permitted"):
                await provider.call_tool("svc_write_thing", {})

    async def test_conversation_filter_intersects_the_agent_filter(self) -> None:
        from fastmcp import FastMCP

        server: FastMCP[Any] = FastMCP("svc")

        @server.tool
        async def alpha() -> str:
            """A."""
            return "a"

        @server.tool
        async def beta() -> str:
            """B."""
            return "b"

        @server.tool
        async def gamma() -> str:
            """C."""
            return "c"

        app = App(
            t0=_fake_t0(), servers=[MCPServer(server=server, namespace="svc")]
        )

        @app.agent("chat", tools=["svc_alpha", "svc_beta"])
        async def chat(q: str) -> str:
            """A filtered agent."""

        async with app.lifespan():
            conv = chat.conversation(tools=["svc_beta", "svc_gamma"])
            await conv._prepare_send()
            provider = conv._conversation_tool_provider()
            assert provider is not None
            names = {tool.name for tool in await provider.list_tools()}
            assert names == {"svc_beta"}

    async def test_empty_tools_is_a_toolless_conversation(self) -> None:
        app, chat = self._app_with_tools()
        async with app.lifespan():
            conv = chat.conversation(tools=[])
            await conv._prepare_send()
            provider = conv._conversation_tool_provider()
            assert provider is not None
            assert await provider.list_tools() == []

    async def test_dead_pattern_warns_at_resolution(self) -> None:
        app, chat = self._app_with_tools()
        async with app.lifespan():
            conv = chat.conversation(tools=["nope_*"])
            with pytest.warns(UserWarning, match="matched no available tool"):
                await conv._prepare_send()

    async def test_tools_without_mcp_surface_fails_loud(self) -> None:
        app = App(t0=_fake_t0())

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A plain agent."""

        async with app.lifespan():
            conv = chat.conversation(tools=["anything"])
            with pytest.raises(ConfigurationError, match="nothing to filter"):
                await conv._prepare_send()
