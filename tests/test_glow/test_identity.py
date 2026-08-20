"""Identity isolation battery — the M4 security gate.

One long-lived App serving many identities must never leak across
conversations.  Every test here drives the FULL conversation path
(``conversation(...)`` → send → per-run override → runtime view → pooled
client → real FastMCP boundary), asserting the four isolation planes:

- **auth**: each conversation's credentials reach the wire, witnessed by
  a real HTTP server echoing the identity header back through the tool
  result;
- **elicitation**: concurrent HITL prompts land on their own
  conversation's handler, with in-stream events carrying the
  forwarder's resolvable id;
- **memory**: per-conversation providers never observe another
  conversation's traffic;
- **tools**: per-conversation visibility is what inference actually
  sees.

Plus the pool-model math (``client_counts``) and the degenerate-case
guarantee that an identity-less conversation creates nothing new.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from fastmcp import Context, FastMCP
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    RequestEnvelope,
)
from t0_agents.glow import App, MCPServer
from t0_agents.memory.base import MemoryRecord
from t0_agents.stream.events import (
    ElicitationRequestedEvent,
    ElicitationResolvedEvent,
)
from t0_agents.testing import MockGateway, tool_call
from tests.test_glow.test_mcp_authed_mounts import _ServiceAuth
from tests.test_mcp.test_runtime import _ephemeral_http_server


def _keyed_gateway(
    tool_name: str | None,
) -> tuple[MockGateway, dict[str, list[str]], dict[str, list[str]]]:
    """A concurrency-safe scripted gateway keyed by episode id.

    A fresh user turn returns a call to ``tool_name`` (or plain text
    when ``None``); a turn whose last message carries a tool result
    collects it and returns ``"done"`` — so multi-send episodes replay
    the same tool-then-done shape per send.  Records, per episode: the
    tool-result strings the model would see (``witness``) and the
    advertised tool names (``tools_seen``).
    """
    witness: dict[str, list[str]] = {}
    tools_seen: dict[str, list[str]] = {}

    def source(**kwargs: Any) -> list[Any]:
        episode = str(kwargs.get("episode_id"))
        advertised = kwargs.get("additional_tools") or []
        tools_seen[episode] = sorted(tool["name"] for tool in advertised)
        last_content = kwargs["input"]["messages"][-1].get("content")
        results = [
            block_result
            for block in (
                last_content if isinstance(last_content, list) else []
            )
            if isinstance(
                (block_result := getattr(block, "result", None)), str
            )
        ]
        if results:
            witness.setdefault(episode, []).extend(results)
            return [Text(type="text", text="done")]
        if tool_name is not None:
            return [tool_call(tool_name)]
        return [Text(type="text", text="done")]

    return MockGateway.canned(source), witness, tools_seen


def _whoami_server() -> FastMCP[Any]:
    from fastmcp.server.dependencies import get_http_headers

    server: FastMCP[Any] = FastMCP("recorder")

    @server.tool
    async def whoami() -> str:
        """Echo the caller's identity header."""
        return get_http_headers().get("x-user", "<none>")

    return server


def _elicit_tag_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("hitl")

    @server.tool
    async def ask(ctx: Context) -> str:
        """Ask the human for a tag."""
        result = await ctx.elicit("Which tag?", response_type=str)
        if result.action != "accept":
            return "declined"
        return f"got:{result.data}"

    return server


def _tag_handler(tag: str, log: list[str]) -> Any:
    async def handler(
        message: str, response_type: Any, params: Any, context: Any
    ) -> Any:
        log.append(tag)
        return {"value": tag}

    return handler


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


# ---------------------------------------------------------------------------
# (a) auth — the wire witness
# ---------------------------------------------------------------------------


async def test_two_identities_on_one_app_reach_the_wire_isolated() -> None:
    """Two concurrent conversations with distinct credentials: each tool
    RESULT proves its own header reached the server (stronger than a
    seen-set — the response is paired to the caller)."""
    gateway, witness, _ = _keyed_gateway("svc_whoami")
    async with _ephemeral_http_server(_whoami_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, gateway),
            servers=[
                MCPServer(
                    url=url,
                    namespace="svc",
                    auth=_ServiceAuth("service-token"),
                )
            ],
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """An identity-aware agent."""

        async with app.lifespan():
            conv_a = chat.conversation(auth={"svc": {"X-User": "u-a"}})
            conv_b = chat.conversation(auth={"svc": {"X-User": "u-b"}})
            replies = await asyncio.gather(
                conv_a.send("who am I?"), conv_b.send("who am I?")
            )
            assert list(replies) == ["done", "done"]
            runtime = app._owned_client
            assert runtime is not None
            counts = runtime.client_counts()
            assert counts.identity == 2
            assert counts.conversation == 0

    assert witness[conv_a.id] == ["u-a"]
    assert witness[conv_b.id] == ["u-b"]


async def test_credential_rotation_mints_a_fresh_pooled_client() -> None:
    """Reopening a conversation with rotated credentials routes the next
    send through a NEW pooled client carrying the new header."""
    gateway, witness, _ = _keyed_gateway("svc_whoami")
    async with _ephemeral_http_server(_whoami_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, gateway),
            servers=[MCPServer(url=url, namespace="svc")],
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """An identity-aware agent."""

        async with app.lifespan():
            conversation_id = None
            conv_1 = chat.conversation(auth={"svc": {"X-User": "u-old"}})
            conversation_id = conv_1.id
            await conv_1.send("hi")

            conv_2 = chat.conversation(
                id=conversation_id, auth={"svc": {"X-User": "u-new"}}
            )
            await conv_2.send("hi again")

            runtime = app._owned_client
            assert runtime is not None
            assert runtime.client_counts().identity == 2

    assert witness[conversation_id] == ["u-old", "u-new"]


# ---------------------------------------------------------------------------
# (b) elicitation — concurrent routing + (h) in-stream id unification
# ---------------------------------------------------------------------------


async def test_interleaved_elicits_route_to_their_own_conversations() -> None:
    """Two conversations elicit concurrently over multiple rounds; every
    prompt lands on its own conversation's handler and every tool result
    carries its own conversation's tag."""
    gateway, witness, _ = _keyed_gateway("svc_ask")
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[
            MCPServer(server=_elicit_tag_server(), namespace="svc", elicits=True)
        ],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A HITL agent."""

    async with app.lifespan():
        log_a: list[str] = []
        log_b: list[str] = []
        conv_a = chat.conversation(elicitation_handler=_tag_handler("a", log_a))
        conv_b = chat.conversation(elicitation_handler=_tag_handler("b", log_b))

        for _ in range(3):
            await asyncio.gather(conv_a.send("go"), conv_b.send("go"))

        assert log_a == ["a", "a", "a"]
        assert log_b == ["b", "b", "b"]
        runtime = app._owned_client
        assert runtime is not None
        counts = runtime.client_counts()
        assert counts.conversation == 2
        assert counts.identity == 0

    assert witness[conv_a.id] == ["got:a"] * 3
    assert witness[conv_b.id] == ["got:b"] * 3


async def test_in_stream_elicitation_events_carry_the_forwarder_id() -> None:
    """A streaming send through an eliciting tool emits
    ElicitationRequested/Resolved whose ``id`` IS the forwarder's
    ``elicitation_id`` — the id an out-of-band resolver answers (M7's
    unification contract).  Scripted on FakeT0: MockGateway has no
    streaming surface, and events are streaming-only."""
    from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk, tool_call_chunk

    gateway = FakeT0(
        responses=[
            [
                chat_chunk(
                    [
                        tool_call_chunk(
                            raw_name="svc_ask",
                            raw_arguments="{}",
                            block_id="c1",
                        )
                    ]
                ),
                final_chunk(),
            ],
            [
                chat_chunk([text_chunk("done", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[
            MCPServer(server=_elicit_tag_server(), namespace="svc", elicits=True)
        ],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A HITL agent."""

    envelopes: list[RequestEnvelope] = []
    cell: list[ElicitationForwarder] = []

    async def on_request(envelope: RequestEnvelope) -> None:
        envelopes.append(envelope)
        await cell[0].resolve(
            envelope["elicitation_id"],
            {"action": "accept", "content": {"value": "tag"}},
        )

    async with app.lifespan():
        forwarder = ElicitationForwarder(on_request=on_request)
        cell.append(forwarder)
        conv = chat.conversation(elicitation_handler=forwarder)

        events: list[Any] = []
        async for event in conv.send("go").events():
            events.append(event)

    requested = [e for e in events if isinstance(e, ElicitationRequestedEvent)]
    resolved = [e for e in events if isinstance(e, ElicitationResolvedEvent)]
    assert len(requested) == 1 and len(resolved) == 1
    assert len(envelopes) == 1
    assert requested[0].id == envelopes[0]["elicitation_id"]
    assert resolved[0].id == requested[0].id
    assert resolved[0].action == "accept"
    assert requested[0].message == "Which tag?"
    assert requested[0].schema_json != ""


async def test_in_stream_elicitation_events_survive_a_derived_send() -> None:
    """Consuming a ``send(...).with_(...)``-derived handle still delivers the
    in-stream ElicitationRequested/Resolved pair.  ``.with_()`` derives a NEW
    FluentRun, so the conversation must observe the run actually consumed —
    not the handle ``send()`` returned (the v2.2.0 bug consumers patched by
    re-pointing ``conv._pending`` by hand)."""
    from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk, tool_call_chunk

    gateway = FakeT0(
        responses=[
            [
                chat_chunk(
                    [
                        tool_call_chunk(
                            raw_name="svc_ask",
                            raw_arguments="{}",
                            block_id="c1",
                        )
                    ]
                ),
                final_chunk(),
            ],
            [
                chat_chunk([text_chunk("done", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[
            MCPServer(server=_elicit_tag_server(), namespace="svc", elicits=True)
        ],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A HITL agent."""

    cell: list[ElicitationForwarder] = []

    async def on_request(envelope: RequestEnvelope) -> None:
        await cell[0].resolve(
            envelope["elicitation_id"],
            {"action": "accept", "content": {"value": "tag"}},
        )

    async with app.lifespan():
        forwarder = ElicitationForwarder(on_request=on_request)
        cell.append(forwarder)
        conv = chat.conversation(elicitation_handler=forwarder)

        events: list[Any] = []
        derived = conv.send("go").with_(tags={"topic": "hitl"})
        async for event in derived.events():
            events.append(event)

    requested = [e for e in events if isinstance(e, ElicitationRequestedEvent)]
    resolved = [e for e in events if isinstance(e, ElicitationResolvedEvent)]
    assert len(requested) == 1 and len(resolved) == 1
    assert resolved[0].id == requested[0].id
    assert resolved[0].action == "accept"


def _elicit_turn_scripts() -> list[Any]:
    """Two scripted model turns: call ``svc_ask``, then close with text."""
    from tests._fake_t0 import chat_chunk, final_chunk, text_chunk, tool_call_chunk

    return [
        [
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="svc_ask",
                        raw_arguments="{}",
                        block_id="c1",
                    )
                ]
            ),
            final_chunk(),
        ],
        [
            chat_chunk([text_chunk("done", block_id="t1")]),
            final_chunk(),
        ],
    ]


async def test_elicitation_survives_a_rebuilt_conversation_handle() -> None:
    """A consumer that reopens a conversation per send (a fresh Conversation
    object over one id — the documented resume pattern) still gets in-stream
    elicitation events on the send actually being driven.  The pooled
    per-conversation elicit client outlives the handle that minted it, so its
    dispatch must follow the CURRENT handle: a frozen first-handle binding
    routes every later prompt onto the first handle's settled run, where it
    is dropped as ``run buffer already closed`` — and the resolve lands on
    the first handle's (abandoned) forwarder."""
    from tests._fake_t0 import FakeT0

    gateway = FakeT0(responses=_elicit_turn_scripts() + _elicit_turn_scripts())
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[
            MCPServer(server=_elicit_tag_server(), namespace="svc", elicits=True)
        ],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A HITL agent."""

    def _resolving_forwarder(
        envelopes: list[RequestEnvelope],
    ) -> ElicitationForwarder:
        cell: list[ElicitationForwarder] = []

        async def on_request(envelope: RequestEnvelope) -> None:
            envelopes.append(envelope)
            await cell[0].resolve(
                envelope["elicitation_id"],
                {"action": "accept", "content": {"value": "tag"}},
            )

        forwarder = ElicitationForwarder(on_request=on_request)
        cell.append(forwarder)
        return forwarder

    async with app.lifespan():
        envelopes_1: list[RequestEnvelope] = []
        conv_1 = chat.conversation(
            elicitation_handler=_resolving_forwarder(envelopes_1)
        )
        async for _ in conv_1.send("go").events():
            pass

        envelopes_2: list[RequestEnvelope] = []
        conv_2 = chat.conversation(
            id=conv_1.id, elicitation_handler=_resolving_forwarder(envelopes_2)
        )
        events: list[Any] = []
        async for event in conv_2.send("go").events():
            events.append(event)

    requested = [e for e in events if isinstance(e, ElicitationRequestedEvent)]
    resolved = [e for e in events if isinstance(e, ElicitationResolvedEvent)]
    assert len(envelopes_1) == 1
    assert len(envelopes_2) == 1
    assert len(requested) == 1 and len(resolved) == 1
    assert requested[0].id == envelopes_2[0]["elicitation_id"]
    assert resolved[0].id == requested[0].id
    assert resolved[0].action == "accept"


async def test_elicitation_survives_a_rebuilt_handle_with_a_bare_handler() -> None:
    """The same reopened-conversation shape through a plain
    ``ElicitationHandler`` (the ``_ObservedHandler`` wrap): the second send's
    stream still carries the Requested/Resolved pair and the second handle's
    handler answers.  Covers the non-forwarder emit path — the fix lives
    below both observer shapes."""
    from tests._fake_t0 import FakeT0

    gateway = FakeT0(responses=_elicit_turn_scripts() + _elicit_turn_scripts())
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[
            MCPServer(server=_elicit_tag_server(), namespace="svc", elicits=True)
        ],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A HITL agent."""

    handled: list[str] = []
    async with app.lifespan():
        conv_1 = chat.conversation(
            elicitation_handler=_tag_handler("one", handled)
        )
        async for _ in conv_1.send("go").events():
            pass

        conv_2 = chat.conversation(
            id=conv_1.id, elicitation_handler=_tag_handler("two", handled)
        )
        events: list[Any] = []
        async for event in conv_2.send("go").events():
            events.append(event)

    requested = [e for e in events if isinstance(e, ElicitationRequestedEvent)]
    resolved = [e for e in events if isinstance(e, ElicitationResolvedEvent)]
    assert handled == ["one", "two"]
    assert len(requested) == 1 and len(resolved) == 1
    assert resolved[0].id == requested[0].id
    assert resolved[0].action == "accept"


async def test_undeclared_elicit_fails_loud_and_never_misroutes() -> None:
    """A server that elicits WITHOUT declaring elicits=True rides a
    handler-less identity client: the elicit fails loud (capability not
    advertised) and the conversation's handler is never consulted."""
    gateway, witness, _ = _keyed_gateway("svc_ask")
    async with _ephemeral_http_server(_elicit_tag_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, gateway),
            servers=[MCPServer(url=url, namespace="svc")],  # NOT elicits=True
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A HITL agent whose server forgot to declare elicits."""

        async with app.lifespan():
            log: list[str] = []
            conv = chat.conversation(
                auth={"svc": {"X-User": "u-a"}},
                elicitation_handler=_tag_handler("a", log),
            )
            reply = await conv.send("go")
            assert reply == "done"
            assert log == []

    (result_text,) = witness[conv.id]
    assert result_text.startswith("ERROR:")


# ---------------------------------------------------------------------------
# (c) memory isolation
# ---------------------------------------------------------------------------


async def test_memory_never_crosses_conversations() -> None:
    base = RecordingMemory()
    memory_a = RecordingMemory()
    memory_b = RecordingMemory()
    gateway, _, _ = _keyed_gateway(None)
    app = App(t0=cast(AsyncTensorZeroGateway, gateway))

    @app.agent("chat", memory=base)
    async def chat(q: str) -> str:
        """A memory-backed agent."""

    conv_a = chat.conversation(memory=memory_a)
    conv_b = chat.conversation(memory=memory_b)

    for _ in range(2):
        await asyncio.gather(conv_a.send("hi"), conv_b.send("hi"))

    assert memory_a.loads == [conv_a.id, conv_a.id]
    assert memory_b.loads == [conv_b.id, conv_b.id]
    assert len(memory_a.saves) == 2
    assert len(memory_b.saves) == 2
    assert not base.loads and not base.saves


# ---------------------------------------------------------------------------
# (d) tool visibility is what inference sees
# ---------------------------------------------------------------------------


async def test_per_conversation_visibility_reaches_inference() -> None:
    server: FastMCP[Any] = FastMCP("svc")

    @server.tool
    async def read_thing() -> str:
        """Read."""
        return "read"

    @server.tool
    async def write_thing() -> str:
        """Write."""
        return "write"

    gateway, _, tools_seen = _keyed_gateway(None)
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[MCPServer(server=server, namespace="svc")],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A scoped agent."""

    async with app.lifespan():
        narrow = chat.conversation(tools=["svc_read_*"])
        wide = chat.conversation()
        await asyncio.gather(narrow.send("go"), wide.send("go"))

    assert tools_seen[narrow.id] == ["svc_read_thing"]
    assert tools_seen[wide.id] == ["svc_read_thing", "svc_write_thing"]


# ---------------------------------------------------------------------------
# (f) pool-model math + (i) the degenerate case
# ---------------------------------------------------------------------------


async def test_client_counts_match_the_pool_model() -> None:
    """Static servers share one client; authed servers pool per identity;
    elicits=True servers cost one client per CONVERSATION — never
    conversations x servers."""
    gateway, _, _ = _keyed_gateway(None)
    async with _ephemeral_http_server(_whoami_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, gateway),
            servers=[
                MCPServer(server=_named("public", "info"), namespace="public"),
                MCPServer(url=url, namespace="authed"),
                MCPServer(
                    server=_elicit_tag_server(),
                    namespace="hitl",
                    elicits=True,
                ),
            ],
        )

        @app.agent("chat")
        async def chat(q: str) -> str:
            """A multi-server agent."""

        async with app.lifespan():
            runtime = app._owned_client
            assert runtime is not None
            same_identity = {"authed": {"X-User": "shared"}}
            log: list[str] = []
            conv_a = chat.conversation(
                auth=same_identity, elicitation_handler=_tag_handler("a", log)
            )
            conv_b = chat.conversation(
                auth=same_identity, elicitation_handler=_tag_handler("b", log)
            )
            conv_c = chat.conversation(
                auth={"authed": {"X-User": "other"}},
                elicitation_handler=_tag_handler("c", log),
            )

            for conv in (conv_a, conv_b, conv_c):
                provider = conv._conversation_tool_provider()
                assert provider is not None
                await provider.call_tool("public_info", {})
                await provider.call_tool("authed_whoami", {})
                await provider.call_tool("hitl_ask", {})

            counts = runtime.client_counts()
            assert counts.static == 3
            assert counts.identity == 2  # {shared, other} on the authed server
            assert counts.conversation == 3  # one per conversation on hitl
            assert log == ["a", "b", "c"]


def _named(server_name: str, tool_name: str) -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP(server_name)

    async def impl() -> str:
        return f"{server_name}:{tool_name}"

    impl.__name__ = tool_name
    impl.__doc__ = f"Return the {tool_name} marker."
    server.tool(impl)
    return server


async def test_identity_less_conversation_creates_no_pool_entries() -> None:
    gateway, witness, _ = _keyed_gateway("svc_info")
    app = App(
        t0=cast(AsyncTensorZeroGateway, gateway),
        servers=[MCPServer(server=_named("svc", "info"), namespace="svc")],
    )

    @app.agent("chat")
    async def chat(q: str) -> str:
        """A plain agent."""

    async with app.lifespan():
        conv = chat.conversation()
        reply = await conv.send("go")
        assert reply == "done"
        runtime = app._owned_client
        assert runtime is not None
        counts = runtime.client_counts()
        assert (counts.identity, counts.conversation) == (0, 0)
    assert witness[conv.id] == ["svc:info"]
