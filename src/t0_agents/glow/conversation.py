"""``Conversation`` — a multi-turn handle over one TensorZero episode.

``agent.conversation()`` is glow's answer to "how do I say a second thing?".
It mints an episode, threads it through every :meth:`~Conversation.send`, keeps
an in-process transcript, and pins the invariants that make multi-turn safe:

- **``conv.id == episode_id``** — the conversation *is* its episode.  Minted as
  a UUIDv7 (TensorZero's requirement) at construction, or supplied via
  ``conversation(id=...)`` to resume an existing one.
- **memory-strategy-aware history** — a send always carries ``episode_id``; when
  the agent runs ``memory_strategy="messages"`` the send passes *only* that and
  lets the memory provider weave history from ``load_history`` (passing
  ``prior_messages`` too would silently suppress it — the E1 double-injection
  trap).  Otherwise the send seeds the in-process transcript as
  ``prior_messages`` (the ``Run.fork`` pattern), so multi-turn works for *any*
  agent, memory-backed or not.
- **aligned tags** — ``user=`` / ``tags=`` set the write tags stamped on every
  send *and* the read filter used by :meth:`hydrate` and ``app.conversations``,
  from one dict — closing the mismatched-tags-read-nothing trap.
- **sequential sends** — a send left unconsumed blocks the next one, so turns
  cannot interleave on a shared episode.

The cross-conversation *inbox* is deliberately not here: listing lives on
``app.conversations`` (dev-grade) or a pluggable
:class:`~t0_agents.core.conversation.ConversationIndex` (production).  This
class only ever owns *one* conversation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast
from uuid import uuid4

import httpx

from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    _response_type_from_schema,
)
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.mcp.runtime import ServerCredentials
from t0_agents.memory.tensorzero import TensorZeroMemory
from t0_agents.stream.events import (
    ElicitationRequestedEvent,
    ElicitationResolvedEvent,
)
from t0_agents.tooling import FilteredToolProvider

if TYPE_CHECKING:
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core._runtime import EventBuffer
    from t0_agents.core.elicitation_forwarder import RequestEnvelope
    from t0_agents.core.types import AgentResult, MemoryProvider, ToolProvider
    from t0_agents.glow._spec import _AgentSpec
    from t0_agents.glow.app import App
    from t0_agents.glow.handle import FluentRun

__all__ = ["Conversation"]

R = TypeVar("R")

_VALID_ACTIONS = ("accept", "decline", "cancel")


def _normalize_credential_overlays(
    auth: Mapping[str, ServerCredentials | Mapping[str, str] | httpx.Auth]
    | None,
) -> dict[str, ServerCredentials]:
    """Resolve ``conversation(auth=...)`` sugar to ``ServerCredentials``.

    A plain mapping is headers-only (the common per-user-identity case),
    a bare ``httpx.Auth`` is auth-only; ``ServerCredentials`` passes
    through.  Anything else fails loud — a credential value that
    silently applies as nothing is a security bug.
    """
    resolved: dict[str, ServerCredentials] = {}
    for namespace, value in (auth or {}).items():
        if isinstance(value, ServerCredentials):
            resolved[namespace] = value
        elif isinstance(value, httpx.Auth):
            resolved[namespace] = ServerCredentials(auth=value)
        elif isinstance(value, Mapping):
            resolved[namespace] = ServerCredentials(headers=dict(value))
        else:
            raise ConfigurationError(
                f"conversation(auth=...): value for namespace "
                f"{namespace!r} must be ServerCredentials, a headers "
                f"mapping, or an httpx.Auth; got {type(value).__name__}."
            )
    return resolved


def _settled_action(action: str) -> Literal["accept", "decline", "cancel"]:
    if action in _VALID_ACTIONS:
        return cast('Literal["accept", "decline", "cancel"]', action)
    return "decline"


class Conversation(Generic[R]):
    """A multi-turn handle over one episode of a glow agent.

    Construct via :meth:`~t0_agents.glow.decorators._AgentWrapper.conversation`
    (``agent.conversation(...)``), never directly.  ``R`` is the agent's return
    type, so :meth:`send` yields a ``FluentRun[R]`` — the exact handle a plain
    ``agent(msg)`` returns, with ``.stream_text()`` / ``.json()`` / ``.with_()``
    intact.

    :attr:`transcript` is the in-process record of the turns *this* handle has
    completed (plus any :meth:`hydrate`-d prefix), as boundary pairs — it
    inherits :meth:`Run.transcript`'s limitation of capturing only each turn's
    final assistant reply, not intermediate tool turns.
    """

    def __init__(
        self,
        app: App,
        spec: _AgentSpec,
        *,
        id: str,
        tags: dict[str, str],
        memory_strategy: str,
        needs_hydration: bool,
        auth: dict[str, ServerCredentials] | None = None,
        elicitation: ElicitationForwarder | ElicitationHandler | None = None,
        memory: MemoryProvider | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self._app = app
        self._spec = spec
        self._id = id
        # Write tags stamped on every send AND the read filter for hydrate /
        # listing — one dict, so the two never drift (the E1 tag-alignment fix).
        self._tags = tags
        self._memory_strategy = memory_strategy
        self._needs_hydration = needs_hydration
        self._hydrated = False
        # Identity legs.  ``_auth`` maps server namespaces to credential
        # overlays (resolved per send into the runtime's identity pool);
        # ``elicitation_handler`` is public and rebindable — the runtime
        # trampoline re-resolves it at elicit time, so swapping handlers
        # between turns takes effect immediately; ``_memory`` swaps the
        # memory provider for every send of THIS conversation (tenant
        # scoping on a shared agent).
        self._auth: dict[str, ServerCredentials] = dict(auth or {})
        self.elicitation_handler: (
            ElicitationForwarder | ElicitationHandler | None
        ) = elicitation
        self._memory = memory
        # Per-conversation tool visibility (``@app.agent(tools=)``
        # grammar); resolved against the live tool list on the first
        # send, then intersected with the agent-level filter.
        self._tools = tools
        self._resolved_tools: set[str] | None = None
        # Boundary-pair display transcript; also the prior_messages seed for
        # non-"messages" agents.
        self._transcript: list[dict[str, str]] = []
        self._pending: FluentRun[R] | None = None

    @property
    def id(self) -> str:
        """The conversation's identity — equal to its TensorZero ``episode_id``.

        A UUIDv7 (minted at construction, or the ``id=`` passed to resume).
        Every :meth:`send` runs under this episode; the equality is the pinned
        P5 contract that lets a caller store one id and later resume.
        """
        return self._id

    @property
    def transcript(self) -> list[Mapping[str, Any]]:
        """The turns completed on this handle, as ``{"role", "content"}`` pairs.

        A copy, oldest-first: any :meth:`hydrate`-d history followed by one
        ``user`` + one ``assistant`` pair per completed send.  Only each turn's
        *final* assistant reply is captured (intermediate tool turns are not) —
        the limitation inherited from :meth:`Run.transcript`.
        """
        return list(self._transcript)

    def send(self, message: str) -> FluentRun[R]:
        """Send one turn, returning the same ``FluentRun[R]`` a call would.

        Consume it however you like — ``await conv.send(q)``,
        ``conv.send(q).stream_text()``, ``.json(Model)``, ``.with_(...)`` — the
        turn is recorded to :attr:`transcript` once the run completes
        successfully, on whichever path drains it.

        Raises :class:`ConfigurationError` if the previous send has not been
        consumed: turns share one episode and must be sequential, so the prior
        ``FluentRun`` must be driven (awaited or its stream drained) before the
        next :meth:`send`.
        """
        if self._pending is not None and not self._pending.consumed:
            raise ConfigurationError(
                "Conversation.send(): the previous send() has not been "
                "consumed.  Await it, or drain its .stream_text() / .events(), "
                "before sending the next turn — turns share one episode "
                f"({self._id}) and must be sequential."
            )
        # "messages" mode: the provider weaves history from load_history, so
        # passing prior_messages would suppress it (the double-injection trap).
        # Every other strategy: seed the in-process transcript for continuity.
        prior = None if self._memory_strategy == "messages" else self._prior_snapshot
        fluent = cast(
            "FluentRun[R]",
            self._app._invoke_in_conversation(
                self._spec,
                message,
                episode_id=self._id,
                tags=self._tags,
                prior_messages=prior,
                on_result=lambda result: self._record(message, result),
                extra_prepare=self._prepare_send,
                tool_provider_factory=self._conversation_tool_provider,
                memory=self._memory,
            ),
        )
        self._pending = fluent
        return fluent

    async def hydrate(self) -> None:
        """Reconstruct :attr:`transcript` from the stored episode, once.

        Fetches the episode's prior messages via
        :meth:`TensorZeroMemory.load_history` (thoughts already stripped) and
        replaces the transcript with them.  Idempotent — the first hydrate
        wins, so it never clobbers turns recorded afterwards.  Called
        automatically before the first :meth:`send` when the conversation was
        opened with ``hydrate=True``; call it explicitly to inspect
        :attr:`transcript` *before* sending.
        """
        await self._hydrate()

    # ------------------------------------------------------------------
    # Internals — identity legs
    # ------------------------------------------------------------------

    async def _prepare_send(self) -> None:
        """Per-send prepare hook: hydration plus tool-filter resolution.

        Runs inside glow's auto-build prepare (after the App is booted),
        so the runtime exists when the ``tools=`` grammar is resolved
        against the live tool list.  Both halves are once-latched.
        """
        await self._maybe_hydrate()
        await self._resolve_conversation_tools()

    async def _resolve_conversation_tools(self) -> None:
        if self._tools is None or self._resolved_tools is not None:
            return
        runtime = self._app._owned_client
        if runtime is None:
            raise ConfigurationError(
                f"conversation(tools=...) on agent {self._spec.name!r}: "
                f"the App mounts no MCP servers or @app.tool functions, "
                f"so there is nothing to filter."
            )
        from t0_agents.glow.app import _resolve_tool_filter

        live_names = {tool.name for tool in await runtime.list_tools()}
        self._resolved_tools = _resolve_tool_filter(
            self._spec.name, self._tools, live_names
        )

    def _conversation_tool_provider(self) -> ToolProvider | None:
        """The per-send provider override, or ``None`` on the degenerate path.

        A conversation carrying no identity (no ``auth=``, no
        conversation-level elicitation handler, no ``tools=``) changes
        nothing: the run uses the agent's own provider exactly as before.
        With identity, every send gets a runtime view binding this
        conversation's overlays and (current) elicitation handler; the
        agent-level ``tools=`` filter — intersected with the
        conversation's own, when set — is re-applied on top so the
        override never widens the agent's tool surface.
        """
        if (
            not self._auth
            and self.elicitation_handler is None
            and self._tools is None
        ):
            return None
        runtime = self._app._owned_client
        if runtime is None:
            if self._auth:
                raise ConfigurationError(
                    f"conversation(auth=...) on agent {self._spec.name!r}: "
                    f"the App mounts no MCP servers, so there is nothing "
                    f"to authenticate against."
                )
            return None
        provider: ToolProvider = runtime.for_conversation(
            self._id,
            overlays=self._auth,
            elicitation_factory=self._current_elicitation_handler,
        )
        allow: set[str] | None = (
            set(self._spec.resolved_tools)
            if self._spec.resolved_tools is not None
            else None
        )
        if self._resolved_tools is not None:
            allow = (
                self._resolved_tools
                if allow is None
                else allow & self._resolved_tools
            )
        if allow is not None:
            provider = FilteredToolProvider(provider, allow)
        return provider

    def _current_elicitation_handler(self) -> ElicitationHandler | None:
        """Resolve the handler the runtime trampoline should use NOW.

        Reads the public (rebindable) :attr:`elicitation_handler` on
        every elicitation, falling back to the App-level handler.  A
        forwarder is observed in place — the in-stream event ``id`` is
        the forwarder's own ``elicitation_id``, so a remote resolver can
        answer it; a bare handler is wrapped with a glow-minted id
        (self-consistent in-stream, nothing out-of-band to unify with).
        """
        source = self.elicitation_handler
        if source is None:
            app_handler = self._app._elicitation_handler
            if app_handler is None:
                return None
            return cast(
                "ElicitationHandler", _ObservedHandler(app_handler, self)
            )
        if isinstance(source, ElicitationForwarder):
            source.observer = _ForwarderObserver(self)
            return source.handler
        return cast("ElicitationHandler", _ObservedHandler(source, self))

    def _active_buffer(self) -> EventBuffer | None:
        """The event buffer of the run this send actually built, if any.

        Reads the pending handle's lineage rather than the handle itself:
        ``send()`` returns the root ``FluentRun`` but callers routinely
        consume a derived one (``send(...).with_(...)`` / ``.json(...)``),
        and elicitation events must land on the run being driven.
        """
        pending = self._pending
        if pending is None:
            return None
        run = pending._lineage.run
        if run is None:
            return None
        return run._buffer

    async def _emit(
        self, event: ElicitationRequestedEvent | ElicitationResolvedEvent
    ) -> None:
        """Push an elicitation event onto the active run, best-effort.

        Events are streaming-only by architecture; a result-mode run
        buffers them harmlessly.  A buffer that already closed (the run
        settled while the elicitation was resolving) drops the event —
        never fails the elicitation.
        """
        buffer = self._active_buffer()
        if buffer is None:
            return
        try:
            await buffer.put(event)
        except RuntimeError:
            return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prior_snapshot(self) -> list[Mapping[str, Any]] | None:
        """Snapshot the transcript as ``prior_messages`` (``None`` when empty).

        Resolved lazily inside the build closure (see
        :meth:`App._invoke_in_conversation`), so it reflects hydration and every
        turn recorded up to the moment the send is consumed.
        """
        return list(self._transcript) if self._transcript else None

    def _record(self, message: str, result: AgentResult[Any]) -> None:
        """Append one completed turn's boundary pair to the transcript."""
        self._transcript.append({"role": "user", "content": message})
        self._transcript.append({"role": "assistant", "content": result.response})

    async def _maybe_hydrate(self) -> None:
        """Auto-hydrate on the first send when opened with ``hydrate=True``."""
        if self._needs_hydration and not self._hydrated:
            await self._hydrate()

    async def _hydrate(self) -> None:
        if self._hydrated:
            return
        # Latch before the await so a concurrent first-send prepare cannot
        # double-fetch (sends are sequential, but the flag keeps hydrate once).
        self._hydrated = True
        try:
            memory: Any = self._memory
            if memory is None:
                t0 = await self._app._ensure_t0()
                memory = TensorZeroMemory(
                    t0,
                    function_name=self._spec.function_name,
                    tag_filter=self._tags or None,
                )
            elif not hasattr(memory, "load_history"):
                raise ConfigurationError(
                    f"conversation(memory=...) provider "
                    f"{type(memory).__name__} has no load_history — "
                    f"hydrate() reconstructs the transcript from it, so a "
                    f"history-capable provider is required."
                )
            history = await memory.load_history(self._id)
            if history:
                self._transcript = _history_to_pairs(history)
        except Exception:
            # A transient load_history failure must stay retryable: un-latch so
            # the next send()/hydrate() re-attempts, then re-raise (fail loud).
            self._hydrated = False
            raise


class _ForwarderObserver:
    """Project a forwarder's lifecycle onto the conversation's run stream.

    Installed on the conversation's :class:`ElicitationForwarder` when the
    handler resolves, so the in-stream
    :class:`~t0_agents.stream.events.ElicitationRequestedEvent` carries
    the forwarder's REAL ``elicitation_id`` — the id an out-of-band
    resolver answers — making a full in-stream cutover possible.
    """

    def __init__(self, conversation: Conversation[Any]) -> None:
        self._conversation = conversation

    async def requested(self, envelope: RequestEnvelope) -> None:
        schema = envelope.get("schema")
        await self._conversation._emit(
            ElicitationRequestedEvent(
                id=str(envelope["elicitation_id"]),
                message=str(envelope.get("message") or ""),
                response_type=envelope.get("response_type"),
                schema_json=json.dumps(schema) if schema is not None else "",
            )
        )

    async def resolved(self, elicitation_id: str, action: str) -> None:
        await self._conversation._emit(
            ElicitationResolvedEvent(
                id=elicitation_id, action=_settled_action(action)
            )
        )


class _ObservedHandler:
    """Wrap a bare elicitation handler with glow-minted event ids.

    Used when the conversation (or the App) supplies a plain
    ``ElicitationHandler`` rather than a forwarder: there is no
    out-of-band id to unify with, so glow mints one per elicitation and
    emits a self-consistent Requested/Resolved pair around the inner
    handler.  A plain (non-``ElicitResult``) return is an accept by
    fastmcp's contract; an exception resolves as a decline before
    propagating.
    """

    def __init__(
        self, inner: ElicitationHandler, conversation: Conversation[Any]
    ) -> None:
        self._inner = inner
        self._conversation = conversation

    async def __call__(
        self,
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> Any:
        elicitation_id = str(uuid4())
        schema = getattr(params, "requestedSchema", None)
        await self._conversation._emit(
            ElicitationRequestedEvent(
                id=elicitation_id,
                message=message,
                response_type=_response_type_from_schema(
                    schema, response_type
                ),
                schema_json=json.dumps(schema) if schema is not None else "",
            )
        )
        action = "decline"
        try:
            result = await self._inner(message, response_type, params, context)
            action = str(getattr(result, "action", "accept"))
            return result
        finally:
            await self._conversation._emit(
                ElicitationResolvedEvent(
                    id=elicitation_id, action=_settled_action(action)
                )
            )


def _history_to_pairs(history: Sequence[Any]) -> list[dict[str, str]]:
    """Convert a ``load_history`` ``list[Message]`` to boundary-pair dicts.

    One ``{"role", "content"}`` entry per message that carried text; messages
    that reduce to no text (e.g. a tool-call-only assistant turn) are dropped,
    so the display transcript stays text-only.
    """
    pairs: list[dict[str, str]] = []
    for message in history:
        text = _message_text(message)
        if text:
            pairs.append({"role": str(message["role"]), "content": text})
    return pairs


def _message_text(message: Any) -> str:
    """Join the text of a runtime TZ ``Message``, or ``""``.

    A ``tensorzero.Message`` is mapping-accessed (``message["content"]``) while
    its content blocks are attribute-accessed (``Text.text`` / ``RawText.value``)
    — so message fields use subscript, blocks use ``getattr``.  Blocks that are
    not text (tool calls / results) contribute nothing.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        value = getattr(block, "value", None)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)
