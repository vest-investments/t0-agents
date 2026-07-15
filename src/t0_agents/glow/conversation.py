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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.memory.tensorzero import TensorZeroMemory

if TYPE_CHECKING:
    from t0_agents.core.types import AgentResult
    from t0_agents.glow._spec import _AgentSpec
    from t0_agents.glow.app import App
    from t0_agents.glow.handle import FluentRun

__all__ = ["Conversation"]

R = TypeVar("R")


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
                extra_prepare=self._maybe_hydrate,
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
            t0 = await self._app._ensure_t0()
            memory = TensorZeroMemory(
                t0,
                function_name=self._spec.function_name,
                tag_filter=self._tags or None,
            )
            history = await memory.load_history(self._id)
            if history:
                self._transcript = _history_to_pairs(history)
        except Exception:
            # A transient load_history failure must stay retryable: un-latch so
            # the next send()/hydrate() re-attempts, then re-raise (fail loud).
            self._hydrated = False
            raise


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
