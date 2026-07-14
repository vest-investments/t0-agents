"""Event selectors for :meth:`FluentRun.events`.

A :class:`Selector` pairs an event class with an optional predicate
(filter) and an optional transform (map).  The seven module-level
instances — ``Thought``, ``ToolCall``, ``Text``, ``ToolResult``,
``TurnEnd``, ``Done``, ``Error`` — cover the v1.0 :class:`AgentEvent`
union surface and are the import points users reach for::

    from t0_agents.glow import Thought, ToolCall, Text

    async for thought in tutor("Explain X").events(Thought):
        print(">>", thought.text)

    async for status in tutor("Search X").events(
        ToolCall.map(lambda tc: f"Calling {tc.name}...")
    ):
        print(status)

Selectors compose via :meth:`Selector.where` (AND-combines predicates)
and :meth:`Selector.map` (replaces transform).  Both return *new*
:class:`Selector` instances so the module-level constants stay frozen.

``Text = Selector(TextEndEvent)`` — the whole-text variant is the
useful default because the LLM only emits one terminal text payload
per turn.  Delta filtering remains reachable via
``Selector(TextDeltaEvent).where(...)`` if a caller wants it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextEndEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)

__all__ = [
    "Done",
    "Error",
    "Selector",
    "Text",
    "Thought",
    "ToolCall",
    "ToolResult",
    "TurnEnd",
]

E = TypeVar("E", bound=AgentEvent)
R = TypeVar("R")
R2 = TypeVar("R2")


def _passthrough_predicate(_event: Any) -> bool:
    """Default predicate — every event matches."""
    return True


def _passthrough_transform(event: Any) -> Any:
    """Default transform — yield the event unchanged."""
    return event


class Selector(Generic[E, R]):
    """Filter + map specification for :meth:`FluentRun.events`.

    Three pieces of state:

    * ``event_type`` — the :class:`AgentEvent` subclass this selector
      targets (``isinstance`` check).
    * ``predicate`` — additional filter on top of the type check.
    * ``transform`` — applied to matching events before they are
      yielded.

    :meth:`where` and :meth:`map` both return *new* :class:`Selector`
    instances so the module-level constants below stay immutable.
    """

    __slots__ = ("event_type", "predicate", "transform")

    def __init__(
        self,
        event_type: type[E],
        predicate: Callable[[E], bool] = _passthrough_predicate,
        transform: Callable[[E], R] = _passthrough_transform,
    ) -> None:
        self.event_type: type[E] = event_type
        self.predicate: Callable[[E], bool] = predicate
        self.transform: Callable[[E], R] = transform

    def where(self, predicate: Callable[[E], bool]) -> Selector[E, R]:
        """Return a new :class:`Selector` with an AND-chained predicate.

        Multiple :meth:`where` calls compose: an event matches only when
        every chained predicate returns ``True``.  Use for narrowing
        within an event type — e.g. ``Thought.where(lambda t: "X" in t.text)``.
        """
        existing = self.predicate

        def _combined(event: E) -> bool:
            return existing(event) and predicate(event)

        return Selector(self.event_type, _combined, self.transform)

    def map(self, transform: Callable[[E], R2]) -> Selector[E, R2]:
        """Return a new :class:`Selector` with the transform replaced.

        Use for emitting a derived value rather than the raw event —
        e.g. ``ToolCall.map(lambda tc: f"Calling {tc.name}")``.
        :meth:`map` replaces (not chains) so a downstream transform
        always sees the original event shape.
        """
        return Selector(self.event_type, self.predicate, transform)

    def matches(self, event: Any) -> bool:
        """``True`` when ``event`` matches this selector's type + predicate."""
        return isinstance(event, self.event_type) and self.predicate(event)


# ---------------------------------------------------------------------------
# Module-level selectors covering the v1.0 ``AgentEvent`` union surface.
# Users import these as the canonical entry points: ``from t0_agents.glow
# import Thought, ToolCall, Text``.
# ---------------------------------------------------------------------------

Thought: Selector[ThoughtEvent, ThoughtEvent] = Selector(ThoughtEvent)
ToolCall: Selector[ToolCallEvent, ToolCallEvent] = Selector(ToolCallEvent)
# Whole-text variant — the LLM emits one terminal text payload per
# assistant turn; ``Selector(TextDeltaEvent).where(...)`` remains
# reachable for callers who want token-by-token deltas.
Text: Selector[TextEndEvent, TextEndEvent] = Selector(TextEndEvent)
ToolResult: Selector[ToolResultEvent, ToolResultEvent] = Selector(ToolResultEvent)
TurnEnd: Selector[TurnEndEvent, TurnEndEvent] = Selector(TurnEndEvent)
Done: Selector[DoneEvent, DoneEvent] = Selector(DoneEvent)
Error: Selector[ErrorEvent, ErrorEvent] = Selector(ErrorEvent)
