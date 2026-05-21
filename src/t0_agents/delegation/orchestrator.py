"""Parent-episode-id ContextVar plumbing for sub-agent correlation.

The sub-agent dispatch surface lives in
:class:`t0_agents.delegation.provider.SubAgentToolProvider`; this
module is now the home of the ContextVar that carries the orchestrator's
``episode_id`` into delegated inferences so ClickHouse can reconstruct
the delegation tree via ``tags[PARENT_EPISODE_ID_TAG]``.

The variable is isolated per :class:`contextvars.ContextVar`, so
concurrent :meth:`Agent.invoke` calls on one orchestrator do not
cross-contaminate each other's bindings.
:class:`~t0_agents.core.middleware.builtins.SubAgentMiddleware` enters
:func:`bind_parent_episode_id` at run start and (when the orchestrator
did not pre-allocate an episode) chains an ``on_inference`` refresh hook
that calls :func:`update_parent_episode_id` once the gateway reports the
real id.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

#: Tag key carried on every sub-agent inference so ClickHouse can
#: reconstruct the delegation tree via ``tags[PARENT_EPISODE_ID_TAG]``.
PARENT_EPISODE_ID_TAG = "t0_agents.parent_episode_id"

_PARENT_EPISODE_ID: ContextVar[str | None] = ContextVar(
    "t0_agents_parent_episode_id",
    default=None,
)


def get_parent_episode_id() -> str | None:
    """Read the current orchestrator's episode id, if any.

    Returns ``None`` when called outside an orchestrator's run (e.g.
    from a sub-agent invoked directly by the user).  Callers should
    treat ``None`` as "no correlation available" rather than a
    configuration bug.
    """
    return _PARENT_EPISODE_ID.get()


@contextlib.contextmanager
def bind_parent_episode_id(episode_id: str | None) -> Iterator[None]:
    """Bind ``episode_id`` as the current parent episode for the block.

    ``None`` is a valid input — sub-agents will then see
    ``get_parent_episode_id() is None`` and fall back to the
    ``"unknown"`` correlation tag.  This keeps :meth:`Agent.invoke` from
    having to short-circuit the bind when the caller passes no
    ``episode_id`` for a fresh orchestrator run.
    """
    token = _PARENT_EPISODE_ID.set(episode_id)
    try:
        yield
    finally:
        _PARENT_EPISODE_ID.reset(token)


def update_parent_episode_id(episode_id: str) -> None:
    """Overwrite the currently-bound parent-episode id in place.

    :meth:`Agent.invoke` enters
    :func:`bind_parent_episode_id` at run start, but passes ``None`` when
    the caller did not supply one — TensorZero only allocates the real
    episode id on the first inference response.  Without a refresh,
    every sub-agent invocation during that same orchestrator run would
    tag ``t0_agents.parent_episode_id=unknown`` even after the id is
    known.

    Must be called inside an enclosing :func:`bind_parent_episode_id`
    context.  The returned :class:`~contextvars.Token` is intentionally
    discarded: :meth:`ContextVar.reset` on the outer bind's token
    reverts the ContextVar to its pre-bind value regardless of how
    many intermediate ``set`` calls landed in between, so the outer
    :func:`bind_parent_episode_id` still owns cleanup on exit.  This
    is a documented Python semantic (``contextvars`` module docs), not
    a quirk of this module.
    """
    _PARENT_EPISODE_ID.set(episode_id)
