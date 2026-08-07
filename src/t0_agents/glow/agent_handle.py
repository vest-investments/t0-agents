"""The ``llm()`` phantom-body sentinel and the ``AgentHandle`` protocol.

``@app.agent`` never executes the decorated function's body — its
signature is the public interface and its docstring is the system
prompt.  Writing ``return llm()`` gives the body a real, mypy-strict
statement (so it needs no ``# type: ignore[empty-body]``) and raises
loudly if the phantom body is ever actually executed.

``AgentHandle[P, R]`` is the callable type of an ``@app.agent`` —
``(*P) -> FluentRun[R]``.  Use it to annotate a sub-agent ``Depends(...)``
parameter so the orchestrator body stays typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, ParamSpec, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core.elicitation_forwarder import ElicitationForwarder
    from t0_agents.core.types import MemoryProvider
    from t0_agents.glow.conversation import Conversation
    from t0_agents.glow.handle import FluentRun
    from t0_agents.mcp.runtime import ServerCredentials

__all__ = ["AgentHandle", "llm"]

P = ParamSpec("P")
R = TypeVar("R")


def llm() -> NoReturn:
    """Phantom-body sentinel — write ``return llm()`` in an ``@app.agent`` body.

    Satisfies any non-``None`` return annotation with a real statement,
    so the body needs no ``# type: ignore[empty-body]``.  Raises if
    executed: ``@app.agent`` never runs the decorated body (the return
    type is taken from the annotation, the behaviour from the gateway).
    """
    raise RuntimeError(
        "llm() is a phantom-body sentinel; @app.agent bodies are never executed."
    )


class AgentHandle(Protocol[P, R]):
    """Callable type of an ``@app.agent`` — ``(*P) -> FluentRun[R]``.

    Annotate a sub-agent ``Depends(...)`` parameter with it so the
    orchestrator body type-checks (``await research("x")`` is ``R``)::

        @app.agent("orchestrator")
        async def orchestrator(
            q: str, research: AgentHandle[..., str] = Depends(researcher)
        ) -> str:
            \"\"\"Coordinate research, then formatting.\"\"\"
            return llm()
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> FluentRun[R]: ...

    def conversation(
        self,
        *,
        id: str | None = None,
        hydrate: bool = False,
        user: str | None = None,
        tags: Mapping[str, str] | None = None,
        auth: Mapping[str, ServerCredentials | Mapping[str, str] | httpx.Auth]
        | None = None,
        elicitation_handler: ElicitationForwarder | ElicitationHandler | None = None,
        memory: MemoryProvider | None = None,
        tools: list[Any] | None = None,
    ) -> Conversation[R]:
        """Open a multi-turn :class:`Conversation` over one episode.

        Lets a conversation-capable handle be typed without importing
        the private wrapper.  The signature mirrors the ``@app.agent``
        wrapper's, so the identity dimension — ``auth``,
        ``elicitation_handler``, ``memory``, ``tools`` — type-checks
        through this Protocol with no ``cast``.  See
        :doc:`/glow/multi-tenant` for what each one does.
        """
        ...
