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

from typing import TYPE_CHECKING, NoReturn, ParamSpec, Protocol, TypeVar

if TYPE_CHECKING:
    from t0_agents.glow.handle import FluentRun

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
