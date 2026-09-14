"""Internal ``_AgentSpec`` shared by :mod:`t0_agents.glow.app` and
:mod:`t0_agents.glow.decorators`.

Captures the registration-time state of an ``@app.agent``-decorated
function: name, function_name, the user's docstring as system prompt,
the inferred :class:`OutputMode`, the return type (kept around for
introspection / future tooling), an optional ``value_transform`` that
the wrapper layer applies to ``AgentResult.value`` so the user-visible
value matches the original return annotation (e.g. ``list[BaseModel]``
is unwrapped from the synthetic-list wrapper), and the list of
sub-agent specs harvested from ``Depends(glow_wrapper)`` defaults at
decoration time (consumed by :meth:`~t0_agents.glow.app.App._lookup_or_build`
to thread ``AgentConfig.sub_agents=[...]``).

The ``agent`` slot is mutated by :class:`~t0_agents.glow.app.App._lookup_or_build`
on first invocation (or eagerly via ``await app.build()``).  Keeping
the spec ``frozen=False`` is deliberate — the dataclass acts as a tiny
mutable handle that survives across invocations and remains the single
source of truth for whether the underlying :class:`~t0_agents.core.agent.Agent`
has been built yet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from t0_agents.core.config import resolve_function_name
from t0_agents.core.output import OutputMode
from t0_agents.core.types import Guardrail

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


@dataclass(slots=True)
class _AgentSpec:
    """Registration-time descriptor for an ``@app.agent``-decorated function."""

    name: str
    # The raw declaration, ``None`` whenever the caller omitted it.  The
    # routing target is ``resolved_function_name`` below.
    function_name: str | None
    fn: Callable[..., Awaitable[Any]]
    system_prompt: str | None
    output_mode: OutputMode[Any]
    return_type: type[Any] | Any
    value_transform: Callable[[Any], Any] | None = None
    agent: Agent | None = field(default=None)
    sub_agent_specs: list[_AgentSpec] = field(default_factory=list)
    # Guardrails harvested from ``Annotated[T, Guardrail(...)]`` on any
    # of the decorated function's parameters or its return type.  Folded
    # into ``AgentConfig.guardrails`` at first build; the runtime path
    # through ``GuardrailsMiddleware`` is unchanged.
    guardrails: list[Guardrail] = field(default_factory=list)
    # Remaining ``AgentConfig`` fields passed as ``@app.agent`` keyword
    # arguments (``max_turns``, ``tool_timeout``, ``tags``, ``namespace``,
    # ``memory``, ``middleware``, …).  Only knobs the caller actually
    # supplied are present.  Merged into ``AgentConfig(**config_kwargs)``
    # by :meth:`~t0_agents.glow.app.App._lookup_or_build`, layered over any
    # App-level defaults so agent-level values win.
    config_overrides: dict[str, Any] = field(default_factory=dict)
    # Per-agent ``tools=`` filter, verbatim from ``@app.agent(tools=[...])``.
    # ``None`` = no filter (the agent sees every composed tool — back-compat);
    # ``[]`` = explicitly toolless; a non-empty list holds ``@app.tool``
    # callables, ``MCPServer`` classes/instances, and glob strings.
    tools: list[Any] | None = None
    # Concrete tool names ``tools`` resolved to, computed once in
    # :meth:`~t0_agents.glow.app.App.build` (needs the connected client's live
    # tool list).  Mirrors ``tools``: ``None`` = no filter, an empty set = a
    # resolved toolless agent.  Consumed by
    # :meth:`~t0_agents.glow.app.App._lookup_or_build` to wrap the provider.
    resolved_tools: set[str] | None = None

    @property
    def resolved_function_name(self) -> str | None:
        """The function this agent's inferences actually land under.

        ``function_name`` is ``None`` for both ``@app.agent("x")`` and
        ``@app.agent("x", model=...)``; only the resolution tells those
        apart, and only it matches the built
        :class:`~t0_agents.core.config.AgentConfig`.
        """
        return resolve_function_name(
            self.function_name, self.config_overrides.get("model")
        )
