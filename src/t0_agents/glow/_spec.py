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

from t0_agents.core.output import OutputMode
from t0_agents.core.types import Guardrail

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


@dataclass(slots=True)
class _AgentSpec:
    """Registration-time descriptor for an ``@app.agent``-decorated function."""

    name: str
    function_name: str
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
