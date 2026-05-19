"""Topology composers for t0-agents.

Nine factories — :func:`sequential`, :func:`parallel`, :func:`race`,
:func:`consensus`, :func:`map_reduce`, :func:`conditional`,
:func:`saga`, :func:`retry_chain`, :func:`circuit_breaker` — return
:class:`Composer` instances whose :meth:`invoke` yields a :class:`Run`
uniform with :meth:`Agent.invoke`.  Composers compose Composables
(Agents *or* other Composers); nesting is arbitrary.

Sub-agent orchestration (``AgentConfig.sub_agents=[...]``,
:func:`agent_as_tool`, :func:`install_sub_agents`) has moved to
:mod:`t0_agents.delegation`; this submodule no longer re-exports it.
"""

from t0_agents.compose.builtins import (
    CircuitBreakerComposer,
    ConditionalComposer,
    ConsensusComposer,
    MapReduceComposer,
    ParallelComposer,
    RaceComposer,
    RetryChainComposer,
    SagaComposer,
    SequentialComposer,
    circuit_breaker,
    conditional,
    consensus,
    map_reduce,
    parallel,
    race,
    retry_chain,
    saga,
    sequential,
)
from t0_agents.compose.topology import (
    DEFAULT_CANCEL_TIMEOUT,
    Composable,
    Composer,
    ComposerError,
    CompositeRun,
)

__all__ = [
    "DEFAULT_CANCEL_TIMEOUT",
    "CircuitBreakerComposer",
    "Composable",
    "Composer",
    "ComposerError",
    "CompositeRun",
    "ConditionalComposer",
    "ConsensusComposer",
    "MapReduceComposer",
    "ParallelComposer",
    "RaceComposer",
    "RetryChainComposer",
    "SagaComposer",
    "SequentialComposer",
    "circuit_breaker",
    "conditional",
    "consensus",
    "map_reduce",
    "parallel",
    "race",
    "retry_chain",
    "saga",
    "sequential",
]
