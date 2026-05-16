"""Multi-agent composition for t0-agents.

Two layers coexist:

- **Declarative sub-agents** (M2): ``AgentConfig.sub_agents=[...]``
  registers each as an MCP tool via :func:`agent_as_tool` and
  :func:`install_sub_agents`.  The orchestrator's LLM sees them as
  ``_ask`` tools and delegates by tool call.  Unchanged in M4.

- **Topology composers** (M4): :func:`sequential`, :func:`parallel`,
  :func:`race`, :func:`consensus`, :func:`map_reduce`,
  :func:`conditional`, :func:`saga`, :func:`retry_chain`,
  :func:`circuit_breaker`.  Each returns a :class:`Composer` whose
  :meth:`invoke` returns a :class:`Run`, uniform with
  :meth:`Agent.invoke`.  Composers compose Composables (Agents *or*
  other Composers); nesting is arbitrary.
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
from t0_agents.compose.delegation import agent_as_tool
from t0_agents.compose.orchestrator import (
    bind_parent_episode_id,
    get_parent_episode_id,
    install_sub_agents,
    update_parent_episode_id,
)
from t0_agents.compose.routing import assert_unique_names, default_tool_name
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
    "agent_as_tool",
    "assert_unique_names",
    "bind_parent_episode_id",
    "circuit_breaker",
    "conditional",
    "consensus",
    "default_tool_name",
    "get_parent_episode_id",
    "install_sub_agents",
    "map_reduce",
    "parallel",
    "race",
    "retry_chain",
    "saga",
    "sequential",
    "update_parent_episode_id",
]
