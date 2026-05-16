from __future__ import annotations


class T0AgentsError(Exception):
    """Base class for every exception raised by t0-agents."""


class T0AgentsFatalError(T0AgentsError):
    """Marker class for exceptions that must abort an ``Agent.run()``.

    ``mcp/loop.py::_execute_single_tool_call`` re-raises this class before
    its broad ``except Exception`` belt-and-braces clause, so guardrail
    hard-blocks (and any future fatal hooks) propagate out of the loop
    instead of being stringified into a tool error.
    """


class MaxTurnsExceededError(T0AgentsError):
    """Raised when the agent loop exceeds the maximum number of turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")


class ConfigurationError(T0AgentsError):
    """Raised when AgentConfig or Agent construction parameters are invalid."""


class GuardrailBlockedError(T0AgentsFatalError):
    """Raised by a guardrail to hard-abort an ``Agent.run()``.

    Short-circuits any tool calls that were about to run in the current
    turn and propagates out of the agent loop so the caller sees a clear
    policy violation instead of a stringified tool error.
    """


class SubAgentError(T0AgentsError):
    """Wraps a non-fatal error raised by a delegated sub-agent.

    The generated tool handler in ``t0_agents.compose.delegation``
    catches plain :class:`Exception` and re-raises as ``SubAgentError``
    so the orchestrator's LLM sees a clean, stringifiable tool error
    and can choose to retry or re-delegate.  Subclasses of
    :class:`T0AgentsFatalError` (for example
    :class:`GuardrailBlockedError` bubbling up from a sub-agent's own
    guardrail) propagate unchanged so the orchestrator still honours
    hard-block contracts end-to-end.
    """


class RunCancelledError(T0AgentsError):
    """Raised by ``await run.result()`` after ``await run.cancel()``.

    Surfaced through the awaitable consumption channel so callers can
    distinguish a user-initiated cancellation from a genuine failure.
    ``run.events()`` consumers see the events emitted up to the
    cancellation point followed by an end-of-stream — the iterator
    does not raise.
    """


class RunFailedError(T0AgentsError):
    """Raised by ``await run.result()`` when the underlying drive failed.

    Wraps the original exception on the ``cause`` attribute so callers
    can introspect or re-raise the root exception.  ``run.cancel()``
    transitions to :class:`RunCancelledError` instead; only non-cancel
    failures produce :class:`RunFailedError`.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"Run failed: {cause!r}")
        self.cause = cause
