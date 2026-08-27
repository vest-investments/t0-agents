from __future__ import annotations


class T0AgentsError(Exception):
    """Base class for every exception raised by t0-agents."""


class T0AgentsFatalError(T0AgentsError):
    """Marker class for exceptions that must abort an ``Agent.invoke()``.

    Every broad ``except Exception`` belt in the library re-raises this
    class first, so a fatal failure propagates to the terminal instead of
    being degraded into the belt's fallback behaviour:
    ``mcp/loop.py::_execute_single_tool_call`` would otherwise stringify
    it into a tool error, and ``MemoryMiddleware`` would otherwise log it
    as a best-effort memory warning.  Backends and hooks therefore raise
    a subclass of this when their failure invalidates the run — that is
    the documented escape hatch out of the non-fatal memory layer.
    """


class MaxTurnsExceededError(T0AgentsError):
    """Raised when the agent loop exceeds the maximum number of turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")


class ConfigurationError(T0AgentsError):
    """Raised when AgentConfig or Agent construction parameters are invalid."""


class GuardrailBlockedError(T0AgentsFatalError):
    """Raised by a guardrail to hard-abort an ``Agent.invoke()``.

    Short-circuits any tool calls that were about to run in the current
    turn and propagates out of the agent loop so the caller sees a clear
    policy violation instead of a stringified tool error.
    """


class MemoryHistoryUnavailableError(T0AgentsFatalError):
    """Raised when ``memory_strategy="messages"`` cannot read the history.

    That strategy exists to thread the prior conversation through
    ``input.messages``, and :meth:`AgentConfig.__post_init__` refuses to
    construct without a provider that implements ``load_history`` — so a
    failed read leaves the turn unable to honour its own contract.
    Answering anyway would reply as if the conversation had never
    happened, which is worse than not replying: the caller can retry a
    failure, but cannot un-see an amnesiac answer.  The provider's own
    exception is chained on ``__cause__``.

    **An empty episode is not this error.**  ``load_history`` returning
    ``None`` (or an empty list) means "no inferences yet", and a fresh
    session still starts clean.  Only a raised failure ends the turn;
    conflating the two is the defect this class exists to prevent.

    Consumers distinguish it from a provider or inference failure by
    class name on every surface — a typed ``except``,
    :attr:`~t0_agents.stream.events.RunErrorEvent.error_type`, and
    ``detail["exc_class"]`` on a
    :class:`~t0_agents.wire.frames.ErrorFrame` — so a re-drive policy can
    treat "the gateway could not be read" differently from "the model
    call failed".  Note that it does **not** cover a read that succeeds
    against stale data: with the gateway's ``observability.async_writes``
    enabled, a lagging store answers with an empty history that is
    indistinguishable from a new episode.  See ``docs/memory.md``
    § Read-after-write timing.
    """


class SubAgentError(T0AgentsError):
    """Wraps a non-fatal error raised by a delegated sub-agent.

    :class:`~t0_agents.delegation.SubAgentToolProvider` catches plain
    :class:`Exception` from a delegated ``Agent.invoke().result()`` and
    re-raises as ``SubAgentError`` so the orchestrator's LLM sees a
    clean, stringifiable tool error and can choose to retry or re-delegate.  Subclasses of
    :class:`T0AgentsFatalError` (for example
    :class:`GuardrailBlockedError` bubbling up from a sub-agent's own
    guardrail) propagate unchanged so the orchestrator still honours
    hard-block contracts end-to-end.
    """


class RunCancelledError(T0AgentsError):
    """Raised after ``await run.cancel()``, on every consumption channel.

    Surfaced through the awaitable channel — ``await run.result()`` and
    plain ``await run`` — so callers can distinguish a user-initiated
    cancellation from a genuine failure.  ``run.events()`` consumers see
    the events emitted up to the cancellation point and then this
    exception raised out of the iterator, in **both** ``on_error`` modes:
    unlike a failure, a cancellation is never converted into a terminal
    ``RunErrorEvent``.
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
