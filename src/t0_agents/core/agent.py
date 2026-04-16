"""Declarative Agent wrapper around ``run_agent``."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import ChatInferenceResponse

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError, T0AgentsFatalError
from t0_agents.core.types import (
    AgentResult,
    GuardrailLogEntry,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.runtime import GuardedToolProvider
from t0_agents.mcp.loop import run_agent
from t0_agents.memory.base import MemoryRecord, render_memory_summary

logger = logging.getLogger(__name__)

# Default tool timeout used when AgentConfig.tool_timeout is None.
# Matches run_agent's own default so behaviour is identical.
_DEFAULT_TOOL_TIMEOUT: float = 30.0


def _make_cost_tracker(
    ctx: GuardrailContext,
) -> Callable[[ChatInferenceResponse], Awaitable[None]]:
    """Build the ``on_inference`` closure that mutates a run's guardrail context.

    Treats ``response.usage is None`` and ``response.usage.cost is None`` as
    zero so the guardrail still sees a monotonically-increasing cost even
    when the provider omits pricing (e.g. a locally-hosted model).  Also
    increments ``ctx.turn`` so developer-defined guardrails can key behaviour
    on turn count without re-implementing the counter.
    """

    async def _on_inference(response: ChatInferenceResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            ctx.cost_so_far += usage.cost or 0.0
        ctx.turn += 1

    return _on_inference


class Agent:
    """Declarative wrapper around :func:`run_agent`.

    Encapsulates configuration in an :class:`AgentConfig` dataclass and
    exposes a clean ``.run(message)`` method that delegates to the existing
    ``run_agent`` loop.  The caller owns the lifecycle of both the TensorZero
    gateway and the tool provider (typically a ``fastmcp.Client``).

    Example::

        config = AgentConfig(name="assistant", function_name="agent")
        agent = Agent(config, t0=gateway, tool_provider=mcp_client)
        result = await agent.run("What is 2 + 3?")
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider,
    ) -> None:
        if not isinstance(tool_provider, ToolProvider):
            raise ConfigurationError(
                f"tool_provider must implement the ToolProvider protocol "
                f"(list_tools + call_tool), got {type(tool_provider).__name__}"
            )
        if config.memory is not None and not isinstance(config.memory, MemoryProvider):
            raise ConfigurationError(
                f"config.memory must implement the MemoryProvider protocol "
                f"(load + save + search), got {type(config.memory).__name__}"
            )
        self._config = config
        self._t0 = t0
        self._tool_provider = tool_provider
        logger.debug(
            "Agent %r created: function=%s, max_turns=%d",
            config.name,
            config.function_name,
            config.max_turns,
        )

    @property
    def config(self) -> AgentConfig:
        """Read-only access to the agent's configuration."""
        return self._config

    async def warmup(self) -> None:
        """Optional lifespan-phase readiness hook.

        Called by :func:`t0_agents.serve.create_app` during FastAPI
        startup so deployments can front-load one-time checks.  The
        default implementation is a no-op: TensorZero v2026.4.0 does
        not expose a public ``health()`` method (validated at
        ``.venv/.../tensorzero/tensorzero.pyi:1017``), so any built-in
        probe would either couple the Agent to the per-request MCP
        ``Client`` lifecycle or burn provider tokens via
        ``inference(dryrun=True)``.  ``docker compose`` already runs a
        gateway-side healthcheck and the first ``/chat`` surfaces any
        real failure.

        Subclasses may override this method to add custom probes (for
        example pinging an auxiliary tool provider or warming a cache).
        """
        return None

    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
    ) -> AgentResult:
        """Run the agent loop and return the final response.

        When ``config.memory`` is set, memory records are loaded before the
        loop (if *episode_id* is provided) and a new record is persisted
        after the loop completes.  When ``config.guardrails`` is non-empty,
        the tool provider is wrapped in a :class:`GuardedToolProvider` that
        fires every hook around every ``call_tool`` and tracks per-run
        counters on a fresh :class:`GuardrailContext`; the resulting
        :class:`AgentResult` carries a populated ``guardrail_log``.

        Args:
            user_message: The user's query as a plain string.
            episode_id: Existing TensorZero episode id to continue, or
                ``None`` for a fresh episode.

        Returns:
            :class:`~t0_agents.core.types.AgentResult` with the joined
            text of the terminating turn, episode metadata, turn count,
            and (when guardrails are configured) the per-hook audit log.

        Raises:
            MaxTurnsExceededError: If the loop exceeds ``config.max_turns``.
            GuardrailBlockedError: If any guardrail hard-aborts the run.
        """
        # --- 1. Load memory ---
        memory_records: list[MemoryRecord] = []
        if self._config.memory is not None and episode_id is not None:
            try:
                raw_records = await self._config.memory.load(episode_id)
                memory_records = cast(list[MemoryRecord], raw_records)
            except Exception:
                logger.warning(
                    "Memory load failed for episode %s; proceeding without context",
                    episode_id,
                    exc_info=True,
                )

        # --- 2. Prepare injection ---
        effective_message, extra_kwargs = self._prepare_memory_injection(
            user_message, memory_records,
        )

        # --- 3. Set up guardrails (Phase C) ---
        # Empty-list case: preserve Phase B byte-identity by forwarding the
        # raw tool_provider and leaving on_inference at None so run_agent's
        # code path is unchanged.
        guard_ctx: GuardrailContext | None = None
        guard_log: list[GuardrailLogEntry] = []
        tool_provider: ToolProvider = self._tool_provider
        on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None

        if self._config.guardrails:
            guard_ctx = GuardrailContext(
                agent_name=self._config.name,
                function_name=self._config.function_name,
                user_message=user_message,
            )
            tool_provider = GuardedToolProvider(
                self._tool_provider,
                list(self._config.guardrails),
                guard_ctx,
                guard_log,
            )
            on_inference = _make_cost_tracker(guard_ctx)

        # --- 4. Delegate to run_agent ---
        # ``before_run`` lives inside the try so a fatal abort still runs
        # ``after_run`` via the finally block — docs/guardrails.md promises
        # after_run fires even when GuardrailBlockedError propagates.
        tool_timeout = (
            self._config.tool_timeout
            if self._config.tool_timeout is not None
            else _DEFAULT_TOOL_TIMEOUT
        )
        result: AgentResult | None = None
        try:
            if guard_ctx is not None:
                for guardrail in self._config.guardrails:
                    try:
                        await guardrail.before_run(guard_ctx)
                    except T0AgentsFatalError:
                        raise
                    except Exception:
                        logger.exception(
                            "Guardrail %r before_run raised a non-fatal exception; "
                            "continuing",
                            guardrail.name,
                        )
                    else:
                        guard_log.append(
                            GuardrailLogEntry(
                                guardrail_name=guardrail.name,
                                hook="before_run",
                                kind="pass",
                            )
                        )

            result = await run_agent(
                self._t0,
                cast(Client[Any], tool_provider),
                effective_message,
                self._config.function_name,
                episode_id=episode_id,
                max_turns=self._config.max_turns,
                tool_timeout=tool_timeout,
                parallel_tools=self._config.parallel_tools,
                dryrun=self._config.dryrun,
                tool_choice=self._config.tool_choice,
                tags=self._config.tags,
                namespace=self._config.namespace,
                on_inference=on_inference,
                **extra_kwargs,
            )
        finally:
            if guard_ctx is not None:
                for guardrail in self._config.guardrails:
                    try:
                        await guardrail.after_run(guard_ctx, result)
                    except Exception:
                        logger.exception(
                            "Guardrail %r after_run raised a non-fatal exception; "
                            "continuing",
                            guardrail.name,
                        )
                    else:
                        guard_log.append(
                            GuardrailLogEntry(
                                guardrail_name=guardrail.name,
                                hook="after_run",
                                kind="pass",
                            )
                        )
                if result is not None:
                    result.guardrail_log = list(guard_log)

        # Control reaches here only when ``run_agent`` returned normally
        # (no exception propagated past the ``try``).  ``result`` is always
        # populated at this point; the narrowing keeps mypy strict happy.
        assert result is not None

        # --- 5. Save memory ---
        # Skip when episode_id is missing so we never persist records under an
        # empty-string key (which would collide across runs and be unretrievable).
        if self._config.memory is not None and result.episode_id:
            record = MemoryRecord(
                episode_id=result.episode_id,
                user_message=user_message,
                agent_response=result.response,
                turns=result.turns,
                inference_ids=list(result.inference_ids),
            )
            try:
                await self._config.memory.save(record)
            except Exception:
                logger.warning(
                    "Memory save failed for episode %s; run result is still valid",
                    result.episode_id,
                    exc_info=True,
                )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_memory_strategy(
        self,
    ) -> Literal["system_template", "prepended_message"]:
        """Resolve ``"auto"`` to a concrete strategy.

        The TensorZero SDK (v2026.4.0) does not expose a
        ``get_function_config`` API, so runtime introspection of the
        system schema is not possible.  ``"auto"`` therefore always
        resolves to ``"prepended_message"`` — the strategy that works
        with any function/variant without TOML changes.
        """
        # TODO: When the TensorZero SDK gains a function-config
        # introspection API, inspect whether the system schema declares
        # a ``memory_context`` property and prefer ``"system_template"``
        # when it does.
        strategy = self._config.memory_strategy
        if strategy == "auto":
            return "prepended_message"
        return strategy

    def _prepare_memory_injection(
        self,
        user_message: str,
        records: list[MemoryRecord],
    ) -> tuple[str, dict[str, Any]]:
        """Build the effective user message and extra kwargs for run_agent.

        Returns:
            A ``(effective_message, extra_kwargs)`` tuple.  *extra_kwargs*
            is empty for the ``"prepended_message"`` strategy and contains
            ``system_arguments`` for the ``"system_template"`` strategy.
        """
        if not records:
            return user_message, {}

        summary = render_memory_summary(records)
        if not summary:
            return user_message, {}

        strategy = self._resolve_memory_strategy()

        if strategy == "system_template":
            return user_message, {
                "system_arguments": {"memory_context": summary},
            }

        # "prepended_message" — prepend context to the user message.
        return (
            f"{summary}\n---\n{user_message}",
            {},
        )
