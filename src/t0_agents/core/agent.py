"""Declarative Agent wrapper around ``run_agent``."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Literal, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import ChatInferenceResponse, JsonInferenceResponse

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError, T0AgentsFatalError
from t0_agents.core.structured import (
    ParseStrategy,
    StructuredOutput,
    permissive_parse,
    run_structured_inference,
)
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
from t0_agents.stream.events import AgentEvent, DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop

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


def _make_json_cost_tracker(
    ctx: GuardrailContext,
) -> Callable[[JsonInferenceResponse], Awaitable[None]]:
    """JSON-function analogue of :func:`_make_cost_tracker`.

    Deliberate duplication of the chat tracker — both ``ChatInferenceResponse``
    and ``JsonInferenceResponse`` expose a ``usage`` attribute but they are
    not related by inheritance, and parameterising the chat tracker with a
    structural ``Union`` would push the typing burden onto every caller.  The
    closure fires once per inference, so a JSON run that retries N times
    invokes it ``1 + N`` times and accumulates cost across every attempt
    without additional wiring.
    """

    async def _on_inference(response: JsonInferenceResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            ctx.cost_so_far += usage.cost or 0.0
        ctx.turn += 1

    return _on_inference


_RETRY_HINT_BASE = (
    "Your previous response was not valid JSON. "
    "Return ONLY a valid JSON object that matches the schema, "
    "with no commentary or prose."
)


def _build_retry_hint(schema: Mapping[str, Any] | None) -> str:
    """Compose the corrective user-turn message sent on a JSON retry.

    Starts from :data:`_RETRY_HINT_BASE` (the cross-call invariant the
    retry tests can pattern-match on), then optionally enriches the hint
    with schema metadata when a non-trivial ``schema`` is available:

    - ``title`` — when the schema's top-level ``title`` is a non-empty
      string, the hint mentions it so the model has a name to anchor on.
    - ``required`` — when the top-level ``required`` array is a non-empty
      list of strings, the hint enumerates the required field names.

    Anything else (nested ``required``, ``properties``, ``oneOf``) is
    deliberately ignored: surfacing it would balloon prompt cost on every
    retry and risk crowding out the model's context.  Diverges from the
    M3 spec §4.5 (which proposed a fixed hard-coded English hint); the
    additive enrichment is opt-out-able by setting
    ``permissive_json_parsing=False, max_json_retries=0``.
    """
    parts: list[str] = [_RETRY_HINT_BASE]
    if schema is None:
        return parts[0]
    title = schema.get("title")
    if isinstance(title, str) and title:
        parts.append(f"The schema is named '{title}'.")
    required = schema.get("required")
    if isinstance(required, list):
        names = [str(n) for n in required if isinstance(n, str)]
        if names:
            joined = ", ".join(repr(n) for n in names)
            parts.append(f"Required top-level fields: {joined}.")
    return " ".join(parts)


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

        # Register each sub-agent as an MCP tool on the orchestrator's
        # in-process FastMCP server.  No-op when config.sub_agents is
        # empty, so single-agent constructions pay no cost.
        self._sub_agents_by_tool_name: dict[str, Agent] = {}
        if config.sub_agents:
            from t0_agents.compose.orchestrator import install_sub_agents
            from t0_agents.compose.routing import default_tool_name

            install_sub_agents(self)
            # Cache the tool-name -> sub-agent map so ``Agent.run`` /
            # ``Agent.stream`` can wrap ``tool_provider`` with
            # ``_OrchestrationToolProvider`` cheaply per run.  The
            # orchestration wrapper needs this mapping to dispatch
            # sub-agent calls inline without crossing FastMCP's
            # transport task boundary.
            self._sub_agents_by_tool_name = {
                default_tool_name(sub): sub for sub in config.sub_agents
            }

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
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        json_mode: Literal["off", "on", "strict", "tool"] | None = None,
        max_json_retries: int | None = None,
    ) -> AgentResult:
        """Run the agent loop and return the final response.

        Dispatches on ``config.function_kind``:

        - ``"chat"`` (default) — runs the existing tool loop via
          :func:`~t0_agents.mcp.loop.run_agent`.  Behaviour is byte-identical
          to v0.9.0 for any caller that does not set the new
          ``output_schema`` / ``json_mode`` kwargs.
        - ``"json"`` — single-shot JSON inference via
          :func:`~t0_agents.core.structured.run_structured_inference`.
          The resulting :class:`AgentResult` carries
          ``function_kind="json"``, ``parsed`` (the JSON-parsed object or
          ``None`` when TensorZero could not parse the model output), and
          ``response`` set to the raw JSON string.

        When ``config.memory`` is set, memory records are loaded before the
        loop (if *episode_id* is provided) and a new record is persisted
        after the loop completes.  When ``config.guardrails`` is non-empty,
        the tool provider is wrapped in a :class:`GuardedToolProvider` that
        fires every hook around every ``call_tool`` and tracks per-run
        counters on a fresh :class:`GuardrailContext`; the resulting
        :class:`AgentResult` carries a populated ``guardrail_log``.  The
        JSON path skips the tool-level hooks (``before_tool``/``after_tool``)
        because JSON functions never call tools, but ``before_run`` and
        ``after_run`` fire normally.

        Args:
            user_message: The user's query as a plain string.
            episode_id: Existing TensorZero episode id to continue, or
                ``None`` for a fresh episode.
            tags: Per-call tags that shallow-merge on top of
                ``config.tags`` for observability — values here win on key
                collisions, and ``None`` leaves the configured tags
                untouched.  The configured deployment label (if any) is
                already part of ``config.tags`` and is preserved by the
                merge.
            system_arguments: Optional dict forwarded to TensorZero as
                ``input.system`` for the system-template render.  When
                the function declares ``schemas.system.path``, the dict
                must match the schema directly — its keys are the
                schema-described properties (no ``arguments`` wrapper).
                Useful when the function's system template declares
                variables beyond what the memory strategy injects (e.g.
                ``{{ max_turns }}``, ``{{ user_name }}``).  Shallow-merged
                with — not overwritten by — the ``system_arguments``
                produced by ``memory_strategy="system_template"``: caller
                keys take precedence on collision.  ``None`` (the
                default) preserves today's behaviour exactly.
            output_schema: Per-call override for the JSON function's static
                schema (TZ guide §4).  Only valid when
                ``config.function_kind == "json"``; raises
                :class:`ConfigurationError` on chat agents.  Takes
                precedence over ``config.output_schema`` when both are set.
            json_mode: Per-call override for the TensorZero ``json_mode``
                (TZ guide §5).  Only valid when
                ``config.function_kind == "json"``; raises
                :class:`ConfigurationError` on chat agents.  Takes
                precedence over ``config.json_mode`` when both are set.
            max_json_retries: Per-call override for
                ``config.max_json_retries``.  ``None`` (the default)
                uses the configured value.  ``0`` disables the retry
                loop for this call only — the permissive layer still
                runs unless ``config.permissive_json_parsing`` is
                ``False``.  Only meaningful for JSON agents; ignored on
                chat agents (no error: chat callers may pass it
                unconditionally without branching).

        Returns:
            :class:`~t0_agents.core.types.AgentResult` with the joined
            text of the terminating turn, episode metadata, turn count,
            and (when guardrails are configured) the per-hook audit log.
            For JSON runs ``response`` is the raw text of the final
            attempt (valid or not), ``parsed`` carries the recovered
            object or ``None``, ``turns`` is ``1 + json_parse_attempts``,
            ``inference_ids`` contains one id per attempt in order, and
            ``parse_strategy`` reports which layer produced ``parsed``:

            - ``"strict"`` — TensorZero's parser succeeded on the first
              attempt.
            - ``"permissive"`` — the in-Python recovery layer extracted
              JSON from prose or a fenced block on the first attempt.
            - ``"retry"`` — at least one retry was needed before a valid
              ``parsed`` was produced (by either strict or permissive).
            - ``"none"`` — every layer and every attempt failed;
              ``parsed`` is ``None``.

        Raises:
            MaxTurnsExceededError: If the loop exceeds ``config.max_turns``.
            GuardrailBlockedError: If any guardrail hard-aborts the run.
            ConfigurationError: If ``output_schema`` or ``json_mode`` is
                passed on an agent with ``config.function_kind == "chat"``,
                or if a JSON run is configured with
                ``memory_strategy == "messages"`` (use ``"prepended_message"``
                or ``"system_template"`` instead).
        """
        # Structured-output dispatch: JSON functions bypass the tool loop
        # and route to ``run_structured_inference``.  The chat path below
        # is unchanged from v0.9.0 for any caller that does not pass the
        # new kwargs.
        if self._config.function_kind == "json":
            return await self._run_json(
                user_message,
                episode_id=episode_id,
                tags=tags,
                system_arguments=system_arguments,
                output_schema=output_schema,
                json_mode=json_mode,
                max_json_retries=max_json_retries,
            )
        if output_schema is not None or json_mode is not None:
            raise ConfigurationError(
                "output_schema/json_mode kwargs are only valid when "
                "AgentConfig.function_kind == 'json'; got function_kind="
                f"{self._config.function_kind!r}."
            )

        # --- 1. Load memory ---
        # ``messages`` strategy uses load_history (called from
        # _prepare_memory_injection); records are unused on that path so
        # we skip the load() round-trip entirely.
        memory_records: list[MemoryRecord] = []
        strategy = self._resolve_memory_strategy()
        if (
            self._config.memory is not None
            and episode_id is not None
            and strategy != "messages"
        ):
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
        effective_message, extra_kwargs = await self._prepare_memory_injection(
            user_message, memory_records, episode_id=episode_id,
        )
        self._merge_caller_system_arguments(extra_kwargs, system_arguments)

        # --- 3. Set up guardrails ---
        # Empty-list case: forward the raw tool_provider and leave
        # on_inference at None so run_agent's code path is unchanged.
        guard_ctx: GuardrailContext | None = None
        guard_log: list[GuardrailLogEntry] = []
        tool_provider: ToolProvider = self._tool_provider
        on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None

        if self._config.sub_agents:
            # Wrap before guardrails so orchestrator-level guardrails
            # still observe sub-agent delegations, while the inline
            # dispatch keeps the parent-episode ContextVar visible (the
            # FastMCP in-process transport otherwise snapshots an empty
            # context into its server sub-task before Agent.run can bind
            # anything).
            from t0_agents.compose.orchestrator import (
                _OrchestrationToolProvider,
            )

            tool_provider = cast(
                ToolProvider,
                _OrchestrationToolProvider(
                    tool_provider, self._sub_agents_by_tool_name,
                ),
            )

        if self._config.guardrails:
            guard_ctx = GuardrailContext(
                agent_name=self._config.name,
                function_name=self._config.function_name,
                user_message=user_message,
            )
            tool_provider = GuardedToolProvider(
                tool_provider,
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

            with contextlib.ExitStack() as stack:
                if self._config.sub_agents:
                    # Bind the orchestrator's episode id so sub-agent
                    # tool handlers tag their inferences with it for
                    # ClickHouse correlation.
                    from t0_agents.compose.orchestrator import (
                        bind_parent_episode_id,
                        update_parent_episode_id,
                    )

                    stack.enter_context(bind_parent_episode_id(episode_id))

                    if episode_id is None:
                        # Fresh orchestrator episode: TensorZero only
                        # allocates the real id on the first inference.
                        # Wrap on_inference so the ContextVar is
                        # refreshed before the turn's tool calls
                        # dispatch — otherwise every sub-agent
                        # invocation during this run tags
                        # t0_agents.parent_episode_id=unknown for the
                        # entire orchestrator run.
                        base_on_inference = on_inference

                        async def _refresh_parent_episode_id(
                            response: ChatInferenceResponse,
                        ) -> None:
                            if response.episode_id:
                                update_parent_episode_id(str(response.episode_id))
                            if base_on_inference is not None:
                                await base_on_inference(response)

                        on_inference = _refresh_parent_episode_id
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
                    tags=self._merge_tags(tags),
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

    async def stream(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Streaming analogue of :meth:`run`.

        Same lifecycle — memory hydration, guardrail scaffolding, tool
        execution — except the turn loop is driven by
        :class:`~t0_agents.stream.runner.StreamingAgentLoop` and every
        intermediate token / tool-call / thought is surfaced as an
        :class:`~t0_agents.stream.events.AgentEvent`.

        The terminal event is always a
        :class:`~t0_agents.stream.events.DoneEvent` carrying a fully-
        populated :class:`~t0_agents.core.types.AgentResult` (including
        the ``guardrail_log`` when guardrails are configured).

        Args:
            user_message: The user's query as a plain string.
            episode_id: Existing TensorZero episode id to continue, or
                ``None`` for a fresh episode.
            tags: Per-call tags that shallow-merge on top of
                ``config.tags``.  Critical for the WebSocket serving path
                where one :class:`Agent` is shared across many frames —
                without this kwarg there is no way to vary tags
                per-frame.  Matches :meth:`run`'s semantics.
            system_arguments: Optional dict forwarded to TensorZero as
                ``input.system`` for the system-template render.  Same
                merge semantics as :meth:`run` — caller keys take
                precedence over the ``memory_context`` injected by
                ``memory_strategy="system_template"``.  ``None`` (the
                default) preserves today's behaviour exactly.

        Yields:
            :class:`~t0_agents.stream.events.AgentEvent` instances as the
            LLM streams them and tools execute.  The last event is always
            a :class:`DoneEvent`.

        Raises:
            MaxTurnsExceededError: If the loop exceeds ``config.max_turns``.
            GuardrailBlockedError: If any guardrail hard-aborts the run.
        """
        # --- 1. Load memory --- (mirrors run())
        # ``messages`` strategy uses load_history; skip the records load
        # for that path.
        memory_records: list[MemoryRecord] = []
        strategy = self._resolve_memory_strategy()
        if (
            self._config.memory is not None
            and episode_id is not None
            and strategy != "messages"
        ):
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
        effective_message, extra_kwargs = await self._prepare_memory_injection(
            user_message, memory_records, episode_id=episode_id,
        )
        self._merge_caller_system_arguments(extra_kwargs, system_arguments)

        # --- 3. Set up guardrails ---
        guard_ctx: GuardrailContext | None = None
        guard_log: list[GuardrailLogEntry] = []
        tool_provider: ToolProvider = self._tool_provider
        on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None

        if self._config.sub_agents:
            # Mirror Agent.run: orchestration wrapping sits inside the
            # guardrail wrapper so the streaming loop's tool dispatch
            # hits the inline sub-agent path and keeps the parent-
            # episode ContextVar visible to ``get_parent_episode_id``.
            from t0_agents.compose.orchestrator import (
                _OrchestrationToolProvider,
            )

            tool_provider = cast(
                ToolProvider,
                _OrchestrationToolProvider(
                    tool_provider, self._sub_agents_by_tool_name,
                ),
            )

        if self._config.guardrails:
            guard_ctx = GuardrailContext(
                agent_name=self._config.name,
                function_name=self._config.function_name,
                user_message=user_message,
            )
            tool_provider = GuardedToolProvider(
                tool_provider,
                list(self._config.guardrails),
                guard_ctx,
                guard_log,
            )
            on_inference = _make_cost_tracker(guard_ctx)

        tool_timeout = (
            self._config.tool_timeout
            if self._config.tool_timeout is not None
            else _DEFAULT_TOOL_TIMEOUT
        )

        # --- 4. before_run hooks ---
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

        # --- 5. Drive the streaming loop ---
        # DoneEvent is buffered so we can attach the full guardrail_log
        # (populated by after_run hooks below) before surfacing it.  If the
        # stream aborts (e.g. GuardrailBlockedError), done_result stays
        # None and the caller sees the exception propagate.
        streamer = StreamingAgentLoop()
        done_result: AgentResult | None = None
        try:
            with contextlib.ExitStack() as stack:
                if self._config.sub_agents:
                    # Same binding as Agent.run — sub-agents invoked
                    # from a streaming orchestrator see the same
                    # ContextVar.
                    from t0_agents.compose.orchestrator import (
                        bind_parent_episode_id,
                        update_parent_episode_id,
                    )

                    stack.enter_context(bind_parent_episode_id(episode_id))

                    if episode_id is None:
                        # Mirror Agent.run: refresh the parent-episode
                        # ContextVar on the first inference so
                        # mid-stream sub-agent delegations correlate
                        # against the orchestrator's real episode id
                        # instead of the "unknown" fallback.  The
                        # streaming runner fires on_inference after
                        # final_usage_chunk.episode_id is populated and
                        # before tool dispatch, so the refresh lands in
                        # time for every sub-agent call in this turn.
                        base_on_inference = on_inference

                        async def _refresh_parent_episode_id(
                            response: ChatInferenceResponse,
                        ) -> None:
                            if response.episode_id:
                                update_parent_episode_id(str(response.episode_id))
                            if base_on_inference is not None:
                                await base_on_inference(response)

                        on_inference = _refresh_parent_episode_id
                async for event in streamer.stream(
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
                    tags=self._merge_tags(tags),
                    namespace=self._config.namespace,
                    on_inference=on_inference,
                    **extra_kwargs,
                ):
                    if isinstance(event, DoneEvent):
                        done_result = event.result
                        # Don't yield DoneEvent yet — finally block populates
                        # guardrail_log and persists memory first.
                        break
                    yield event
        finally:
            # --- 6. after_run hooks (fire even on error, parity with run()) ---
            if guard_ctx is not None:
                for guardrail in self._config.guardrails:
                    try:
                        await guardrail.after_run(guard_ctx, done_result)
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
                if done_result is not None:
                    done_result.guardrail_log = list(guard_log)

            # --- 7. Persist memory ---
            if (
                self._config.memory is not None
                and done_result is not None
                and done_result.episode_id
            ):
                record = MemoryRecord(
                    episode_id=done_result.episode_id,
                    user_message=user_message,
                    agent_response=done_result.response,
                    turns=done_result.turns,
                    inference_ids=list(done_result.inference_ids),
                )
                try:
                    await self._config.memory.save(record)
                except Exception:
                    logger.warning(
                        "Memory save failed for episode %s; run result is still valid",
                        done_result.episode_id,
                        exc_info=True,
                    )

        # --- 8. Terminal DoneEvent (outside try so after_run fires first) ---
        if done_result is not None:
            yield DoneEvent(result=done_result)

    async def _run_json(
        self,
        user_message: str,
        *,
        episode_id: str | None,
        tags: dict[str, str] | None,
        system_arguments: dict[str, Any] | None,
        output_schema: Mapping[str, Any] | None,
        json_mode: Literal["off", "on", "strict", "tool"] | None,
        max_json_retries: int | None = None,
    ) -> AgentResult:
        """JSON-function dispatch path for :meth:`run`.

        Wraps up to ``max_json_retries + 1`` calls to
        :func:`run_structured_inference` in the same memory + run-level
        guardrail scaffolding as the chat path.  When TensorZero's
        strict parser returns ``parsed == None``, the orchestration
        consults the in-Python recovery layer (:func:`permissive_parse`,
        gated by ``AgentConfig.permissive_json_parsing``); when that
        also fails and the retry budget is non-zero, it re-emits the
        inference with the previous (invalid) assistant turn and a
        corrective user-turn hint until either a valid parse is
        produced or the budget is exhausted.

        No tool loop, no sub-agent orchestration, no
        ``before_tool``/``after_tool`` hooks — JSON functions never call
        tools.  ``before_run`` and ``after_run`` fire exactly once
        around the whole retry sequence; the cost-tracker closure fires
        once per inference and so accumulates cost across every attempt
        automatically.

        Sub-agent composition is rejected at config construction
        (``AgentConfig.__post_init__`` raises when
        ``function_kind == "json"`` is combined with ``sub_agents``), so
        this method does not exercise the ``_OrchestrationToolProvider``
        path used by chat.
        """
        # --- 1. Load memory (mirrors chat run()) ---
        # ``memory_strategy="messages"`` would require turning prior-episode
        # records into ``prior_messages`` for the helper, which the M3
        # surface deliberately does not do — ``prior_messages`` is reserved
        # for the retry loop below.  Reject the strategy here at the
        # strategy level (rather than after a load round-trip that might
        # silently see an empty history) so the misconfiguration is caught
        # uniformly regardless of episode state.
        strategy = self._resolve_memory_strategy()
        if strategy == "messages":
            raise ConfigurationError(
                "memory_strategy='messages' is not supported for "
                "function_kind='json' — JSON functions are single-shot. "
                "Use memory_strategy='prepended_message' (the default) "
                "or 'system_template' instead."
            )

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

        # --- 2. Prepare memory injection ---
        effective_message, extra_kwargs = await self._prepare_memory_injection(
            user_message, memory_records, episode_id=episode_id,
        )
        self._merge_caller_system_arguments(extra_kwargs, system_arguments)

        # --- 3. Set up guardrail scaffolding ---
        # JSON has no tool loop, so the tool-provider is never wrapped
        # and ``before_tool``/``after_tool`` never fire.  Run-level
        # hooks (``before_run``/``after_run``) fire just like the chat
        # path so cost-tracking and policy guardrails work uniformly.
        guard_ctx: GuardrailContext | None = None
        guard_log: list[GuardrailLogEntry] = []
        on_inference: (
            Callable[[JsonInferenceResponse], Awaitable[None]] | None
        ) = None

        if self._config.guardrails:
            guard_ctx = GuardrailContext(
                agent_name=self._config.name,
                function_name=self._config.function_name,
                user_message=user_message,
            )
            on_inference = _make_json_cost_tracker(guard_ctx)

        # --- 4. Resolve effective output_schema / json_mode ---
        # Per-call kwargs take precedence over the config-level fields.
        # ``AgentConfig.__post_init__`` resolves ``Path`` schemas into
        # ``Mapping`` at construction time, so the config field is
        # ``Mapping | None`` here.
        config_schema = self._config.output_schema
        effective_schema: Mapping[str, Any] | None
        if output_schema is not None:
            effective_schema = output_schema
        elif config_schema is not None:
            effective_schema = cast(Mapping[str, Any], config_schema)
        else:
            effective_schema = None
        effective_mode = (
            json_mode if json_mode is not None else self._config.json_mode
        )

        # --- 5. Delegate to run_structured_inference (with retry loop) ---
        # Per-call ``max_json_retries`` wins over the config value, and ``0``
        # disables the retry layer entirely (the permissive layer still
        # runs unless it too is disabled at the config level).
        retry_budget = (
            max_json_retries
            if max_json_retries is not None
            else self._config.max_json_retries
        )
        permissive_enabled = self._config.permissive_json_parsing

        attempts = 0
        prior_msgs: list[Mapping[str, Any]] = []
        current_user = effective_message
        inference_ids: list[str] = []
        captured_episode: str | None = None
        final_parsed: Any | None = None
        final_raw: str = ""
        final_strategy: ParseStrategy = "none"

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

            while True:
                # ``captured_episode`` is None on attempt 0 — use the
                # caller-supplied ``episode_id`` so the first inference may
                # mint a fresh episode (when both are None) or continue an
                # existing one.  Subsequent attempts always reuse the id
                # TZ assigned the first response, so the whole retry chain
                # belongs to one episode.
                structured: StructuredOutput = await run_structured_inference(
                    self._t0,
                    function_name=self._config.function_name,
                    user_message=current_user,
                    prior_messages=prior_msgs if prior_msgs else None,
                    output_schema=effective_schema,
                    json_mode=effective_mode,
                    system_arguments=extra_kwargs.get("system_arguments"),
                    episode_id=(
                        captured_episode
                        if captured_episode is not None
                        else episode_id
                    ),
                    tags=self._merge_tags(tags),
                    namespace=self._config.namespace,
                    dryrun=self._config.dryrun,
                    on_inference=on_inference,
                )
                inference_ids.append(structured.inference_id)
                if captured_episode is None:
                    captured_episode = structured.episode_id
                final_raw = structured.raw

                if structured.parsed is not None:
                    final_parsed = structured.parsed
                    final_strategy = "retry" if attempts > 0 else "strict"
                    break

                if permissive_enabled:
                    candidate, _ = permissive_parse(structured.raw)
                    if candidate is not None:
                        final_parsed = candidate
                        final_strategy = (
                            "retry" if attempts > 0 else "permissive"
                        )
                        break

                if attempts >= retry_budget:
                    final_parsed = None
                    final_strategy = "none"
                    break

                prior_msgs.append(
                    {"role": "user", "content": current_user}
                )
                prior_msgs.append(
                    {"role": "assistant", "content": structured.raw}
                )
                current_user = _build_retry_hint(effective_schema)
                attempts += 1

            result = AgentResult(
                response=final_raw,
                episode_id=captured_episode,
                inference_ids=inference_ids,
                turns=1 + attempts,
                function_kind="json",
                parsed=final_parsed,
                parse_strategy=final_strategy,
                json_parse_attempts=attempts,
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

        # Control reaches here only when the structured inference returned
        # normally (no exception propagated past the ``try``).
        assert result is not None

        # --- 6. Save memory (mirrors chat run() ordering) ---
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

    def _merge_tags(
        self,
        override: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Shallow-merge a per-call ``tags`` override onto ``config.tags``.

        Matches :meth:`t0_agents.serve.session.SessionFactory._merge_tags`
        so the streaming-WS path (one :class:`Agent` shared across frames)
        and the per-HTTP path (fresh :class:`Agent` per request) stay
        behaviourally identical.  Returns ``None`` when both inputs are
        empty — preserves the byte-identity guarantee for callers that
        never supply tags.
        """
        base = self._config.tags or {}
        extra = override or {}
        if not base and not extra:
            return None
        return {**base, **extra}

    @staticmethod
    def _merge_caller_system_arguments(
        extra_kwargs: dict[str, Any],
        caller_args: dict[str, Any] | None,
    ) -> None:
        """Merge caller-supplied ``system_arguments`` into ``extra_kwargs`` in-place.

        Caller keys take precedence over memory-strategy keys on collision —
        explicit caller intent should not be silently overwritten by an
        implicit memory injection.  An empty merged dict collapses to "no
        ``system_arguments`` key" so the gateway never sees
        ``input.system={}``.

        No-op when ``caller_args`` is ``None``: byte-identical to today's
        behaviour for callers that omit the kwarg.
        """
        if caller_args is None:
            return
        merged = dict(extra_kwargs.pop("system_arguments", None) or {})
        merged.update(caller_args)
        if merged:
            extra_kwargs["system_arguments"] = merged

    def _resolve_memory_strategy(
        self,
    ) -> Literal["system_template", "prepended_message", "messages"]:
        """Resolve ``"auto"`` to a concrete strategy.

        The TensorZero SDK (v2026.4.0) does not expose a
        ``get_function_config`` API, so runtime introspection of the
        system schema is not possible.  ``"auto"`` therefore always
        resolves to ``"prepended_message"`` — the strategy that works
        with any function/variant without TOML changes.

        ``"messages"`` is never selected by ``"auto"``; callers must
        opt in explicitly because it requires the memory provider to
        implement ``load_history``.
        """
        # TODO: When the TensorZero SDK gains a function-config
        # introspection API, inspect whether the system schema declares
        # a ``memory_context`` property and prefer ``"system_template"``
        # when it does.
        strategy = self._config.memory_strategy
        if strategy == "auto":
            return "prepended_message"
        return strategy

    async def _prepare_memory_injection(
        self,
        user_message: str,
        records: list[MemoryRecord],
        *,
        episode_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build the effective user message and extra kwargs for run_agent.

        Dispatches on the resolved memory strategy.  For
        ``"prepended_message"`` and ``"system_template"`` the *records*
        list (already loaded by the caller via ``memory.load``) is rendered
        into a string summary and surfaced either inside the user message
        or as ``system_arguments``.  For ``"messages"`` the records list
        is unused — instead this method calls ``memory.load_history`` to
        retrieve a TZ-native ``list[Message]`` and returns it under the
        ``prior_messages`` key in *extra_kwargs*.

        Returns:
            A ``(effective_message, extra_kwargs)`` tuple.  *extra_kwargs*
            is empty for the ``"prepended_message"`` strategy, contains
            ``system_arguments`` for the ``"system_template"`` strategy,
            and contains ``prior_messages`` for the ``"messages"`` strategy.
        """
        strategy = self._resolve_memory_strategy()

        if strategy == "messages":
            if self._config.memory is None or episode_id is None:
                return user_message, {}
            try:
                history = await self._config.memory.load_history(  # type: ignore[attr-defined]
                    episode_id,
                )
            except Exception:
                logger.warning(
                    "Memory load_history failed for episode %s; proceeding "
                    "without prior context",
                    episode_id,
                    exc_info=True,
                )
                return user_message, {}
            if not history:
                return user_message, {}
            return user_message, {"prior_messages": list(history)}

        if not records:
            return user_message, {}

        summary = render_memory_summary(records)
        if not summary:
            return user_message, {}

        if strategy == "system_template":
            return user_message, {
                "system_arguments": {"memory_context": summary},
            }

        # "prepended_message" — prepend context to the user message.
        return (
            f"{summary}\n---\n{user_message}",
            {},
        )
