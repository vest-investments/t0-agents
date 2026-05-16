"""Declarative Agent wrapper around ``run_agent``."""

from __future__ import annotations

import logging
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import JsonInferenceResponse

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
)
from t0_agents.core.middleware.base import RunContext, build_pipeline
from t0_agents.core.run import Run, build_run_input
from t0_agents.core.structured import (
    ParseStrategy,
    StructuredOutput,
    permissive_parse,
    run_structured_inference,
)
from t0_agents.core.types import (
    AgentResult,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.mcp.loop import run_agent
from t0_agents.memory.base import MemoryRecord, render_memory_summary
from t0_agents.stream.events import AgentEvent, DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop

logger = logging.getLogger(__name__)

# Default tool timeout used when AgentConfig.tool_timeout is None.
# Matches run_agent's own default so behaviour is identical.
_DEFAULT_TOOL_TIMEOUT: float = 30.0


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
    retry and risk crowding out the model's context.  The enrichment is
    additive on top of a fixed English base sentence; opt out entirely by
    setting ``permissive_json_parsing=False, max_json_retries=0``.
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
    exposes ``agent.invoke(message) -> Run`` as the v1.0 canonical
    entry point.  The caller owns the lifecycle of both the TensorZero
    gateway and the tool provider (when one is supplied — typically a
    ``fastmcp.Client``).

    The ``tool_provider`` kwarg is **required for chat-mode agents**
    (``function_kind="chat"``) because the tool loop calls
    ``list_tools()`` / ``call_tool()`` on it every turn.  For JSON-mode
    agents (``function_kind="json"``) it is **optional**: JSON functions
    are single-shot and never dispatch MCP tools, so a JSON agent can be
    constructed with just the gateway.

    Cross-cutting concerns (memory, guardrails, cost-tracking, sub-agent
    orchestration, JSON retry) are materialised by
    :meth:`AgentConfig.__post_init__` as a middleware chain on
    ``config.middleware``.  The agent itself is now a thin dispatch
    layer that builds a :class:`RunContext`, composes the middleware
    chain around the appropriate terminal callable, and drives it.

    Example (chat)::

        config = AgentConfig(name="assistant", function_name="agent")
        agent = Agent(config, t0=gateway, tool_provider=mcp_client)
        result = await agent.invoke("What is 2 + 3?")

    Example (JSON, no tool provider)::

        config = AgentConfig(
            name="extractor",
            function_name="extract_customer",
            function_kind="json",
            json_mode="strict",
        )
        agent = Agent(config, t0=gateway)
        result = await agent.invoke("Hi, I'm Sarah Johnson <sarah@example.com>")
        print(result.parsed)
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider | None = None,
    ) -> None:
        if tool_provider is not None and not isinstance(tool_provider, ToolProvider):
            raise ConfigurationError(
                f"tool_provider must implement the ToolProvider protocol "
                f"(list_tools + call_tool), got {type(tool_provider).__name__}"
            )
        if config.function_kind == "chat" and tool_provider is None:
            raise ConfigurationError(
                "Agent with function_kind='chat' requires a tool_provider. "
                "Pass a fastmcp.Client wrapping your tool server, or set "
                "AgentConfig(function_kind='json') for a single-shot JSON "
                "agent that does not need a tool provider."
            )
        if config.memory is not None and not isinstance(config.memory, MemoryProvider):
            raise ConfigurationError(
                f"config.memory must implement the MemoryProvider protocol "
                f"(load + save + search), got {type(config.memory).__name__}"
            )
        self._config = config
        self._t0 = t0
        self._tool_provider: ToolProvider | None = tool_provider
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
            # Cache the tool-name -> sub-agent map so the
            # ``SubAgentMiddleware`` (which wraps ``tool_provider`` with
            # ``_OrchestrationToolProvider``) can dispatch sub-agent
            # calls inline without crossing FastMCP's transport task
            # boundary.
            self._sub_agents_by_tool_name = {
                default_tool_name(sub): sub for sub in config.sub_agents
            }

        # JSON-mode safety state.  ``_tool_provider_verified_empty`` is
        # a positive latch — set to ``True`` after the first successful
        # ``list_tools()`` round-trip that confirms the provider is
        # empty.  We cannot decide at construction time whether the
        # check is needed: ``Agent.invoke`` accepts a per-call
        # ``json_mode`` override, so a config-level ``"strict"`` agent
        # may still receive ``invoke(..., json_mode="tool")`` and must
        # verify the provider lazily.  ``_warned_anthropic_combination``
        # latches the once-per-instance Anthropic extended-thinking
        # compatibility warning (TZ guide §11.4).
        self._tool_provider_verified_empty = False
        self._warned_anthropic_combination = False

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

    # ------------------------------------------------------------------
    # v1.0 canonical entry point + deprecation shims
    # ------------------------------------------------------------------

    def invoke(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
        json_mode: Literal["off", "on", "strict", "tool"] | None = None,
        max_json_retries: int | None = None,
        prior_messages: Sequence[Mapping[str, Any]] | None = None,
        middleware: Sequence[Any] | None = None,
        output: Any | None = None,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
    ) -> Run[Any]:
        """v1.0 canonical entry point — returns a :class:`Run` handle.

        The single entry point for both chat and JSON agents; the
        :class:`Run` dispatches the drive mode on first consumer
        (``result()`` selects the non-streaming chat path or the JSON
        path; ``events()`` selects the streaming chat path or raises
        :class:`ConfigurationError` for JSON agents).

        Args:
            user_message: The user's query as a plain string.
            episode_id: Existing TensorZero episode id, or ``None`` to
                let the gateway mint a fresh one.
            tags: Per-call tags shallow-merged on top of ``config.tags``.
            system_arguments: Template arguments forwarded as
                ``input.system``.
            system: Raw system prompt forwarded as ``input.system``
                (string form).  Mutually exclusive with
                ``system_arguments`` and with
                ``memory_strategy="system_template"``.
            output_schema: **M1-deprecated** per-call schema override
                for JSON agents.  Removed in M3 alongside the
                :class:`AgentConfig` field; use the M3 ``output=``
                replacement when it lands.
            json_mode: **M1-deprecated** per-call ``json_mode`` override
                for JSON agents.  Removed in M3.
            max_json_retries: **M1-deprecated** per-call retry budget
                override for JSON agents.  Removed in M3.
            prior_messages: TensorZero-shape ``{role, content}`` history
                to seed the inference loop; used by :meth:`Run.fork` to
                continue from a recorded transcript.  Overrides any
                ``prior_messages`` the memory strategy would supply.
            middleware: Explicit middleware chain that overrides
                ``config.middleware`` for this Run only.  ``None`` (the
                default) uses the config-materialised chain.
            output: **M3 placeholder** — stored on the Run, not yet
                consulted by the orchestration.
            eager: When ``True`` (default), the Run's drive task is
                scheduled immediately.  ``eager=False`` defers task
                start to the first consumer of ``result()`` /
                ``events()`` so callers can queue Runs without firing
                inference.
            context: User-supplied metadata dict copied onto the Run.

        Returns:
            A :class:`Run` whose lifecycle is the canonical v1.0 shape.
            Awaiting the Run is equivalent to ``await run.result()``.

        Raises:
            ValueError: When ``system`` is set alongside
                ``system_arguments`` or alongside
                ``memory_strategy='system_template'``.
            ConfigurationError: When per-call ``output_schema`` or
                ``json_mode`` is passed against a chat agent, or when
                a JSON agent is configured with
                ``memory_strategy='messages'``.
        """
        # Sync pre-flight so the synchronous-call surface stays
        # byte-identical with v0.12.  ``Agent.run`` and ``Agent.stream``
        # raised these eagerly; in v1.0 they surface from the shim's
        # synchronous ``self.invoke(...)`` call before the Run is even
        # constructed.
        self._validate_system_kwargs(system, system_arguments)
        if self._config.function_kind == "chat" and (
            output_schema is not None or json_mode is not None
        ):
            raise ConfigurationError(
                "output_schema/json_mode kwargs are only valid when "
                "AgentConfig.function_kind == 'json'; got function_kind="
                f"{self._config.function_kind!r}."
            )
        if self._config.function_kind == "json":
            # Mirror the JSON terminal's step 0 sync validations so they
            # raise before the Run task spawns — preserves the v0.12
            # caller experience (the same exception is raised from the
            # same await).
            effective_mode_for_safety = (
                json_mode if json_mode is not None else self._config.json_mode
            )
            self._maybe_warn_anthropic_quirk(effective_mode_for_safety)
            if self._resolve_memory_strategy() == "messages":
                raise ConfigurationError(
                    "memory_strategy='messages' is not supported for "
                    "function_kind='json' — JSON functions are single-shot. "
                    "Use memory_strategy='prepended_message' (the default) "
                    "or 'system_template' instead."
                )

        input_ = build_run_input(
            user_message=user_message,
            function_name=self._config.function_name,
            function_kind=self._config.function_kind,
            episode_id=episode_id,
            tags=tags,
            system=system,
            system_arguments=system_arguments,
            output_schema=output_schema,
            json_mode=json_mode,
            max_json_retries=max_json_retries,
            prior_messages=prior_messages,
        )
        return Run(
            self,
            input_,
            eager=eager,
            context=context,
            middleware=middleware,
            output=output,
        )

    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
        json_mode: Literal["off", "on", "strict", "tool"] | None = None,
        max_json_retries: int | None = None,
    ) -> AgentResult:
        """Deprecated v0.12 entry point — delegates to :meth:`invoke`.

        Emits ``DeprecationWarning`` once per source line (Python's
        default filter dedups by stacklevel).  Output stays byte-
        identical to v0.12; the only stderr change is the warning.
        Removed in v1.1.
        """
        warnings.warn(
            "Agent.run() is deprecated and will be removed in t0-agents 1.1. "
            "Use `await agent.invoke(msg, ...)` instead — Run is awaitable "
            "and returns the same AgentResult.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.invoke(
            user_message,
            episode_id=episode_id,
            tags=tags,
            system_arguments=system_arguments,
            system=system,
            output_schema=output_schema,
            json_mode=json_mode,
            max_json_retries=max_json_retries,
        ).result()

    async def stream(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Deprecated v0.12 streaming entry point — delegates to :meth:`invoke`.

        Emits ``DeprecationWarning`` once per source line.  Output
        stays byte-identical to v0.12; the only stderr change is the
        warning.  Removed in v1.1.
        """
        warnings.warn(
            "Agent.stream() is deprecated and will be removed in t0-agents 1.1. "
            "Use `async for ev in agent.invoke(msg, ...).events()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        async for ev in self.invoke(
            user_message,
            episode_id=episode_id,
            tags=tags,
            system_arguments=system_arguments,
            system=system,
        ).events():
            yield ev

    # ------------------------------------------------------------------
    # Pipeline driver + terminals (v1.0 orchestration)
    # ------------------------------------------------------------------

    async def _drive(
        self,
        ctx: RunContext,
        *,
        is_streaming: bool,
        middleware: Sequence[Any] | None = None,
    ) -> AgentResult:
        """Build the middleware pipeline and drive it.

        Called by :class:`~t0_agents.core.run.Run` after the first
        consumer (``result()`` or ``events()``) picks the drive mode.
        Selects the terminal by ``(function_kind, is_streaming)`` and
        wraps the supplied middleware chain around it.

        ``middleware=None`` (the default) reads from
        ``ctx.config.middleware`` — the v0.12-compat chain materialised
        by :meth:`AgentConfig.__post_init__`.  A per-Run override (the
        ``middleware=`` kwarg to :meth:`Agent.invoke`) takes precedence
        and is threaded in through ``Run._drive``.  An empty list (the
        explicit "no middleware" override) bypasses the chain entirely;
        the terminal runs unwrapped.
        """
        terminal: Callable[[RunContext], Awaitable[AgentResult]]
        if self._config.function_kind == "json":
            terminal = self._json_terminal
        elif is_streaming:
            terminal = self._chat_streaming_terminal
        else:
            terminal = self._chat_terminal

        if middleware is None:
            effective: list[Any] = list(ctx.config.middleware or [])
        else:
            effective = list(middleware)
        pipeline = build_pipeline(effective, terminal)
        return await pipeline(ctx)

    async def _chat_terminal(self, ctx: RunContext) -> AgentResult:
        """Non-streaming chat terminal — drives the inner tool loop.

        Calls :func:`run_agent` with the (possibly memory-injected)
        ``ctx.user_message`` and the (possibly wrapped)
        ``ctx.tool_provider``.  Memory hydration, guardrail wrapping,
        sub-agent orchestration, and cost accumulation already happened
        upstream in the middleware chain — the terminal is now the
        slim "drive the inference loop" body.
        """
        tool_provider = ctx.tool_provider
        assert tool_provider is not None, (
            "Chat agents require a tool_provider; rejected at Agent.__init__."
        )
        tool_timeout = (
            self._config.tool_timeout
            if self._config.tool_timeout is not None
            else _DEFAULT_TOOL_TIMEOUT
        )
        extra_kwargs: dict[str, Any] = {}
        # Empty system_arguments dict collapses to "no key" so callers
        # passing ``system_arguments={}`` produce input without
        # ``input.system`` — matches v0.12 ``_merge_caller_system_arguments``
        # semantics.
        if ctx.system_arguments:
            extra_kwargs["system_arguments"] = ctx.system_arguments
        if ctx.prior_messages is not None:
            extra_kwargs["prior_messages"] = list(ctx.prior_messages)

        return await run_agent(
            self._t0,
            cast(Client[Any], tool_provider),
            ctx.user_message,
            self._config.function_name,
            episode_id=ctx.episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._merge_tags(ctx.tags),
            namespace=self._config.namespace,
            on_inference=ctx.on_inference,
            system=ctx.system,
            **extra_kwargs,
        )

    async def _chat_streaming_terminal(self, ctx: RunContext) -> AgentResult:
        """Streaming chat terminal — pushes events to the Run buffer, returns final result.

        Mirrors :meth:`_chat_terminal` but drives via
        :class:`StreamingAgentLoop`.  Every event yielded by the
        streaming runner (except the terminal :class:`DoneEvent`) is
        pushed to ``ctx.run._buffer`` so :meth:`Run.events` consumers
        see them as they happen.  The :class:`DoneEvent`'s
        ``AgentResult`` is captured and returned — the
        :class:`~t0_agents.core.run.Run` driver pushes a synthetic
        :class:`DoneEvent` with the result (after any outer middleware
        modifies it, e.g.  :class:`GuardrailsMiddleware` populating
        ``guardrail_log``) once the chain unwinds.
        """
        tool_provider = ctx.tool_provider
        assert tool_provider is not None, (
            "Chat agents require a tool_provider; rejected at Agent.__init__."
        )
        tool_timeout = (
            self._config.tool_timeout
            if self._config.tool_timeout is not None
            else _DEFAULT_TOOL_TIMEOUT
        )
        extra_kwargs: dict[str, Any] = {}
        # Empty system_arguments dict collapses to "no key" (parity with
        # :meth:`_chat_terminal`).
        if ctx.system_arguments:
            extra_kwargs["system_arguments"] = ctx.system_arguments
        if ctx.prior_messages is not None:
            extra_kwargs["prior_messages"] = list(ctx.prior_messages)

        streamer = StreamingAgentLoop()
        done_result: AgentResult | None = None
        run = ctx.run
        assert run is not None, "_chat_streaming_terminal requires a live Run"

        async for event in streamer.stream(
            self._t0,
            cast(Client[Any], tool_provider),
            ctx.user_message,
            self._config.function_name,
            episode_id=ctx.episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._merge_tags(ctx.tags),
            namespace=self._config.namespace,
            on_inference=ctx.on_inference,
            system=ctx.system,
            **extra_kwargs,
        ):
            if isinstance(event, DoneEvent):
                done_result = event.result
                # Don't push the streaming runner's DoneEvent — outer
                # middlewares may still mutate ``result.guardrail_log``
                # before the Run driver pushes a synthetic DoneEvent
                # with the final, populated value.
                break
            await run._buffer.put(event)

        if done_result is None:
            raise RuntimeError(
                "Streaming runner completed without a DoneEvent — "
                "this indicates a bug in the streaming runner."
            )
        return done_result

    async def _json_terminal(self, ctx: RunContext) -> AgentResult:
        """JSON function dispatch — retry loop wrapping ``run_structured_inference``.

        Reproduces the v0.12 retry semantics inline.  In M2 the
        :class:`JsonRetryMiddleware` is a no-op placeholder; M3 will
        invert this — the retry loop moves into the middleware
        (consulting ``OutputMode.recover()``) and the terminal becomes
        a single ``run_structured_inference`` call.

        Memory load/save, ``before_run``/``after_run`` hooks, and
        cost accumulation are handled by the outer middleware chain;
        this terminal only manages the retry budget and the
        ``permissive_parse`` recovery layer.
        """
        # --- 0. Safety check: json_mode='tool' against non-empty provider ---
        # Uses ``self._tool_provider`` directly (not ``ctx.tool_provider``)
        # so the check sees the raw agent-level provider rather than any
        # GuardedToolProvider wrap.  Latched to once per Agent instance.
        run_input = ctx.run.input if ctx.run is not None else None
        per_call_json_mode = run_input.json_mode if run_input is not None else None
        effective_mode_for_safety = (
            per_call_json_mode
            if per_call_json_mode is not None
            else self._config.json_mode
        )
        if (
            effective_mode_for_safety == "tool"
            and not self._tool_provider_verified_empty
        ):
            if self._tool_provider is None:
                self._tool_provider_verified_empty = True
            else:
                advertised_tools = await self._tool_provider.list_tools()
                if advertised_tools:
                    raise ConfigurationError(
                        "function_kind='json' with json_mode='tool' is "
                        "incompatible with a tool_provider that advertises "
                        "tools (TensorZero guide §11.1).  Either change "
                        "json_mode to 'strict'/'on'/'off' or use an empty "
                        "tool_provider."
                    )
                self._tool_provider_verified_empty = True

        # --- 1. Resolve effective output_schema / json_mode / max_json_retries ---
        # Per-call kwargs (M1-deprecated; M3-removed) take precedence over
        # the config-level fields.  ``AgentConfig.__post_init__`` resolves
        # ``Path`` schemas into ``Mapping`` at construction time, so the
        # config field is ``Mapping | None`` here.
        per_call_output_schema = (
            run_input.output_schema if run_input is not None else None
        )
        per_call_max_retries = (
            run_input.max_json_retries if run_input is not None else None
        )
        config_schema = self._config.output_schema
        effective_schema: Mapping[str, Any] | None
        if per_call_output_schema is not None:
            effective_schema = per_call_output_schema
        elif config_schema is not None:
            effective_schema = cast(Mapping[str, Any], config_schema)
        else:
            effective_schema = None
        effective_mode = (
            per_call_json_mode
            if per_call_json_mode is not None
            else self._config.json_mode
        )

        # --- 2. Retry budget + per-call override ---
        retry_budget = (
            per_call_max_retries
            if per_call_max_retries is not None
            else self._config.max_json_retries
        )
        permissive_enabled = self._config.permissive_json_parsing

        attempts = 0
        prior_msgs: list[Mapping[str, Any]] = []
        current_user = ctx.user_message
        inference_ids: list[str] = []
        captured_episode: str | None = None
        final_parsed: Any | None = None
        final_raw: str = ""
        final_strategy: ParseStrategy = "none"

        on_inference_cb = cast(
            Callable[[JsonInferenceResponse], Awaitable[None]] | None,
            ctx.on_inference,
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
                # Empty dict collapses to None (parity with
                # ``_merge_caller_system_arguments`` in v0.12).
                system_arguments=ctx.system_arguments or None,
                system=ctx.system,
                episode_id=(
                    captured_episode
                    if captured_episode is not None
                    else ctx.episode_id
                ),
                tags=self._merge_tags(ctx.tags),
                namespace=self._config.namespace,
                dryrun=self._config.dryrun,
                on_inference=on_inference_cb,
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
                    final_strategy = "retry" if attempts > 0 else "permissive"
                    break

            if attempts >= retry_budget:
                final_parsed = None
                final_strategy = "none"
                break

            prior_msgs.append({"role": "user", "content": current_user})
            prior_msgs.append({"role": "assistant", "content": structured.raw})
            current_user = _build_retry_hint(effective_schema)
            attempts += 1

        return AgentResult(
            response=final_raw,
            episode_id=captured_episode,
            inference_ids=inference_ids,
            turns=1 + attempts,
            function_kind="json",
            parsed=final_parsed,
            parse_strategy=final_strategy,
            json_parse_attempts=attempts,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_warn_anthropic_quirk(
        self,
        effective_mode: Literal["off", "on", "strict", "tool"] | None,
    ) -> None:
        """Emit the Anthropic extended-thinking compatibility warning once.

        TensorZero guide §11.4 documents that Anthropic's
        extended-thinking models reject ``json_mode`` values other than
        ``"strict"`` or ``"off"``.  The SDK does not expose provider
        routing, so the warning is generic — operators verify the
        deployment themselves.  The latch (``_warned_anthropic_combination``)
        keeps the warning to once per :class:`Agent` instance so a
        long-lived agent that invokes repeatedly does not emit log
        spam.  Modes ``"strict"`` and ``"off"`` are safe and do not
        trigger the warning.

        Takes the *effective* mode (per-call override resolved against
        the config field) so a chat-agent-style config that gets
        ``invoke(..., json_mode="on")`` per call still surfaces the
        warning.
        """
        if self._warned_anthropic_combination:
            return
        if effective_mode in {"on", "tool"}:
            logger.warning(
                "AgentConfig/Agent.invoke(json_mode=%r) — Anthropic "
                "extended-thinking models reject this combination "
                "(TensorZero guide §11.4).  Verify your provider routing.",
                effective_mode,
            )
            self._warned_anthropic_combination = True

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

    def _validate_system_kwargs(
        self,
        system: str | None,
        system_arguments: dict[str, Any] | None,
    ) -> None:
        """Reject illegal combinations of ``system`` / ``system_arguments``.

        ``system`` (raw string) and ``system_arguments`` (template dict)
        both target ``input.system`` and are mutually exclusive at the
        gateway level.  Memory strategy ``"system_template"`` also writes
        to ``input.system`` via a memory-derived dict, so combining it
        with a raw ``system`` string would silently drop the memory
        context; that combination is rejected up front for the same
        reason the gateway would reject mixed shapes.

        ``"prepended_message"`` and ``"messages"`` memory strategies
        leave ``input.system`` untouched and are compatible with both
        kwargs.
        """
        if system is None:
            return
        if system_arguments is not None:
            raise ValueError(
                "Pass either `system` (raw string for input.system) or "
                "`system_arguments` (dict for template variables), not both."
            )
        if self._resolve_memory_strategy() == "system_template":
            raise ValueError(
                "system=<str> is incompatible with "
                "memory_strategy='system_template' — that strategy injects "
                "memory_context into input.system via the template, which "
                "requires a dict.  Either pass `system_arguments` (with a "
                "`memory_context` key) or switch memory_strategy to "
                "'prepended_message' or 'messages'."
            )

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
