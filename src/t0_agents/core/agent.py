"""Declarative Agent wrapper around ``run_agent``."""

from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar, cast, overload

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
)
from t0_agents.core.middleware.base import RunContext, build_pipeline
from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputMode,
)
from t0_agents.core.run import Run, build_run_input
from t0_agents.core.types import (
    AgentResult,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.mcp.loop import run_agent
from t0_agents.memory.base import MemoryRecord, render_memory_summary
from t0_agents.stream.events import DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Agent:
    """Declarative wrapper around :func:`run_agent`.

    Encapsulates configuration in an :class:`AgentConfig` dataclass and
    exposes ``agent.invoke(message) -> Run`` as the v1.0 canonical
    entry point.  The caller owns the lifecycle of both the TensorZero
    gateway and the tool provider (when one is supplied — typically a
    ``fastmcp.Client``).

    The ``tool_provider`` kwarg is **optional for every output mode**.
    Chat-mode agents (``output=ChatOutput()``, the default) call
    ``list_tools()`` / ``call_tool()`` on it every turn; when the
    caller passes ``None``, that ``None`` flows straight through to
    :func:`run_agent` (and to :class:`StreamingAgentLoop` for the
    streaming path).  The inner loop skips ``list_tools()``, the LLM
    sees no tools and produces a text response, and the loop
    terminates after the first turn — no MCP server is instantiated.
    This is the "toolless chat" convenience for thin chatbot-style
    use cases.  Single-shot agents (``output=JsonOutput(...)``,
    ``output=StructuredOutput(Model)``, ``output=ToolCallOutput()``)
    never dispatch MCP tools, so the kwarg has always been optional
    for them.

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
            output=JsonOutput(json_mode="strict"),
        )
        agent = Agent(config, t0=gateway)
        result = await agent.invoke("Hi, I'm Sarah Johnson <sarah@example.com>")
        print(result.parsed)

    Example (Pydantic)::

        from pydantic import BaseModel
        class Customer(BaseModel):
            name: str
            email: str
        config = AgentConfig(
            name="extractor",
            function_name="extract_customer",
            output=StructuredOutput(Customer),
        )
        agent = Agent(config, t0=gateway)
        result = await agent.invoke("Hi, Sarah <sarah@example.com>").result()
        print(result.value)        # Customer(name='Sarah', email='sarah@example.com')
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

        # Wrap the tool provider with SubAgentToolProvider when
        # ``config.sub_agents`` is non-empty: the provider synthesises a
        # tool surface from the sub-agent list and dispatches matching
        # names inline, replacing the legacy server-side ``add_tool``
        # registration + middleware-side ``_OrchestrationToolProvider``
        # wrap.  No-op when ``config.sub_agents`` is empty, so
        # single-agent constructions pay no cost.  ``inner=tool_provider``
        # preserves any real MCP tools the orchestrator also wants; the
        # synthetic delegation tools are appended after the inner list.
        if config.sub_agents:
            from t0_agents.delegation.provider import SubAgentToolProvider

            self._tool_provider = SubAgentToolProvider(
                config.sub_agents,
                inner=tool_provider,
            )

        # JSON-mode safety state.  ``_tool_provider_verified_empty`` is
        # a positive latch — set to ``True`` after the first successful
        # ``list_tools()`` round-trip that confirms the provider is
        # empty.  We cannot decide at construction time whether the
        # check is needed: ``Agent.invoke`` accepts a per-call
        # ``output=`` override, so a config-level ``ChatOutput`` agent
        # may still receive ``invoke(..., output=JsonOutput(json_mode="tool"))``
        # and must verify the provider lazily.
        # ``_warned_anthropic_combination`` latches the once-per-instance
        # Anthropic extended-thinking compatibility warning (TZ guide §11.4).
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
    # v1.0 canonical entry point
    # ------------------------------------------------------------------

    @overload
    def invoke(
        self,
        user_message: str,
        *,
        output: OutputMode[T],
        episode_id: str | None = ...,
        tags: dict[str, str] | None = ...,
        system_arguments: dict[str, Any] | None = ...,
        system: str | None = ...,
        prior_messages: Sequence[Mapping[str, Any]] | None = ...,
        middleware: Sequence[Any] | None = ...,
        eager: bool = ...,
        context: Mapping[str, Any] | None = ...,
    ) -> Run[T]: ...

    @overload
    def invoke(
        self,
        user_message: str,
        *,
        output: None = ...,
        episode_id: str | None = ...,
        tags: dict[str, str] | None = ...,
        system_arguments: dict[str, Any] | None = ...,
        system: str | None = ...,
        prior_messages: Sequence[Mapping[str, Any]] | None = ...,
        middleware: Sequence[Any] | None = ...,
        eager: bool = ...,
        context: Mapping[str, Any] | None = ...,
    ) -> Run[Any]: ...

    def invoke(
        self,
        user_message: str,
        *,
        output: OutputMode[Any] | None = None,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        prior_messages: Sequence[Mapping[str, Any]] | None = None,
        middleware: Sequence[Any] | None = None,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
    ) -> Run[Any]:
        """v1.0 canonical entry point — returns a :class:`Run` handle.

        Awaitable: ``result = await agent.invoke(msg)`` is the shortest
        path from query to answer.  Use :meth:`Run.events` for
        streaming, :meth:`Run.replay` / :meth:`Run.fork` for
        re-invocation, :meth:`Run.cancel` for in-flight cancellation.

        Per-call kwargs override their :class:`AgentConfig` analogues.
        ``output=`` narrows ``Run[T]`` via overload.  ``middleware=[]``
        is an explicit "no middleware" override (drops cost tracking
        too — see :doc:`run`).

        See :doc:`run` for the full surface, lifecycle, and recipes.
        """
        effective_output: OutputMode[Any] = (
            output if output is not None else self._config.output
        )

        # Sync pre-flight so the synchronous-call surface stays
        # byte-identical with v0.12.
        self._validate_system_kwargs(system, system_arguments)
        effective_json_mode = _resolve_json_mode(effective_output)
        if effective_json_mode is not None:
            self._maybe_warn_anthropic_quirk(effective_json_mode)
        if (
            not effective_output.expects_tool_loop
            and self._resolve_memory_strategy() == "messages"
        ):
            raise ConfigurationError(
                f"memory_strategy='messages' is not supported for "
                f"output={effective_output.name!r} (single-shot mode — "
                f"there is no multi-turn message history to thread).  Use "
                f"memory_strategy='prepended_message' (the default) or "
                f"'system_template' instead."
            )

        input_ = build_run_input(
            user_message=user_message,
            function_name=self._config.function_name,
            episode_id=episode_id,
            tags=tags,
            system=system,
            system_arguments=system_arguments,
            prior_messages=prior_messages,
        )
        return Run(
            self,
            input_,
            effective_output,
            eager=eager,
            context=context,
            middleware=middleware,
        )

    # ------------------------------------------------------------------
    # Pipeline driver + terminals (v1.0 orchestration)
    # ------------------------------------------------------------------

    async def _drive(
        self,
        ctx: RunContext,
        *,
        is_streaming: bool,
        middleware: Sequence[Any] | None = None,
    ) -> AgentResult[Any]:
        """Build the middleware pipeline and drive it.

        Called by :class:`~t0_agents.core.run.Run` after the first
        consumer (``result()`` or ``events()``) picks the drive mode.
        Selects the terminal by ``(output.expects_tool_loop, is_streaming)``
        and wraps the supplied middleware chain around it.

        ``middleware=None`` (the default) reads from
        ``ctx.config.middleware`` — the v0.12-compat chain materialised
        by :meth:`AgentConfig.__post_init__`.  A per-Run override (the
        ``middleware=`` kwarg to :meth:`Agent.invoke`) takes precedence
        and is threaded in through ``Run._drive``.  An empty list (the
        explicit "no middleware" override) bypasses the chain entirely;
        the terminal runs unwrapped.
        """
        terminal: Callable[[RunContext], Awaitable[AgentResult[Any]]]
        if ctx.output is None or not ctx.output.expects_tool_loop:
            terminal = self._single_shot_terminal
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

    async def _chat_terminal(self, ctx: RunContext) -> AgentResult[Any]:
        """Non-streaming chat terminal — drives the inner tool loop.

        Calls :func:`run_agent` with the (possibly memory-injected)
        ``ctx.user_message`` and the (possibly wrapped)
        ``ctx.tool_provider``.  ``ctx.tool_provider`` flows through
        unchanged — including ``None``, which :func:`run_agent`
        interprets as "skip ``list_tools()``; treat the single
        inference as the entire run".  Memory hydration, guardrail
        wrapping, sub-agent orchestration, and cost accumulation
        already happened upstream in the middleware chain — the
        terminal is now the slim "drive the inference loop" body.
        """
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
            cast(Client[Any] | None, ctx.tool_provider),
            ctx.user_message,
            self._config.function_name,
            episode_id=ctx.episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=self._config.tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._merge_tags(ctx.tags),
            namespace=self._config.namespace,
            on_inference=ctx.on_inference,
            system=ctx.system,
            model_name=self._config.model,
            variant_name=self._config.variant,
            params=self._config.params,
            cache_options=self._config.cache,
            **extra_kwargs,
        )

    async def _chat_streaming_terminal(
        self, ctx: RunContext
    ) -> AgentResult[Any]:
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
        extra_kwargs: dict[str, Any] = {}
        # Empty system_arguments dict collapses to "no key" (parity with
        # :meth:`_chat_terminal`).
        if ctx.system_arguments:
            extra_kwargs["system_arguments"] = ctx.system_arguments
        if ctx.prior_messages is not None:
            extra_kwargs["prior_messages"] = list(ctx.prior_messages)

        streamer = StreamingAgentLoop()
        done_result: AgentResult[Any] | None = None
        run = ctx.run
        assert run is not None, "_chat_streaming_terminal requires a live Run"

        async for event in streamer.stream(
            self._t0,
            cast(Client[Any] | None, ctx.tool_provider),
            ctx.user_message,
            self._config.function_name,
            episode_id=ctx.episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=self._config.tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._merge_tags(ctx.tags),
            namespace=self._config.namespace,
            on_inference=ctx.on_inference,
            system=ctx.system,
            progress_sink=run._buffer.put,
            model_name=self._config.model,
            variant_name=self._config.variant,
            params=self._config.params,
            cache_options=self._config.cache,
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

    async def _single_shot_terminal(
        self, ctx: RunContext
    ) -> AgentResult[Any]:
        """Single-shot terminal — one inference, delegated to ``ctx.output``.

        Replaces v0.12's ``_json_terminal`` retry loop.  The terminal
        is now flat:

        1. Build an :class:`InferenceRequest` from ``ctx`` (input
           messages, tags, namespace, etc.).
        2. Apply ``ctx.output.shape_inference(req)`` so the
           :class:`OutputMode` injects its specific kwargs (JSON mode,
           output schema, tool_choice).
        3. Call ``t0.inference(...)``.
        4. Fire ``ctx.on_inference(response)`` so middlewares (cost
           tracking) observe the usage.
        5. Call ``ctx.output.materialize(response, attempts)`` and
           return the typed :class:`AgentResult`.

        Retry orchestration moved into
        :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`,
        which calls this terminal in a loop and consults
        ``ctx.output.recover()`` between attempts.  The middleware
        threads its accumulated attempts list via
        ``ctx.extras["_output_attempts"]`` so :meth:`OutputMode.materialize`
        can choose the right ``parse_strategy`` value (``"strict"`` /
        ``"permissive"`` on the first attempt, ``"retry"`` after one
        or more retries).
        """
        assert ctx.output is not None, (
            "_single_shot_terminal requires ctx.output to be set"
        )

        # --- json_mode='tool' safety check (latched once per Agent) ---
        # Replicates the v0.12 _json_terminal step-0 check: a tool-mode
        # JSON inference combined with a tool-advertising provider is
        # incompatible (TZ guide §11.1).
        json_mode = _resolve_json_mode(ctx.output)
        if (
            json_mode == "tool"
            and not self._tool_provider_verified_empty
        ):
            if self._tool_provider is None:
                self._tool_provider_verified_empty = True
            else:
                advertised_tools = await self._tool_provider.list_tools()
                if advertised_tools:
                    raise ConfigurationError(
                        "output=JsonOutput(json_mode='tool') is incompatible "
                        "with a tool_provider that advertises tools "
                        "(TensorZero guide §11.1).  Either switch to "
                        "json_mode='strict'/'on'/'off' or use an empty "
                        "tool_provider."
                    )
                self._tool_provider_verified_empty = True

        # --- 1. Build the inference request ---
        messages: list[Mapping[str, Any]] = []
        if ctx.prior_messages:
            messages.extend(ctx.prior_messages)
        messages.append({"role": "user", "content": ctx.user_message})
        input_dict: dict[str, Any] = {"messages": messages}
        if ctx.system is not None:
            input_dict["system"] = ctx.system
        elif ctx.system_arguments:
            input_dict["system"] = dict(ctx.system_arguments)

        req = InferenceRequest(
            function_name=self._config.function_name,
            input=input_dict,
            additional_tools=None,
            tool_choice=self._config.tool_choice,
            tags=self._merge_tags(ctx.tags),
            namespace=self._config.namespace,
            dryrun=self._config.dryrun,
            output_schema=None,
            # shape_inference mutates params in place (setdefault +
            # key assignment) — seed a deep copy so the long-lived
            # AgentConfig.params dict is never contaminated across runs.
            params=copy.deepcopy(self._config.params),
            model_name=self._config.model,
            variant_name=self._config.variant,
            cache_options=self._config.cache,
        )

        # --- 2. Apply the output mode's request shaping ---
        req = ctx.output.shape_inference(req)

        # --- 3. Call TensorZero ---
        inference_kwargs: dict[str, Any] = {
            "function_name": req.function_name,
            "input": req.input,
            "episode_id": ctx.episode_id,
            "tags": req.tags,
            "namespace": req.namespace,
            "dryrun": req.dryrun,
        }
        if req.model_name is not None:
            inference_kwargs["model_name"] = req.model_name
        if req.variant_name is not None:
            inference_kwargs["variant_name"] = req.variant_name
        if req.cache_options is not None:
            inference_kwargs["cache_options"] = req.cache_options
        if req.output_schema is not None:
            inference_kwargs["output_schema"] = dict(req.output_schema)
        if req.params is not None:
            inference_kwargs["params"] = req.params
        if req.tool_choice is not None:
            inference_kwargs["tool_choice"] = req.tool_choice
        if req.additional_tools is not None:
            inference_kwargs["additional_tools"] = req.additional_tools

        response = await self._t0.inference(**inference_kwargs)

        # --- 4. Notify middlewares (cost tracking, observability) ---
        if ctx.on_inference is not None:
            await ctx.on_inference(response)

        # --- 5. Materialise via the output mode ---
        attempts: list[Attempt] = ctx.extras.get("_output_attempts", [])
        result = await ctx.output.materialize(response, attempts)
        return result

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

        Takes the *effective* mode (resolved from the per-call or
        config-level :class:`OutputMode`).
        """
        if self._warned_anthropic_combination:
            return
        if effective_mode in {"on", "tool"}:
            logger.warning(
                "OutputMode(json_mode=%r) — Anthropic "
                "extended-thinking models reject this combination "
                "(TensorZero guide §11.4).  Verify your provider routing.",
                effective_mode,
            )
            self._warned_anthropic_combination = True

    def _merge_tags(
        self,
        override: dict[str, str] | None,
    ) -> dict[str, str]:
        """Merge the auto-tag, ``config.tags``, and a per-call override.

        Every inference carries ``t0_agents.agent = config.name`` so
        per-agent observability survives shared functions and the
        functionless passthrough (whose inferences all land under
        ``tensorzero::default``).  Precedence: per-call ``override``
        beats ``config.tags`` beats the auto-tag.

        :meth:`t0_agents.serve.session.SessionFactory._merge_tags` feeds
        ``config.tags`` upstream of this merge, so the serve path
        inherits the auto-tag without duplicating it.
        """
        return {
            "t0_agents.agent": self._config.name,
            **(self._config.tags or {}),
            **(override or {}),
        }

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

        ``system_arguments`` additionally requires a declared function
        with a system template — the functionless passthrough behind
        ``model=`` has none, so that combination is rejected up front
        rather than surfacing as an opaque gateway 400.
        """
        if system_arguments is not None and self._config.model is not None:
            raise ValueError(
                "system_arguments= requires a tensorzero.toml function with "
                "a system template; this agent routes via model= (the "
                "functionless passthrough has no templates). Pass system= "
                "(a raw prompt string), or declare the function and use "
                "function_name=."
            )
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


def _resolve_json_mode(
    output: OutputMode[Any],
) -> Literal["off", "on", "strict", "tool"] | None:
    """Extract ``json_mode`` from an :class:`OutputMode`, when applicable.

    Chat / streaming / tool-call modes have no ``json_mode``;
    :class:`JsonOutput` and :class:`StructuredOutput` expose it as an
    instance field.  Used by :meth:`Agent.invoke` to decide whether to
    fire the Anthropic extended-thinking warning, and by
    :meth:`Agent._single_shot_terminal` to drive the once-per-Agent
    ``json_mode='tool'`` safety check.
    """
    mode = getattr(output, "json_mode", None)
    if mode in {"off", "on", "strict", "tool"}:
        return cast(Literal["off", "on", "strict", "tool"], mode)
    return None
