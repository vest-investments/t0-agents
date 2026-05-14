"""Single-shot JSON-function inference helper.

Sibling of :func:`t0_agents.mcp.loop.run_agent` for TensorZero functions
whose ``type = "json"``.  JSON functions are single-shot by contract —
there is no tool loop, no tool dispatch, no termination criterion to
evaluate; the gateway returns ``output.raw`` (the model's JSON string)
and ``output.parsed`` (a JSON-parsed mapping or ``None``).  This module
exposes that one inference call with the kwargs that make sense for a
JSON function.

The helper deliberately omits ``parallel_tools``, ``tool_choice``,
``tool_timeout``, ``prior_messages``, and ``max_turns`` — none of them
are meaningful for a single-shot JSON inference.

Only ``mcp/`` is forbidden from importing higher-level modules.  This
``core/structured.py`` lives alongside ``core/agent.py``,
``core/config.py``, and ``core/types.py`` and follows the same dependency
hygiene: it imports only from ``tensorzero`` and the standard library.

The ``parse_strategy`` field carries the four-value vocabulary the
later milestones will populate.  In M1 the helper produces only
``"strict"`` (TZ parsed the JSON) or ``"none"`` (TZ returned ``parsed
== None``).  M3 will fill in ``"permissive"`` and ``"retry"``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    Usage,
)

ParseStrategy = Literal["strict", "permissive", "retry", "none"]
"""How :class:`StructuredOutput.parsed` came to be populated.

- ``"strict"`` — TensorZero's own parse populated ``output.parsed``.
- ``"permissive"`` — M3's in-Python recovery layer extracted JSON from
  ``output.raw`` after TZ returned ``parsed == None``.
- ``"retry"`` — M3's retry loop re-ran the inference (with a corrective
  user-turn hint) and one of the subsequent attempts produced
  parseable output.
- ``"none"`` — every layer failed; ``parsed`` is ``None``.
"""


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    """Result of a single TensorZero JSON-function inference.

    Carries every field the caller is likely to need for both happy-path
    consumption (``parsed``) and observability (``raw``, ``parse_strategy``,
    ``retry_attempts``, ``usage``).  Mirrors the shape of
    ``tensorzero.types.JsonInferenceResponse`` but flattens
    ``response.output.raw`` and ``response.output.parsed`` onto the top
    level for ergonomic access.

    ``parsed`` is typed as ``Any | None`` rather than the SDK's stricter
    ``Mapping[str, Any] | None`` so that M3's permissive layer can surface
    top-level arrays or other shapes without widening a published type.
    """

    raw: str
    parsed: Any | None
    episode_id: str
    inference_id: str
    usage: Usage | None
    parse_strategy: ParseStrategy = "strict"
    retry_attempts: int = 0


async def run_structured_inference(
    t0: AsyncTensorZeroGateway,
    function_name: str,
    user_message: str,
    *,
    output_schema: Mapping[str, Any] | None = None,
    json_mode: Literal["off", "on", "strict", "tool"] | None = None,
    system_arguments: dict[str, Any] | None = None,
    episode_id: str | None = None,
    tags: dict[str, str] | None = None,
    namespace: str | None = None,
    dryrun: bool = False,
    on_inference: (
        Callable[[JsonInferenceResponse], Awaitable[None]] | None
    ) = None,
) -> StructuredOutput:
    """Run one TensorZero JSON-function inference.

    Args:
        t0: An :class:`AsyncTensorZeroGateway` (typically built via
            ``AsyncTensorZeroGateway.build_http(gateway_url=...)``).
        function_name: The name of a TensorZero function configured with
            ``type = "json"``.  Pointing this at a chat function will
            raise ``RuntimeError`` once the gateway returns a
            ``ChatInferenceResponse``.
        user_message: The user turn to send.  JSON functions are
            single-shot so this is the only message the gateway sees
            unless ``system_arguments`` carries a templated system
            prompt.
        output_schema: Optional dynamic schema override (TZ guide §4).
            Sent as the request's top-level ``output_schema`` field.
            When ``None``, the function's static configuration applies.
        json_mode: Optional inference-time ``json_mode`` override
            (TZ guide §5).  Nested under
            ``params.chat_completion.json_mode`` and takes precedence
            over the variant configuration.  When ``None``, the
            configured variant value applies.
        system_arguments: Template arguments forwarded as
            ``input.system``.  Matches the semantics of
            :func:`t0_agents.mcp.loop.run_agent`'s ``system_arguments``.
        episode_id: Continues an existing TensorZero episode when set;
            ``None`` lets the gateway mint a fresh one.
        tags: Free-form tags forwarded for observability.
        namespace: TensorZero namespace for experimentation routing.
        dryrun: When ``True``, TZ executes the inference without
            persisting it.  Useful for tests that exercise the request
            shape against a live gateway.
        on_inference: Async callback invoked exactly once with the raw
            :class:`JsonInferenceResponse` before this function returns.
            Mirrors :func:`run_agent`'s hook and is the entry point for
            cost-tracking or other per-call observability.

    Returns:
        :class:`StructuredOutput` whose ``parse_strategy`` is
        ``"strict"`` when the gateway returned a non-null ``parsed``
        and ``"none"`` otherwise.  M3 will extend this with
        ``"permissive"`` and ``"retry"`` strategies.

    Raises:
        RuntimeError: If the gateway returned a ``ChatInferenceResponse``
            (the configured function is not a JSON function), or if the
            call returned a streaming iterator (we never set
            ``stream=True``, but the SDK declares the union).
    """
    input_dict: dict[str, Any] = {
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_arguments is not None:
        input_dict["system"] = system_arguments

    params: dict[str, Any] | None = None
    if json_mode is not None:
        params = {"chat_completion": {"json_mode": json_mode}}

    raw = await t0.inference(
        function_name=function_name,
        input=input_dict,
        output_schema=dict(output_schema) if output_schema is not None else None,
        params=params,
        episode_id=episode_id,
        tags=tags,
        namespace=namespace,
        dryrun=dryrun,
    )

    if not isinstance(raw, JsonInferenceResponse):
        raise RuntimeError(
            f"Function {function_name!r} did not return a "
            f"JsonInferenceResponse (got {type(raw).__name__}). "
            f"Verify that the function is configured as type=\"json\" "
            f"in your tensorzero.toml."
        )

    response: JsonInferenceResponse = raw

    if on_inference is not None:
        await on_inference(response)

    parsed_value: Any | None = cast(Any, response.output.parsed)
    raw_value: str = response.output.raw or ""
    strategy: ParseStrategy = "strict" if parsed_value is not None else "none"

    return StructuredOutput(
        raw=raw_value,
        parsed=parsed_value,
        episode_id=str(response.episode_id),
        inference_id=str(response.inference_id),
        usage=response.usage,
        parse_strategy=strategy,
        retry_attempts=0,
    )
