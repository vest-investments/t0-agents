"""Single-shot JSON-function inference helper.

Sibling of :func:`t0_agents.mcp.loop.run_agent` for TensorZero functions
whose ``type = "json"``.  JSON functions are single-shot by contract —
there is no tool loop, no tool dispatch, no termination criterion to
evaluate; the gateway returns ``output.raw`` (the model's JSON string)
and ``output.parsed`` (a JSON-parsed mapping or ``None``).  This module
exposes that one inference call with the kwargs that make sense for a
JSON function, plus an in-Python :func:`permissive_parse` recovery
function used by :meth:`Agent._run_json` when TensorZero returned
``parsed == None`` but ``raw`` still contains JSON wrapped in prose or
a fenced code block.

The helper deliberately omits ``parallel_tools``, ``tool_choice``,
``tool_timeout``, and ``max_turns`` — none of them are meaningful for a
JSON inference.  ``prior_messages`` *is* exposed so the retry
orchestration in :meth:`Agent._run_json` can thread a corrective hint
back into the conversation after an invalid attempt; callers using the
helper directly should leave it at its ``None`` default unless they
have built their own retry loop.

Only ``mcp/`` is forbidden from importing higher-level modules.  This
``core/structured.py`` lives alongside ``core/agent.py``,
``core/config.py``, and ``core/types.py`` and follows the same dependency
hygiene: it imports only from ``tensorzero`` and the standard library.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
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
- ``"permissive"`` — the in-Python recovery layer extracted JSON from
  ``output.raw`` after TZ returned ``parsed == None``.
- ``"retry"`` — the retry orchestration in :meth:`Agent._run_json`
  re-ran the inference (with a corrective user-turn hint) and one of
  the subsequent attempts produced parseable output.
- ``"none"`` — every layer failed; ``parsed`` is ``None``.
"""

PermissiveStrategy = Literal["strict", "permissive", "none"]
""":func:`permissive_parse`'s outcome.

A narrower alias of :data:`ParseStrategy` — :func:`permissive_parse` is a
pure-string utility and never observes the orchestrator-level ``"retry"``
outcome.  :class:`AgentResult` and :class:`StructuredOutput` carry the
wider :data:`ParseStrategy` alias because the orchestrator may upgrade a
``"strict"`` or ``"permissive"`` outcome to ``"retry"`` when it took
more than one attempt to get there.
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


def _iter_balanced_spans(
    text: str, opener: str, closer: str
) -> Iterator[str]:
    """Yield every balanced ``opener``…``closer`` span in ``text``, in order.

    Iterates through every occurrence of ``opener`` as a candidate
    starting position.  When a starting position produces a balanced
    span (closer matched at depth 0, respecting JSON string-literal
    state with ``\\``-escapes), the span is yielded and iteration
    resumes after its closer.  A literal ``opener`` in prose that never
    closes is silently skipped — the scanner advances to the next
    occurrence rather than aborting.

    Yielding every balanced span — rather than returning only the first
    — lets :func:`permissive_parse` retry ``json.loads`` past spans
    that are syntactically balanced but not valid JSON (for example
    ``{key: value}`` without quoted keys, which models occasionally
    produce in apology text).
    """
    curr = 0
    while True:
        start = text.find(opener, curr)
        if start < 0:
            return

        depth = 0
        in_string = False
        escape = False
        end = -1

        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end >= 0:
            yield text[start : end + 1]
            curr = end + 1
        else:
            # Unbalanced from this starting position — advance past it
            # and retry from the next occurrence of ``opener``.
            curr = start + 1


def _iter_fenced_code(text: str) -> Iterator[str]:
    """Yield the contents of every fenced code block in ``text``, in order.

    Recognises both `````json … ````` and
    plain ````` … ````` blocks; any optional
    language tag on the opening fence is skipped (the scanner advances
    past the first newline after the opening backticks).  Each yielded
    block has its trailing ``\\n`` stripped.  Iteration stops at the
    first opening backticks not followed by a complete block (no
    newline, or no matching closing backticks).

    Chain-of-thought models sometimes emit a prose fence first and a
    JSON fence second — :func:`permissive_parse` tries every block
    until one parses, so the JSON in the second fence is recovered
    even when the first fence does not parse.
    """
    pos = 0
    while True:
        open_idx = text.find("```", pos)
        if open_idx < 0:
            return
        content_start = open_idx + 3
        newline = text.find("\n", content_start)
        if newline < 0:
            return
        content_start = newline + 1
        close_idx = text.find("```", content_start)
        if close_idx < 0:
            return
        yield text[content_start:close_idx].rstrip("\n")
        pos = close_idx + 3


def _ordered_pairs_for(text: str) -> list[tuple[str, str]]:
    """Return ``[("{", "}"), ("[", "]")]`` ordered by which opener
    appears first in ``text``.

    Used by :func:`permissive_parse` so a top-level array containing
    objects (``"[{"k": 1}, {"k": 2}]"``) extracts the array rather than
    the first object — whichever opener appears first in the source
    wins.  Pairs whose opener does not appear in ``text`` are still
    included (at the back) so the fall-back path can still try them
    when the earlier opener produced an unbalanced span.
    """
    brace = text.find("{")
    bracket = text.find("[")
    if brace < 0 and bracket < 0:
        return [("{", "}"), ("[", "]")]
    if brace < 0:
        return [("[", "]"), ("{", "}")]
    if bracket < 0:
        return [("{", "}"), ("[", "]")]
    if brace <= bracket:
        return [("{", "}"), ("[", "]")]
    return [("[", "]"), ("{", "}")]


def permissive_parse(raw: str) -> tuple[Any | None, PermissiveStrategy]:
    """Try increasingly relaxed JSON extraction strategies on ``raw``.

    The recovery ladder, applied in order:

    1. ``json.loads(raw)`` — when the model returned valid JSON directly,
       even if TensorZero's parser disagreed (shouldn't happen, but cheap
       to verify).  Returns ``(value, "strict")``.
    2. Balanced-brace scan for the first ``{…}`` or ``[…]`` span,
       preferring whichever opener appears earliest in ``raw`` — so a
       top-level array containing objects is recovered as the array.
       Within a chosen pair, each occurrence of the opener is tried in
       turn: a literal brace or bracket in prose that never closes is
       skipped, not fatal.  Returns ``(value, "permissive")``.
    3. Every fenced code block (`````json … `````
       or plain ````` … `````) in order
       — recovers Markdown-wrapped JSON even when an earlier reasoning
       fence does not parse.  Returns ``(value, "permissive")``.

    No exceptions escape: every ``json.JSONDecodeError`` along the way is
    silently swallowed and the next strategy is attempted.  When all
    strategies fail, returns ``(None, "none")``.
    """
    if raw:
        try:
            return json.loads(raw), "strict"
        except json.JSONDecodeError:
            pass

    for opener, closer in _ordered_pairs_for(raw):
        for span in _iter_balanced_spans(raw, opener, closer):
            try:
                return json.loads(span), "permissive"
            except json.JSONDecodeError:
                pass

    for fenced in _iter_fenced_code(raw):
        try:
            return json.loads(fenced), "permissive"
        except json.JSONDecodeError:
            pass
        for opener, closer in _ordered_pairs_for(fenced):
            for span in _iter_balanced_spans(fenced, opener, closer):
                try:
                    return json.loads(span), "permissive"
                except json.JSONDecodeError:
                    pass

    return None, "none"


async def run_structured_inference(
    t0: AsyncTensorZeroGateway,
    function_name: str,
    user_message: str,
    *,
    output_schema: Mapping[str, Any] | None = None,
    json_mode: Literal["off", "on", "strict", "tool"] | None = None,
    system_arguments: dict[str, Any] | None = None,
    prior_messages: list[Mapping[str, Any]] | None = None,
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
        prior_messages: Optional list of ``{role, content}`` message
            dicts to prepend before the ``user_message`` turn.  Used by
            :meth:`Agent._run_json`'s retry orchestration to thread the
            previous (invalid) assistant turn and the corrective hint
            back into the conversation.  ``None`` (the default) sends a
            single-turn ``[{"role": "user", "content": user_message}]``
            payload — byte-identical to the M2 shape.
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
    messages: list[Mapping[str, Any]] = []
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_message})
    input_dict: dict[str, Any] = {"messages": messages}
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
