"""Shared turn engine for the two agent loops.

:func:`t0_agents.mcp.loop.run_agent` (non-streaming) and
:class:`t0_agents.stream.runner.StreamingAgentLoop` (streaming) drive
the same conversation shape — validate inputs, seed messages, call
``t0.inference``, recover from recoverable gateway errors, append the
assistant turn, dispatch tools, terminate on a tool-free turn — and
historically duplicated that logic block for block.  The duplication
produced real bugs fixed twice (the thought-strip on replay, fixed
independently in each loop) and three behavioral divergences.  This
module is the single home for every piece of turn logic that does not
depend on how the inference response is acquired.

Deliberately NOT in the engine (the differences between the loops, plus
import-direction constraints — this module never imports ``mcp/`` or
``stream/``):

* the ``for turn in range(max_turns)`` skeleton and per-loop logging
  that names the entry point;
* inference acquisition — plain call + cast vs ``stream=True`` +
  ``TurnStreamAssembler`` + final-chunk tap;
* event yielding (streaming only);
* ``list_tools`` → ``convert_tools`` → executor pick (I/O plus
  ``mcp.adapter``/executor imports that must stay out of ``core/``);
* tool dispatch and the single-user-message append of results — two
  lines per loop; the shared error contract already lives in
  ``t0_agents.mcp.loop._execute_single_tool_call``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, NoReturn

from tensorzero import Message, Text, Thought

from t0_agents.core.exceptions import MaxTurnsExceededError, T0AgentsFatalError
from t0_agents.core.types import AgentResult

if TYPE_CHECKING:
    from tensorzero import ToolChoice
    from tensorzero.types import ChatInferenceResponse, TensorZeroError

logger = logging.getLogger(__name__)

# Patterns in TensorZeroError.text that indicate a recoverable inference
# failure.  When a match is found AND the status code is in
# ``RECOVERABLE_INFERENCE_STATUS_CODES``, the loop injects a corrective
# user message and retries instead of propagating the exception.  Extend
# either list to handle similar artifacts from other models / providers.
RECOVERABLE_INFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"not in request\.tools"),
]

# HTTP status codes that TensorZero may use when wrapping a provider-side
# validation failure.  ``400`` is the direct passthrough; ``502`` is what
# v2026.4.0 returns when an upstream provider (e.g. Groq) rejects the
# request with 400 — the gateway classifies the whole request as
# "upstream failed" and responds with a gateway-level Bad Gateway.
# ``503`` is intentionally excluded: Service Unavailable is a
# rate-limit / outage signal, not a format issue.
RECOVERABLE_INFERENCE_STATUS_CODES: frozenset[int] = frozenset({400, 502})


def strip_thoughts(content_blocks: list[Any]) -> list[Any]:
    """Return ``content_blocks`` without any ``Thought`` blocks.

    Applied to the assistant turn before it is appended to the replay
    ``messages``: TensorZero rejects input thought blocks for thought-less
    providers (e.g. groq) and drops them with a WARN, so replaying them is
    never useful. Mirrors the cross-conversation strip-on-replay rule in
    ``t0_agents.memory.tensorzero`` (``_to_native_message`` /
    ``_output_to_content_blocks``).
    """
    return [b for b in content_blocks if not isinstance(b, Thought)]


def validate_run_inputs(
    *,
    user_message: str,
    function_name: str | None,
    model_name: str | None,
    system: str | None,
    system_arguments: dict[str, Any] | None,
    entry_point: str,
) -> None:
    """Reject invalid run-input combinations before any I/O.

    Raises ``ValueError`` when both ``system`` and ``system_arguments``
    are set, when ``system_arguments`` is combined with ``model_name``
    (the functionless passthrough has no system template), or when
    ``user_message`` / the routing target are left at their empty
    defaults.  ``entry_point`` names the caller in the last message so
    each loop keeps its historical error text.
    """
    if system is not None and system_arguments is not None:
        raise ValueError(
            "Pass either `system` (raw string for input.system) or "
            "`system_arguments` (dict for template variables), not both."
        )
    if system_arguments is not None and model_name is not None:
        raise ValueError(
            "system_arguments requires a tensorzero.toml function with a "
            "system template; model_name routes via the functionless "
            "passthrough, which has none. Pass system= (a raw prompt "
            "string), or declare the function and use function_name=."
        )
    if not user_message or not (function_name or model_name):
        raise ValueError(
            f"`{entry_point}` requires non-empty `user_message` and a "
            "routing target: `function_name` or `model_name`. The `\"\"` "
            "defaults exist only as a Python-syntax workaround so "
            "`tool_provider` can default to `None` at position 2."
        )


def seed_messages(
    prior_messages: list[Message] | None, user_message: str
) -> list[Message]:
    """Seed the run's ``messages`` with prior history plus the new user turn."""
    messages: list[Message] = list(prior_messages) if prior_messages else []
    messages.append(Message(role="user", content=user_message))
    return messages


def build_inference_kwargs(
    *,
    function_name: str | None,
    model_name: str | None,
    variant_name: str | None,
    episode_id: str | None,
    messages: list[Message],
    t0_tools: list[dict[str, Any]],
    dryrun: bool,
    tool_choice: ToolChoice | None,
    tags: dict[str, str] | None,
    namespace: str | None,
    system: str | None,
    system_arguments: dict[str, Any] | None,
    params: dict[str, Any] | None,
    cache_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the ``t0.inference(...)`` kwargs shared by both loops.

    ``input.system`` carries ``system`` (raw string) or, failing that,
    ``system_arguments`` (template dict); neither key is present when
    both are ``None``.  Empty ``function_name`` coalesces to ``None`` so
    a ``model_name``-routed call omits the field.  The streaming loop
    adds ``stream=True`` on top.
    """
    input_dict: dict[str, Any] = {"messages": messages}
    if system is not None:
        input_dict["system"] = system
    elif system_arguments is not None:
        input_dict["system"] = system_arguments
    return {
        "function_name": function_name or None,
        "model_name": model_name,
        "variant_name": variant_name,
        "episode_id": episode_id,
        "input": input_dict,
        "additional_tools": t0_tools,
        "dryrun": dryrun,
        "tool_choice": tool_choice,
        "tags": tags,
        "namespace": namespace,
        "params": params,
        "cache_options": cache_options,
    }


def is_recoverable(err: TensorZeroError) -> bool:
    """``True`` when the gateway error warrants a corrective retry."""
    if err.status_code not in RECOVERABLE_INFERENCE_STATUS_CODES:
        return False
    if not err.text:
        return False
    return any(p.search(err.text) for p in RECOVERABLE_INFERENCE_PATTERNS)


def corrective_retry(
    err_text: str | None, t0_tools: list[dict[str, Any]], turn: int
) -> tuple[Message, Message, str]:
    """Build the corrective-retry trio for a recoverable inference error.

    Returns ``(empty assistant turn, corrective user message, sentinel
    inference id)``.  The empty assistant turn keeps the role alternation
    valid; the sentinel (``error-recoverable-turn-N``, 1-based) marks the
    retried turn in ``AgentResult.inference_ids``.  Logs the WARN both
    loops historically emitted at this point.
    """
    logger.warning(
        "Turn %d: recoverable TensorZeroError caught: %s", turn + 1, err_text
    )
    tool_names_available = [t["name"] for t in t0_tools]
    correction = (
        f"The previous response contained an invalid tool call "
        f"that was rejected: {err_text}\n"
        f"Available tools: {tool_names_available}. "
        f"Please retry with a valid tool name, stripping any "
        f"special tokens."
    )
    return (
        Message(role="assistant", content=[]),
        Message(role="user", content=correction),
        f"error-recoverable-turn-{turn + 1}",
    )


@dataclass
class TurnState:
    """Mutable per-run state threaded through the turn functions.

    ``episode_id`` starts as the caller-provided id (or ``None`` for a
    fresh episode) and is overwritten with the gateway-assigned id after
    every completed turn.
    """

    messages: list[Message]
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)


async def complete_assistant_turn(
    state: TurnState,
    *,
    content_blocks: list[Any],
    response: ChatInferenceResponse | None,
    on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None,
) -> None:
    """Post-inference sequence shared by both loops.

    Fires ``on_inference`` (BEFORE tool dispatch, so cost guardrails can
    block the next tool call), captures the episode / inference ids, and
    appends the thought-stripped assistant turn to the replay
    ``messages``.  ``response`` is ``None`` only on the streaming
    degenerate path where the stream ended without a usage chunk — the
    assistant turn is still appended, ids are not captured.

    Hook-before-capture is ``run_agent``'s historical order; the
    streaming loop appended the assistant turn before firing the hook.
    Unified on the former — the hook only reads ``response.usage``, so
    the order relative to state mutation is unobservable to it.
    """
    if response is not None:
        if on_inference is not None:
            await on_inference(response)
        state.episode_id = str(response.episode_id)
        state.inference_ids.append(str(response.inference_id))
    state.messages.append(
        Message(role="assistant", content=strip_thoughts(content_blocks))
    )


def raise_missing_tool_provider(*, entry_point: str) -> NoReturn:
    """Fail fatally when the LLM emitted tool calls but no provider exists.

    Call under the caller's own ``if tool_provider is None`` so mypy
    narrows the provider afterwards.  ``entry_point`` names the caller so
    each loop keeps its historical error text.  Raises
    ``T0AgentsFatalError`` — this must escape the loop rather than
    becoming a stringified tool result.
    """
    raise T0AgentsFatalError(
        "Agent loop received tool calls from the LLM but no "
        "`tool_provider` was configured. Pass a connected "
        f"`fastmcp.Client` to `{entry_point}(..., tool_provider=...)` "
        "if your agent is expected to dispatch tools."
    )


_JSON_WHITESPACE: Final[str] = " \t\n\r"
"""RFC 8259 insignificant whitespace — the same set ``json.loads`` skips."""

_TRAILING_NOISE: Final[str] = "}]" + _JSON_WHITESPACE
"""Every character :func:`salvage_tool_arguments` tolerates after the object."""

_DECODER: Final[json.JSONDecoder] = json.JSONDecoder()


class SalvagedArguments(NamedTuple):
    """Outcome of :func:`salvage_tool_arguments`.

    ``arguments`` is the recovered object; ``discarded`` counts the trailing
    characters dropped after it.  Callers log that count — never the payload.
    """

    arguments: dict[str, Any]
    discarded: int


def salvage_tool_arguments(raw: str) -> SalvagedArguments | None:
    """Recover tool-call arguments from a complete object followed by closing noise.

    Some providers emit a correct JSON object and then keep going — an extra
    ``}`` or ``]`` after the final brace is the dominant shape.  Strict parsing
    rejects the whole call and the model is re-prompted for arguments it
    already produced.  This helper accepts exactly that class and nothing
    wider:

    * the value must start at the first non-whitespace character and be a
      complete JSON **object** (``raw_decode`` succeeds with a ``dict``);
    * the remainder after it may contain only ``}``, ``]`` and JSON whitespace.

    Anything else returns ``None`` and the caller keeps its strict error path:
    a second value (``{"a": 1}{"b": 2}``, ``{"a": 1}, …``) is ambiguous about
    which arguments the model meant, prose is not arguments, and interior
    corruption (a missing comma, a truncated object) never reaches a
    successful ``raw_decode`` in the first place.  Syntactically valid JSON
    returns ``None`` as well: no parser rejected it, so a caller holding
    ``arguments=None`` for such a string was told so by schema validation
    (TensorZero zeroes arguments that fail the tool's parameter schema), and
    a schema verdict is not this helper's to overrule.  Deliberately narrower than
    :func:`~t0_agents.core.structured.permissive_parse`, which mines JSON out
    of prose and fences — the right tolerance for reading an answer, the wrong
    one for choosing the arguments of a call that has side effects.

    Shared by the streaming assembler and the dispatch executor so the two
    loops accept the same inputs — the parity this module exists for.
    """
    stripped = raw.lstrip(_JSON_WHITESPACE)
    try:
        value, end = _DECODER.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    remainder = stripped[end:]
    if not remainder.strip(_JSON_WHITESPACE) or remainder.strip(_TRAILING_NOISE):
        return None
    return SalvagedArguments(value, len(remainder))


def mine_final_text(blocks: list[Any]) -> tuple[str, Literal["text", "thought"]]:
    """Join ``Text`` blocks; fall back to mining thought content.

    When no ``Text`` block carries content, the turn's ``Thought``
    blocks are mined so a thought-only final turn never yields an empty
    response.  Reasoning models surface the answer via either
    ``Thought.text`` (raw) or ``Thought.summary[*].text`` (provider
    summary, e.g. gpt-oss harmony's analysis channel); both are mined.
    The source is flagged ``"thought"`` so consumers branch on
    provenance, not on model name.
    """
    text = "\n".join(b.text for b in blocks if isinstance(b, Text) and b.text)
    if text:
        return text, "text"
    thought_texts: list[str] = []
    for b in blocks:
        if not isinstance(b, Thought):
            continue
        if b.text:
            thought_texts.append(b.text)
        if b.summary:
            thought_texts.extend(s.text for s in b.summary if s.text)
    if thought_texts:
        return "\n".join(thought_texts), "thought"
    return "", "text"


def build_agent_result(
    text: str,
    source: Literal["text", "thought"],
    *,
    state: TurnState,
    turns: int,
) -> AgentResult[str]:
    """Materialise the run's ``AgentResult`` from the final turn."""
    return AgentResult(
        value=text,
        episode_id=state.episode_id,
        inference_ids=list(state.inference_ids),
        turns=turns,
        response_source=source,
    )


def raise_max_turns(max_turns: int, episode_id: str | None) -> NoReturn:
    """Warn and raise once the loop exhausts ``max_turns`` without terminating."""
    logger.warning(
        "Agent loop exceeded max_turns=%d, episode_id=%s", max_turns, episode_id
    )
    raise MaxTurnsExceededError(max_turns)
