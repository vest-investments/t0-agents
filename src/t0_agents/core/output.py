"""Output polymorphism — the :class:`OutputMode` protocol and helpers.

v1.0 collapses the chat/JSON dichotomy that v0.12 hardcoded in
``AgentConfig.function_kind`` into a single polymorphic surface:

- :class:`OutputMode` — protocol with three methods (``shape_inference``,
  ``materialize``, ``recover``) and two capability flags
  (``expects_tool_loop``, ``supports_streaming``).
- :class:`InferenceRequest` — mutable dataclass passed through
  ``shape_inference`` so each :class:`OutputMode` contributes its own
  TensorZero kwargs (``json_mode``, ``output_schema``, ``tool_choice``).
- :class:`OutputDiagnostics` — per-result observability payload that
  :attr:`AgentResult.diagnostics` exposes (parse strategy, attempt
  count, raw text).
- :class:`Attempt` / :class:`ParseFailure` / :class:`Recovery` — the
  retry handshake :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`
  uses to drive :meth:`OutputMode.recover`.

Built-in implementations live under ``core/outputs/`` — five modes ship
in v1.0 (:class:`~t0_agents.core.outputs.ChatOutput`,
:class:`~t0_agents.core.outputs.JsonOutput`,
:class:`~t0_agents.core.outputs.StructuredOutput`,
:class:`~t0_agents.core.outputs.ToolCallOutput`,
:class:`~t0_agents.core.outputs.StreamingOutput`).  Third parties can
plug in custom modes (YAML extraction, Instructor-style retry,
SQLAlchemy row materialisation) without forking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from t0_agents.core.structured import ParseStrategy

if TYPE_CHECKING:
    from t0_agents.core.types import AgentResult


__all__ = [
    "Attempt",
    "InferenceRequest",
    "OutputDiagnostics",
    "OutputMode",
    "ParseFailure",
    "ParseStrategy",
    "Recovery",
]


T_co = TypeVar("T_co")


@dataclass
class InferenceRequest:
    """Mutable request descriptor threaded through :meth:`OutputMode.shape_inference`.

    Built by the single-shot terminal from a :class:`RunContext` and
    mutated in-place by every :class:`OutputMode` that wraps non-chat
    behaviour.  The terminal then projects the fields onto
    :meth:`AsyncTensorZeroGateway.inference` kwargs.

    ``params`` follows the TensorZero shape:
    ``{"chat_completion": {"json_mode": "strict"}}`` — set by
    :meth:`JsonOutput.shape_inference`.  ``output_schema`` likewise maps
    onto the top-level TZ kwarg of the same name.  ``tool_choice``
    accepts :class:`tensorzero.ToolChoice` values (``"auto"``,
    ``"required"``, ``"off"``, or ``{"specific": "<tool>"}``).
    """

    function_name: str
    input: dict[str, Any]
    additional_tools: list[Any] | None = None
    tool_choice: Any | None = None
    tags: dict[str, str] | None = None
    namespace: str | None = None
    dryrun: bool = False
    output_schema: Mapping[str, Any] | None = None
    params: dict[str, Any] | None = None


@dataclass(slots=True)
class Attempt:
    """One past inference attempt as seen by :meth:`OutputMode.recover`.

    Populated by :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`
    after each failed materialisation.  Passed as the ``attempts`` list
    so the recovery policy can implement budget-based or content-aware
    decisions (e.g. give up after 2 attempts, or accept a partial parse
    on the third).
    """

    inference_id: str
    raw_text: str | None
    parsed: Any | None
    usage: Any | None


@dataclass(slots=True)
class ParseFailure:
    """Failure description handed to :meth:`OutputMode.recover`.

    Built by :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`
    when :meth:`OutputMode.materialize` produced an
    :class:`AgentResult` whose ``diagnostics.parse_strategy == "none"``.
    Carries the raw assistant text so the recovery policy can build a
    schema-aware corrective hint or accept a manually-coerced value.

    ``reason`` is a stable short tag (``"parse_failed"``,
    ``"validation_error"``, ...); ``detail`` carries the original
    exception message or the validator's field-path list.
    """

    raw_text: str
    inference_id: str
    reason: str
    detail: str


@dataclass(slots=True)
class Recovery:
    """Outcome of :meth:`OutputMode.recover` — retry, accept, or fail.

    Use the classmethods to construct: :meth:`Recovery.retry` to drive
    the inner middleware to re-emit inference with a corrective user-turn
    hint; :meth:`Recovery.accept` to bypass parsing and take a manually
    coerced value; :meth:`Recovery.fail` to surface the failure to the
    caller (the middleware returns the failed :class:`AgentResult` with
    ``parse_strategy == "none"`` so v0.12 byte-identity is preserved).
    """

    kind: Literal["retry", "accept", "fail"]
    hint: str | None = None
    value: Any = None

    @classmethod
    def retry(cls, hint: str) -> Recovery:
        """Drive a retry with the supplied corrective user-turn message."""
        return cls(kind="retry", hint=hint)

    @classmethod
    def accept(cls, value: Any) -> Recovery:
        """Bypass parsing and use ``value`` as the final result."""
        return cls(kind="accept", value=value)

    @classmethod
    def fail(cls) -> Recovery:
        """Surface the failure to the caller — no retry, no override."""
        return cls(kind="fail")


@dataclass(slots=True)
class OutputDiagnostics:
    """Per-result observability payload attached to :attr:`AgentResult.diagnostics`.

    Carries the JSON-path observability fields v0.12 exposed inline on
    :class:`AgentResult`:

    - ``parse_strategy`` — which layer in the recovery ladder produced
      the parsed value (``"strict"`` / ``"permissive"`` / ``"retry"`` /
      ``"none"``).
    - ``attempts`` — number of corrective retry attempts (≥ 0).
    - ``raw_text`` — the model's last raw response (helpful for
      debugging permissive recoveries and for echoing the original text
      via ``AgentResult.response``).
    - ``failure_detail`` — non-recovery context attached when an
      :class:`OutputMode` reports ``parse_strategy="none"``.  Plain
      JSON parse failures leave this ``None``;
      :class:`~t0_agents.core.outputs.StructuredOutput` populates it
      with the Pydantic ``ValidationError`` summary so
      :meth:`StructuredOutput.recover` can build a field-aware
      corrective hint.

    Chat-mode results leave ``diagnostics=None``; the v0.12 property
    aliases on :class:`AgentResult` fall back to defaults in that case.
    """

    parse_strategy: ParseStrategy = "strict"
    attempts: int = 0
    raw_text: str | None = None
    failure_detail: str | None = None


@runtime_checkable
class OutputMode(Protocol[T_co]):
    """Polymorphic output strategy for an :class:`Agent` run.

    Three methods own the lifecycle of a non-chat inference:

    - :meth:`shape_inference` mutates an :class:`InferenceRequest` so
      TensorZero receives the kwargs this mode needs (``json_mode``,
      ``output_schema``, ``tool_choice``).
    - :meth:`materialize` turns the raw inference response into an
      :class:`AgentResult` (typed ``T_co``).
    - :meth:`recover` decides what to do when :meth:`materialize`
      reported a parse failure (``parse_strategy == "none"``): retry
      with a hint, accept a coerced value, or surface the failure.

    Two capability flags drive the dispatcher:

    - ``expects_tool_loop`` — ``True`` for :class:`ChatOutput` (the
      MCP tool loop runs); ``False`` for single-shot modes.
    - ``supports_streaming`` — ``True`` if ``run.events()`` /
      ``agent.stream()`` are legal; ``False`` raises
      :class:`ConfigurationError` early when streaming is attempted.

    ``name`` is a short identifier used in error messages and
    diagnostics.

    Custom modes only need to satisfy the protocol — no base class
    required.
    """

    expects_tool_loop: bool
    supports_streaming: bool
    name: str

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        """Mutate ``req`` (or return a new one) to encode this mode's TZ kwargs."""
        ...

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[T_co]:
        """Build an :class:`AgentResult` from a TensorZero inference response.

        On parse failure return an :class:`AgentResult` whose
        ``diagnostics.parse_strategy == "none"`` so the retry middleware
        can ask :meth:`recover` what to do next.  Do *not* raise — the
        middleware drives the recovery loop.
        """
        ...

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        """Decide how to handle a parse failure: retry, accept, or fail."""
        ...


