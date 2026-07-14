"""JsonOutput — single-shot JSON inference with retry-and-recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputDiagnostics,
    ParseFailure,
    Recovery,
)
from t0_agents.core.structured import permissive_parse
from t0_agents.core.types import AgentResult

__all__ = ["JsonOutput"]


@dataclass(slots=True)
class JsonOutput:
    """Single-shot JSON-mode :class:`~t0_agents.core.output.OutputMode`.

    Replaces v0.12's ``AgentConfig(function_kind="json", json_mode=...,
    output_schema=..., permissive_json_parsing=..., max_json_retries=...)``
    cluster with one canonical surface:

        ``AgentConfig(output=JsonOutput(schema=S, json_mode="strict",
                                        permissive=True, max_retries=2))``

    Fields:
        schema: Optional dynamic schema override (per TensorZero guide
            §4).  Threaded onto :attr:`InferenceRequest.output_schema`
            so the gateway validates the model's output against it.
            ``None`` defers to the function's static configuration.
        json_mode: TensorZero ``json_mode`` (``"off"`` / ``"on"`` /
            ``"strict"`` / ``"tool"``).  Nested under
            ``params.chat_completion.json_mode``; takes precedence over
            the variant configuration.
        permissive: When ``True`` (default), the in-Python recovery
            ladder (:func:`permissive_parse`) runs after the gateway
            returns ``parsed == None``.  Set ``False`` to disable
            in-Python recovery (any TZ-side parse failure becomes a
            retry candidate).
        max_retries: Bound on the number of corrective retries before
            :meth:`recover` returns :meth:`Recovery.fail`.

    The retry-with-hint mechanism is split across two collaborators:

    - :meth:`materialize` reports the parse outcome on
      :attr:`AgentResult.diagnostics.parse_strategy`
      (``"strict"``/``"permissive"``/``"retry"``/``"none"``).
    - :meth:`recover` builds the corrective user-turn hint sent on
      retry, optionally enriching it with ``schema.title`` and the
      top-level ``required`` field list.  The hint logic lifted
      verbatim from v0.12 ``Agent._build_retry_hint`` so existing
      pattern-matching tests continue to assert against the same
      shape via :attr:`JsonOutput.RETRY_HINT_BASE`.

    :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`
    drives the loop: it calls the terminal, inspects
    ``result.diagnostics.parse_strategy``, and on ``"none"`` consults
    :meth:`recover` to retry or fail.
    """

    schema: Mapping[str, Any] | None = None
    json_mode: Literal["off", "on", "strict", "tool"] = "strict"
    permissive: bool = True
    max_retries: int = 2

    expects_tool_loop: bool = False
    supports_streaming: bool = False
    requires_json_function: bool = True
    name: str = "json"

    #: Fixed English base sentence used by :meth:`_build_retry_hint`.
    #: Stable across versions — tests pattern-match against this
    #: substring so the retry behaviour is observable from the outside.
    RETRY_HINT_BASE: ClassVar[str] = (
        "Your previous response was not valid JSON. "
        "Return ONLY a valid JSON object that matches the schema, "
        "with no commentary or prose."
    )

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        """Inject ``json_mode`` (under ``params.chat_completion``) and
        ``output_schema`` onto the request."""
        params = req.params if req.params is not None else {}
        chat = params.setdefault("chat_completion", {})
        chat["json_mode"] = self.json_mode
        req.params = params
        if self.schema is not None:
            req.output_schema = self.schema
        return req

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[dict[str, Any]]:
        """Read ``response.output.parsed``; fall back to permissive parsing.

        Recovery ladder:

        1. ``response.output.parsed`` non-None → ``parse_strategy``
           is ``"strict"`` (first attempt) or ``"retry"`` (after one or
           more failed prior attempts).
        2. ``self.permissive`` and step 1 returned ``None`` →
           :func:`permissive_parse` runs against ``response.output.raw``.
           Success → ``parse_strategy`` is ``"permissive"`` (first
           attempt) or ``"retry"`` (after prior failures).
        3. Both layers failed → ``parse_strategy="none"``,
           ``AgentResult.value`` is ``cast(..., None)`` (mypy-narrowed
           via the diagnostics field).  The retry middleware reads this
           and calls :meth:`recover`.
        """
        from tensorzero.types import (  # local import — keeps mcp/ standalone
            JsonInferenceResponse,
        )

        if not isinstance(response, JsonInferenceResponse):
            raise TypeError(
                "JsonOutput.materialize expected JsonInferenceResponse, "
                f"got {type(response).__name__}"
            )

        raw_text: str = response.output.raw or ""
        parsed: Any = response.output.parsed
        attempt_count = len(attempts)
        accumulated_ids: list[str] = [
            a.inference_id for a in attempts if a.inference_id
        ]
        accumulated_ids.append(str(response.inference_id))

        if parsed is not None:
            return AgentResult(
                value=cast(dict[str, Any], parsed),
                episode_id=str(response.episode_id),
                inference_ids=accumulated_ids,
                turns=1 + attempt_count,
                diagnostics=OutputDiagnostics(
                    parse_strategy="retry" if attempt_count > 0 else "strict",
                    attempts=attempt_count,
                    raw_text=raw_text,
                ),
            )

        if self.permissive:
            candidate, _strategy = permissive_parse(raw_text)
            if candidate is not None:
                return AgentResult(
                    value=cast(dict[str, Any], candidate),
                    episode_id=str(response.episode_id),
                    inference_ids=accumulated_ids,
                    turns=1 + attempt_count,
                    diagnostics=OutputDiagnostics(
                        parse_strategy=(
                            "retry" if attempt_count > 0 else "permissive"
                        ),
                        attempts=attempt_count,
                        raw_text=raw_text,
                    ),
                )

        return AgentResult(
            value=cast(dict[str, Any], None),
            episode_id=str(response.episode_id),
            inference_ids=accumulated_ids,
            turns=1 + attempt_count,
            diagnostics=OutputDiagnostics(
                parse_strategy="none",
                attempts=attempt_count,
                raw_text=raw_text,
            ),
        )

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        """Drive a retry with a schema-aware corrective hint, up to ``max_retries``.

        ``attempts`` includes the current failed attempt, so the
        predicate is ``len(attempts) > max_retries`` — matching v0.12's
        ``max_json_retries=N`` semantics (N retries allowed, N+1 total
        inferences).
        """
        if len(attempts) > self.max_retries:
            return Recovery.fail()
        return Recovery.retry(self._build_retry_hint())

    def _build_retry_hint(self) -> str:
        """Compose the corrective user-turn message sent on a JSON retry.

        Starts from :attr:`RETRY_HINT_BASE` (the cross-call invariant the
        retry tests can pattern-match on), then optionally enriches the
        hint with schema metadata when ``self.schema`` is non-trivial:

        - ``title`` — when the schema's top-level ``title`` is a
          non-empty string, the hint mentions it so the model has a name
          to anchor on.
        - ``required`` — when the top-level ``required`` array is a
          non-empty list of strings, the hint enumerates the required
          field names.

        Anything else (nested ``required``, ``properties``, ``oneOf``)
        is deliberately ignored: surfacing it would balloon prompt cost
        on every retry and risk crowding out the model's context.
        """
        parts: list[str] = [self.RETRY_HINT_BASE]
        if self.schema is None:
            return parts[0]
        title = self.schema.get("title")
        if isinstance(title, str) and title:
            parts.append(f"The schema is named '{title}'.")
        required = self.schema.get("required")
        if isinstance(required, list):
            names = [str(n) for n in required if isinstance(n, str)]
            if names:
                joined = ", ".join(repr(n) for n in names)
                parts.append(f"Required top-level fields: {joined}.")
        return " ".join(parts)
