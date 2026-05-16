"""StructuredOutput — Pydantic-validated single-shot JSON inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputDiagnostics,
    ParseFailure,
    Recovery,
)
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.structured import permissive_parse
from t0_agents.core.types import AgentResult

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["StructuredOutput"]


M = TypeVar("M", bound="BaseModel")


def _require_pydantic() -> Any:
    """Lazy import of :mod:`pydantic` — raises a discoverable error when missing.

    :class:`StructuredOutput` is the only built-in mode that depends on
    Pydantic.  We defer the import until first construction so installs
    that never use structured outputs (chat, plain JSON, tool-call) don't
    pay the cost.
    """
    try:
        import pydantic
    except ImportError as exc:
        raise ConfigurationError(
            "StructuredOutput requires pydantic>=2.0.  "
            "Install with: pip install 't0-agents[pydantic]'."
        ) from exc
    return pydantic


@dataclass
class StructuredOutput(Generic[M]):
    """Pydantic-validated :class:`~t0_agents.core.output.OutputMode`.

    Derives the JSON schema from ``model.model_json_schema()`` so the
    TensorZero gateway validates the raw response against it, and then
    materialises the parsed dict into an instance of ``model`` via
    ``model.model_validate(...)``.  ``AgentResult.value`` is statically
    typed as ``M`` — so a caller using
    ``output=StructuredOutput(Customer)`` gets
    ``result.value: Customer`` mypy-verified (per-call overload on
    :meth:`Agent.invoke`).

    Fields:
        model: A Pydantic v2 ``BaseModel`` subclass.  The class itself
            (not an instance) — :meth:`StructuredOutput.materialize`
            calls ``model.model_validate(parsed_dict)`` to build the
            value.
        json_mode: TensorZero ``json_mode`` (defaults to ``"strict"``).
        permissive: When ``True``, falls back to
            :func:`permissive_parse` if TZ returns ``parsed == None``.
            ``False`` keeps a stricter contract — any TZ-side parse
            failure goes straight to the retry loop.
        max_retries: Bound on corrective retries.  Reused for *both*
            JSON parse failures and Pydantic validation failures.

    The recovery hint includes the Pydantic
    ``ValidationError.errors()`` summary so the model receives concrete
    field-level feedback (``"field 'email' missing"``,
    ``"field 'age' must be int, got str"``).
    """

    model: type[M]
    json_mode: Literal["off", "on", "strict", "tool"] = "strict"
    permissive: bool = True
    max_retries: int = 2

    expects_tool_loop: bool = field(default=False, init=False)
    supports_streaming: bool = field(default=False, init=False)
    #: Strict-tightened JSON schema derived from ``model`` at
    #: construction time.  ``model_json_schema()`` is expensive on
    #: complex models, and both :meth:`shape_inference` (called once
    #: per attempt in the retry loop) and :meth:`recover` (called on
    #: parse failure) need the schema — caching here keeps the work
    #: O(1) regardless of how many retries fire.
    _strict_schema: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        pydantic = _require_pydantic()
        base_model = pydantic.BaseModel
        if not (isinstance(self.model, type) and issubclass(self.model, base_model)):
            raise ConfigurationError(
                "StructuredOutput.model must be a pydantic.BaseModel "
                f"subclass, got {self.model!r}."
            )
        self._strict_schema = _enforce_strict_object_schema(
            self.model.model_json_schema()
        )

    @property
    def name(self) -> str:
        """Human-readable identifier — ``structured/<ModelName>``."""
        return f"structured/{self.model.__name__}"

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        """Derive the schema from the Pydantic model and apply JSON-mode kwargs.

        The Pydantic-emitted schema is tightened before send-off so
        strict-JSON providers (Groq, OpenAI) accept it: every object
        gets ``additionalProperties: false`` and ``required`` is
        expanded to cover every key in ``properties``.  Pydantic v2
        emits neither by default.
        """
        params = req.params if req.params is not None else {}
        chat = params.setdefault("chat_completion", {})
        chat["json_mode"] = self.json_mode
        req.params = params
        req.output_schema = self._strict_schema
        return req

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[M]:
        """Parse the response and ``model.model_validate`` the result.

        Two failure modes both surface as ``parse_strategy="none"``:

        - The response was unparseable JSON even after permissive
          recovery.  ``failure_detail`` is left ``None``;
          :meth:`recover` issues the generic JSON hint.
        - The response parsed but failed Pydantic validation.
          ``failure_detail`` carries the ``ValidationError.errors()``
          summary so :meth:`recover` can name the offending fields.
        """
        from tensorzero.types import (  # local import — keeps mcp/ standalone
            JsonInferenceResponse,
        )

        pydantic = _require_pydantic()

        if not isinstance(response, JsonInferenceResponse):
            raise TypeError(
                "StructuredOutput.materialize expected JsonInferenceResponse, "
                f"got {type(response).__name__}"
            )

        raw_text: str = response.output.raw or ""
        parsed: Any = response.output.parsed
        attempt_count = len(attempts)
        used_permissive = False
        accumulated_ids: list[str] = [
            a.inference_id for a in attempts if a.inference_id
        ]
        accumulated_ids.append(str(response.inference_id))

        if parsed is None and self.permissive:
            candidate, _strategy = permissive_parse(raw_text)
            if candidate is not None:
                parsed = candidate
                used_permissive = True

        if parsed is None:
            return AgentResult(
                value=cast(M, None),
                episode_id=str(response.episode_id),
                inference_ids=accumulated_ids,
                turns=1 + attempt_count,
                diagnostics=OutputDiagnostics(
                    parse_strategy="none",
                    attempts=attempt_count,
                    raw_text=raw_text,
                ),
            )

        try:
            instance = self.model.model_validate(parsed)
        except pydantic.ValidationError as exc:
            return AgentResult(
                value=cast(M, None),
                episode_id=str(response.episode_id),
                inference_ids=accumulated_ids,
                turns=1 + attempt_count,
                diagnostics=OutputDiagnostics(
                    parse_strategy="none",
                    attempts=attempt_count,
                    raw_text=raw_text,
                    failure_detail=_summarise_validation_error(exc),
                ),
            )

        # Mirror JsonOutput's three-way precedence: a retry beats
        # permissive (because retry implies prior failures), and
        # permissive beats strict (because strict means TZ already
        # delivered a clean parse).
        if attempt_count > 0:
            strategy: Literal["strict", "permissive", "retry"] = "retry"
        elif used_permissive:
            strategy = "permissive"
        else:
            strategy = "strict"
        return AgentResult(
            value=instance,
            episode_id=str(response.episode_id),
            inference_ids=accumulated_ids,
            turns=1 + attempt_count,
            diagnostics=OutputDiagnostics(
                parse_strategy=strategy,
                attempts=attempt_count,
                raw_text=raw_text,
            ),
        )

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        """Drive a field-aware retry; fall back to JSON-shape hint when no fields.

        ``attempts`` includes the current failed attempt, so the
        predicate is ``len(attempts) > max_retries`` — matching v0.12's
        ``max_json_retries=N`` semantics (N retries allowed, N+1 total
        inferences).
        """
        if len(attempts) > self.max_retries:
            return Recovery.fail()
        if failure.detail and "validation" in failure.reason:
            return Recovery.retry(
                f"Validation failed: {failure.detail}.  "
                "Re-emit the JSON conforming to the schema."
            )
        # Plain JSON parse failure — reuse JsonOutput's hint shape so the
        # model sees the same base sentence regardless of which mode
        # surfaced the failure.
        hint_helper = JsonOutput(schema=self._strict_schema, json_mode=self.json_mode)
        return Recovery.retry(hint_helper._build_retry_hint())


def _enforce_strict_object_schema(schema: Any) -> Any:
    """Tighten every object node so strict-JSON providers accept it.

    Pydantic v2's ``model_json_schema()`` emits a permissive shape:
    ``additionalProperties`` is unset (treated as ``true``) and
    ``required`` lists only fields without defaults.  Strict-JSON
    providers (Groq, OpenAI) reject both, surfacing as ``HTTP 400`` with
    ``invalid JSON schema for response_format``.  This helper walks the
    schema (including ``$defs``/``definitions`` blocks and
    ``items``/``anyOf``/``oneOf`` subtrees) and on every object node:

    1. Sets ``additionalProperties: false`` if absent.
    2. Sets ``required`` to the full list of ``properties`` keys (in
       declaration order) if ``properties`` is present.

    Mutates the dict in place and returns it for chaining convenience.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False
            properties = schema.get("properties")
            if isinstance(properties, dict) and properties:
                schema["required"] = list(properties.keys())
        for value in schema.values():
            _enforce_strict_object_schema(value)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_strict_object_schema(item)
    return schema


def _summarise_validation_error(exc: Any) -> str:
    """Compact one-line summary of a Pydantic ``ValidationError``.

    Lists up to the first three offending field paths and their
    error types (e.g. ``"email: missing; age: int_type"``).  Trimmed so
    the hint fits in a single user turn without blowing up prompt cost
    on schemas with dozens of fields.
    """
    errors = exc.errors() if callable(getattr(exc, "errors", None)) else []
    parts: list[str] = []
    for entry in errors[:3]:
        loc = entry.get("loc") if isinstance(entry, dict) else None
        err_type = entry.get("type") if isinstance(entry, dict) else None
        if loc and isinstance(loc, (list, tuple)):
            path = ".".join(str(p) for p in loc)
        else:
            path = "<root>"
        parts.append(f"{path}: {err_type or 'invalid'}")
    if not parts:
        return str(exc)
    if len(errors) > 3:
        parts.append(f"...and {len(errors) - 3} more")
    return "; ".join(parts)
