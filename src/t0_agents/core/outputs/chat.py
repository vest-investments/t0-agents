"""ChatOutput — boundary assistant text from the MCP tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputDiagnostics,
    ParseFailure,
    Recovery,
)
from t0_agents.core.types import AgentResult

__all__ = ["ChatOutput"]


@dataclass(slots=True)
class ChatOutput:
    """The default :class:`~t0_agents.core.output.OutputMode`.

    Runs the MCP tool loop (:func:`t0_agents.mcp.loop.run_agent`) and
    materialises the boundary assistant text as ``AgentResult[str]``.

    ``shape_inference`` is a no-op — chat inferences use the function's
    static TensorZero configuration without per-call schema overrides.
    The :class:`Agent` dispatcher consults
    :attr:`expects_tool_loop` to route to the chat terminal, which
    calls ``run_agent`` directly; :meth:`materialize` is provided for
    protocol completeness and for callers that drive a single chat
    inference without the loop (rare).

    :meth:`recover` always returns :meth:`Recovery.fail` — chat does
    not have a parse-failure semantic; retry decisions for the tool
    loop are handled by ``mcp.run_agent`` itself.
    """

    expects_tool_loop: bool = True
    supports_streaming: bool = True
    requires_json_function: bool = False
    name: str = "chat"

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        """No-op: chat uses the function's static TZ configuration."""
        return req

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[str]:
        """Build ``AgentResult[str]`` from a ``ChatInferenceResponse``.

        Joins ``Text`` blocks with newlines; falls back to ``Thought``
        blocks (with ``response_source="thought"``) when no plain-text
        content is present.  Identical to the boundary-pair logic in
        :func:`t0_agents.mcp.loop.run_agent` so a caller using this
        method directly observes the same text as the tool-loop path.

        Not normally called by the dispatcher: chat agents go through
        ``run_agent`` which builds the :class:`AgentResult` directly.
        Provided for protocol uniformity and for users building custom
        single-shot chat terminals.
        """
        from tensorzero.types import (  # local import — keeps mcp/ standalone
            ChatInferenceResponse,
            Text,
            Thought,
        )

        if not isinstance(response, ChatInferenceResponse):
            raise TypeError(
                "ChatOutput.materialize expected ChatInferenceResponse, "
                f"got {type(response).__name__}"
            )

        text_parts: list[str] = []
        thought_parts: list[str] = []
        for block in response.content:
            if isinstance(block, Text) and block.text is not None:
                text_parts.append(block.text)
            elif isinstance(block, Thought) and block.text is not None:
                thought_parts.append(block.text)

        response_source: Literal["text", "thought"]
        if text_parts:
            text = "\n".join(text_parts)
            response_source = "text"
        elif thought_parts:
            text = "\n".join(thought_parts)
            response_source = "thought"
        else:
            text = ""
            response_source = "text"

        return AgentResult[str](
            value=text,
            episode_id=str(response.episode_id),
            inference_ids=[str(response.inference_id)],
            turns=1,
            response_source=response_source,
            diagnostics=OutputDiagnostics(
                parse_strategy="strict",
                attempts=len(attempts),
                raw_text=text or None,
            ),
        )

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        """Chat does not retry parse failures — always fail."""
        return Recovery.fail()
