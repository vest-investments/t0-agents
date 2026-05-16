"""ToolCallOutput — extract tool calls from a chat response without executing them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputDiagnostics,
    ParseFailure,
    Recovery,
)
from t0_agents.core.types import AgentResult

if TYPE_CHECKING:
    from tensorzero.types import ToolCall

__all__ = ["ToolCallOutput"]


@dataclass(slots=True)
class ToolCallOutput:
    """Router-pattern :class:`~t0_agents.core.output.OutputMode`.

    Issues a single chat-style inference and **returns the model's
    proposed tool calls without executing them**.  Used to delegate
    routing decisions to the model:

        ``output=ToolCallOutput(tool_choice="required")``

    The caller inspects ``result.value`` (a ``list[ToolCall]``) and
    dispatches manually.  This is intentionally *not* the MCP tool loop
    — :attr:`expects_tool_loop` is ``False`` so the single-shot terminal
    runs once and returns.

    ``tool_choice`` accepts the full :class:`tensorzero.ToolChoice`
    surface (``"auto"`` / ``"required"`` / ``"off"`` /
    ``{"specific": "<tool>"}``); defaults to ``"required"`` since the
    common router use case wants exactly one tool call.

    :meth:`recover` always returns :meth:`Recovery.fail` — router runs
    do not retry parse failures; the caller decides whether to re-run
    on an empty tool-call list.
    """

    tool_choice: Any = "required"

    expects_tool_loop: bool = False
    supports_streaming: bool = False
    name: str = "tool_call"

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        """Set ``req.tool_choice`` so the gateway honours the routing constraint."""
        req.tool_choice = self.tool_choice
        return req

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[list[ToolCall]]:
        """Extract :class:`ToolCall` blocks from a chat response.

        Any accompanying ``Text`` and ``Thought`` blocks (reasoning the
        model produced alongside the tool calls) are joined with
        newlines and stashed on ``diagnostics.raw_text`` — mirroring
        the streaming-parity convention used elsewhere — so callers
        debugging a router agent can still see the model's reasoning
        via ``result.response`` (the v0.12 read-side property), even
        though ``result.value`` is the structured tool-call list.
        """
        from tensorzero.types import (  # local import — keeps mcp/ standalone
            ChatInferenceResponse,
            Text,
            Thought,
            ToolCall,
        )

        if not isinstance(response, ChatInferenceResponse):
            raise TypeError(
                "ToolCallOutput.materialize expected ChatInferenceResponse, "
                f"got {type(response).__name__}"
            )

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in response.content:
            if isinstance(block, ToolCall):
                tool_calls.append(block)
            elif isinstance(block, (Text, Thought)) and block.text:
                text_parts.append(block.text)

        return AgentResult(
            value=tool_calls,
            episode_id=str(response.episode_id),
            inference_ids=[str(response.inference_id)],
            turns=1 + len(attempts),
            diagnostics=OutputDiagnostics(
                parse_strategy="strict" if tool_calls else "none",
                attempts=len(attempts),
                raw_text="\n".join(text_parts) if text_parts else None,
            ),
        )

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        """Router runs do not retry parse failures — always fail."""
        return Recovery.fail()
