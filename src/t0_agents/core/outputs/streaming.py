"""StreamingOutput — placeholder for future streaming JSON support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputMode,
    ParseFailure,
    Recovery,
)
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.types import AgentResult

__all__ = ["StreamingOutput"]


@dataclass
class StreamingOutput:
    """Streaming-capable :class:`OutputMode` wrapper.

    Wraps a ``base`` mode and forwards every protocol method.  In v1.0
    only ``base=ChatOutput()`` is supported — TensorZero does not yet
    emit ``output.parsed`` mid-stream, so ``StreamingOutput(base=
    JsonOutput(...))`` would silently degrade to a non-streaming
    pathway.  The constructor refuses bases whose
    :attr:`supports_streaming` is ``False`` with a discoverable
    :class:`ConfigurationError` (open question Q11 — deferred to a
    post-v1.0 milestone once TZ ships streaming JSON).

    ``StreamingOutput(base=ChatOutput())`` is functionally equivalent
    to plain ``ChatOutput()`` — the wrapper is intentional documentation
    that the user is opting into streaming explicitly.
    """

    base: OutputMode[Any] = field(default_factory=ChatOutput)
    expects_tool_loop: bool = field(init=False)
    supports_streaming: bool = field(default=True, init=False)
    name: str = field(default="streaming", init=False)

    def __post_init__(self) -> None:
        if not getattr(self.base, "supports_streaming", False):
            raise ConfigurationError(
                f"StreamingOutput(base={getattr(self.base, 'name', type(self.base).__name__)!r}) "
                "is not yet supported.  Only base=ChatOutput() works in v1.0; "
                "streaming JSON awaits TZ upstream (open question Q11)."
            )
        # Inherit the base's tool-loop flag — chat streams through the
        # tool loop, future modes (e.g. streaming JSON) may not.
        self.expects_tool_loop = self.base.expects_tool_loop

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest:
        return self.base.shape_inference(req)

    async def materialize(
        self, response: Any, attempts: list[Attempt]
    ) -> AgentResult[Any]:
        return await self.base.materialize(response, attempts)

    async def recover(
        self, failure: ParseFailure, attempts: list[Attempt]
    ) -> Recovery:
        return await self.base.recover(failure, attempts)
