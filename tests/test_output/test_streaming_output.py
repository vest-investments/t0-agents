"""Unit tests for :class:`~t0_agents.core.outputs.StreamingOutput`."""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import (
    ChatOutput,
    JsonOutput,
    StreamingOutput,
)


class TestConstruction:
    def test_streaming_chat_works(self) -> None:
        out = StreamingOutput(base=ChatOutput())
        assert out.expects_tool_loop is True
        assert out.supports_streaming is True
        assert out.name == "streaming"

    def test_streaming_json_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not yet supported"):
            StreamingOutput(base=JsonOutput(json_mode="strict"))

    def test_default_base_is_chat(self) -> None:
        out = StreamingOutput()
        assert out.expects_tool_loop is True
        assert out.supports_streaming is True


class TestDelegation:
    """The wrapper delegates every protocol method to ``self.base``."""

    def test_shape_inference_delegates(self) -> None:
        from t0_agents.core.output import InferenceRequest

        base = ChatOutput()
        out = StreamingOutput(base=base)
        req = InferenceRequest(
            function_name="agent", input={"messages": []}
        )
        result = out.shape_inference(req)
        # ChatOutput.shape_inference is a no-op — same request returns.
        assert result is req
