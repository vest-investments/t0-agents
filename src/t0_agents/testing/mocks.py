"""Deterministic mocks for the TensorZero + FastMCP surface the agent uses.

**Test-only.**  Never import or construct any symbol from this module in a
production code path.  ``MockGateway`` and ``MockToolProvider`` bypass real
inference and tool dispatch respectively — they exist to unit-test code
that exercises :class:`~t0_agents.core.agent.Agent`,
:func:`~t0_agents.mcp.loop.run_agent`, or anything else that depends on
the TensorZero / FastMCP surface.

The mocks duck-type the real classes (not subclass) so they stay
obviously replaceable and can be injected anywhere a real
``AsyncTensorZeroGateway`` or a ``fastmcp.Client`` is expected.  A
shape-contract test in ``tests/test_testing/test_mock_shape.py`` asserts
that the mocks accept every kwarg the real loop passes, so drift is
caught early.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast
from uuid import UUID, uuid4

from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool
from tensorzero.types import (
    ChatInferenceResponse,
    ContentBlock,
    Text,
    Usage,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Return type of a :class:`MockGateway` ``source`` callable.  A source may
#: return either a list of :class:`~tensorzero.types.ContentBlock` (wrapped
#: into a fresh :class:`ChatInferenceResponse` by the gateway) or a fully
#: built response when the caller needs to pin inference/episode IDs.
_SourceReturn: TypeAlias = list[ContentBlock] | ChatInferenceResponse

#: Handler registered in :class:`MockToolProvider`.  Receives the tool
#: arguments as a dict and returns any JSON-serialisable payload (strings
#: pass through; everything else is ``json.dumps``'d by
#: :func:`~t0_agents.mcp.adapter.serialize_tool_result`).
ToolHandler: TypeAlias = (
    Callable[[dict[str, Any]], Any]
    | Callable[[dict[str, Any]], Awaitable[Any]]
)


# ---------------------------------------------------------------------------
# MockChatResponse
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MockChatResponse:
    """Builder for a single :class:`ChatInferenceResponse`-shaped object.

    Keeps the content block list + overrides explicit so tests can refer
    back to the original inputs when debugging a failure.  Use
    :meth:`to_response` to materialise a response with deterministic or
    caller-supplied IDs.
    """

    content: list[ContentBlock]
    usage: Usage | None = None
    variant_name: str = "mock_variant"
    finish_reason: Any = None

    def to_response(
        self,
        *,
        inference_id: UUID | None = None,
        episode_id: UUID | None = None,
    ) -> ChatInferenceResponse:
        """Materialise a :class:`ChatInferenceResponse` from the builder."""
        return ChatInferenceResponse(
            inference_id=inference_id or uuid4(),
            episode_id=episode_id or uuid4(),
            variant_name=self.variant_name,
            content=list(self.content),
            usage=self.usage or Usage(input_tokens=0, output_tokens=0),
            finish_reason=self.finish_reason,
        )


# ---------------------------------------------------------------------------
# MockGateway
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MockFeedbackResponse:
    """Duck-type of :class:`tensorzero.types.FeedbackResponse`."""

    feedback_id: UUID


@dataclass(slots=True)
class _RecordedInference:
    """One captured ``(kwargs, response)`` pair from :class:`RecordingGateway`."""

    kwargs: dict[str, Any]
    response: ChatInferenceResponse


class MockGateway:
    """A duck-typed :class:`~tensorzero.AsyncTensorZeroGateway` for unit tests.

    Three construction patterns:

    * :meth:`script` — list of per-turn content blocks; each
      ``inference()`` call pops the next entry.  When the script is
      exhausted, the gateway returns an empty :class:`~tensorzero.Text`
      turn so the agent loop terminates cleanly (avoids hanging tests).
    * :meth:`canned` — callable receiving the inference kwargs and
      returning either a content list or a fully built response.
      Behaviour-driven; ideal when a test wants to branch on the
      assistant's previous message or tool results.
    * :meth:`recording` — wrap a real gateway, forward ``inference()``
      through it, and capture ``(kwargs, response)`` pairs.  Call
      :meth:`RecordingGateway.replay` afterwards to turn the capture
      into a deterministic :class:`MockGateway`.

    The gateway records every call on :attr:`calls` (and
    :attr:`feedback_calls` for the feedback path) so tests can assert
    on the kwargs the loop passed — notably ``additional_tools``,
    ``tool_choice``, ``tags``, and ``namespace``.
    """

    def __init__(
        self,
        source: Callable[..., _SourceReturn],
        *,
        episode_id: UUID | None = None,
    ) -> None:
        self._source = source
        # MockGateway keeps the same episode id for every turn of a run
        # so ``Agent.run(episode_id=None)`` sees a stable id across turns,
        # matching the real gateway's contract (episode_id first allocated
        # on inference #1, reused thereafter).
        self._sticky_episode_id: UUID | None = episode_id
        self.calls: list[dict[str, Any]] = []
        # Parallel to ``calls``: the ``ChatInferenceResponse`` the gateway
        # returned for each entry.  Enables assertions on what the mock
        # *model* emitted (tool calls, text, etc.) without requiring
        # callers to replay the script themselves.
        self.responses: list[ChatInferenceResponse] = []
        self.feedback_calls: list[dict[str, Any]] = []

    # --- Factory methods ---------------------------------------------------

    @classmethod
    def script(
        cls,
        turns: list[list[ContentBlock]],
        *,
        episode_id: UUID | None = None,
    ) -> MockGateway:
        """Queue-driven responses, one content list per turn."""
        queue: deque[list[ContentBlock]] = deque(turns)

        def source(**_: Any) -> _SourceReturn:
            if queue:
                return queue.popleft()
            return [Text(text="")]

        return cls(source, episode_id=episode_id)

    @classmethod
    def canned(
        cls,
        fn: Callable[..., _SourceReturn],
        *,
        episode_id: UUID | None = None,
    ) -> MockGateway:
        """Callable-driven responses; receives full inference kwargs."""
        return cls(fn, episode_id=episode_id)

    @classmethod
    def recording(cls, real_gateway: Any) -> RecordingGateway:
        """Wrap a real gateway to capture ``(kwargs, response)`` pairs.

        Returns a :class:`RecordingGateway` (itself a gateway-shaped
        object).  After the run, call :meth:`RecordingGateway.replay`
        to convert the capture into a deterministic
        :class:`MockGateway`.
        """
        return RecordingGateway(real_gateway)

    # --- Gateway surface (duck-typed) --------------------------------------

    async def inference(self, **kwargs: Any) -> ChatInferenceResponse:
        """Record the call and return the next scripted response."""
        self.calls.append(dict(kwargs))
        result = self._source(**kwargs)
        if isinstance(result, ChatInferenceResponse):
            response: ChatInferenceResponse = result
        else:
            # The source returned a content list — wrap it with a stable episode id.
            if self._sticky_episode_id is None:
                self._sticky_episode_id = uuid4()
            response = MockChatResponse(list(result)).to_response(
                inference_id=uuid4(),
                episode_id=self._sticky_episode_id,
            )
        self.responses.append(response)
        return response

    async def feedback(self, **kwargs: Any) -> _MockFeedbackResponse:
        """Record a feedback call and return an ephemeral response."""
        self.feedback_calls.append(dict(kwargs))
        return _MockFeedbackResponse(feedback_id=uuid4())


class RecordingGateway:
    """Capture :meth:`inference` calls against a real gateway for later replay.

    Use via :meth:`MockGateway.recording`.  Forwards every call to the
    wrapped gateway, stores the kwargs + response, and exposes
    :meth:`replay` to snap a deterministic :class:`MockGateway` from
    the capture.  Handy for converting a flaky integration test into a
    fast unit test without hand-writing the expected content blocks.
    """

    def __init__(self, real_gateway: Any) -> None:
        self._real = real_gateway
        self._captured: list[_RecordedInference] = []
        self.calls: list[dict[str, Any]] = []
        self.responses: list[ChatInferenceResponse] = []
        self.feedback_calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> ChatInferenceResponse:
        self.calls.append(dict(kwargs))
        raw = await self._real.inference(**kwargs)
        response = cast(ChatInferenceResponse, raw)
        self._captured.append(
            _RecordedInference(kwargs=dict(kwargs), response=response)
        )
        self.responses.append(response)
        return response

    async def feedback(self, **kwargs: Any) -> Any:
        self.feedback_calls.append(dict(kwargs))
        return await self._real.feedback(**kwargs)

    @property
    def captured(self) -> list[_RecordedInference]:
        """Read-only view of the captured inference pairs."""
        return list(self._captured)

    def replay(self) -> MockGateway:
        """Build a deterministic :class:`MockGateway` from the capture."""
        queue: deque[ChatInferenceResponse] = deque(
            rec.response for rec in self._captured
        )

        def source(**_: Any) -> _SourceReturn:
            if queue:
                return queue.popleft()
            return [Text(text="")]

        return MockGateway(source)


# ---------------------------------------------------------------------------
# MockToolProvider
# ---------------------------------------------------------------------------


class MockToolProvider:
    """A dict-backed :class:`~t0_agents.core.types.ToolProvider` for unit tests.

    Keys are tool names; values are sync or async callables receiving a
    ``dict[str, Any]`` arguments payload.  The return value is attached
    to a real :class:`fastmcp.client.client.CallToolResult`'s ``data``
    field so the production serializer
    (:func:`t0_agents.mcp.adapter.serialize_tool_result`) produces the
    same string a real tool would — strings pass through, everything
    else is ``json.dumps``'d with ``default=str``.

    Raise a :class:`ToolError` or any other exception from the handler
    to exercise the loop's error contract — the loop stringifies the
    exception into an ``"ERROR: ..."`` tool result the LLM can read on
    the next turn.
    """

    def __init__(
        self,
        tools: dict[str, ToolHandler],
        *,
        schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._tools = dict(tools)
        self._schemas = dict(schemas or {})
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools.

        Real ``mcp.types.Tool`` instances are yielded — not fabricated
        anonymous classes — so the converter in
        :func:`t0_agents.mcp.adapter.convert_tools` sees the exact same
        shape it would from a live FastMCP server.  Custom schemas can
        be passed via the ``schemas=`` constructor kwarg; otherwise a
        no-argument schema (``{"type": "object", "properties": {}}``)
        is used.
        """
        tools: list[MCPTool] = []
        for name in self._tools:
            schema = self._schemas.get(
                name,
                {"type": "object", "properties": {}},
            )
            tools.append(
                MCPTool(
                    name=name,
                    description=f"Mock tool {name!r}",
                    inputSchema=schema,
                )
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> CallToolResult:
        """Dispatch to the registered handler and wrap the result.

        ``timeout`` is accepted for interface parity with
        :class:`fastmcp.Client.call_tool` but intentionally ignored —
        the mock runs inline, never over a transport.  Unknown tool
        names raise :class:`KeyError` so the test surfaces the
        misconfiguration immediately rather than silently succeeding.
        """
        self.calls.append({"name": name, "arguments": dict(arguments)})
        handler = self._tools.get(name)
        if handler is None:
            raise KeyError(
                f"MockToolProvider: tool {name!r} not registered "
                f"(available: {sorted(self._tools)!r})"
            )
        if asyncio.iscoroutinefunction(handler):
            async_handler = cast(
                Callable[[dict[str, Any]], Awaitable[Any]], handler
            )
            result = await async_handler(dict(arguments))
        else:
            sync_handler = cast(Callable[[dict[str, Any]], Any], handler)
            result = sync_handler(dict(arguments))
        return CallToolResult(
            content=[],
            structured_content=None,
            meta=None,
            data=result,
            is_error=False,
        )


__all__ = [
    "MockChatResponse",
    "MockGateway",
    "MockToolProvider",
    "RecordingGateway",
    "ToolHandler",
]
