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
import copy
import json
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
    JsonInferenceOutput,
    JsonInferenceResponse,
    Text,
    Usage,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Return type of a :class:`MockGateway` ``source`` callable.  A source may
#: return any of:
#:
#: * a list of :class:`~tensorzero.types.ContentBlock` — wrapped into a fresh
#:   :class:`ChatInferenceResponse` by the gateway (chat path);
#: * a fully built :class:`ChatInferenceResponse` — used as-is (chat path);
#: * a fully built :class:`JsonInferenceResponse` — used as-is (JSON path);
#: * a ``dict`` payload with at least a ``raw`` key — materialised into a
#:   :class:`JsonInferenceResponse` by the gateway (JSON path).  Extra keys
#:   (``parsed``, ``episode_id``, ``inference_id``, ``usage``,
#:   ``variant_name``, ``finish_reason``) override the defaults; an absent
#:   ``parsed`` key triggers a best-effort :func:`json.loads` of ``raw``.
_SourceReturn: TypeAlias = (
    list[ContentBlock]
    | ChatInferenceResponse
    | JsonInferenceResponse
    | dict[str, Any]
)

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
# MockJsonResponse
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MockJsonResponse:
    """Builder for a single :class:`JsonInferenceResponse`-shaped object.

    Symmetric counterpart to :class:`MockChatResponse` for JSON-flavoured
    agents.  Keeps ``raw`` and ``parsed`` explicit so retry tests can
    script precise invalid → valid transitions; the chat-builder pattern
    of "list of content blocks" maps poorly to JSON functions, where the
    response is a single (raw, parsed) pair.
    """

    raw: str | None
    parsed: Any | None = None
    usage: Usage | None = None
    variant_name: str = "mock_variant"
    finish_reason: Any = None

    def to_response(
        self,
        *,
        inference_id: UUID | None = None,
        episode_id: UUID | None = None,
    ) -> JsonInferenceResponse:
        """Materialise a :class:`JsonInferenceResponse` from the builder."""
        return JsonInferenceResponse(
            inference_id=inference_id or uuid4(),
            episode_id=episode_id or uuid4(),
            variant_name=self.variant_name,
            output=JsonInferenceOutput(raw=self.raw, parsed=self.parsed),
            usage=self.usage or Usage(input_tokens=0, output_tokens=0),
            finish_reason=self.finish_reason,
        )


# ---------------------------------------------------------------------------
# JSON dict helpers
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str | None) -> Any | None:
    """Best-effort :func:`json.loads`; returns ``None`` on any failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _coerce_uuid(value: Any, fallback: UUID) -> UUID:
    """Accept ``UUID``, ``str``, or ``None`` and return a ``UUID``.

    ``None`` and the absence of a key both fall back to *fallback* so
    callers can pin specific IDs when they care and leave the rest to
    the gateway's defaults.
    """
    if value is None:
        return fallback
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _materialise_json_dict(
    payload: dict[str, Any],
    *,
    sticky_episode_id: UUID,
) -> JsonInferenceResponse:
    """Build a :class:`JsonInferenceResponse` from a ``script_json`` dict.

    Recognised keys (all optional except ``raw``):

    * ``raw`` — required.  ``str | None``; the JSON string the model
      "returned" (``None`` is legal for tests of the empty-output edge).
    * ``parsed`` — when the key is **absent**, the gateway auto-derives
      via :func:`json.loads` (returning ``None`` on parse failure); when
      the key is **present** (even as ``None``), the value is used
      verbatim so retry scripts can pin a `(valid raw, parsed=None)` pair
      and observe the permissive layer kick in.
    * ``episode_id`` — ``UUID | str | None``; ``None`` reuses the
      gateway's sticky episode id (mirrors the real gateway's contract
      that an episode is allocated on inference #1 and reused).
    * ``inference_id`` — ``UUID | str | None``; ``None`` mints a fresh
      :func:`uuid.uuid4`.
    * ``usage`` — :class:`~tensorzero.Usage` or ``None``; defaults to
      ``Usage(input_tokens=0, output_tokens=0)``.
    * ``variant_name`` — ``str``; defaults to ``"mock_variant"``.
    * ``finish_reason`` — passed through unchanged.
    """
    if "raw" not in payload:
        raise KeyError(
            "MockGateway.script_json/canned_json: dict entry missing "
            "required 'raw' key"
        )
    raw_value = payload["raw"]
    if "parsed" in payload:
        parsed_value = payload["parsed"]
    else:
        parsed_value = _try_parse_json(raw_value)

    episode_id = _coerce_uuid(payload.get("episode_id"), sticky_episode_id)
    inference_id = _coerce_uuid(payload.get("inference_id"), uuid4())
    usage = payload.get("usage") or Usage(input_tokens=0, output_tokens=0)
    variant_name = payload.get("variant_name", "mock_variant")
    finish_reason = payload.get("finish_reason")

    return JsonInferenceResponse(
        inference_id=inference_id,
        episode_id=episode_id,
        variant_name=variant_name,
        output=JsonInferenceOutput(raw=raw_value, parsed=parsed_value),
        usage=usage,
        finish_reason=finish_reason,
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
    response: ChatInferenceResponse | JsonInferenceResponse


class MockGateway:
    """A duck-typed :class:`~tensorzero.AsyncTensorZeroGateway` for unit tests.

    Five construction patterns:

    * :meth:`script` — list of per-turn content blocks; each
      ``inference()`` call pops the next entry.  When the script is
      exhausted, the gateway returns an empty :class:`~tensorzero.Text`
      turn so the agent loop terminates cleanly (avoids hanging tests).
    * :meth:`canned` — callable receiving the inference kwargs and
      returning either a content list or a fully built response.
      Behaviour-driven; ideal when a test wants to branch on the
      assistant's previous message or tool results.
    * :meth:`script_json` — JSON-mode sibling of :meth:`script`.  Each
      script entry is either a dict (``{"raw": ..., "parsed": ...,
      ...}``) or a pre-built :class:`JsonInferenceResponse`.  When the
      script is exhausted, the gateway raises ``AssertionError`` —
      JSON functions are single-shot per call, so an exhausted script
      almost always means the test forgot to script a retry response.
    * :meth:`canned_json` — JSON-mode sibling of :meth:`canned`.  Same
      dict / pre-built return shape as :meth:`script_json`.
    * :meth:`recording` — wrap a real gateway, forward ``inference()``
      through it, and capture ``(kwargs, response)`` pairs.  Call
      :meth:`RecordingGateway.replay` afterwards to turn the capture
      into a deterministic :class:`MockGateway`.

    The gateway records every call on :attr:`calls` (and
    :attr:`feedback_calls` for the feedback path) so tests can assert
    on the kwargs the loop passed — notably ``additional_tools``,
    ``tool_choice``, ``tags``, ``namespace`` (chat path), and
    ``output_schema``, ``params`` (JSON path).
    """

    def __init__(
        self,
        source: Callable[..., _SourceReturn],
        *,
        episode_id: UUID | None = None,
    ) -> None:
        self._source = source
        # MockGateway keeps the same episode id for every turn of a run
        # so ``Agent.invoke(episode_id=None)`` sees a stable id across turns,
        # matching the real gateway's contract (episode_id first allocated
        # on inference #1, reused thereafter).  The same id is reused for
        # JSON retries via the script_json dict's ``episode_id: None``
        # default — Agent._run_json reads ``structured.episode_id`` on
        # the first attempt and threads it into subsequent retries.
        self._sticky_episode_id: UUID | None = episode_id
        self.calls: list[dict[str, Any]] = []
        # Parallel to ``calls``: the response object the gateway returned
        # for each entry.  May be a chat or JSON response depending on
        # the source; assertion helpers narrow with ``isinstance`` when
        # they need a specific shape.
        self.responses: list[ChatInferenceResponse | JsonInferenceResponse] = []
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
    def script_json(
        cls,
        turns: list[dict[str, Any] | JsonInferenceResponse],
        *,
        episode_id: UUID | None = None,
    ) -> MockGateway:
        """Queue-driven JSON responses, one entry per ``inference()`` call.

        Each entry is either a dict (see :func:`_materialise_json_dict`
        for the recognised keys) or a pre-built
        :class:`JsonInferenceResponse`.  Dicts are the common case for
        single-shot tests; pre-built responses are useful when a test
        needs to pin a specific :class:`~tensorzero.Usage` shape or
        re-use the same response across attempts.

        When the queue is empty and ``inference()`` is called again, the
        gateway raises ``AssertionError`` — JSON functions do not have a
        "graceful empty turn" the way the chat loop does, so an
        exhausted JSON script almost always indicates the test scripted
        too few responses for the retry budget.
        """
        queue: deque[dict[str, Any] | JsonInferenceResponse] = deque(turns)

        def source(**_: Any) -> _SourceReturn:
            if not queue:
                raise AssertionError(
                    "MockGateway.script_json: ran out of canned responses; "
                    "scripted N retries but Agent._run_json requested "
                    "more inferences than the script provides"
                )
            return queue.popleft()

        return cls(source, episode_id=episode_id)

    @classmethod
    def canned_json(
        cls,
        fn: Callable[..., dict[str, Any] | JsonInferenceResponse],
        *,
        episode_id: UUID | None = None,
    ) -> MockGateway:
        """Callable-driven JSON responses; receives full inference kwargs.

        ``fn`` returns either a dict (same shape as the entries
        :meth:`script_json` accepts) or a pre-built
        :class:`JsonInferenceResponse`.  Useful when the response must
        depend on the captured kwargs — e.g. branching on
        ``kwargs.get("output_schema")`` to verify that the per-call
        override threaded through to the gateway.
        """
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

    async def inference(
        self, **kwargs: Any
    ) -> ChatInferenceResponse | JsonInferenceResponse:
        """Record the call and return the next scripted response.

        ``kwargs`` are deep-copied before being appended to :attr:`calls`
        so the recording is a true snapshot of the state the caller
        passed on that turn.  Agent loops typically mutate
        ``input["messages"]`` in place across turns; without the deep
        copy every recorded entry would end up pointing at the same
        mutating list.

        The return type is the SDK's ``InferenceResponse`` union —
        chat sources yield :class:`ChatInferenceResponse`, JSON sources
        yield :class:`JsonInferenceResponse`.  A single gateway instance
        can mix both shapes if the source decides per-call (useful when
        wrapping a real gateway that hosts both chat and JSON functions).
        """
        self.calls.append(copy.deepcopy(kwargs))
        result = self._source(**kwargs)
        response: ChatInferenceResponse | JsonInferenceResponse
        if isinstance(result, ChatInferenceResponse | JsonInferenceResponse):
            response = result
        elif isinstance(result, dict):
            # JSON dict shape — materialise into a JsonInferenceResponse.
            if self._sticky_episode_id is None:
                self._sticky_episode_id = uuid4()
            response = _materialise_json_dict(
                result, sticky_episode_id=self._sticky_episode_id
            )
        else:
            # Content-block list (chat path) — wrap with a stable episode id.
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
        self.responses: list[ChatInferenceResponse | JsonInferenceResponse] = []
        self.feedback_calls: list[dict[str, Any]] = []

    async def inference(
        self, **kwargs: Any
    ) -> ChatInferenceResponse | JsonInferenceResponse:
        # One deep copy is used for both ``.calls`` and ``_captured`` so the
        # two records stay in sync and neither can be mutated by the loop's
        # in-place ``messages.append`` after the call returns.
        kwargs_snapshot = copy.deepcopy(kwargs)
        self.calls.append(kwargs_snapshot)
        raw = await self._real.inference(**kwargs)
        response = cast(
            "ChatInferenceResponse | JsonInferenceResponse", raw
        )
        self._captured.append(
            _RecordedInference(kwargs=kwargs_snapshot, response=response)
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
        queue: deque[ChatInferenceResponse | JsonInferenceResponse] = deque(
            rec.response for rec in self._captured
        )

        def source(**_: Any) -> _SourceReturn:
            if queue:
                return queue.popleft()
            # Capture exhausted — fall back to an empty chat turn so the
            # agent loop terminates cleanly.  Replay is a chat-leaning
            # idiom; JSON callers should pad their capture explicitly.
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
        # Pass-through when the handler already built a full result (e.g. to
        # exercise the ``is_error=True`` branch of the production serializer).
        # Wrapping it again would nest the ``CallToolResult`` inside ``.data``
        # and break ``serialize_tool_result``.
        if isinstance(result, CallToolResult):
            return result
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
    "MockJsonResponse",
    "MockToolProvider",
    "RecordingGateway",
    "ToolHandler",
]
