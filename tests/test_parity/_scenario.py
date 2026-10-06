"""Shared scripted conversation for the loop/runner parity gates.

One canonical three-turn conversation — tool dispatch, corrective retry,
final answer — expressed twice: as non-streaming ``FakeChatResponse``
entries for :func:`t0_agents.mcp.loop.run_agent` and as streaming
``StreamScript`` chunk lists for
:class:`t0_agents.stream.runner.StreamingAgentLoop`.  Every id is pinned
so the golden snapshots in ``test_golden_turn.py`` are deterministic.

The scenario deliberately uses only valid tool names and valid JSON
arguments and never wires progress — so the goldens freeze the turn
*engine* behavior and stay green across the parity-fix commits, whose
behavior changes are covered by their own regression tests instead.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import Text, Thought, ToolCall, ToolResult
from tensorzero.types import TensorZeroError

from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import (
    FakeChatResponse,
    chat_chunk,
    final_chunk,
    text_chunk,
    thought_chunk,
    tool_call_chunk,
)

EPISODE_ID = UUID("00000000-0000-4000-8000-0000000000e1")
INFERENCE_ID_TURN_1 = UUID("00000000-0000-4000-8000-0000000000a1")
INFERENCE_ID_TURN_3 = UUID("00000000-0000-4000-8000-0000000000a3")

USER_MESSAGE = "Add 1 and 2, then greet x."
FUNCTION_NAME = "parity_agent"
RECOVERABLE_ERROR_TEXT = "tool 'bogus' not in request.tools"
FINAL_ANSWER = "The sum is 3 and the greeting was sent."
THOUGHT_SIGNATURE = "sig-th-1"
THOUGHT_PROVIDER = "test-provider"

TOOLS = [
    MCPTool(
        name="add",
        description="Add two integers.",
        inputSchema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    ),
    MCPTool(
        name="greet",
        description="Greet someone by name.",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
]


def _tool_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        data=text,
        is_error=False,
        meta=None,
    )


def make_mcp_client() -> FakeMCPClient:
    return FakeMCPClient(
        tools=TOOLS,
        call_results={"add": _tool_result("3"), "greet": _tool_result("Hello, x!")},
    )


def make_nonstreaming_responses() -> list[Any]:
    """Turn 1: thought + text + two tool calls · turn 2: recoverable error · turn 3: final text."""
    return [
        FakeChatResponse(
            content=[
                Thought(
                    type="thought",
                    text="I should add first.",
                    signature=THOUGHT_SIGNATURE,
                    provider_type=THOUGHT_PROVIDER,
                ),
                Text(type="text", text="Working on it."),
                ToolCall(
                    type="tool_call",
                    id="tc-1",
                    name="add",
                    raw_name="add",
                    arguments={"a": 1, "b": 2},
                    raw_arguments='{"a": 1, "b": 2}',
                ),
                ToolCall(
                    type="tool_call",
                    id="tc-2",
                    name="greet",
                    raw_name="greet",
                    arguments={"name": "x"},
                    raw_arguments='{"name": "x"}',
                ),
            ],
            episode_id=EPISODE_ID,
            inference_id=INFERENCE_ID_TURN_1,
        ),
        TensorZeroError(status_code=400, text=RECOVERABLE_ERROR_TEXT),
        FakeChatResponse(
            content=[Text(type="text", text=FINAL_ANSWER)],
            episode_id=EPISODE_ID,
            inference_id=INFERENCE_ID_TURN_3,
        ),
    ]


def make_streaming_responses() -> list[Any]:
    """The same conversation as chunk scripts (same pinned ids, same block order)."""
    return [
        [
            chat_chunk(
                [
                    thought_chunk(
                        "I should add first.",
                        block_id="th-1",
                        signature=THOUGHT_SIGNATURE,
                        provider_type=THOUGHT_PROVIDER,
                    )
                ],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            chat_chunk(
                [text_chunk("Working on it.", block_id="t-1")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            chat_chunk(
                [tool_call_chunk("add", '{"a": 1, "b": 2}', block_id="tc-1")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            chat_chunk(
                [tool_call_chunk("greet", '{"name": "x"}', block_id="tc-2")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            final_chunk(episode_id=EPISODE_ID, inference_id=INFERENCE_ID_TURN_1),
        ],
        TensorZeroError(status_code=400, text=RECOVERABLE_ERROR_TEXT),
        [
            chat_chunk(
                [text_chunk(FINAL_ANSWER, block_id="t-2")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_3,
            ),
            final_chunk(episode_id=EPISODE_ID, inference_id=INFERENCE_ID_TURN_3),
        ],
    ]


TERMINAL_TOOLS = ("greet",)
TERMINAL_TEXT = "Saying hello now."


def make_terminal_nonstreaming_responses() -> list[Any]:
    """One turn: text plus a terminal tool call, and nothing scripted after it.

    FakeT0 raises once its queue is empty, so the single entry is itself
    the assertion that no second inference is issued.
    """
    return [
        FakeChatResponse(
            content=[
                Text(type="text", text=TERMINAL_TEXT),
                ToolCall(
                    type="tool_call",
                    id="tc-1",
                    name="greet",
                    raw_name="greet",
                    arguments={"name": "x"},
                    raw_arguments='{"name": "x"}',
                ),
            ],
            episode_id=EPISODE_ID,
            inference_id=INFERENCE_ID_TURN_1,
        ),
    ]


def make_terminal_streaming_responses() -> list[Any]:
    """The same terminal turn as a chunk script (same pinned ids, same order)."""
    return [
        [
            chat_chunk(
                [text_chunk(TERMINAL_TEXT, block_id="t-1")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            chat_chunk(
                [tool_call_chunk("greet", '{"name": "x"}', block_id="tc-1")],
                episode_id=EPISODE_ID,
                inference_id=INFERENCE_ID_TURN_1,
            ),
            final_chunk(episode_id=EPISODE_ID, inference_id=INFERENCE_ID_TURN_1),
        ],
    ]


GEMINI_SIGNATURES = ("CiQBjz1rX+8kq/4Tz0Wm==", "CiQBjz1rXwPaW1w7Lnq3==", "CiQBjz1rX3fV6dQe")
GEMINI_PROVIDER = "gcp_vertex_gemini"
GEMINI_INFERENCE_IDS = tuple(
    UUID(f"00000000-0000-4000-8000-0000000000b{n}") for n in (1, 2, 3)
)


def _gemini_rounds(*, preamble: bool) -> list[list[Any]]:
    """Gemini 3's three rounds as gateway 2026.6.0 returns them.

    Each ``thoughtSignature`` is split off the part it signs into an empty
    ``Thought`` directly ahead of it; the final answer's trails its text.
    ``preamble`` adds the sentence Gemini writes before a call when asked to.
    """

    def signed(n: int) -> Thought:
        return Thought(
            type="thought",
            text=None,
            signature=GEMINI_SIGNATURES[n],
            provider_type=GEMINI_PROVIDER,
        )

    def lead(text: str) -> list[Any]:
        return [Text(type="text", text=text)] if preamble else []

    return [
        [
            *lead("Adding first."),
            signed(0),
            ToolCall(
                type="tool_call",
                id="tc-1",
                name="add",
                raw_name="add",
                arguments={"a": 1, "b": 2},
                raw_arguments='{"a": 1, "b": 2}',
            ),
        ],
        [
            *lead("Now the greeting."),
            signed(1),
            ToolCall(
                type="tool_call",
                id="tc-2",
                name="greet",
                raw_name="greet",
                arguments={"name": "x"},
                raw_arguments='{"name": "x"}',
            ),
        ],
        [Text(type="text", text=FINAL_ANSWER), signed(2)],
    ]


def make_gemini_nonstreaming_responses(*, preamble: bool) -> list[Any]:
    return [
        FakeChatResponse(content=blocks, episode_id=EPISODE_ID, inference_id=inference_id)
        for blocks, inference_id in zip(
            _gemini_rounds(preamble=preamble), GEMINI_INFERENCE_IDS, strict=True
        )
    ]


def make_gemini_streaming_responses(*, preamble: bool) -> list[Any]:
    """The same rounds as chunk scripts, one chunk per block, gateway ids."""

    def chunk(block: Any) -> Any:
        if isinstance(block, Text):
            return text_chunk(block.text, block_id="0")
        if isinstance(block, Thought):
            return thought_chunk(
                block_id="1", signature=block.signature, provider_type=block.provider_type
            )
        return tool_call_chunk(block.raw_name, block.raw_arguments, block_id=block.id)

    return [
        [
            *(
                chat_chunk([chunk(b)], episode_id=EPISODE_ID, inference_id=inference_id)
                for b in blocks
            ),
            final_chunk(episode_id=EPISODE_ID, inference_id=inference_id),
        ]
        for blocks, inference_id in zip(
            _gemini_rounds(preamble=preamble), GEMINI_INFERENCE_IDS, strict=True
        )
    ]


def block_to_dict(block: Any) -> dict[str, Any]:
    """Stable dict projection of a TensorZero content block for snapshots."""
    if isinstance(block, Text):
        return {"kind": "text", "text": block.text}
    if isinstance(block, Thought):
        return {
            "kind": "thought",
            "text": block.text,
            "signature": block.signature,
            "provider_type": block.provider_type,
        }
    if isinstance(block, ToolCall):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "raw_name": block.raw_name,
            "arguments": block.arguments,
            "raw_arguments": block.raw_arguments,
        }
    if isinstance(block, ToolResult):
        return {
            "kind": "tool_result",
            "id": block.id,
            "name": block.name,
            "result": block.result,
        }
    raise AssertionError(f"unexpected block type: {type(block).__name__}")


def message_to_dict(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    return {
        "role": message["role"],
        "content": content
        if isinstance(content, str)
        else [block_to_dict(b) for b in content],
    }


def call_to_dict(call: dict[str, Any]) -> dict[str, Any]:
    """Slim projection of one captured ``FakeT0.inference`` call.

    Freezes the turn-engine-owned request shape (routing, episode
    threading, message evolution, advertised tool names) without
    freezing the adapter's full JSON Schema output, which is not this
    gate's contract.
    """
    return {
        "function_name": call.get("function_name"),
        "model_name": call.get("model_name"),
        "episode_id": str(call["episode_id"]) if call.get("episode_id") else None,
        "stream": call.get("stream"),
        "tools": [t["name"] for t in call.get("additional_tools") or []],
        "messages": [message_to_dict(m) for m in call["input"]["messages"]],
    }


def result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "value": result.value,
        "episode_id": result.episode_id,
        "inference_ids": list(result.inference_ids),
        "turns": result.turns,
        "response_source": result.response_source,
    }
