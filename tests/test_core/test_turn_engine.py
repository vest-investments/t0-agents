"""Unit tests for :mod:`t0_agents.core.turn_engine`."""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest
from tensorzero import Message, Text, Thought, ToolCall
from tensorzero.types import ChatInferenceResponse, TensorZeroError, ThoughtSummaryBlock

from t0_agents.core.exceptions import MaxTurnsExceededError, T0AgentsFatalError
from t0_agents.core.turn_engine import (
    TurnState,
    build_agent_result,
    build_inference_kwargs,
    complete_assistant_turn,
    corrective_retry,
    is_recoverable,
    mine_final_text,
    raise_max_turns,
    raise_missing_tool_provider,
    seed_messages,
    strip_thoughts,
    validate_run_inputs,
)


def _tool_call(name: str = "add") -> ToolCall:
    return ToolCall(
        type="tool_call",
        id="tc-1",
        name=name,
        raw_name=name,
        arguments={},
        raw_arguments="{}",
    )


# ---------------------------------------------------------------------------
# validate_run_inputs
# ---------------------------------------------------------------------------


def test_validate_rejects_system_and_system_arguments() -> None:
    with pytest.raises(ValueError, match="not both"):
        validate_run_inputs(
            user_message="hi",
            function_name="f",
            model_name=None,
            system="raw",
            system_arguments={"k": "v"},
            entry_point="run_agent",
        )


def test_validate_rejects_system_arguments_with_model_name() -> None:
    with pytest.raises(ValueError, match="functionless"):
        validate_run_inputs(
            user_message="hi",
            function_name=None,
            model_name="openai::gpt",
            system=None,
            system_arguments={"k": "v"},
            entry_point="run_agent",
        )


@pytest.mark.parametrize("entry_point", ["run_agent", "StreamingAgentLoop.stream"])
def test_validate_rejects_empty_inputs_naming_the_entry_point(
    entry_point: str,
) -> None:
    with pytest.raises(ValueError, match=f"`{entry_point}` requires"):
        validate_run_inputs(
            user_message="",
            function_name="",
            model_name=None,
            system=None,
            system_arguments=None,
            entry_point=entry_point,
        )


def test_validate_accepts_model_name_routing() -> None:
    validate_run_inputs(
        user_message="hi",
        function_name="",
        model_name="openai::gpt",
        system="raw",
        system_arguments=None,
        entry_point="run_agent",
    )


# ---------------------------------------------------------------------------
# seed_messages / build_inference_kwargs
# ---------------------------------------------------------------------------


def test_seed_messages_appends_user_turn_to_prior() -> None:
    prior = [Message(role="user", content="old"), Message(role="assistant", content=[])]
    messages = seed_messages(prior, "new")
    assert len(messages) == 3
    assert messages[-1] == {"role": "user", "content": "new"}
    assert len(prior) == 2  # caller's list untouched


def test_seed_messages_without_prior() -> None:
    assert seed_messages(None, "q") == [Message(role="user", content="q")]


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        function_name="f",
        model_name=None,
        variant_name=None,
        episode_id=None,
        messages=[Message(role="user", content="q")],
        t0_tools=[],
        dryrun=False,
        tool_choice=None,
        tags=None,
        namespace=None,
        system=None,
        system_arguments=None,
        params=None,
        cache_options=None,
    )
    base.update(overrides)
    return build_inference_kwargs(**base)


def test_inference_kwargs_omit_system_when_unset() -> None:
    assert "system" not in _kwargs()["input"]


def test_inference_kwargs_system_string_wins() -> None:
    assert _kwargs(system="raw")["input"]["system"] == "raw"


def test_inference_kwargs_system_arguments_dict() -> None:
    assert _kwargs(system_arguments={"k": "v"})["input"]["system"] == {"k": "v"}


def test_inference_kwargs_empty_function_name_coalesces_to_none() -> None:
    assert _kwargs(function_name="", model_name="openai::gpt")["function_name"] is None


# ---------------------------------------------------------------------------
# is_recoverable / corrective_retry
# ---------------------------------------------------------------------------


def test_recoverable_requires_status_and_pattern() -> None:
    assert is_recoverable(
        TensorZeroError(status_code=400, text="tool 'x' not in request.tools")
    )
    assert is_recoverable(
        TensorZeroError(status_code=502, text="not in request.tools")
    )
    assert not is_recoverable(TensorZeroError(status_code=500, text="not in request.tools"))
    assert not is_recoverable(TensorZeroError(status_code=400, text="boom"))
    assert not is_recoverable(TensorZeroError(status_code=400, text=""))


def test_corrective_retry_trio() -> None:
    assistant, user, sentinel = corrective_retry(
        "tool 'x' not in request.tools", [{"name": "add"}, {"name": "greet"}], turn=1
    )
    assert assistant == {"role": "assistant", "content": []}
    assert user["role"] == "user"
    assert "tool 'x' not in request.tools" in user["content"]
    assert "['add', 'greet']" in user["content"]
    assert sentinel == "error-recoverable-turn-2"


# ---------------------------------------------------------------------------
# complete_assistant_turn
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self) -> None:
        self.episode_id = UUID("00000000-0000-4000-8000-0000000000e1")
        self.inference_id = UUID("00000000-0000-4000-8000-0000000000a1")


async def test_complete_assistant_turn_hook_fires_before_capture() -> None:
    state = TurnState(messages=[])
    order: list[str] = []
    response = _Response()

    async def hook(r: ChatInferenceResponse) -> None:
        order.append("hook")
        assert state.episode_id is None  # capture happens after the hook

    await complete_assistant_turn(
        state,
        content_blocks=[Thought(type="thought", text="t"), Text(type="text", text="x")],
        response=cast(ChatInferenceResponse, response),
        on_inference=hook,
    )
    assert order == ["hook"]
    assert state.episode_id == str(response.episode_id)
    assert state.inference_ids == [str(response.inference_id)]
    (appended,) = state.messages
    assert appended["role"] == "assistant"
    assert [type(b).__name__ for b in appended["content"]] == ["Text"]


async def test_complete_assistant_turn_without_response_appends_only() -> None:
    state = TurnState(messages=[], episode_id="keep")
    await complete_assistant_turn(
        state,
        content_blocks=[Text(type="text", text="x")],
        response=None,
        on_inference=None,
    )
    assert state.episode_id == "keep"
    assert state.inference_ids == []
    assert len(state.messages) == 1


# ---------------------------------------------------------------------------
# require_tool_provider / strip_thoughts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", ["run_agent", "StreamingAgentLoop.stream"])
def test_raise_missing_tool_provider_names_entry_point(entry_point: str) -> None:
    with pytest.raises(T0AgentsFatalError, match=f"`{entry_point}\\("):
        raise_missing_tool_provider(entry_point=entry_point)


def test_strip_thoughts_drops_only_thoughts() -> None:
    text = Text(type="text", text="x")
    call = _tool_call()
    assert strip_thoughts([Thought(type="thought", text="t"), text, call]) == [text, call]


# ---------------------------------------------------------------------------
# mine_final_text / build_agent_result / raise_max_turns
# ---------------------------------------------------------------------------


def test_mine_final_text_prefers_text() -> None:
    blocks = [Thought(type="thought", text="t"), Text(type="text", text="answer")]
    assert mine_final_text(blocks) == ("answer", "text")


def test_mine_final_text_mines_thought_text_and_summary() -> None:
    blocks = [
        Thought(
            text="raw",
            summary=[ThoughtSummaryBlock(text="summarised")],
        )
    ]
    assert mine_final_text(blocks) == ("raw\nsummarised", "thought")


def test_mine_final_text_empty() -> None:
    assert mine_final_text([]) == ("", "text")


def test_build_agent_result_copies_inference_ids() -> None:
    state = TurnState(messages=[], episode_id="ep", inference_ids=["a"])
    result = build_agent_result("txt", "text", state=state, turns=2)
    state.inference_ids.append("b")
    assert result.value == "txt"
    assert result.episode_id == "ep"
    assert result.inference_ids == ["a"]
    assert result.turns == 2
    assert result.response_source == "text"


def test_raise_max_turns() -> None:
    with pytest.raises(MaxTurnsExceededError):
        raise_max_turns(5, "ep")
