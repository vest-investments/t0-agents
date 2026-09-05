"""Unit tests for :mod:`t0_agents.core.turn_engine`."""

from __future__ import annotations

import json
import random
from typing import Any, cast
from uuid import UUID

import pytest
from tensorzero import Message, Text, Thought, ToolCall
from tensorzero.types import ChatInferenceResponse, TensorZeroError, ThoughtSummaryBlock

from t0_agents.core.exceptions import MaxTurnsExceededError, T0AgentsFatalError
from t0_agents.core.turn_engine import (
    SalvagedArguments,
    TurnState,
    build_agent_result,
    build_inference_kwargs,
    complete_assistant_turn,
    corrective_retry,
    is_recoverable,
    is_terminal_round,
    mine_final_text,
    raise_max_turns,
    raise_missing_tool_provider,
    salvage_tool_arguments,
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


# ---------------------------------------------------------------------------
# salvage_tool_arguments
# ---------------------------------------------------------------------------

_CLOSING_NOISE = "}] \t\n\r"


def _chart_spec() -> dict[str, Any]:
    """A ~2 KB chart-like payload — the size class observed in the wild."""
    return {
        "type": "line",
        "title": "Portfolio value vs. benchmark, last 40 sessions",
        "x": [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(40)],
        "series": [
            {"name": name, "y": [round(100 + i * k * 0.37, 2) for i in range(40)]}
            for k, name in enumerate(("portfolio", "benchmark", "cash", "bonds"), start=1)
        ],
    }


def test_salvage_recovers_large_object_followed_by_one_extra_brace() -> None:
    spec = _chart_spec()
    raw = json.dumps(spec)
    assert len(raw) > 1800
    assert salvage_tool_arguments(raw + "}") == SalvagedArguments(spec, len(raw), 1)


@pytest.mark.parametrize(
    "suffix",
    ["}", "]", "}}}}", "\n}\n", " } ] \r\n"],
    ids=["brace", "bracket", "braces", "newline-brace", "mixed"],
)
def test_salvage_tolerates_only_closing_noise(suffix: str) -> None:
    spec = {"city": "Añasco", "q": "a}b]c", "n": [1, 2, {"k": None}]}
    encoded = json.dumps(spec, ensure_ascii=False)
    assert salvage_tool_arguments(encoded + suffix) == SalvagedArguments(
        spec, len(encoded), len(suffix)
    )


def test_salvage_tolerates_leading_whitespace() -> None:
    """Leading whitespace is skipped and counted as neither parsed nor discarded."""
    assert salvage_tool_arguments('\n  {"a": 1}}') == SalvagedArguments({"a": 1}, 8, 1)


def test_salvage_leaves_valid_json_to_the_schema_validator() -> None:
    """No parser rejected valid JSON — ``arguments=None`` for it means TensorZero's
    schema validation refused the call, and the salvage must not overrule that."""
    assert salvage_tool_arguments('{"a": 1}') is None
    assert salvage_tool_arguments('  {"a": 1}\n\t') is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1, "b": 2 "c": 3}}',
        '{"a": 1',
        '"hello"}',
        "[1, 2]}",
        '{"a": 1}{"b": 2}',
        '{"a": 1},',
        '{"a": 1}, {"b": 2}',
        '{"a": 1} done',
        'here: {"a": 1}',
        '{"a": 1}"',
        '{"a": 1}}x',
        "",
        "   ",
    ],
    ids=[
        "missing-comma",
        "truncated",
        "string",
        "array",
        "second-object",
        "trailing-comma",
        "comma-then-object",
        "prose-after",
        "prose-before",
        "stray-quote",
        "brace-then-text",
        "empty",
        "blank",
    ],
)
def test_salvage_refuses_everything_else(raw: str) -> None:
    assert salvage_tool_arguments(raw) is None


_FUZZ_ALPHABET = 'ab{}[]",:\\ ñ'


def _random_value(rng: random.Random, depth: int) -> Any:
    kind = rng.randrange(7 if depth else 5)
    if kind == 0:
        return rng.randint(-10_000, 10_000)
    if kind == 1:
        return round(rng.uniform(-1e6, 1e6), rng.randrange(4))
    if kind == 2:
        return "".join(rng.choice(_FUZZ_ALPHABET) for _ in range(rng.randrange(12)))
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return [rng.randint(0, 9) for _ in range(rng.randrange(6))]
    if kind == 5:
        return [_random_value(rng, depth - 1) for _ in range(rng.randrange(4))]
    return _random_object(rng, depth - 1)


def _random_object(rng: random.Random, depth: int) -> dict[str, Any]:
    return {f"k{i}": _random_value(rng, depth) for i in range(rng.randrange(1, 6))}


def _without_member_commas(spec: dict[str, Any]) -> str:
    members = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in spec.items()]
    return "{" + " ".join(members) + "}"


def test_salvage_fuzz_round() -> None:
    """Closing noise after a complete object is salvaged; damage inside never is."""
    rng = random.Random(20260824)
    for _ in range(300):
        spec = _random_object(rng, depth=2)
        encoded = json.dumps(spec, ensure_ascii=rng.random() < 0.5)
        chars = [rng.choice("}]"), *(rng.choice(_CLOSING_NOISE) for _ in range(rng.randrange(8)))]
        rng.shuffle(chars)
        noise = "".join(chars)

        assert salvage_tool_arguments(encoded) is None
        assert salvage_tool_arguments(encoded + "  ") is None
        assert salvage_tool_arguments(encoded + noise) == SalvagedArguments(
            spec, len(encoded), len(noise)
        )
        assert salvage_tool_arguments(encoded + noise + encoded) is None
        assert salvage_tool_arguments(encoded[: rng.randrange(1, len(encoded))]) is None
        if len(spec) > 1:
            assert salvage_tool_arguments(_without_member_commas(spec) + noise) is None


# ---- is_terminal_round ------------------------------------------------

_RENDER = ("present_*",)


def test_terminal_round_ends_a_text_plus_render_round() -> None:
    assert is_terminal_round(
        _RENDER,
        tool_names=["present_chart", "present_quick_replies"],
        errored=[False, False],
        text="Here is your portfolio.",
        source="text",
    )


def test_terminal_round_matches_a_literal_name() -> None:
    assert is_terminal_round(
        ("present_chart",),
        tool_names=["present_chart"],
        errored=[False],
        text="Here it is.",
        source="text",
    )


def test_terminal_round_is_off_when_no_tool_is_declared_terminal() -> None:
    assert not is_terminal_round(
        (),
        tool_names=["present_chart"],
        errored=[False],
        text="Here is your portfolio.",
        source="text",
    )


def test_terminal_round_needs_tool_calls() -> None:
    """A text-only round is an ordinary turn end, not a terminal round."""
    assert not is_terminal_round(
        _RENDER,
        tool_names=[],
        errored=[],
        text="Here is your portfolio.",
        source="text",
    )


def test_terminal_round_needs_text_from_this_round() -> None:
    assert not is_terminal_round(
        _RENDER,
        tool_names=["present_chart"],
        errored=[False],
        text="",
        source="text",
    )


def test_terminal_round_refuses_blank_text() -> None:
    """A model that emits only newlines before its tool call has said nothing.

    The wire counts a delta as text by ``delta.strip()``; terminating on a
    blank round would leave the belt to write EMPTY_TURN_RETRY_TEXT under
    the render the tool just produced.
    """
    assert not is_terminal_round(
        _RENDER,
        tool_names=["present_chart"],
        errored=[False],
        text="   \n\n   ",
        source="text",
    )


def test_terminal_round_refuses_a_thought_sourced_answer() -> None:
    assert not is_terminal_round(
        _RENDER,
        tool_names=["present_chart"],
        errored=[False],
        text="the user asked for a chart, so I will render one",
        source="thought",
    )


def test_terminal_round_refuses_an_errored_terminal_result() -> None:
    assert not is_terminal_round(
        _RENDER,
        tool_names=["present_chart"],
        errored=[True],
        text="Here is your portfolio.",
        source="text",
    )


def test_terminal_round_refuses_a_round_that_also_fetches_data() -> None:
    assert not is_terminal_round(
        _RENDER,
        tool_names=["present_chart", "customers_retrieve_account_overview"],
        errored=[False, False],
        text="Here is your portfolio.",
        source="text",
    )


def test_terminal_round_refuses_a_hallucinated_tool_name() -> None:
    assert not is_terminal_round(
        _RENDER,
        tool_names=[None],
        errored=[False],
        text="Here is your portfolio.",
        source="text",
    )
