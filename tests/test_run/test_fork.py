"""Tests for :meth:`Run.fork`.

Fork extends the parent Run's transcript with an additional user turn.
M1's transcript is the boundary pair — the original user message plus
the agent's final response — sufficient to continue the conversation
in a follow-up Run.  Intermediate tool turns are deferred to M2/M3
when the middleware-level message history becomes available.
"""

from __future__ import annotations

from typing import cast

from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


def _resp(text: str = "answer") -> FakeChatResponse:
    return FakeChatResponse(content=[Text(type="text", text=text)])


def _make_agent(t0: FakeT0) -> Agent:
    return Agent(
        AgentConfig(name="primary", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


class TestTranscriptCapture:
    async def test_transcript_is_user_plus_assistant_boundary(self) -> None:
        agent = _make_agent(FakeT0([_resp("first answer")]))
        run = agent.invoke("what is x?")
        await run.result()
        transcript = run.transcript()
        assert transcript == [
            {"role": "user", "content": "what is x?"},
            {"role": "assistant", "content": "first answer"},
        ]

    async def test_transcript_before_run_is_empty(self) -> None:
        agent = _make_agent(FakeT0([_resp("first answer")]))
        run = agent.invoke("hello")
        # No await — run has not completed yet.
        assert run.transcript() == []


class TestForkContinuation:
    async def test_fork_passes_transcript_as_prior_messages(self) -> None:
        # First Run produces "first answer".  Fork issues a new
        # inference seeded with the prior turns + an extra user
        # message.  The FakeT0 captures every call, so we can verify
        # the prior_messages were threaded through.
        fake_t0 = FakeT0([_resp("first answer"), _resp("second answer")])
        agent = _make_agent(fake_t0)
        original = agent.invoke("what is x?")
        await original.result()
        forked = await original.fork("and what is y?")
        await forked.result()
        # The second call to the FakeT0 must include the prior turns
        # at the head of input.messages, then the new user message.
        second_call = fake_t0.calls[1]
        messages = second_call["input"]["messages"]
        # message[0] is the original user turn, [1] the assistant turn,
        # [2] is the new user message threaded by run_agent.
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "what is x?"
        assert messages[1]["role"] == "assistant"
        # ``run_agent`` materialises the new user prompt as a
        # ``Message(role="user", content="and what is y?")``.
        last = messages[-1]
        # Last entry is a tensorzero.Message instance which compares
        # equal as a dict over (role, content).  Inspect attributes.
        role = getattr(last, "role", None) or last["role"]
        content = getattr(last, "content", None) or last["content"]
        assert role == "user"
        assert content == "and what is y?"

    async def test_forked_runs_have_distinct_episodes(self) -> None:
        fake_t0 = FakeT0([_resp("first answer"), _resp("second answer")])
        agent = _make_agent(fake_t0)
        original = agent.invoke("first")
        await original.result()
        forked = await original.fork("second")
        await forked.result()
        # Independence: each Run gets its own episode_id from the
        # FakeT0 (FakeChatResponse mints uuid4 per instance).
        assert original.episode_id != forked.episode_id

    async def test_fork_is_independent_after_cancellation(self) -> None:
        # Cancel the original AFTER it produced its transcript; the
        # fork must still complete.
        fake_t0 = FakeT0([_resp("first answer"), _resp("second answer")])
        agent = _make_agent(fake_t0)
        original = agent.invoke("first")
        await original.result()
        forked = await original.fork("second")
        # Cancel the original now (no-op since it terminated, but the
        # contract is that the fork is unaffected).
        await original.cancel()
        # Forked Run still runs to completion.
        result = await forked.result()
        assert result.response == "second answer"
