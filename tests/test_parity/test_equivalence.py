"""Engine-level equivalence: both loops drive the same conversation identically.

Unlike the frozen goldens (which pin each loop's output byte-for-byte),
this test asserts the two loops against EACH OTHER on the shared
scenario: same final ``AgentResult``, same request evolution — message
stacks, episode threading, advertised tools — modulo the ``stream``
flag.  A future change that alters both loops in the same wrong way
slips past this test but trips the goldens; a change that alters only
one loop trips this test with a readable diff.
"""

from __future__ import annotations

from typing import Any, cast

from t0_agents.mcp.loop import run_agent
from t0_agents.stream.events import DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_t0 import FakeT0
from tests.test_parity._scenario import (
    FUNCTION_NAME,
    USER_MESSAGE,
    call_to_dict,
    make_mcp_client,
    make_nonstreaming_responses,
    make_streaming_responses,
    result_to_dict,
)


async def test_loop_and_runner_produce_equivalent_runs() -> None:
    t0_loop = FakeT0(responses=make_nonstreaming_responses())
    loop_result = await run_agent(
        cast(Any, t0_loop), cast(Any, make_mcp_client()), USER_MESSAGE, FUNCTION_NAME
    )

    t0_stream = FakeT0(responses=make_streaming_responses())
    events = [
        e
        async for e in StreamingAgentLoop().stream(
            cast(Any, t0_stream),
            cast(Any, make_mcp_client()),
            USER_MESSAGE,
            FUNCTION_NAME,
        )
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]

    assert result_to_dict(loop_result) == result_to_dict(done.result)

    loop_calls = [call_to_dict(c) for c in t0_loop.calls]
    stream_calls = [call_to_dict(c) for c in t0_stream.calls]
    assert all(c.pop("stream") is None for c in loop_calls)
    assert all(c.pop("stream") is True for c in stream_calls)
    assert loop_calls == stream_calls
