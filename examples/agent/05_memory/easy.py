"""Easy tier — store primitives without an Agent.

Runs in plain asyncio: no Docker, no TensorZero gateway, no MCP client.
Exercises ``InMemoryMemory`` directly to show the shape of a
``MemoryRecord``, the output of ``render_memory_summary``, and how
``search`` ranks matches.
"""

from __future__ import annotations

import asyncio

from t0_agents.memory import (
    InMemoryMemory,
    MemoryRecord,
    render_memory_summary,
)


async def main() -> None:
    store = InMemoryMemory()

    await store.save(
        MemoryRecord(
            episode_id="demo",
            user_message="My name is Alice.",
            agent_response="Nice to meet you, Alice.",
            turns=1,
        )
    )
    await store.save(
        MemoryRecord(
            episode_id="demo",
            user_message="My favourite colour is teal.",
            agent_response="Got it — teal.",
            turns=1,
        )
    )
    # A record under a different episode — confirms isolation.
    await store.save(
        MemoryRecord(
            episode_id="other",
            user_message="Bob here.",
            agent_response="Hello Bob.",
            turns=1,
        )
    )

    records = await store.load("demo")
    assert len(records) == 2
    print(f"demo episode has {len(records)} records\n")

    # The rendered summary is what an Agent would inject into the prompt.
    summary = render_memory_summary(records)
    print(summary)
    print()

    # Search is case-insensitive and orders most-recent matches first.
    hits = await store.search("COLOUR")
    assert len(hits) == 1
    print(f"search('COLOUR') → {len(hits)} match")
    print(f"  latest match: {hits[0].user_message!r}\n")

    # Episodes stay isolated.
    other_records = await store.load("other")
    assert len(other_records) == 1
    print(f"'other' episode has {len(other_records)} record (isolation ok)")


if __name__ == "__main__":
    asyncio.run(main())
