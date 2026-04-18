"""Advanced tier — custom MemoryProvider from scratch.

Implements the ``MemoryProvider`` Protocol directly (not by wrapping one
of the built-in backends) with two production concerns the built-ins
don't handle:

* **Per-tenant namespacing** — two tenants using the same ``episode_id``
  never see each other's records.
* **Per-episode record cap with LRU eviction** — oldest records drop
  out once the per-episode count exceeds ``max_records_per_episode``.
  Resolves the unbounded-growth caveat documented in ``docs/memory.md``.

Runs in plain asyncio: no Docker, no TensorZero gateway, no MCP client.
Because ``MemoryProvider`` is ``@runtime_checkable``, an ``isinstance``
check at runtime asserts that this class satisfies the Protocol.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from t0_agents.memory import MemoryProvider, MemoryRecord


class NamespacedCappedMemory:
    """Custom MemoryProvider with tenant namespacing and per-episode LRU."""

    def __init__(
        self,
        *,
        tenant: str,
        max_records_per_episode: int = 5,
    ) -> None:
        self.tenant = tenant
        self.max = max_records_per_episode
        # Key: (tenant, episode_id) → list of records (oldest first).
        self._store: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)

    def _key(self, episode_id: str) -> tuple[str, str]:
        return (self.tenant, episode_id)

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        return list(self._store[self._key(episode_id)])

    async def save(self, record: MemoryRecord) -> None:
        bucket = self._store[self._key(record.episode_id)]
        bucket.append(record)
        if len(bucket) > self.max:
            # LRU eviction: drop the oldest records in place so the
            # bucket is never larger than ``max``.
            del bucket[: len(bucket) - self.max]

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        q = query.lower()
        hits: list[MemoryRecord] = []
        for (tenant, _), bucket in self._store.items():
            if tenant != self.tenant:
                continue
            for rec in bucket:
                hay = f"{rec.user_message}\n{rec.agent_response}".lower()
                if q in hay:
                    hits.append(rec)
        hits.sort(key=lambda r: r.created_at, reverse=True)
        return hits[:limit]


async def main() -> None:
    alice = NamespacedCappedMemory(tenant="alice", max_records_per_episode=3)
    bob = NamespacedCappedMemory(tenant="bob", max_records_per_episode=3)

    # Protocol conformance is enforced at runtime.
    assert isinstance(alice, MemoryProvider)
    assert isinstance(bob, MemoryProvider)

    # Write five records for Alice under a single episode.  The cap
    # keeps only the three most recent.
    for i in range(5):
        await alice.save(
            MemoryRecord(
                episode_id="session-1",
                user_message=f"alice msg {i}",
                agent_response=f"alice reply {i}",
                turns=1,
            )
        )
    alice_records = await alice.load("session-1")
    assert len(alice_records) == 3
    assert [r.user_message for r in alice_records] == [
        "alice msg 2",
        "alice msg 3",
        "alice msg 4",
    ]
    print(f"Alice LRU cap enforced: kept {len(alice_records)} of 5 records")

    # Bob uses the same episode_id, but lives in a separate namespace.
    await bob.save(
        MemoryRecord(
            episode_id="session-1",
            user_message="bob msg",
            agent_response="bob reply",
            turns=1,
        )
    )
    bob_records = await bob.load("session-1")
    assert len(bob_records) == 1
    assert bob_records[0].user_message == "bob msg"
    print(f"Bob isolation verified: {len(bob_records)} record under 'session-1'")

    # Search respects tenant boundaries.
    alice_hits = await alice.search("msg")
    bob_hits = await bob.search("msg")
    assert all("alice" in h.user_message for h in alice_hits)
    assert all("bob" in h.user_message for h in bob_hits)
    print(
        f"search 'msg' → alice:{len(alice_hits)}  bob:{len(bob_hits)}  "
        "(no cross-tenant leakage)"
    )


if __name__ == "__main__":
    asyncio.run(main())
