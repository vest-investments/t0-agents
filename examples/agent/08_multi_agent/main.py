"""Multi-agent composition demo.

Spins up three :class:`Agent` instances — a *researcher* with arXiv
tool access, a *writer* with markdown tooling, and an *orchestrator*
that sees the first two as ``researcher_ask`` and ``writer_ask`` MCP
tools via ``AgentConfig.sub_agents`` — then asks the orchestrator to
write a report.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in your environment (or in ``.env`` at the
       project root).
    2. Start the gateway:
       ``docker compose -f examples/agent/08_multi_agent/docker-compose.yml up -d``
    3. Run: ``uv run python examples/agent/08_multi_agent/main.py``

Alternative entry-point (CLI factory loader):
    ``uv run t0-agents run examples.agent.08_multi_agent.agents:orchestrator_factory``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Dir-name ``08_multi_agent`` starts with a digit, so we cannot import
# siblings as ``from examples.agent.08_multi_agent.agents import ...``.  Add
# the example directory to ``sys.path`` and import ``agents`` as a
# top-level module — same trick ``examples/agent/10_cli/agent.py`` uses to
# reach its ``server`` sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import orchestrator_factory  # type: ignore[import-not-found]


async def main() -> None:
    async with orchestrator_factory() as orchestrator:
        result = await orchestrator.invoke(
            "Write a short report on RLHF advances in 2025. "
            "Use the researcher to find relevant papers, then use the "
            "writer to format the final deliverable as a markdown report.",
        )

    print("\n=== Orchestrator response ===\n")
    print(result.response)
    print("\n=== Run metadata ===")
    print(f"Episode ID    : {result.episode_id}")
    print(f"Turns         : {result.turns}")
    print(f"Inference IDs : {len(result.inference_ids)} total")


if __name__ == "__main__":
    asyncio.run(main())
