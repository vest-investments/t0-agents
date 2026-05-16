"""Advanced multi-agent composition demo — nested orchestrators.

Builds a two-level delegation tree on top of the same arXiv-flavoured
researcher and markdown writer used by ``main.py``:

* ``researcher`` — leaf agent with arXiv tool access.
* ``writer`` — leaf agent with markdown tooling.
* ``report_drafter`` — inner orchestrator; sees ``researcher_ask`` and
  ``writer_ask`` as tools and composes the report.
* ``project_lead`` — outer orchestrator; sees exactly one tool,
  ``report_drafter_ask``, and decides when to delegate the drafting.

Each level is an independent TensorZero episode.  ClickHouse shows
four episode chains per run:

* ``project_lead`` — the outer orchestrator's inferences.
* ``report_drafter`` — the inner orchestrator's inferences, tagged
  ``t0_agents.parent_episode_id=<project_lead episode>``.
* ``researcher`` and ``writer`` — leaf inferences, each tagged
  ``t0_agents.parent_episode_id=<report_drafter episode>``.

Prerequisites are identical to ``main.py``:

    1. ``GROQ_API_KEY`` set in your environment (or ``.env``).
    2. ``docker compose -f examples/08_multi_agent/docker-compose.yml up -d``
    3. ``uv run python examples/08_multi_agent/advanced.py``

Both the inner and the outer orchestrator use the same TensorZero
function (``orchestrator``) — the role-specific behaviour comes from
which sub-agents each one can see.  In a production setup you would
typically declare a separate function per orchestrator role with its
own system template.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

# Dir-name ``14_multi_agent`` starts with a digit, so we cannot import
# siblings via dotted-path imports.  Add the example directory to
# ``sys.path`` and import the leaf servers as top-level modules — same
# trick ``main.py`` + ``agents.py`` use.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_server import mcp as arxiv_server  # type: ignore[import-not-found]
from markdown_server import mcp as markdown_server  # type: ignore[import-not-found]


@asynccontextmanager
async def nested_orchestrator_factory() -> AsyncIterator[Agent]:
    """Yield the outer ``project_lead`` orchestrator, then tear down."""
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    # Every agent needs its own fastmcp.Client — the inline dispatch
    # that carries the parent-episode id relies on each orchestrator
    # owning the in-process FastMCP server it registers tools on.
    async with (
        await gateway as t0,
        Client(arxiv_server) as arxiv_client,
        Client(markdown_server) as markdown_client,
        Client(FastMCP("report-drafter-server")) as drafter_client,
        Client(FastMCP("project-lead-server")) as lead_client,
    ):
        researcher = Agent(
            AgentConfig(
                name="researcher",
                function_name="research",
                max_turns=6,
                tags={"example": "14_multi_agent", "role": "researcher"},
            ),
            t0=t0,
            tool_provider=cast(Any, arxiv_client),
        )
        writer = Agent(
            AgentConfig(
                name="writer",
                function_name="writing",
                max_turns=6,
                tags={"example": "14_multi_agent", "role": "writer"},
            ),
            t0=t0,
            tool_provider=cast(Any, markdown_client),
        )
        # Inner orchestrator: sees researcher_ask + writer_ask.
        report_drafter = Agent(
            AgentConfig(
                name="report_drafter",
                function_name="orchestrator",
                max_turns=10,
                sub_agents=[researcher, writer],
                tags={"example": "14_multi_agent", "role": "report_drafter"},
            ),
            t0=t0,
            tool_provider=cast(Any, drafter_client),
        )
        # Outer orchestrator: sees report_drafter_ask as its only
        # delegation option.  project_lead never talks directly to the
        # leaf agents — it asks the drafter, which in turn composes the
        # researcher + writer call sequence.
        project_lead = Agent(
            AgentConfig(
                name="project_lead",
                function_name="orchestrator",
                max_turns=6,
                sub_agents=[report_drafter],
                tags={"example": "14_multi_agent", "role": "project_lead"},
            ),
            t0=t0,
            tool_provider=cast(Any, lead_client),
        )
        yield project_lead


async def main() -> None:
    async with nested_orchestrator_factory() as project_lead:
        result = await project_lead.run(
            "Produce a one-page markdown report on RLHF advances in 2025. "
            "Delegate the drafting — do not try to write it yourself.",
        )

    print("\n=== project_lead response ===\n")
    print(result.response)
    print("\n=== Run metadata ===")
    print(f"Episode ID    : {result.episode_id}")
    print(f"Turns         : {result.turns}")
    print(f"Inference IDs : {len(result.inference_ids)} total")
    print(
        "\nCheck ClickHouse for the cascade:\n"
        "  SELECT function_name, episode_id,\n"
        "         tags['t0_agents.parent_episode_id'] AS parent\n"
        "  FROM tensorzero.ChatInference\n"
        f"  WHERE episode_id = '{result.episode_id}'\n"
        "     OR tags['t0_agents.parent_episode_id'] IN (\n"
        f"       '{result.episode_id}',\n"
        "       SELECT episode_id FROM tensorzero.ChatInference\n"
        f"        WHERE tags['t0_agents.parent_episode_id'] = '{result.episode_id}'\n"
        "     )\n"
        "  ORDER BY timestamp;",
    )


if __name__ == "__main__":
    asyncio.run(main())
