"""Dynamic schema override + per-call ``json_mode`` swap (M5).

Runs four scenarios against the ``extract_entities`` JSON function:

1. **Default ``json_mode = "strict"``** — the config-level schema with
   the four canonical categories (people, locations, organizations,
   topics) is used.  The variant's ``json_mode = "strict"`` flows
   through unchanged.
2. **Per-call ``output_schema`` override** — replaces the static
   four-category schema with a five-category one (adds ``products``).
   Demonstrates the TensorZero per-inference ``output_schema`` field
   (TZ guide §4) and confirms the gateway parses against the
   per-call schema, not the variant-level one.
3. **Per-call ``json_mode = "tool"`` swap** — the function is
   configured with ``json_mode = "strict"`` and has no MCP tools
   attached, so swapping to ``"tool"`` mode at call time succeeds (M4
   safety guard's positive path).
4. **Recovery demo** — sets ``json_mode = "off"`` per call so the
   provider is free to wrap the JSON in prose; the M3 permissive
   parsing layer absorbs the result and ``result.parse_strategy``
   reports ``"permissive"`` (or ``"retry"`` if a hint round was
   needed).

Prerequisites:
    1. ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Start the gateway with this example's config:
         docker compose -f examples/17_structured/docker-compose.yml \\
             --env-file .env up -d
    3. Run:
         uv run python examples/17_structured/ner_dynamic_schema.py
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, AgentResult
from t0_agents.core.types import ToolProvider

SAMPLE_TEXT = (
    "Ada Lovelace and Charles Babbage met at the University of London in 1843 "
    "to discuss the analytical engine, an early mechanical computer.  Years "
    "later, Grace Hopper would deliver her first compiler at the "
    "Eckert-Mauchly Computer Corporation in Philadelphia, then move to Univac "
    "and on to the Pentagon.  Their work seeded both modern programming "
    "languages and the COBOL standard."
)

SAMPLE_TEXT_WITH_PRODUCT = (
    "Apple shipped the original Macintosh from Cupertino in January 1984.  "
    "Steve Jobs unveiled it at the De Anza Flint Center; Susan Kare designed "
    "the icons."
)

WEAK_PROMPT_TEXT = (
    "Tell me everything you can extract from this snippet: 'Marie Curie "
    "shared the 1903 Nobel Prize in Physics with Pierre Curie and Henri "
    "Becquerel.'  Reply in JSON."
)


# Default four-category schema lives at the gateway — for scenario 2 we
# build a five-category dynamic schema at call time.
FIVE_CATEGORY_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "products": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Commercial products, services, or named devices.",
        },
    },
    "required": [
        "people",
        "locations",
        "organizations",
        "topics",
        "products",
    ],
    "additionalProperties": False,
}


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


def _print_scenario(label: str, text: str, result: AgentResult) -> None:
    print(f"\n=== {label} ===")
    print(f"input:           {text[:80] + ('...' if len(text) > 80 else '')}")
    print(f"function_kind:   {result.function_kind}")
    print(f"parse_strategy:  {result.parse_strategy}")
    print(f"parsed:          {json.dumps(result.parsed, indent=2)}")
    print(f"raw (truncated): {result.response[:200]}")
    print(f"turns:           {result.turns}")
    print(f"inference_ids:   {len(result.inference_ids)} id(s)")


async def main() -> None:
    config = AgentConfig(
        name="entity-extractor",
        function_name="extract_entities",
        function_kind="json",
        json_mode="strict",
    )
    # JSON functions never call tools; the in-process FastMCP server is
    # empty but required by Agent's constructor.
    noop_server: FastMCP[Any] = FastMCP("noop")
    mcp_client: Client[Any] = Client(noop_server)

    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )

        # 1. Default config-level strict mode, four-category schema.
        result_1 = await agent.run(SAMPLE_TEXT)
        _print_scenario(
            "Scenario 1: default config (strict, four categories)",
            SAMPLE_TEXT,
            result_1,
        )

        # 2. Per-call dynamic schema override: five categories.
        result_2 = await agent.run(
            SAMPLE_TEXT_WITH_PRODUCT,
            output_schema=FIVE_CATEGORY_SCHEMA,
        )
        _print_scenario(
            "Scenario 2: per-call schema override (five categories)",
            SAMPLE_TEXT_WITH_PRODUCT,
            result_2,
        )
        if result_2.parsed is not None:
            assert "products" in result_2.parsed, (
                "Per-call schema override should produce a 'products' key."
            )

        # 3. Per-call json_mode swap to "tool" (legal: no MCP tools).
        result_3 = await agent.run(SAMPLE_TEXT, json_mode="tool")
        _print_scenario(
            "Scenario 3: per-call json_mode='tool' swap",
            SAMPLE_TEXT,
            result_3,
        )

        # 4. Recovery demo: json_mode='off' so the provider can wrap
        #    JSON in prose; the permissive layer should kick in.
        result_4 = await agent.run(WEAK_PROMPT_TEXT, json_mode="off")
        _print_scenario(
            "Scenario 4: json_mode='off' + permissive recovery",
            WEAK_PROMPT_TEXT,
            result_4,
        )


if __name__ == "__main__":
    asyncio.run(main())
