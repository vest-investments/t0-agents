"""Local-only demo of the M3 permissive parsing + retry layers.

Runs four scripted scenarios against an in-process fake of
:class:`AsyncTensorZeroGateway`, exercising each ``parse_strategy``
outcome end-to-end through ``Agent.run`` — no docker, no provider
credentials, completes in under a second.

Scenarios:

1. ``"strict"`` — TZ returned valid JSON on the first attempt.
2. ``"permissive"`` — TZ returned ``parsed == None`` but the raw text
   wraps JSON in prose, recovered by :func:`permissive_parse`.
3. ``"retry"`` — first attempt invalid, retry succeeds.
4. ``"none"`` — every attempt invalid, retry budget exhausted.

For a real-provider demo (Groq via docker compose) see
``recovery_real_provider.py`` in this directory.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    Usage,
)
from tensorzero.types import JsonInferenceOutput

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider


def _fake_json_response(
    raw: str | None, parsed: dict[str, Any] | None
) -> JsonInferenceResponse:
    """Build a real :class:`JsonInferenceResponse` from SDK dataclasses.

    Using the real type rather than a duck-typed shim means the
    ``isinstance(response, JsonInferenceResponse)`` runtime check in
    :func:`run_structured_inference` succeeds without a separate branch.
    """
    return JsonInferenceResponse(
        inference_id=uuid4(),
        episode_id=uuid4(),
        variant_name="demo-variant",
        output=JsonInferenceOutput(raw=raw, parsed=parsed),
        usage=Usage(input_tokens=10, output_tokens=20),
    )


class ScriptedGateway:
    """In-process fake that pops one response per ``inference()`` call.

    Mirrors the surface :class:`AsyncTensorZeroGateway` exposes to
    :func:`run_structured_inference` (a single async ``inference()``
    coroutine).  Raises ``AssertionError`` if a scenario exhausts its
    scripted responses — useful for catching off-by-one mistakes in the
    demo wiring itself.
    """

    def __init__(self, responses: list[JsonInferenceResponse]) -> None:
        self._queue: list[JsonInferenceResponse] = list(responses)

    async def inference(self, **_kwargs: Any) -> JsonInferenceResponse:
        if not self._queue:
            raise AssertionError("ScriptedGateway ran out of responses")
        return self._queue.pop(0)


class NoOpToolProvider:
    """JSON-only agents never call tools — return an empty toolset."""

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:  # pragma: no cover - JSON path never reaches here
        raise AssertionError(
            "NoOpToolProvider.call_tool should not be invoked from "
            "a JSON-function agent"
        )


def _make_agent(t0: ScriptedGateway) -> Agent:
    config = AgentConfig(
        name="recovery-demo",
        function_name="extract_customer_info",
        function_kind="json",
        json_mode="strict",
        permissive_json_parsing=True,
        max_json_retries=2,
    )
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, NoOpToolProvider()),
    )


SCENARIOS: list[tuple[str, list[tuple[str | None, dict[str, Any] | None]]]] = [
    (
        "strict",
        [
            ('{"name": "Ada", "email": "ada@example.com"}',
             {"name": "Ada", "email": "ada@example.com"}),
        ],
    ),
    (
        "permissive",
        [
            ('Sure! Here it is:\n```json\n'
             '{"name": "Grace", "email": "grace@example.com"}\n```',
             None),
        ],
    ),
    (
        "retry",
        [
            ("Sorry, I do not know how to format that.", None),
            ('{"name": "Linus", "email": "linus@example.com"}',
             {"name": "Linus", "email": "linus@example.com"}),
        ],
    ),
    (
        "none",
        [
            ("nope", None),
            ("still nope", None),
            ("really, no JSON here", None),
        ],
    ),
]


async def main() -> None:
    rows: list[dict[str, Any]] = []
    for label, script in SCENARIOS:
        gateway = ScriptedGateway(
            [_fake_json_response(raw=r, parsed=p) for r, p in script]
        )
        agent = _make_agent(gateway)
        result = await agent.run("Tell me about a famous engineer.")
        rows.append(
            {
                "scenario": label,
                "parse_strategy": result.parse_strategy,
                "json_parse_attempts": result.json_parse_attempts,
                "inference_ids": len(result.inference_ids),
                "parsed": result.parsed,
            }
        )

    width = max(len(row["scenario"]) for row in rows)
    print(
        f"{'scenario':<{width}}  parse_strategy  attempts  "
        f"inference_ids  parsed"
    )
    print(
        f"{'-' * width}  {'-' * 14}  {'-' * 8}  "
        f"{'-' * 13}  {'-' * 6}"
    )
    for row in rows:
        parsed_repr = str(row["parsed"])
        if len(parsed_repr) > 36:
            parsed_repr = parsed_repr[:33] + "..."
        print(
            f"{row['scenario']:<{width}}  "
            f"{row['parse_strategy']:<14}  "
            f"{row['json_parse_attempts']:<8}  "
            f"{row['inference_ids']:<13}  "
            f"{parsed_repr}"
        )


if __name__ == "__main__":
    asyncio.run(main())
