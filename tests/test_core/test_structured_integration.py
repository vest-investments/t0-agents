"""Integration tests for :func:`run_structured_inference`.

Requires ``docker compose up -d`` from the repo root with
``GROQ_API_KEY`` populated in ``.env``.  The gateway config at
``config/tensorzero.toml`` declares the ``extract_customer_info`` JSON
function used here.

The model output is not asserted exactly (Groq's gpt-oss-20b is a small
model and may produce slight variations).  The assertions cover the
JSON-function contract: the parsed object satisfies the schema's shape
(both keys present, types are string-or-null), the parse strategy is
``"strict"``, and the underlying SDK fields propagated through to the
:class:`StructuredOutput` wrapper.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.structured import StructuredOutput, run_structured_inference

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Helper to narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_run_structured_inference_static_schema() -> None:
    """Happy path: the gateway parses the model output against the static schema."""
    async with await _gateway_factory() as t0:
        result = await run_structured_inference(
            t0,
            function_name="extract_customer_info",
            user_message=(
                "Hi, I'm Sarah Johnson and you can reach me at "
                "sarah.j@example.com"
            ),
            json_mode="strict",
        )

    assert isinstance(result, StructuredOutput)
    assert result.parse_strategy == "strict"
    assert result.parsed is not None
    assert isinstance(result.parsed, dict)
    assert set(result.parsed.keys()) == {"name", "email"}
    assert isinstance(result.parsed["name"], str | type(None))
    assert isinstance(result.parsed["email"], str | type(None))
    assert result.episode_id
    assert result.inference_id
    assert result.raw  # non-empty raw JSON string


async def test_run_structured_inference_dynamic_schema_override() -> None:
    """Per-call ``output_schema`` overrides the function's static schema."""
    dynamic_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "domain": {
                "type": ["string", "null"],
                "description": "Domain portion of the customer's email.",
            },
        },
        "required": ["name", "email", "domain"],
        "additionalProperties": False,
    }

    async with await _gateway_factory() as t0:
        result = await run_structured_inference(
            t0,
            function_name="extract_customer_info",
            user_message=(
                "Hi, I'm Sarah Johnson and you can reach me at "
                "sarah.j@example.com"
            ),
            output_schema=dynamic_schema,
            json_mode="strict",
        )

    assert result.parse_strategy == "strict"
    assert result.parsed is not None
    assert isinstance(result.parsed, dict)
    assert set(result.parsed.keys()) == {"name", "email", "domain"}
