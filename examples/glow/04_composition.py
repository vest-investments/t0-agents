"""Glow ``compose()``: native-collection composition.

Demonstrates the three compose() shapes plus composer-in-composer
nesting — the "stackable by contract" promise from the plan file.

- ``compose([a, b, c])`` → ``sequential(a, b, c)``: pipeline.
- ``compose({a, b}, merge=fn)`` → ``parallel(*, reducer=fn,
  error_policy="continue")``: fan-out + merge.
- ``compose([compose({a, b}, merge=fn), c])``: nested composer as a
  node of an outer pipeline.

Glow's ``@app.agent`` returns a typed wrapper, not the underlying
v1.0 :class:`Agent`.  ``compose()`` accepts wrappers and unwraps them
through ``app._lookup_or_build`` before handing them to the v1.0
composer factories.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/04_composition.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, compose


def _join_string_values(results: list[Any]) -> str:
    """Custom reducer: concatenate string ``AgentResult`` values with a
    bullet prefix.  Any failures (``BaseException`` entries that
    ``parallel(error_policy="continue")`` keeps inline) are noted but
    do not abort the merge."""
    parts: list[str] = []
    for r in results:
        if isinstance(r, BaseException):
            parts.append(f"- (branch failed: {type(r).__name__})")
        else:
            parts.append(f"- {r.value}")
    return "\n".join(parts)


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("optimist")
        async def optimist(query: str) -> str:  # type: ignore[empty-body]
            """Reply with a one-sentence optimistic take on the topic."""

        @app.agent("pessimist")
        async def pessimist(query: str) -> str:  # type: ignore[empty-body]
            """Reply with a one-sentence pessimistic take on the topic."""

        @app.agent("summarizer")
        async def summarizer(query: str) -> str:  # type: ignore[empty-body]
            """Summarize the bulleted input into one short paragraph."""

        async with app.lifespan():
            # ---- Example 5: set ensemble + custom merge -------------
            ensemble = compose({optimist, pessimist}, merge=_join_string_values)
            ensemble_run = ensemble.invoke("the rise of generative AI")
            ensemble_text = (await ensemble_run.result()).value
            print("\n=== Example 5: fan-out + merge ===")
            print(ensemble_text)

            # ---- Example 5b: composer-in-composer -------------------
            # The outer pipeline's first node is the fan-out + merge
            # from above; its second node is the summarizer.  The v1.0
            # ``Composable`` protocol type-checks this nesting for free
            # — both layers obey the same ``invoke(input) -> Run``
            # contract.
            nested = compose([
                compose({optimist, pessimist}, merge=_join_string_values),
                summarizer,
            ])
            nested_run = nested.invoke("the rise of generative AI")
            nested_text = (await nested_run.result()).value
            print("\n=== Example 5b: composer-in-composer ===")
            print(nested_text)

    print("\nSUCCESS: composition ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
