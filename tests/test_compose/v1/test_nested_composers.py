"""Tests for nested Composer composition."""

from __future__ import annotations

from collections import Counter
from typing import Any

from t0_agents import conditional, consensus, race, sequential
from tests.test_compose.v1._helpers import MockComposable


async def test_three_level_nest_resolves_through_layers() -> None:
    fast = MockComposable(name="fast", value="fast_won", delay=0.005)
    slow = MockComposable(name="slow", value="slow_won", delay=0.5)

    happy_voter_1 = MockComposable(name="v1", value="happy")
    happy_voter_2 = MockComposable(name="v2", value="happy")
    sad_voter = MockComposable(name="v3", value="sad")

    happy_path = consensus(
        happy_voter_1, happy_voter_2, sad_voter,
        merge=lambda values: Counter(values).most_common(1)[0][0],
    )
    sad_path = MockComposable(name="sad", value="SAD_PATH")

    router = conditional(
        router=lambda input: "happy" if "won" in str(input) else "sad",
        routes={"happy": happy_path, "sad": sad_path},
    )

    composer = sequential(race(fast, slow), router)
    result = await composer.invoke("seed", cancel_timeout=1.0).result()

    # The router saw "fast_won" → "happy" → consensus → "happy".
    assert result.value == "happy"


async def test_nested_inference_ids_aggregate_through_layers() -> None:
    a = MockComposable(name="a", value="x", inference_ids=["id_a"])
    b = MockComposable(name="b", value="y", inference_ids=["id_b"])
    c = MockComposable(name="c", value="z", inference_ids=["id_c"])

    # sequential(parallel(a, b), c) — parallel collects [x, y], chains to c.
    from t0_agents import parallel

    composer = sequential(parallel(a, b), c)
    result = await composer.invoke("seed").result()

    # All three inference ids present.
    assert "id_a" in result.inference_ids
    assert "id_b" in result.inference_ids
    assert "id_c" in result.inference_ids


async def test_nested_composer_satisfies_composable_protocol() -> None:
    """A nested Composer can itself be a child of another Composer."""
    from t0_agents.compose.topology import Composable, Composer

    inner = race(MockComposable(name="a"), MockComposable(name="b"))
    assert isinstance(inner, Composer)
    assert isinstance(inner, Composable)

    outer = sequential(inner, MockComposable(name="c"))
    assert isinstance(outer, Composer)
    assert isinstance(outer, Composable)


async def test_three_level_inference_id_provenance() -> None:
    """Three-level nesting still aggregates inference_ids correctly."""
    from t0_agents import parallel

    leaf_1 = MockComposable(name="l1", value=1, inference_ids=["i1"])
    leaf_2 = MockComposable(name="l2", value=2, inference_ids=["i2"])
    leaf_3 = MockComposable(name="l3", value=3, inference_ids=["i3"])
    leaf_4 = MockComposable(name="l4", value=4, inference_ids=["i4"])

    # Layer 1: parallel(leaf_1, leaf_2)  → list[int]
    # Layer 2: sequential(layer_1, leaf_3) → leaf_3 gets list[int] as input
    # Layer 3: sequential(layer_2, leaf_4) → final value = leaf_4.value
    layer_1 = parallel(leaf_1, leaf_2, reducer=lambda rs: [r.value for r in rs])
    layer_2 = sequential(layer_1, leaf_3)
    layer_3 = sequential(layer_2, leaf_4)

    result: Any = await layer_3.invoke("seed").result()
    assert set(result.inference_ids) == {"i1", "i2", "i3", "i4"}
