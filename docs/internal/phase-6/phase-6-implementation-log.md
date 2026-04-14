# Phase 6 Implementation Log

> Bridges the design in `docs/implementation-plan/` (M6 in the meta-plan, F6.x
> in the validation checklist) with what we actually shipped. Scope of this
> phase is **observability and feedback**: adding TensorZero metrics
> configuration, validating that all four feedback types (boolean, float,
> comment, demonstration) are accepted by the gateway, verifying dynamic tools
> land in ClickHouse, and adding structured logging to the agent loop.

## What we built

| File | Purpose |
|---|---|
| `config/tensorzero.toml` (edit) | Added `[metrics.task_success]` (boolean, episode, max) and `[metrics.quality_score]` (float, inference, max) for F6.2 and F6.3. |
| `src/t0_agents/__init__.py` (edit) | Added `import logging` and `logging.getLogger("t0_agents").addHandler(logging.NullHandler())` per Python library best practice (prevents "No handler found" warnings). |
| `src/t0_agents/mcp/loop.py` (edit) | Added `import logging`, module-level `logger = logging.getLogger(__name__)`, and structured log statements throughout `_execute_single_tool_call` (DEBUG for tool dispatch/result, ERROR for failures) and `run_agent` (INFO for turn summaries and loop lifecycle, DEBUG for tool-call dispatch counts, WARNING before max-turns raise). |
| `tests/_fake_t0.py` (edit) | Added `FakeFeedbackResponse` dataclass (with `feedback_id: UUID`) and `feedback(**kwargs)` method to `FakeT0` that records calls in `self.feedback_calls`. |
| `tests/test_mcp/test_observability.py` | **NEW.** Nine Phase-6 unit tests across three classes: `TestObservabilityIds` (F6.1), `TestFeedback` (F6.2-F6.5), `TestLogging` (F6.7). Uses `FakeT0`/`FakeMCPClient` for IDs and feedback, pytest `caplog` for logging. |
| `tests/test_mcp/test_integration.py` (edit) | Added a `--- Phase 6: observability and feedback ---` block with six integration tests: `test_f6_1_ids_present_in_agent_result`, `test_f6_2_boolean_feedback_accepted_by_gateway`, `test_f6_3_float_feedback_accepted_by_gateway`, `test_f6_4_comment_feedback_accepted_by_gateway`, `test_f6_5_demonstration_feedback_accepted_by_gateway`, `test_f6_6_dynamic_tools_recorded_in_clickhouse`. |
| `docs/implementation-plan/02-validation-checklist-v3-final.md` (edit) | Ticked F6.1-F6.7 with inline test-function references (Phase 2-5 convention). Bumped Progress summary row for Phase 6 to `7 / 7` checkmark. |

## Decisions taken during planning

| Topic | Options | We chose | Why |
|---|---|---|---|
| F6.2-F6.5 feedback test strategy | dryrun=True on both / dryrun=False to verify ClickHouse / unit only | **dryrun=False on inference + dryrun=True on feedback** (automated) + **manual snippet with dryrun=False on feedback** | The feedback endpoint validates that the target ID exists in ClickHouse, so inference must persist (dryrun=False). Feedback uses dryrun=True to avoid polluting ClickHouse with test feedback rows. Manual snippet in this log for full ClickHouse verification. User-confirmed. |
| F6.6 verification | SDK get_inferences / direct ClickHouse HTTP / manual only | **Direct ClickHouse HTTP query via httpx** | The SDK's `get_inferences()` does not expose the `dynamic_tools` column. `httpx` is already available as a transitive dep. Credentials from `docker-compose.yml`. User-confirmed. |
| F6.7 logging style | Standard logging INFO+DEBUG / INFO only / structured JSON | **Standard logging, INFO+DEBUG** | INFO for turn summaries (inference_id, episode_id, turn count). DEBUG for tool call details (name, args, result). ERROR for tool failures. Plain text, no JSON — structured logging is an application-level concern. `NullHandler` in `__init__.py` per Python library convention. User-confirmed. |
| F6.1 scope | Test-only / code change | **Test-only** | IDs are already captured at `loop.py:221-222` and returned in `AgentResult`. Phase 6 adds explicit F6.1-labelled tests for the validation checklist. |
| Unit test layout | Extend test_loop.py / new module | **New `tests/test_mcp/test_observability.py`** | Matches Phase 4/5 "module per milestone" precedent. |

## Divergences discovered during implementation

1. **Feedback endpoint requires the target ID to exist in ClickHouse.** The
   initial plan called for `dryrun=True` on both inference and feedback. When
   inference uses `dryrun=True`, the IDs are generated but never stored in
   ClickHouse, so `t0.feedback()` rejects them with `"Episode ID: ... does not
   exist"` (HTTP 400). The fix: use `dryrun=False` on inference so the IDs
   persist, then `dryrun=True` on feedback to validate the request structure
   without storing feedback rows. This diverges from the initial plan but is
   the correct approach given the gateway's validation behavior.

2. **F6.6 ClickHouse query returns tool definitions as JSON strings in an
   array.** The `dynamic_tools` column in `ChatInference` is
   `Array(String)`, where each element is a JSON-encoded tool definition.
   The integration test parses each entry and extracts the `name` field to
   verify that the test server's tools (at minimum `add`) are present.

3. **No changes to `AgentResult` were needed.** The existing dataclass
   (`response`, `episode_id`, `inference_ids`, `turns`) already carries all
   the observability data Phase 6 requires. Feedback is a caller concern
   (not part of `run_agent`), so no feedback-related fields were added.

4. **`FakeFeedbackResponse` is a minimal duck-type.** The real
   `tensorzero.types.FeedbackResponse` has only one field (`feedback_id: UUID`).
   The fake mirrors this exactly. The `FakeT0.feedback()` method uses
   `**kwargs` to match the real SDK's keyword-only signature without
   duplicating every parameter.

5. **All 24 integration tests passed on the first full-suite run** (no Groq
   flakes in this run). The F6.2-F6.6 tests are deterministic because they
   don't depend on specific LLM tool-call behavior beyond "use the add tool"
   -- the feedback and ClickHouse queries happen after `run_agent` completes.

## Manual ClickHouse verification (F6.2-F6.5 with dryrun=False)

The automated integration tests use `dryrun=True` on feedback calls.
To verify that feedback rows actually persist in ClickHouse, run this
snippet from a Python REPL (`uv run python -i`). It exercises all four
feedback types with `dryrun=False`.

```python
import asyncio
from collections.abc import Awaitable
from typing import Any, cast

import httpx
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import run_agent
from tests.server import mcp


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)
    async with await gateway as t0, mcp_client:
        result = await run_agent(
            t0, mcp_client,
            "Please use the add tool to compute 2 + 3.",
            "agent",
            dryrun=False,
            max_turns=4,
        )
        print(f"episode_id:    {result.episode_id}")
        print(f"inference_ids: {result.inference_ids}")

        # F6.2: Boolean feedback on episode
        r = await t0.feedback(
            metric_name="task_success",
            episode_id=result.episode_id,
            value=True,
        )
        print(f"\n[F6.2] Boolean feedback_id: {r.feedback_id}")

        # F6.3: Float feedback on inference
        r = await t0.feedback(
            metric_name="quality_score",
            inference_id=result.inference_ids[0],
            value=9.0,
        )
        print(f"[F6.3] Float feedback_id:   {r.feedback_id}")

        # F6.4: Comment feedback
        r = await t0.feedback(
            metric_name="comment",
            inference_id=result.inference_ids[-1],
            value="Tool selection was appropriate.",
        )
        print(f"[F6.4] Comment feedback_id: {r.feedback_id}")

        # F6.5: Demonstration feedback
        r = await t0.feedback(
            metric_name="demonstration",
            inference_id=result.inference_ids[-1],
            value="The sum of 2 and 3 is 5.",
        )
        print(f"[F6.5] Demo feedback_id:    {r.feedback_id}")

    # Verify in ClickHouse
    ch_params = {"user": "chuser", "password": "chpassword", "database": "tensorzero"}
    async with httpx.AsyncClient() as http:
        for table, target_id in [
            ("BooleanMetricFeedback", result.episode_id),
            ("FloatMetricFeedback", result.inference_ids[0]),
            ("CommentFeedback", result.inference_ids[-1]),
            ("DemonstrationFeedback", result.inference_ids[-1]),
        ]:
            resp = await http.get(
                "http://localhost:8123",
                params={
                    "query": f"SELECT * FROM {table} WHERE target_id = '{target_id}' "
                             f"OR inference_id = '{target_id}' FORMAT JSON",
                    **ch_params,
                },
            )
            rows = resp.json().get("data", [])
            print(f"\n[ClickHouse] {table}: {len(rows)} row(s)")
            for row in rows:
                print(f"  {row}")


asyncio.run(main())
```

What to look for:

- Each `feedback_id` is a UUID (not an error).
- Each ClickHouse table has 1 row matching the target ID.
- `BooleanMetricFeedback.value` is `true`.
- `FloatMetricFeedback.value` is `9.0`.
- `CommentFeedback.value` is the comment text.
- `DemonstrationFeedback.value` is the ideal output text.

## Final status

| Item | Validation | Result |
|---|---|---|
| F6.1 / M6.1 (IDs captured) | Unit: `TestObservabilityIds::test_f6_1_ids_captured_on_every_turn` -- 3-turn FakeT0 flow, asserts `episode_id` is UUID, `len(inference_ids) == turns`, all distinct UUIDs. Integration: `test_f6_1_ids_present_in_agent_result` -- live gateway, asserts `episode_id` and `inference_ids` non-empty, length matches `turns`. | checkmark |
| F6.2 / M6.2 (Boolean feedback) | Unit: `TestFeedback::test_f6_2_boolean_feedback_on_episode` + `::test_f6_2_boolean_feedback_false_value` -- FakeT0 records correct params. Integration: `test_f6_2_boolean_feedback_accepted_by_gateway` -- live gateway accepts `metric_name="task_success"`, `value=True`, returns `feedback_id`. | checkmark |
| F6.3 (Float feedback) | Unit: `TestFeedback::test_f6_3_float_feedback_on_inference` -- FakeT0 records `value=8.5` on `inference_id`. Integration: `test_f6_3_float_feedback_accepted_by_gateway` -- live gateway accepts `metric_name="quality_score"`. | checkmark |
| F6.4 (Comment feedback) | Unit: `TestFeedback::test_f6_4_comment_feedback` -- FakeT0 records `metric_name="comment"`. Integration: `test_f6_4_comment_feedback_accepted_by_gateway` -- live gateway accepts built-in `comment` metric. | checkmark |
| F6.5 (Demonstration feedback) | Unit: `TestFeedback::test_f6_5_demonstration_feedback` -- FakeT0 records `metric_name="demonstration"` on `inference_id`. Integration: `test_f6_5_demonstration_feedback_accepted_by_gateway` -- live gateway accepts built-in `demonstration` metric. | checkmark |
| F6.6 (Dynamic tools in ClickHouse) | Integration: `test_f6_6_dynamic_tools_recorded_in_clickhouse` -- `run_agent(dryrun=False)` then ClickHouse HTTP query verifies `dynamic_tools` array is non-empty and contains `"add"` tool definition. | checkmark |
| F6.7 / M6.3 (Structured logging) | Unit: `TestLogging::test_f6_7_logging_shows_tool_calls_and_results` (DEBUG records contain tool name and result), `::test_f6_7_logging_shows_errors` (ERROR record on ToolError), `::test_f6_7_logging_shows_turn_info` (INFO records with turn number, inference_id, loop lifecycle). All via `caplog`. | checkmark |

Test totals after Phase 6: **105 total** (81 unit + 24 integration). Breakdown:
- 81 unit tests: 72 pre-existing + 9 new Phase-6 (`test_observability.py`)
- 24 integration tests: 18 pre-existing + 6 new Phase-6

`uv run ruff check src/ tests/`: clean. `uv run mypy src/ tests/`: clean.
All 24 integration tests passed on first run (no Groq flakes this session).

Phase 6 closed. All seven phases (0-6) of the implementation plan are now
complete.
