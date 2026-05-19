# 06 Feedback Loop

Capture observability data from agent runs and send structured feedback to TensorZero for model optimization.

## What TensorZero tracks automatically

Every call to `run_agent` returns an `AgentResult` containing:

- **`episode_id`** -- A unique identifier for the entire conversation. TensorZero allocates this on the first inference and reuses it for all subsequent turns in the same `run_agent` call.
- **`inference_ids`** -- One ID per inference turn. Use these to attach per-turn feedback (e.g., rating the quality of a specific response).
- **`turns`** -- How many inference rounds the loop took.

These IDs are stored in ClickHouse and link inferences to feedback for downstream optimization.

## The feedback API

TensorZero supports three feedback types, each declared as a metric in `tensorzero.toml`:

| Type | Level | Value | Use case |
|------|-------|-------|----------|
| `boolean` | `episode` | `True` / `False` | Was the overall task successful? |
| `float` | `inference` | `0.0` -- `1.0` | Quality score for a specific turn |
| `comment` | `episode` | Free-form string | Human reviewer notes |

Send feedback via the TensorZero SDK:

```python
# Boolean (episode-level)
await t0.feedback(episode_id=result.episode_id, metric_name="task_success", value=True)

# Float (inference-level)
await t0.feedback(inference_id=result.inference_ids[-1], metric_name="quality_score", value=0.9)

# Comment (episode-level)
await t0.feedback(episode_id=result.episode_id, metric_name="comment", value="Looks good.")
```

## Metric declarations

Every metric name used in `t0.feedback()` must be declared in `tensorzero.toml`. The gateway rejects feedback for undeclared metrics.

```toml
[metrics.task_success]
type = "boolean"
level = "episode"
optimize = "max"

[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"

[metrics.comment]
type = "comment"
level = "episode"
```

## The dryrun caveat

When `run_agent(..., dryrun=True)` is used, TensorZero generates inference IDs but does not persist them in ClickHouse. Any subsequent `t0.feedback()` call targeting those IDs will fail with HTTP 400 (`"Episode ID: ... does not exist"`).

Use `dryrun=False` (the default) when you intend to send feedback after the agent run.

## How TensorZero uses feedback

Feedback stored in ClickHouse is the foundation for TensorZero's optimization features:

- **Model routing** -- Route future inferences to the variant that performs best on your metrics.
- **Fine-tuning** -- Use episode-level boolean feedback to build training datasets.
- **Analytics** -- Query ClickHouse directly to analyze agent performance over time.

## Prerequisites

- Python 3.11+
- `uv sync` (install dependencies)
- `GROQ_API_KEY` in your environment

## Run

```bash
docker compose -f examples/13_observability/docker-compose.yml up -d
uv run python examples/13_observability/main.py
docker compose -f examples/13_observability/docker-compose.yml down
```

## What to observe

1. The agent calls `search_docs`, then `get_doc_content` to fulfill the query.
2. `AgentResult` prints the `episode_id` and `inference_ids` from the run.
3. Three feedback calls succeed and print their `feedback_id` values.
4. The feedback data is now queryable in ClickHouse at `http://localhost:8123`.
