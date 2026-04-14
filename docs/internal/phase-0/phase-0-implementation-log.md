# Phase 0 Implementation Log

> Bridges the design in `docs/implementation-plan/` (M0 in the meta-plan, P0.x in the
> validation checklist) with what we actually shipped, so future work has a record of
> where we diverged and why.

## What we built

| File | Purpose |
|---|---|
| `docker-compose.yml` | Boots the TensorZero gateway with ClickHouse |
| `config/tensorzero.toml` | Chat function `agent` over Groq `gpt-oss-20b` |
| `.env.example` | `GROQ_API_KEY=` template |
| `tests/test_infrastructure.py` | Four validations covering P0.2 / P0.3 / P0.4 / P0.5 |
| `tests/conftest.py` (edit) | Tightened generic to `Client[Any]` so mypy stays clean |
| `.gitignore` (edit) | Anchored the `0X-*.md` design-doc patterns to repo root and allowed `.env.example` |

## Decisions taken during planning (vs. what the docs offered)

| Topic | Plan offered | We chose | Why |
|---|---|---|---|
| DB backend | Postgres (phase-0 guide) or ClickHouse (meta-plan) | **ClickHouse only** | Phase 6 (observability with `dynamic_tools`) expects ClickHouse; one DB is enough |
| LLM provider | OpenAI gpt-4o-mini (reference docs) | **Groq `gpt-oss-20b`** | We have Groq credentials; matches the user's existing `[models.*]` pattern |
| Function name | `generate_haiku` (phase-0 guide) | **`agent`** | Forward-compatible with Phase 3's agent loop |
| P0.2 API | Native SDK or OpenAI-compat | **Native SDK only** | Same code path Phase 1+ will exercise; uses `dryrun=True` |

## Divergences discovered during implementation (not in the plan)

1. **ClickHouse image name was wrong**. First attempt used `clickhouse/clickhouse-server:lts` — that tag does not exist on Docker Hub (`manifest unknown`). Correct image is `clickhouse:lts` from the Docker Hub library namespace, matching TensorZero's official MCP-example compose.

2. **Gateway version had to be pinned**. `tensorzero/gateway:latest` resolved to a stale `2025.10.7` image cached on the host, which rejected the `namespace` field sent by SDK `2026.4.0` (`HTTP 400 unknown field 'namespace'`). Pinned to `tensorzero/gateway:2026.4.0` so the image always matches the SDK pinned in `pyproject.toml`.

3. **`CLICKHOUSE_DB` env var removed**. The gateway creates the `tensorzero` database itself via auto-migrations on first boot — declaring it on the ClickHouse side was redundant.

4. **Health endpoint reports `"postgres":"ok"` even without Postgres**. Observed but harmless: the 2026.x gateway probes Postgres but tolerates its absence. No action needed for our ClickHouse-only setup.

5. **Reasoning models break the literal M0.4 check**. The meta-plan asserts `response.content[0] is Text`. Groq's `gpt-oss-20b` is a reasoning model and emits a `Thought` block before the final `Text`. The test was widened to "at least one `Text` block with non-empty text" — honors the spirit of M0.4 while supporting reasoning models. If we ever need to suppress the `Thought`, add `extra_body = [{ pointer = "/reasoning_format", value = "hidden" }]` to the variant in `config/tensorzero.toml`.

6. **mypy hardening on the test surface**. Three small adjustments not anticipated by the plan:
   - `Client[Any]` generic param in `tests/conftest.py` and `tests/test_infrastructure.py` (the `fastmcp.Client` import is generic).
   - `AsyncTensorZeroGateway.build_http()`'s `.pyi` returns `Union[Awaitable[Gateway], Gateway]`, so we `cast(Awaitable[AsyncTensorZeroGateway], ...)` before `await`.
   - `inference()` return type is also a Union — narrowed with `isinstance(response, ChatInferenceResponse)` before touching `.content`.

## Final status

| Item | Validation | Result |
|---|---|---|
| P0.1 / M0.1 | `curl /health` → 200 with `clickhouse:"ok"` | ✅ |
| P0.2 / M0.4 | `pytest -m integration` passes (`agent` returns Text via Groq) | ✅ |
| P0.3 / M0.2 | FastMCP server with 2 `@mcp.tool` tools | ✅ |
| P0.4 / M0.3 | `Client.list_tools()` returns `{add, greet}` | ✅ |
| P0.5 / M0.3 | `Client.call_tool("add", {a:5,b:3}).data == 8` | ✅ |
| P0.6 / M0.5 | `pyproject.toml:41` has `asyncio_mode = "auto"` | ✅ |

Phase 0 closed. Ready for **M1 — Tools Adapter** (FastMCP → T0).
