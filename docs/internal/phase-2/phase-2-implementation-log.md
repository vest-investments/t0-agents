# Phase 2 Implementation Log

> Bridges the design in `docs/implementation-plan/` (M2 in the meta-plan, F2.x in
> the validation checklist) with what we actually shipped. Scope of this phase
> is the **results adapter** (FastMCP → TensorZero) — serializing a FastMCP
> `CallToolResult` into the `str` TensorZero expects, and wrapping it in a
> `ToolResult` keyed by the original `ToolCall`'s `name` and `id`.

## What we built

| File | Purpose |
|---|---|
| `src/t0_agents/mcp/adapter.py` (edit) | Implemented `serialize_tool_result` and `build_tool_result`; replaced the Phase-2 `NotImplementedError` stubs. Added `import json`. |
| `tests/test_mcp/test_adapter.py` (edit) | Populated `TestSerializeToolResult` (15 cases covering F2.2–F2.6 plus edges) and `TestBuildToolResult` (3 cases covering F2.7 shape + raw_name fallback). |
| `tests/test_mcp/test_integration.py` (edit) | Added `test_add_tool_roundtrip_final_text` (F2.1 + F2.7 + F2.8 end-to-end) and `test_search_pydantic_roundtrip` (F2.1 + F2.3 + F2.7). |
| `docs/implementation-plan/02-validation-checklist-v3-final.md` (edit) | Checked F2.1–F2.8 and bumped the Progress summary row for Phase 2 to 8/8. |

No changes to `src/t0_agents/mcp/loop.py` — those stubs (`run_agent`, `execute_tool_calls`, `has_tool_calls`) belong to Phase 3. The two-turn flow needed for F2.7/F2.8 lives inline in the integration tests.

## Decisions taken during planning

| Topic | Options | We chose | Why |
|---|---|---|---|
| Loop scope | inline in tests / mini-loop in `loop.py` / private helpers in the test module | **Inline in the integration tests** | Keeps Phase 2 strictly scoped to adapter functions; `loop.py` stays a clean Phase-3 surface. |
| F2.8 validation | automated test against `tests/server.py` / + manual against a local MCP server / manual only | **Automated only** | Matches the Phase-1 pattern (`dryrun=True` + existing test server). The user's local MCP servers are available if needed but not required for CI. |
| Error prefix | `"ERROR: "` (FastMCP guide §3.2) / `"Error: "` (integration contract §4.3) | **`"ERROR: "`** | Caps are more explicit for the LLM and the FastMCP guide is the Phase-2 specific reference. |
| Primitive serialization | `str(data)` (Phase-2 guide §3.2) / `json.dumps(data, default=str)` (meta-plan M2.1 + integration contract §4.3) | **`json.dumps(data, default=str, ensure_ascii=False)`** | Initially the entry was answered with `str(data)` based on a misframed interview question that did not surface the meta-plan/guide conflict. Realigned to the meta-plan after the post-implementation double-check. Bool now serializes as JSON-cased (`"true"` / `"false"`) instead of Python-cased (`"True"` / `"False"`). |
| JSON options | `ensure_ascii=True` (default) / `ensure_ascii=False` | **`ensure_ascii=False`** | Preserves Unicode (accents, non-ASCII) verbatim in tool output sent to the LLM. Both the FastMCP guide and the integration contract converge on this. |
| Fallback join | `"\n".join(...)` / `" ".join(...)` / first block only | **`"\n".join(...)`** | Matches every reference snippet in the Phase-2 guides. |

## Divergences discovered during implementation

1. **`build_tool_result` uses `tool_call.name or tool_call.raw_name`** to reconcile the type gap between `ToolCall.name: Optional[str]` (from `tensorzero/types.py:152`) and `ToolResult.name: str` (from `tensorzero/types.py:186`). `raw_name` is never `None`. This keeps mypy strict happy without an `assert` or `cast` and gives us a cheap defense against LLM hallucinations of tool names. Phase 3's `execute_tool_calls` is where hallucinations will be *rejected* (ToolResult with an error payload) — Phase 2 only needs to produce a well-shaped `ToolResult`.

2. **F2.6 is validated only by unit tests, not by an integration test.** `tests/server.py` does not expose any tool that fails on purpose, and adding one would expand the Phase-2 surface beyond the checklist. The serializer's error path is therefore exercised by five unit tests (`test_serialize_is_error_*`). When Phase 5 (Robustness) arrives with F5.1–F5.8, we can add a failing test tool and cover F2.6 end-to-end too.

3. **`test_search_pydantic_roundtrip` does not assert F2.8 as strictly as `test_add_tool_roundtrip_final_text`.** The `search` tool in `tests/server.py` is typed `-> list[dict]` (no type parameters on `dict`), so FastMCP's content-block path sometimes stringifies the structured payload as `["Root()"]`. The LLM reasonably reacts by asking for another tool call rather than emitting text, which makes an `isinstance(block, Text)` assertion flaky. F2.8 is therefore validated end-to-end by the `add` roundtrip, and the `search` roundtrip focuses on F2.1 + F2.3 + F2.7 (that the serialized `list[dict]` round-trips through T0 without raising). If we tighten the `search` return annotation in Phase 3 (to e.g. `list[dict[str, Any]]` or a Pydantic model), this test can be strengthened.

4. **`ensure_ascii=False` matters for non-ASCII payloads.** `test_serialize_dict_preserves_unicode` pins this behavior: a payload with Spanish characters (`"hóla ñ"`) survives serialization literally instead of being escaped to `"h\u00f3la \u00f1"`. This is a strict improvement for LLM consumption — fewer escape sequences, more natural-language context.

5. **`fastmcp.client.client.CallToolResult` is a `@dataclass`**, not a Pydantic model. This means fixture construction in unit tests is trivial: the `_make_result(...)` helper just instantiates the dataclass directly. Verified at `.venv/lib/python3.12/site-packages/fastmcp/client/client.py:114`. The import remains via the internal path (`from fastmcp.client.client import CallToolResult`) — same caveat as Phase 1 divergence #1.

6. **Error content is extracted with `hasattr(block, "text")`** rather than `isinstance(block, TextContent)`. FastMCP/MCP can return `ImageContent`, `AudioContent`, or `EmbeddedResource` alongside text, and we want to (a) silently skip non-textual blocks in the fallback, (b) avoid importing the full `ContentBlock` union for a runtime type check. `TextContent` is the only MCP content type that exposes `.text`; the check is therefore precise.

7. **Post-implementation realignment with meta-plan M2.1 (booleans).** The first cut of `serialize_tool_result` had three explicit branches for `.data` non-`str`: `(int, float, bool)` via `str(data)`, `dict` via `json.dumps`, and a fallback `try`/`except json.dumps`. That followed the Phase-2 FastMCP guide §3.2 literally but diverged from meta-plan v3 M2.1 and integration contract §4.3, which converge on a single non-`str` branch: `json.dumps(data, default=str, ensure_ascii=False)`. The divergence was only visible for booleans (`str(True) == "True"` vs `json.dumps(True) == "true"`); for `int` and `float` both paths produced the same output. After flagging the conflict during the M2 double-check, we collapsed the implementation to the two-branch shape the meta-plan prescribes (str passthrough + `json.dumps`) and updated `test_serialize_bool_as_json` (renamed from `test_serialize_bool_via_str`) to expect `"true"` / `"false"`. The defensive `try`/`except` around `json.dumps` is retained — meta-plan M2.1 does not mention it, but it costs nothing and protects the loop from exotic objects whose `__str__` raises.

## Manual sanity-check against a local MCP server

The automated tests use the in-memory `tests/server.py`. If you want to verify Phase 2 against one of your real MCP servers, run the snippet below from a Python REPL or a one-off script (`uv run python -i`). It exercises the full two-turn flow: tool discovery → inference #1 (forces a tool call) → execute through FastMCP → `serialize_tool_result` + `build_tool_result` → inference #2 (LLM consumes the result).

```python
# adapt the URL / command and the forced tool name to your MCP server
import asyncio
from collections.abc import Awaitable
from typing import cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse

from t0_agents.mcp import build_tool_result, convert_tools


TOOL_TO_FORCE = "your_tool_name"  # must exist in your MCP server
USER_PROMPT = "Do something that requires the tool above."


async def main() -> None:
    # Pick ONE of these transports:
    #   HTTP:   Client("https://your-mcp-server.example.com/mcp")
    #   stdio:  Client("/abs/path/to/your_server.py")
    #   config: Client({"mcpServers": {"x": {"command": "python", "args": ["..."]}}})
    async with Client("http://localhost:8000/mcp") as mcp_client:
        t0_tools = convert_tools(await mcp_client.list_tools())
        print(f"[discover] {[t['name'] for t in t0_tools]}")

        gateway = cast(
            Awaitable[AsyncTensorZeroGateway],
            AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
        )
        async with await gateway as t0:
            first = await t0.inference(
                function_name="agent",
                input={"messages": [{"role": "user", "content": USER_PROMPT}]},
                additional_tools=t0_tools,
                tool_choice={"specific": TOOL_TO_FORCE},
                dryrun=True,
            )
            assert isinstance(first, ChatInferenceResponse)
            tc = next(b for b in first.content if isinstance(b, ToolCall))
            print(f"[F2.1] ToolCall: {tc.name}({tc.arguments})")

            mcp_result = await mcp_client.call_tool(
                tc.name, tc.arguments, raise_on_error=False,
            )
            print(f"[F2.x] is_error={mcp_result.is_error} data={mcp_result.data!r}")

            t0_result = build_tool_result(tc, mcp_result)
            print(f"[F2.7] ToolResult(name={t0_result.name!r}, "
                  f"result={t0_result.result[:120]!r}, id={t0_result.id!r})")

            second = await t0.inference(
                function_name="agent",
                episode_id=first.episode_id,
                input={"messages": [
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": first.content},
                    {"role": "user", "content": [t0_result]},
                ]},
                additional_tools=t0_tools,
                dryrun=True,
            )
            assert isinstance(second, ChatInferenceResponse)
            final = " ".join(b.text for b in second.content if isinstance(b, Text))
            print(f"[F2.8] final text: {final!r}")


asyncio.run(main())
```

What to look for:

- `[F2.1]` — the first turn contains a `ToolCall` and `tc.arguments is not None`.
- `[F2.x]` — `is_error` is `False` for a happy-path tool; `data` shape depends on whether your tool has an output schema.
- `[F2.7]` — the `ToolResult` has a non-empty `result: str`, and `name`/`id` match the original `ToolCall`.
- `[F2.8]` — the second turn produces at least one `Text` block that makes sense given the tool's output.

If `[F2.8]` produces only thoughts or another `ToolCall`, your tool's output was likely ambiguous for the model — inspect `t0_result.result` to see what the LLM actually received.

## Final status

| Item | Validation | Result |
|---|---|---|
| F2.1 / M2.1 | `call_tool(name, arguments)` executes with T0's validated `ToolCall.arguments` (`dict`) — covered by `test_add_tool_roundtrip_final_text` and `test_search_pydantic_roundtrip` | ✅ |
| F2.2 / M2.3 | `.data: str` passthrough; `int`/`float`/`bool` via `json.dumps(data, default=str)` (booleans become JSON-cased `"true"`/`"false"` per meta-plan M2.1) — `test_serialize_str_passthrough`, `test_serialize_int_as_json`, `test_serialize_float_as_json`, `test_serialize_bool_as_json` | ✅ |
| F2.3 / M2.3 | `.data: dict` → `json.dumps(..., default=str, ensure_ascii=False)`; list and Unicode covered as well — `test_serialize_dict_as_json`, `test_serialize_list_as_json`, `test_serialize_dict_preserves_unicode` | ✅ |
| F2.4 / M2.3 | `.data is None` → concatenate `.content[*].text` — `test_serialize_none_fallback_to_content_text`, `test_serialize_none_multiple_content_blocks_joined`, `test_serialize_none_no_content_returns_empty` | ✅ |
| F2.5 / M2.3 | Complex types (`datetime`, `UUID`) survive via `default=str` — `test_serialize_complex_datetime_via_default_str`, `test_serialize_complex_uuid_via_default_str` | ✅ |
| F2.6 / M2.3 | `.is_error=True` → `"ERROR: "` prefix; error content wins over `.data`; empty content falls back to `"Unknown tool error"` — `test_serialize_is_error_*` (5 cases). End-to-end exercise is deferred to Phase 5 (see divergence #2) | ✅ |
| F2.7 / M2.2 | `ToolResult(name, result, id)` is a valid shape T0 accepts in a follow-up `inference()` — `test_add_tool_roundtrip_final_text` and `test_search_pydantic_roundtrip` both complete the second inference without error; `raw_name` fallback covered by unit test | ✅ |
| F2.8 / M2.4 | After receiving the `ToolResult`, the LLM emits a coherent `Text` block that incorporates the tool's output — `test_add_tool_roundtrip_final_text` pins `"5"` in the final answer | ✅ |

Test totals after Phase 2: **31 passed** (Phase 1's 10 + 15 Phase-2 serializer unit tests + 3 Phase-2 `build_tool_result` unit tests + 2 Phase-2 integration tests + 1 Phase-1 infrastructure test already counted). Ruff and mypy strict both clean.

Phase 2 closed. Ready for **M3 — Agent Loop** (implement `run_agent`, `execute_tool_calls`, `has_tool_calls` in `src/t0_agents/mcp/loop.py`).
