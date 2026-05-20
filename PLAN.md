# Optional MCP `tool_provider` in t0-agents core

> Make `tool_provider` truly optional throughout the v1.0 agent stack so devs running pure chat (no tools) don't pay any MCP plumbing cost. Rename the `run_agent` and streaming-runner parameters from `mcp_client` to `tool_provider`, allow `Client[Any] | None = None`, drop the ephemeral `FastMCP("agent-default")` fallback the `Agent` class allocates today, and surface the unlock through docs + a showcase example. v1.0 is still `Unreleased`, so the rename is a controlled breaking change called out in the changelog.

## Context

- **Current state of the half-unlock.** `Agent` already accepts `tool_provider=None` (commit `ee030bc`), but the gap is hidden inside the framework: `_resolve_chat_tool_provider` (`src/t0_agents/core/agent.py:41-59`) allocates an ephemeral `FastMCP("agent-default")` + `Client` per invocation. The low-level `run_agent` (`src/t0_agents/mcp/loop.py:204`) and `StreamingAgentLoop.stream` (`src/t0_agents/stream/runner.py:73`) still require a non-optional `Client[Any]` — toolless callers using the low-level API have to spin up FastMCP anyway, and every chat invoke pays one wasted `list_tools()` roundtrip.
- **Decisions already made (do not relitigate in milestones).**
  - Rename `mcp_client → tool_provider` in `run_agent` and `StreamingAgentLoop.stream` (breaking; v1.0 still `Unreleased`).
  - Type stays `Client[Any] | None = None`, not `ToolProvider` — importing the protocol from `core/types.py` would break the `mcp/` standalone rule (memory: `feedback_mcp_standalone.md`).
  - Delete `_resolve_chat_tool_provider` entirely; pass `ctx.tool_provider` straight through.
  - Ship a new minimal example at `examples/agent/00_chat_only/`.
- **Existing systems / files involved.**
  - `src/t0_agents/mcp/loop.py` — `run_agent`, `_execute_single_tool_call`, `execute_tool_calls`, `execute_tool_calls_parallel`.
  - `src/t0_agents/stream/runner.py` — `StreamingAgentLoop.stream`.
  - `src/t0_agents/core/agent.py` — `_resolve_chat_tool_provider`, `_chat_terminal`, `_chat_streaming_terminal`, `_single_shot_terminal` (already guards `self._tool_provider is None`).
  - `src/t0_agents/delegation/orchestrator.py` (or `src/t0_agents/core/middleware/builtins.py` — confirm during M1) — sub-agent installation that already raises `ConfigurationError`.
  - `examples/agent/01_first_agent/`, `examples/agent/03_output_modes/` — reference examples.
  - `tests/test_mcp/`, `tests/test_stream/`, `tests/test_core/test_agent.py` — existing tests; `test_toolless_chat_invoke_returns_text` is the seed.
  - `README.md`, `docs/getting-started.md`, `docs/agent.md`, `CLAUDE.md`, `CHANGELOG.md`.
- **Known constraints.**
  - Python ≥ 3.11, `mypy --strict`, ruff (`E, F, I, N, UP, RUF`), 100-char lines.
  - `mcp/` submodule must stay standalone (no imports from higher layers; external `fastmcp.Client` is OK).
  - Multi-commit PRs, 5–10 story-telling commits per PR (per `CLAUDE.md`); run the full pytest suite (no `-m` filter) before push.
  - User language: Spanish in conversation, English in code/docs/commits/PRs.

## Out of scope

- Deprecation alias for `mcp_client=` keyword. v1.0 is `Unreleased`; the breaking rename ships without a shim.
- Elicitation subsystem (`mcp/elicitation.py`, `core/elicitation_forwarder.py`). It legitimately needs a `Client`; remains opt-in.
- Glow (`src/t0_agents/glow/`). `App` owns and manages its own `Client`; optional buys nothing there.
- Annotating every existing example with "no tools needed". The new `00_chat_only/` example carries that pedagogy.
- Any performance work beyond removing the per-invoke `FastMCP("agent-default")` allocation that M3 deletes.

## Milestone overview

| # | Title | Optional? |
|---|---|---|
| 1 | `run_agent` accepts optional `tool_provider` | no |
| 2 | Streaming runner mirrors the optional contract | no |
| 3 | Agent class drops the ephemeral FastMCP fallback | no |
| 4 | Public docs, CHANGELOG, sub-agent error polish | no |
| 5 | `examples/agent/00_chat_only/` showcase | yes |

---

## Milestone 1 — `run_agent` accepts optional `tool_provider`

**Slug**: `run-agent-optional-tool-provider`

**Goal**: Devs calling `run_agent` directly can pass `tool_provider=None` (or omit it) for a chat-only inference loop with zero MCP plumbing.

**Scope**:
- Rename the second positional kwarg of `run_agent` from `mcp_client` to `tool_provider`; type becomes `Client[Any] | None = None`.
- Skip `list_tools()` when `tool_provider is None` (`t0_tools = []`).
- Add a defensive guard: if the LLM emits a `ToolCall` while `tool_provider is None`, raise a clear `T0AgentsFatalError`.
- Rename `mcp_client → tool_provider` inside the private helpers (`_execute_single_tool_call`, `execute_tool_calls`, `execute_tool_calls_parallel`) — they remain non-optional (only called inside the tool-call branch).
- New test file covering the toolless path and the hallucination guard.

**Checklist**:
- [ ] `src/t0_agents/mcp/loop.py` — rename `mcp_client` → `tool_provider` in `run_agent` signature; type `Client[Any] | None = None`; update docstring `Args:` block to describe the new contract (when `None`, no tools advertised, single inference + return).
- [ ] `src/t0_agents/mcp/loop.py:323` — `mcp_tools = await tool_provider.list_tools() if tool_provider is not None else []`.
- [ ] `src/t0_agents/mcp/loop.py:459` (tool-dispatch branch) — defensive guard: raise `T0AgentsFatalError` with explanatory message if `tool_provider is None` at the point of executing tool calls.
- [ ] `src/t0_agents/mcp/loop.py` — rename `mcp_client` → `tool_provider` in `_execute_single_tool_call`, `execute_tool_calls`, `execute_tool_calls_parallel` (signatures, docstrings, bodies).
- [ ] `tests/test_mcp/test_loop_toolless.py` — new file: `test_run_agent_toolless_returns_text` (mock gateway returns Text response; assert `list_tools` is never called); `test_run_agent_toolless_raises_on_hallucinated_tool_call` (mock returns a `ToolCall` block; assert `T0AgentsFatalError` with the expected message fragment).
- [ ] Existing `tests/test_mcp/test_*.py` — rename any `mcp_client=` keyword callers to `tool_provider=`. Positional callers stay unaffected.

**Depends on**: none

**Acceptance criteria**:
- [ ] `uv run pytest -m "not integration" tests/test_mcp/` passes.
- [ ] `uv run pytest -m "not integration"` (full offline suite) passes — no regression elsewhere.
- [ ] `uv run mypy src/t0_agents/mcp/` clean.
- [ ] `uv run ruff check src/t0_agents/mcp/ tests/test_mcp/` clean.
- [ ] `grep -rn "mcp_client" src/t0_agents/mcp/` returns zero hits.

**Notes**:
- Keep `Client[Any]` as the type, not `ToolProvider`. Importing the protocol from `core/types.py` would break the `mcp/` standalone rule.
- The defensive guard message must name the public kwarg (`tool_provider`), never the legacy `mcp_client`.
- The existing log line `"tools=%d"` at the start of the loop already prints `0` when the list is empty — nothing to change there.

---

## Milestone 2 — Streaming runner mirrors the optional contract

**Slug**: `stream-runner-optional-tool-provider`

**Goal**: `StreamingAgentLoop.stream` accepts `tool_provider: Client[Any] | None = None` with the same semantics as `run_agent`; streaming chatbots can run toolless.

**Scope**:
- Apply M1's exact contract to `StreamingAgentLoop.stream` and any helpers it owns.
- Same guard message, same defensive raise, same naming.
- New test file covering the streaming toolless path.

**Checklist**:
- [ ] `src/t0_agents/stream/runner.py:73` — change `tool_provider: Client[Any]` → `tool_provider: Client[Any] | None = None`; update docstring (lines 67–71 and the `stream` body docstring).
- [ ] `src/t0_agents/stream/runner.py:114` — `mcp_tools = await tool_provider.list_tools() if tool_provider is not None else []`.
- [ ] `src/t0_agents/stream/runner.py` (tool-dispatch branch ≈ line 245) — same defensive guard as `loop.py`.
- [ ] Rename `tool_provider` inside any streaming-specific helper that takes it; verify reuse of the M1 helpers when shared.
- [ ] `tests/test_stream/test_runner_toolless.py` — new file mirroring M1's two tests for the streaming case (assert no `list_tools` call when `None`; assert defensive raise on hallucinated tool call).
- [ ] Existing `tests/test_stream/*.py` — rename `tool_provider=` keyword callers if any pre-existed under the old name.

**Depends on**: milestone 1 (reuses M1's contract decisions — defensive-guard message format, naming, `Client[Any] | None` type). Read the M1 dev-log before implementing.

**Acceptance criteria**:
- [ ] `uv run pytest -m "not integration" tests/test_stream/` passes.
- [ ] `uv run pytest -m "not integration"` (full offline suite) passes.
- [ ] `uv run mypy src/t0_agents/stream/` clean.
- [ ] `uv run ruff check src/t0_agents/stream/ tests/test_stream/` clean.
- [ ] `python -c "import inspect; from t0_agents.mcp.loop import run_agent; from t0_agents.stream.runner import StreamingAgentLoop; a = inspect.signature(run_agent).parameters['tool_provider']; b = inspect.signature(StreamingAgentLoop.stream).parameters['tool_provider']; assert a.default is None and b.default is None"` exits 0 (signature parity).

**Notes**:
- The streaming tool-dispatch site has a different call shape than the non-streaming loop; confirm the exact line number during implementation.
- The `StreamingAgentLoop` docstring already mentions `tool_provider` (line 70) — that part of the rename is already done; finish the rest.

---

## Milestone 3 — Agent class drops the ephemeral FastMCP fallback

**Slug**: `agent-drop-ephemeral-fallback`

**Goal**: `Agent.invoke(...)` with `tool_provider=None` no longer allocates a `FastMCP("agent-default")` + `Client` per call; it forwards `None` straight into the core loop.

**Scope**:
- Delete `_resolve_chat_tool_provider` and its `FastMCP` import.
- Update `_chat_terminal` and `_chat_streaming_terminal` to pass `ctx.tool_provider` directly.
- Extend `test_toolless_chat_invoke_returns_text` to assert no `FastMCP` instance is constructed during a toolless invoke.
- Improve the sub-agent `ConfigurationError` message in `delegation/orchestrator.py` (or wherever the check lives).

**Checklist**:
- [ ] `src/t0_agents/core/agent.py:41-59` — delete the `_resolve_chat_tool_provider` async context manager.
- [ ] `src/t0_agents/core/agent.py:10` — remove `FastMCP` from the `from fastmcp import …` line if no other call site uses it in this module.
- [ ] `src/t0_agents/core/agent.py:381` — replace the `async with _resolve_chat_tool_provider(...)` block with `return await run_agent(self._t0, ctx.tool_provider, ...)`.
- [ ] `src/t0_agents/core/agent.py:434` — same simplification for `_chat_streaming_terminal` (pass `ctx.tool_provider` straight into `streamer.stream(...)`).
- [ ] `src/t0_agents/core/agent.py:353-365, 400-415` — update terminal docstrings to drop "ephemeral fallback" language; describe the direct semantics.
- [ ] `src/t0_agents/delegation/orchestrator.py` (or the file owning the check identified during M1) — rewrite the `ConfigurationError` message: `"AgentConfig.sub_agents requires the parent's tool_provider to be a fastmcp.Client (got {type}). Pass tool_provider=Client(FastMCP(...)) when constructing the parent Agent."`
- [ ] `tests/test_core/test_agent.py` — extend `test_toolless_chat_invoke_returns_text` (or add a sibling test) with a `monkeypatch` / spy on `fastmcp.FastMCP.__init__` asserting it is not called during a toolless `invoke`.
- [ ] `tests/test_delegation/*.py` — update assertion strings to match the new sub-agent error message.

**Depends on**: milestone 1, milestone 2 (the direct pass-through only works once both terminals accept `None`). Read both prior dev-logs before implementing.

**Acceptance criteria**:
- [ ] `uv run pytest -m "not integration"` (full offline suite) passes.
- [ ] `uv run mypy src/t0_agents/core/` clean.
- [ ] `uv run ruff check src/t0_agents/core/` clean.
- [ ] `grep -rn '_resolve_chat_tool_provider\|FastMCP("agent-default")' src/t0_agents/` returns zero hits.
- [ ] The spy test in `test_agent.py` proves `FastMCP.__init__` is not invoked during toolless chat (test name pattern: `test_*_does_not_allocate_fastmcp`).

**Notes**:
- `_single_shot_terminal` already guards `self._tool_provider is None` at the JSON-mode-`tool` safety check (`agent.py:510-511`); leave that branch alone.
- The sub-agent error-message polish lives in M3 (not M4) because it's about the high-level `Agent` UX — keeps M4 strictly docs.

---

## Milestone 4 — Public docs, CHANGELOG, sub-agent error polish

**Slug**: `docs-changelog-optional-client`

**Goal**: The feature is documented end-to-end and the breaking rename is called out in the v1.0 changelog; new devs reading the README see the toolless pattern.

**Scope**:
- README snippet showing chat-only `Agent` usage without `tool_provider`.
- `docs/getting-started.md` rewording.
- `docs/agent.md` cross-references.
- `CLAUDE.md` updated framing.
- `CHANGELOG.md` Unreleased entry covering both the rename and the optional contract.
- Final audit grep guaranteeing no stale `mcp_client` references survive.

**Checklist**:
- [ ] `README.md` (≈ lines 38–46) — add a "Chat-only" sub-snippet right next to the existing "Agent with tool_provider" snippet, ~6 lines: `Agent(config, t0=gateway)` + `await agent.invoke(...)`, no FastMCP import.
- [ ] `docs/getting-started.md:87` — replace "Both `AsyncTensorZeroGateway` and `Client` must be entered as async context managers before calling `run_agent`" with "`AsyncTensorZeroGateway` must be entered as an async context manager; the optional `tool_provider` (typically a `fastmcp.Client`) is only required when the agent dispatches tools."
- [ ] `docs/agent.md` — add cross-reference to the toolless pattern; reword the "JSON-mode agents never dispatch tools, so `tool_provider` is optional" paragraph to also mention chat-mode parity.
- [ ] `CLAUDE.md:29` — extend the "Agent loop flow" line: "Tools and the MCP `Client` are optional; single-shot output modes and chat-only agents work without them."
- [ ] `CHANGELOG.md` (Unreleased section) — add a **Breaking** bullet: "`run_agent` and `StreamingAgentLoop.stream` rename `mcp_client` → `tool_provider`; the parameter is now optional (`Client[Any] | None = None`). Chat-mode agents with no tools no longer allocate an ephemeral FastMCP server."
- [ ] `grep -rn "mcp_client" docs/ README.md CLAUDE.md` — confirm zero hits after edits.

**Depends on**: milestone 1, milestone 2, milestone 3 (docs describe behaviour that has already shipped). Read all three prior dev-logs.

**Acceptance criteria**:
- [ ] `uv run pytest -m "not integration"` passes (sanity: docs-only PR shouldn't touch source, but guard anyway).
- [ ] `grep -rn "mcp_client" src/ tests/ docs/ README.md CLAUDE.md examples/` returns zero hits.
- [ ] `uv run python -c "from t0_agents import run_agent; import inspect; assert 'tool_provider' in inspect.signature(run_agent).parameters and 'mcp_client' not in inspect.signature(run_agent).parameters"` exits 0.
- [ ] `git diff main -- CHANGELOG.md` shows the new Breaking bullet under Unreleased.

**Notes**:
- After M4 lands, the memory note `feedback_mcp_standalone.md` says `run_agent` signature is frozen — update it in the dev-log to reflect the new baseline (not in the PR diff).
- No emojis in docs (user preference).

---

## Milestone 5 — `examples/agent/00_chat_only/` showcase (OPTIONAL)

**Slug**: `example-chat-only-showcase`

**Goal**: New devs landing in `examples/` see "chat without tools" as the very first example — no FastMCP server in sight.

**Scope**:
- New numbered example sorted before `01_first_agent`.
- ~30-line `main.py`, minimal `tensorzero.toml`, a README that contrasts with `01_first_agent`.
- Reuse a known-good compose pin (per memory: `examples/14_multi_agent` or `examples/17_structured`; avoid `02..07`, `18`, which have a broken clickhouse tag).

**Checklist**:
- [ ] `examples/agent/00_chat_only/main.py` — ~30 lines: import `AsyncTensorZeroGateway`, `Agent`, `AgentConfig`, `ChatOutput`; build `Agent(config, t0=gateway)` without `tool_provider`; one `await agent.invoke(...)`; print `result.response`.
- [ ] `examples/agent/00_chat_only/tensorzero.toml` — minimal chat function pointing at the same model `01_first_agent` uses.
- [ ] `examples/agent/00_chat_only/.env.example` — annotate the provider env var(s) needed (mirror `01_first_agent`).
- [ ] `examples/agent/00_chat_only/docker-compose.yml` — reuse the `examples/14_multi_agent` compose pin (gateway + clickhouse versions known to work).
- [ ] `examples/agent/00_chat_only/README.md` — explain when to use this template vs `01_first_agent`; link both directions.
- [ ] `examples/agent/01_first_agent/README.md` — append a line: "If you only need conversational LLM access without tools, see [`../00_chat_only/`](../00_chat_only/)."

**Depends on**: milestone 1, milestone 2, milestone 3, milestone 4. Read all four prior dev-logs.

**Acceptance criteria**:
- [ ] `python -c "import ast; ast.parse(open('examples/agent/00_chat_only/main.py').read())"` exits 0.
- [ ] `uv run ruff check examples/agent/00_chat_only/` clean.
- [ ] `uv run mypy examples/agent/00_chat_only/main.py` clean.
- [ ] `grep -nE "fastmcp|FastMCP|Client" examples/agent/00_chat_only/main.py` returns zero hits (the example must not import any MCP plumbing).
- [ ] Smoke run recorded in the dev-log: `docker compose --env-file ../../../.env -f docker-compose.yml up -d && uv run python main.py` returns a non-empty `result.response`.

**Notes**:
- `00_` ordering puts the toolless case first in directory listings — pedagogically the "absolute minimum" example.
- If the smoke run fails because the gateway version drifted, fall back to the same compose pin used by `examples/17_structured`.
- Annotating existing `03_output_modes/` examples is out of scope; track separately on `project_pre_public_polish.md` if it surfaces.
