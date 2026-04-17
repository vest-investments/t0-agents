# Phase G Implementation Log

Phase: G (Multi-Agent Composition)
Milestone: M7 — Composed Agents
Date: 2026-04-16

## Divergences from Spec

### D1: `fastmcp.Client` has no `local_provider` attribute — in-process unwrap is `client.transport.server`

**Spec (§4.3 `orchestrator.py`).** Pseudo-code reads:

```python
server = client.local_provider  # FastMCP instance
if server is None:
    raise ConfigurationError(...)
```

**Actual.** Source-validated correction against the installed
`fastmcp` 2.x in `.venv/lib/python3.12/site-packages/fastmcp/`:

- `local_provider` is a property on `fastmcp.server.server.FastMCP`
  (line 430), **not** on `fastmcp.Client`.  On the client it simply
  does not exist — `getattr(client, 'local_provider', None)` returns
  `None` and `client.local_provider` raises `AttributeError`.
- The in-process unwrap path is
  `client.transport` (always set, arbitrary type) narrowed via
  `isinstance(client.transport, FastMCPTransport)`, then
  `client.transport.server` returns the `FastMCP` instance that was
  passed to `Client(mcp)` at construction
  (`fastmcp/client/transports/memory.py:23-29`).

**Implementation.**
`src/t0_agents/compose/orchestrator.py::install_sub_agents` narrows
the tool provider in two steps: first an `isinstance(client, Client)`
guard, then an
`isinstance(getattr(client, 'transport', None), FastMCPTransport)`
guard.  Remote MCP transports (HTTP, SSE, WebSocket) raise
`ConfigurationError` at construction time — runtime-registered tools
cannot travel across those transports.

**Reason.** The Phase G spec was drafted before the installed FastMCP
source was inspected; the precedent here is identical to T0 Q7
(`build_embedded` IS available) and FastMCP Q1 (`add_tool` does NOT
take a `name=` kwarg) in
[`../02-open-questions.md`](../02-open-questions.md).  Same family of
correction: documentation-level assumption, invalidated by a quick
read of the installed source.

**Impact.** No contract amendment.  The behavioural contract in
`00-integration-contract.md §3.2` ("`sub_agents: list[Agent]`
exposed as MCP tools via an in-process registry") stands.  The Phase
G spec at `phases/phase-g-multi-agent.md §4.3` and its §10 risk row
("`client.local_provider` API differs across FastMCP versions") are
reconciled by this log; future readers hitting the spec should find
the log via the phase index.

### D2: `server.list_tools_sync()` does not exist — synchronous enumeration via `_components`, and `add_tool` does NOT raise on duplicates by default

**Spec (§4.3).** Proposed a pre-registration collision check:

```python
existing = {t.name for t in server.list_tools_sync()}
```

**Actual.** Two source-validated findings:

1. `FastMCP` exposes only the async `list_tools()`
   (`fastmcp/server/server.py:559`) and an async-only
   `LocalProvider._list_tools()`.  There is no sync public API for
   enumerating tools.
2. `FastMCP.add_tool` does **not** raise on duplicates by default —
   `LocalProvider._on_duplicate` defaults to `"warn"`, so a duplicate
   silently replaces the existing component with a log line
   (`fastmcp/server/providers/local_provider/local_provider.py:99-107,
   187-200`).  Only when the FastMCP is constructed with
   `on_duplicate="error"` does the `ValueError` branch fire.

Consequence: the collision detection **must** pre-check.  We read
`server.local_provider._components` (the unified component dict) and
filter for `isinstance(c, Tool)`.  The attribute is private in name
but public in spirit — FastMCP's own `LocalProvider._list_tools` walks
the same dict — and its stability is load-bearing for our sync path.
A second `ValueError`-catch around `add_tool` is kept as
defence-in-depth in case the orchestrator's FastMCP is explicitly
constructed with `on_duplicate="error"`.

**Reason.** Initial implementation briefly dropped the pre-check on
the assumption that `add_tool` would raise on duplicates; the Phase G
test
`test_collision_with_existing_tool_raises` surfaced the warn-and-
replace default and forced the pre-check back.  Logged here so future
readers don't make the same assumption.

**Impact.** No contract amendment.  The §3.2 guarantee ("Sub-agent
tool-name collisions raise `ConfigurationError` at construction
time") is satisfied.

### D4: Parent-episode `ContextVar` must refresh on the first inference for fresh orchestrator runs

**Observed.** Gemini flagged on PR #9 (two symmetric inline comments on
`src/t0_agents/core/agent.py:251` / `:439`, plus a summary review):

> The current implementation binds the `episode_id` to the `ContextVar`
> only once at the start of the run. If `episode_id` is `None` (which
> is common for fresh episodes), sub-agents called during the run will
> see `"unknown"` as their parent episode ID for the entire duration
> of the orchestrator's run, even after the first inference has
> populated the actual episode ID.

Reproduction matches the reading: `Agent.run` / `Agent.stream` enter
`bind_parent_episode_id(episode_id)` exactly once before delegating to
`run_agent` / `StreamingAgentLoop.stream`.  With `episode_id=None` the
`ContextVar` holds `None` for the entire orchestrator run;
`mcp/loop.py:355` / `stream/runner.py:168` do capture the real
episode id from the first response but only into a local, never into
the ContextVar.  `compose/delegation.py:65` therefore falls back to
the `"unknown"` sentinel for every sub-agent call.

Contract §5.2 ("sub-agents tag every inference with the orchestrator's
episode id via `t0_agents.parent_episode_id`") was therefore only
honoured for callers that passed an explicit `episode_id`.  The test
suite did not catch this: `TestParentEpisodeContextVar` exercises
ContextVar mechanics in isolation, and the two integration tests in
`tests/test_compose/test_integration.py` assert *episode isolation*
between orchestrator and sub-agent but never inspect the actual
`parent_episode_id` tag value.

**Implementation.**
`src/t0_agents/compose/orchestrator.py` gains a public helper
`update_parent_episode_id(episode_id: str) -> None` that replaces
the value stored by the enclosing `bind_parent_episode_id` context
without disturbing its reset-on-exit `Token`.  `Agent.run` and
`Agent.stream` now wrap their existing `on_inference` callback when
`sub_agents` are configured and `episode_id is None`: the wrapper
calls `update_parent_episode_id(str(response.episode_id))` on every
fired inference (idempotent within one episode) before delegating to
whatever `on_inference` was previously in place — guardrail cost
tracking stays composed.  With an explicit `episode_id` no wrapping
happens, preserving the Phase A byte-identity for the explicit-id
path.

**Reason.** Keeping the fix in `core/agent.py` + `compose/` upholds
the invariant §2.1 ("`mcp/` stays standalone").  Exposing a public
`update_parent_episode_id` rather than importing the private
`_PARENT_EPISODE_ID` symbol keeps `compose/orchestrator.py` the sole
custodian of the ContextVar machinery and documents *why* a callable
outside `bind_parent_episode_id` is allowed to mutate it.

**Impact.** No contract amendment.  The §3.2 / §5.2 guarantees
("sub-agents tag every inference with the orchestrator's episode id")
are now true end-to-end.  Four new unit tests in
`tests/test_compose/test_orchestrator.py::TestParentEpisodeIdRefresh`
+ one helper test lock the behaviour for both `Agent.run` and
`Agent.stream`.

### D5: FastMCP in-process transport snapshots ContextVars into its server sub-task — inline dispatch wrapper required

**Observed.** After shipping D4 (the `on_inference` refresh wrapper),
a live ClickHouse spot-check on the Phase G example still showed
every sub-agent inference tagged ``parent_episode_id='unknown'``:

```
┌─function_name─┬─rows─┬─parents─────┐
│ orchestrator  │   12 │ []          │
│ writing       │    9 │ ['unknown'] │
│ research      │    7 │ ['unknown'] │
└───────────────┴──────┴─────────────┘
```

Source tracing found the root cause.
`fastmcp/client/transports/memory.py::FastMCPTransport.connect_session`
spawns the FastMCP server on an anyio sub-task via
``tg.start_soon(...)``:

```python
async with anyio.create_task_group() as tg:
    tg.start_soon(
        lambda: self.server._mcp_server.run(...),
    )
```

`start_soon` snapshots the current ``contextvars.Context`` at task-
creation time, which is ``async with Client(server):`` entry in
``main.py`` — strictly **before** :meth:`Agent.run` has the chance
to bind the parent-episode id.  The server's tool-handler in
:mod:`t0_agents.compose.delegation` therefore reads a stale
``None`` from ``_PARENT_EPISODE_ID`` regardless of what the
orchestrator's task does mid-run.  D4's ``on_inference`` refresh
correctly updates the ContextVar inside the orchestrator's task
(``TestParentEpisodeIdRefresh`` proves it), but the update never
reaches the server sub-task's snapshotted context.

This is a fundamental constraint of anyio/asyncio task isolation,
not a bug we can fix with more ContextVar plumbing.

**Implementation.** Add a private ``_OrchestrationToolProvider`` in
``src/t0_agents/compose/orchestrator.py`` that wraps the
orchestrator's ``tool_provider`` and dispatches sub-agent tool calls
**inline** — staying in the orchestrator's asyncio task so
``get_parent_episode_id()`` reads the live binding.  Non-sub-agent
tool calls fall through to the inner provider unchanged.
:meth:`Agent.run` and :meth:`Agent.stream` now wrap ``tool_provider``
with this provider *inside* the :class:`GuardedToolProvider` layer,
so orchestrator-level guardrails (e.g. ``max_tool_calls``) still
observe every delegation.  ``install_sub_agents`` continues to
register each sub-agent on the FastMCP server so ``list_tools``
surfaces them to the LLM and external callers (who genuinely have
no orchestrator context) fall back to ``"unknown"`` via the
server-side handler.

``Agent.__init__`` caches ``self._sub_agents_by_tool_name: dict[str,
Agent]`` once at construction to avoid rebuilding the mapping per
run.

**Reason.** Keeps the fix inside ``core/agent.py`` + ``compose/``
(invariant §2.1: ``mcp/`` stays standalone).  Preserves the D4
``on_inference`` wrapper for public-API consistency — callers that
use ``get_parent_episode_id`` in their own same-task code still see
the live binding.  No change to the server-side delegation handler
contract: it remains the fallback path for external callers.

**Impact.** No contract amendment.  §3.2 / §5.2 are now end-to-end
correct for orchestrator-driven sub-agent calls — the primary use
case.  The example's ClickHouse spot-check post-fix returns the
orchestrator's real episode id as the ``parent`` for every
sub-agent row.  Test suite grew from 392 → 399 passes
(+6 unit tests for the wrapper, +1 end-to-end regression using
scripted ``FakeT0`` + real ``FastMCP.Client`` to reproduce the
user-reported failure mode).

### D3: `mypy --strict` rejects `FastMCP.add_tool(tool: Tool | ...)` when the concrete type is `Tool`

**Observed.** Passing a `fastmcp.tools.Tool` to
`FastMCP.add_tool(tool: Tool | Callable[..., Any])` triggers

```
Argument 1 to "add_tool" of "FastMCP" has incompatible type "Tool";
expected "Callable[..., Any]"
```

under `mypy --strict`, even though the runtime signature is a proper
union.  The union collapses at type-check time because `Tool` is a
Pydantic `BaseModel` subclass and `Callable[..., Any]` is inferred as
the wider branch.

**Workaround.**
`cast(Callable[..., Any], tool)` at the single `add_tool` call site
in `install_sub_agents`.  The comment at that call site explains the
cast so future readers don't mistake it for dead defensive typing.
Runtime behaviour is unchanged — `LocalProvider.add_tool` unwraps the
callable vs. Tool cases itself
(`providers/local_provider/decorators/tools.py:151-200`).

**Reason.** FastMCP does not ship `.pyi` stubs and does not use
`@overload` on `add_tool`, so there is no upstream fix available
today.  Filing an upstream type-stub PR is out of scope for Phase G.

**Impact.** Confined to one line of
`src/t0_agents/compose/orchestrator.py`.  Future FastMCP versions
that either add an `@overload` or tighten the Pydantic type can
drop the cast.

## Contract Amendments

None.  The integration contract in
[`../00-integration-contract.md`](../00-integration-contract.md) is
byte-identical after Phase G:

- `Agent` public signature (§3.2) unchanged.
- `run_agent` public signature (§6.1) untouched.
- `mcp/` public exports (§6.3) untouched — `tests/test_mcp/` still
  green.
- The new `sub_agents` field on `AgentConfig` was reserved by §3.2
  ahead of time; adding it matches that reservation.

## Resolved Open Questions

None new during the phase.  The two relevant open questions
(**T0 Q2** — `additional_tools` mutation safety; **FastMCP Q1** —
runtime tool registration) were already resolved in
[`../02-open-questions.md`](../02-open-questions.md) before Phase G
started.  Their resolutions were applied verbatim:

- T0 Q2 → sub-agents are declared at construction time; mid-run
  mutation is deferred.
- FastMCP Q1 → use `Tool.from_function(fn, name=...)` plus
  `server.add_tool(tool)`; no `name=` kwarg on `add_tool` itself.

## Release Notes (0.8.0)

> **Multi-agent composition** — one `Agent` can now delegate to
> others via the new `AgentConfig.sub_agents` field.  Each sub-agent
> becomes a `{name}_ask` MCP tool on the orchestrator's in-process
> FastMCP server, installed at `Agent.__init__` time.  Sub-agent
> errors surface as the new `SubAgentError` (non-fatal; the
> orchestrator LLM decides how to recover) except for fatal errors
> like `GuardrailBlockedError`, which propagate unchanged.
> Sub-agents run in their own TensorZero episodes, correlated back
> to the orchestrator via a `t0_agents.parent_episode_id` tag.  See
> [`docs/multi-agent.md`](../../../docs/multi-agent.md) and the new
> `examples/14_multi_agent/` demo.

## Verification snapshot

```
$ uv run ruff check src/ tests/test_compose/ examples/14_multi_agent/
All checks passed!

$ uv run mypy src/
Success: no issues found in 45 source files

$ uv run pytest -m "not integration"
399 passed, 42 deselected in 2.50s
```

(After D4 the suite grew from 387 → 392: one helper test for
`update_parent_episode_id` + four refresh-wrapper tests in
`TestParentEpisodeIdRefresh`.  After D5 it grew again from
392 → 399: six unit tests for `_OrchestrationToolProvider` + one
scripted-T0 end-to-end regression test in
`TestParentEpisodeIdEndToEnd`.)

Integration verification (`uv run pytest -m integration
tests/test_compose/` + `python -m examples.14_multi_agent.main`)
requires `docker compose up -d` and `GROQ_API_KEY`; to be run
manually by the phase shepherd.
