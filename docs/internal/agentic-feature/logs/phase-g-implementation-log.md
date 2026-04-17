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
387 passed, 42 deselected in 2.56s
```

Integration verification (`uv run pytest -m integration
tests/test_compose/` + `python -m examples.14_multi_agent.main`)
requires `docker compose up -d` and `GROQ_API_KEY`; to be run
manually by the phase shepherd.
