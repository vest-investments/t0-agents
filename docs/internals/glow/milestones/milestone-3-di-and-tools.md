# M3 — `di-and-tools`

> **Status:** Not started.
> **Branch:** `experimental/glow`.
> **Dependencies:** M2 (needs `App` + `@app.agent`).
> **Blocks:** M4 doesn't strictly need M3 but M5 examples assume it.

## Purpose

Land the **Depends + tools + sub-agents** trinity that makes glow
feel like FastAPI for agents. `Depends(...)` becomes the universal
injection point: tool dependencies (db, secrets, http clients),
sub-agents, and per-request context all flow through the same
mechanism. `@app.tool` registers tools backed by FastMCP, with
`Depends` parameters stripped from the auto-generated MCP schema.
Sub-agents declared as `Depends(other_agent)` become declarative
participants in the orchestrator's signature — overrideable in
tests via `app.dependency_overrides`.

After M3, examples 6 and 8 from the plan work; the only
introspective magic glow ships (`Depends(an_agent)` → installed as
`sub_agent` on the parent `AgentConfig`) is in place and tested.

## Scope

### In

- `src/t0_agents/glow/di.py`:
  - `Depends` dataclass (`provider`, `use_cache`).
  - `DependencyResolver` — walks signatures, recursively resolves
    providers (sync + async), per-call cache keyed by
    `id(Depends)`, honors `app.dependency_overrides[provider]`.
- `src/t0_agents/glow/decorators.py` updates:
  - `@app.tool` — registers the function with the App's in-memory
    `FastMCP` server.
  - `@app.tool` strips `Depends` params before FastMCP's
    `Tool.from_function` sees the signature; per-tool wrapper
    re-injects resolved values at call time.
- `glow/app.py` updates:
  - `App._mcp_server: fastmcp.FastMCP` — created lazily in
    `build()` / `lifespan()`; mounts user tools.
  - `App._mcp_client: fastmcp.Client` — composed from the in-memory
    server; becomes `tool_provider` on every constructed `Agent`.
  - `App.dependency_overrides` populated and consulted by
    `DependencyResolver`.
- `Depends(agent)` special case:
  - Detected during `@app.agent` registration when a parameter is
    typed `Agent` (or the glow wrapper) and the provider is a
    glow-registered agent.
  - Adds the provider's underlying `Agent` to
    `AgentConfig.sub_agents` instead of being injected as a value.
- Tests:
  - `tests/test_glow/test_di.py` — unit tests for the resolver.
  - `tests/test_glow/test_tool_decorator.py` — unit tests for
    `@app.tool` (schema stripping, `Depends` re-injection).
  - `tests/test_glow/test_integration.py` extensions — orchestrator
    + 2 sub-agents wired via `Depends`, plus one tool using
    `Depends(get_db)` with an overridden `InMemoryDB`.

### Out

- `compose()` (M4).
- MCP typing (M4).
- Guardrail metadata (M4).
- `AsyncIterator` tool yields (deferred to a follow-up; out of
  scope for v0).

## Checklist

- [ ] **Commit 6: feat(glow): `Depends` + `DependencyResolver`**
  - [ ] `glow/di.py`:
        - [ ] `@dataclass(frozen=True, slots=True) class Depends`
              with `provider: Callable[..., Any]` and
              `use_cache: bool = True`.
        - [ ] `class DependencyResolver`:
              - [ ] `async def resolve(self, fn, app, call_cache)` —
                    walks `inspect.signature(fn)`, finds `Depends`
                    defaults, recursively resolves them.
              - [ ] Per-call cache keyed by `id(Depends)`.
              - [ ] `app.dependency_overrides[provider]` consulted
                    *every call* (overrides not snapshotted at
                    registration).
              - [ ] Async providers `await`-ed; sync providers
                    called inline.
        - [ ] Returned mapping `{param_name: value}` merged into
              the call kwargs of the decorated fn.
  - [ ] Wire `App.dependency_overrides: dict[Callable, Callable]`
        — empty dict by default.
  - [ ] Unit tests `tests/test_glow/test_di.py`:
        - [ ] Single `Depends(get_db)` resolves and injects.
        - [ ] Nested `Depends`: provider depends on another
              `Depends`.
        - [ ] `use_cache=True` (default): same `Depends` instance
              resolved twice in one call returns the same value.
        - [ ] `use_cache=False`: each resolution invokes the
              provider freshly.
        - [ ] `app.dependency_overrides[get_db] = mock_db` replaces
              the resolved value.
        - [ ] Async + sync providers both work.

- [ ] **Commit 7: feat(glow): `@app.tool` decorator**
  - [ ] `glow/app.py` updates:
        - [ ] Lazily build `App._mcp_server = FastMCP("glow")` and
              `App._mcp_client = Client(self._mcp_server)` inside
              `build()` (called by first invocation).
        - [ ] Register decorated tools on `_mcp_server` via the
              FastMCP `add_tool` API.
        - [ ] `Agent` constructor uses `tool_provider=self._mcp_client`.
  - [ ] `glow/decorators.py`:
        - [ ] `App.tool(fn)` — extract `Depends` params from
              `inspect.signature(fn)`, build a stripped signature
              that FastMCP can introspect, register a wrapper that
              re-injects resolved `Depends` at call time.
        - [ ] Tool docstring → MCP tool description (FastMCP
              already does this by default).
  - [ ] Unit tests `tests/test_glow/test_tool_decorator.py`:
        - [ ] Tool with no `Depends` registers and the FastMCP
              schema matches the function signature.
        - [ ] Tool with `Depends(get_db)`: the auto-generated
              MCP tool schema does **not** include the `db`
              parameter (schema has only the non-`Depends` params).
        - [ ] When the tool is called via the MCP client, the `db`
              kwarg is populated from the resolver.
        - [ ] `app.dependency_overrides[get_db] = mock_db` makes
              the tool see the mock.

- [ ] **Commit 12: feat(glow): sub-agents via `Depends(agent)`**
  - [ ] During `@app.agent` registration, scan parameters for
        `Depends(provider)` where `provider` is a glow-registered
        agent wrapper. Record these on the `_AgentSpec` as
        `sub_agent_specs: list[_AgentSpec]`.
  - [ ] When building the `Agent`, pass `sub_agents=[spec._agent
        for spec in sub_agent_specs]` to `AgentConfig`.
  - [ ] The parameter is **not** present in the runtime call
        signature (the resolver removes it).
  - [ ] Document this prominently — it's the only piece of
        introspective magic glow ships.
  - [ ] Unit tests:
        - [ ] `@app.agent("orchestrator")` declaring
              `Depends(researcher)` adds `researcher._agent` to
              the orchestrator's `AgentConfig.sub_agents`.
  - [ ] Integration test mirroring `examples/08_multi_agent`:
        - [ ] Orchestrator + researcher + writer, all via glow.
        - [ ] Orchestrator's signature includes
              `research: Agent = Depends(researcher)` and
              `write: Agent = Depends(writer)`.
        - [ ] End-to-end run produces a non-empty article.

- [ ] **Examples:**
  - [ ] `examples/glow/05_di_tools.py` — `@app.tool` with
        `Depends(get_db)` + override in a test block.

- [ ] **Run validation:**
  ```bash
  uv run ruff check src/ tests/
  uv run mypy src/
  uv run pytest -m "not integration"

  docker compose up -d
  uv run pytest tests/test_glow/ -q
  docker compose down
  ```

## Acceptance criteria

- `tests/test_glow/test_di.py` covers single, nested, cached,
  uncached, async, sync, and overridden cases.
- `tests/test_glow/test_tool_decorator.py` proves the FastMCP
  schema does not include `Depends` params, and that runtime
  injection happens.
- Integration test for the orchestrator-with-sub-agents
  (mirroring `examples/08_multi_agent`) runs against a live
  gateway and the orchestrator's response references both
  sub-agents' contributions.
- `examples/glow/05_di_tools.py` runs cold and prints a success
  banner.

## Verification commands

```bash
uv run ruff check src/t0_agents/glow/ tests/test_glow/
uv run mypy src/t0_agents/glow/
uv run pytest tests/test_glow/ -q -m "not integration"

docker compose up -d
uv run pytest tests/test_glow/ -q
uv run python examples/glow/05_di_tools.py
docker compose down
```

## Cross-references

- Plan §"What `glow` adds" items 4, 9.
- Plan §"API design" examples 6 and 8.
- Plan §"Technical architecture" → "Dependency Injection" and
  "Depends(an_agent) special case" at
  `~/.claude/plans/la-idea-es-tener-glowing-gem.md`.
- v1.0 source: `src/t0_agents/core/config.py:24` (`AgentConfig.sub_agents`),
  `src/t0_agents/mcp/loop.py:323` (tool auto-discovery), FastMCP
  `Tool.from_function` (external).

## Dev-log

`logs/milestone-3-dev-log.md` at end.
