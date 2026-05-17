# M4 — `composition-mcp-guardrails`

> **Status:** Not started.
> **Branch:** `experimental/glow`.
> **Dependencies:** M2 (App + decorator); M3 helpful but not
> strictly required.
> **Blocks:** M5 examples.

## Purpose

Deliver the three high-leverage glue features that round out the
public surface:

1. **`compose(spec, *, merge=None)`** — native-collection
   composition. `list/tuple` → sequential; `set/frozenset` →
   fan-out + merge (every agent runs, errors don't kill siblings,
   reducer fuses results); `dict[str, Composable]` → LLM-routed
   dispatch. Composers nest freely because v1.0's `Composable`
   protocol already guarantees stack-ability.
2. **Typed MCP integration** — `class GithubMCP(MCPServer, url=...)`
   and `Annotated[MCPClient, MCPServer(url=...)]` patterns; auto-mount
   via `FastMCP.mount()` (mirroring `examples/14_multi_server`).
3. **`Annotated[T, Guardrail(...)]`** — guardrail metadata
   extracted at registration time and folded into
   `AgentConfig.guardrails`. Runtime path unchanged
   (`GuardrailsMiddleware` does the work).

After M4, examples 5, 5b, 7, and 9 from the plan work. The
"stackable by contract" promise gets its acceptance test here.

## Scope

### In

- `src/t0_agents/glow/compose.py`:
  - `compose(spec: Any, *, merge: Callable | None = None) -> Composer`.
  - `_default_merge(results)` — concatenates strings; raises
    `TypeError` for mixed types (forcing the user to pass `merge=`).
  - `_llm_routed(routes: dict[str, Composable]) -> Composer` —
    synthetic `ToolCallOutput` classifier + `conditional` dispatch.
  - `_make_router_agent(keys)` — internal helper.
- `src/t0_agents/glow/mcp_typing.py`:
  - `class MCPServer` base class with `__init_subclass__(url, transport, namespace)`
    that records the spec.
  - `MCPServerSpec` dataclass for `Annotated[MCPClient, MCPServer(url=...)]` use.
  - `MCPServerRegistry` on `App` — discovered specs accumulate
    there.
  - `App.lifespan()` builds a composed `FastMCP` server with
    `mount(server, namespace=spec.namespace)` and passes the
    composed `Client` as `tool_provider`.
- `src/t0_agents/glow/guardrails_typing.py`:
  - Extracts `Guardrail` instances from `Annotated[T, Guardrail(...)]`
    on both input parameters and the return annotation.
  - Folds them into `AgentConfig.guardrails` at registration time
    (input guardrails wrap user message; output guardrails wrap
    response — see `GuardrailsMiddleware` for the existing
    semantics).
- Tests:
  - `tests/test_glow/test_compose.py` — unit tests for the
    dispatcher.
  - `tests/test_glow/test_compose_nested.py` — composer-in-composer
    acceptance test.
  - `tests/test_glow/test_mcp_typing.py` — unit tests for the
    server class + spec extraction; integration test for the
    actual mount.
  - `tests/test_glow/test_guardrails_typing.py` — unit + integration.

### Out

- `AsyncIterator` tools (deferred).
- Per-agent MCP subsets (one App = one composed client view across
  every agent it owns; deferred per plan).
- Streaming JSON (deferred to a v1.1 follow-up).

## Checklist

- [ ] **Commit 8: feat(glow): `compose(spec, *, merge)` native-type dispatcher**
  - [ ] `glow/compose.py`:
        - [ ] `def compose(spec, *, merge=None)`:
              - [ ] `isinstance(spec, (list, tuple))` →
                    `sequential(*spec)`. Spec nodes are passed
                    verbatim; the `Composable` protocol means an
                    `Agent`, a `Composer`, or a glow wrapper all
                    work.
              - [ ] `isinstance(spec, (set, frozenset))` →
                    `parallel(*spec, reducer=(merge or _default_merge),
                                  error_policy="continue")`.
              - [ ] `isinstance(spec, dict)` → `_llm_routed(spec)`.
              - [ ] Else → raise `TypeError` with the exhaustive
                    message from the plan.
        - [ ] `_default_merge` per the plan.
  - [ ] **Adapter for glow wrappers**: glow's `@app.agent` returns
        a callable wrapper, not the raw `Agent`. `compose()` accepts
        the wrapper and unwraps via `.{_agent or _composer}`
        before passing to `sequential`/`parallel`. Document this.
  - [ ] Unit tests `tests/test_glow/test_compose.py`:
        - [ ] `compose([a, b])` returns a `Composer` whose
              `invoke("x")` produces sequential execution.
        - [ ] `compose({a, b}, merge=concat)` runs both and feeds
              their `AgentResult` list to `concat`.
        - [ ] `compose({a, b})` with all-string outputs uses
              `_default_merge`.
        - [ ] `compose({a, b})` with mixed types raises
              `TypeError` mentioning `merge=`.
        - [ ] `compose([1, 2])` (invalid spec) raises `TypeError`.
        - [ ] `compose(["string"])` (invalid spec) raises
              `TypeError`.
        - [ ] **Composer-in-composer**: `compose([compose([a, b]), c])`
              succeeds; the inner composer is a node of the outer.

- [ ] **Commit 9: feat(glow): LLM-routed `dict` composer**
  - [ ] `_make_router_agent(keys)`:
        - [ ] Internal `@app.agent`-like registration with a
              `ToolCallOutput(tool_choice=AUTO)` config and one
              synthetic tool per route key.
        - [ ] System prompt: "You are a router. Given the user
              query, pick exactly one tool whose name matches the
              intended branch. Do not produce additional text."
  - [ ] `_llm_routed(routes)`:
        - [ ] Returns `sequential(classifier, conditional(router=..., routes=routes))`.
        - [ ] `routes` values can be `Composable` (Agent, Composer,
              or glow wrapper unwrapped to the underlying object).
  - [ ] Integration test (marked):
        - [ ] Two routes: `"math"` agent and `"code"` agent.
        - [ ] Query "Solve x^2 = 4" → routed to "math".
        - [ ] Query "Write a Python function" → routed to "code".
        - [ ] Composer-as-route case: one route is `compose([a, b])`.

- [ ] **Commit 10: feat(glow): typed `MCPServer` integration**
  - [ ] `glow/mcp_typing.py`:
        - [ ] `class MCPServer`:
              - [ ] `__init_subclass__(cls, *, url, transport="http", namespace=None)`
                    records the spec on the class.
              - [ ] Class-based usage: `class GithubMCP(MCPServer, url="..."): ...`.
        - [ ] `MCPServerSpec(url, transport, namespace)` dataclass
              for the `Annotated[MCPClient, MCPServer(url=...)]`
              pattern — extracted from parameter annotations at
              registration.
        - [ ] `App._mcp_specs: list[MCPServerSpec]` accumulates
              specs from both pattern variants and from
              `App(servers=[...])` constructor arg.
  - [ ] `App.lifespan()`:
        - [ ] Build composed `FastMCP("glow-composed")`; mount each
              spec via `composed.mount(server, namespace=spec.namespace)`.
        - [ ] `App._mcp_client = Client(composed)` — used as
              `tool_provider` on every constructed `Agent`.
        - [ ] Tear down on context exit.
  - [ ] Integration test:
        - [ ] In-memory FastMCP with two tools (math + text).
        - [ ] Two `MCPServer` subclasses mounted via the App.
        - [ ] Decorated agent uses both via injected parameters.

- [ ] **Commit 11: feat(glow): `Annotated[T, Guardrail(...)]` extraction**
  - [ ] `glow/guardrails_typing.py`:
        - [ ] Walk `inspect.signature`. For each parameter and the
              return annotation:
              - [ ] `Annotated[T, Guardrail(...)]` → collect the
                    `Guardrail` instances.
              - [ ] Input guardrails attached as
                    `before_message` checks; output guardrails as
                    `after_run` checks. Use existing
                    `t0_agents.guardrails` API (whatever
                    `GuardrailsMiddleware` consumes).
        - [ ] Fold into `AgentConfig.guardrails` at registration.
  - [ ] Unit test:
        - [ ] A regex guardrail on the return annotation triggers
              `GuardrailBlockedError` when the response matches the
              pattern (mock `Agent` returns a string with PII).
  - [ ] Integration test:
        - [ ] Output guardrail catches an SSN-like pattern in a
              real response (use a system prompt asking the model
              to output a placeholder SSN — verify the guardrail
              blocks).

- [ ] **Examples:**
  - [ ] `examples/glow/04_composition.py` — covers Example 5 and
        5b from the plan (set-based ensemble + composer-in-composer).
  - [ ] `examples/glow/06_mcp_typing.py` — class-based MCP usage.

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

- `compose()` returns an instance of `Composer` (asserted via
  `isinstance(result, Composer)`) for every valid spec.
- The composer-in-composer test runs end-to-end: `compose([a,
  compose({b, c}, merge=fn), d])("input")` produces a result that
  is observably the concatenation of three stages.
- `dict[str, Composable]` routing dispatches with composer-valued
  routes (verified in integration).
- The class-based `MCPServer` and the `Annotated` pattern both
  result in working multi-MCP wiring; the agent sees tools from
  both servers.
- An output guardrail expressed via `Annotated[str, no_pii]`
  blocks the response in a controlled test.

## Verification commands

```bash
uv run ruff check src/t0_agents/glow/ tests/test_glow/
uv run mypy src/t0_agents/glow/
uv run pytest tests/test_glow/ -q -m "not integration"

docker compose up -d
uv run pytest tests/test_glow/ -q
uv run python examples/glow/04_composition.py
uv run python examples/glow/06_mcp_typing.py
docker compose down
```

## Cross-references

- Plan §"What `glow` adds" items 5, 6, 8.
- Plan §"Composability by contract — already guaranteed by v1.0".
- Plan §"API design" examples 5, 5b, 7, 9.
- Plan §"Technical architecture" → "Native composition via
  `compose(spec, *, merge=None)`", "Typed MCP integration", "Guardrails
  via `Annotated`".
- v1.0 source:
  `src/t0_agents/compose/topology.py:49-94` (`Composable` Protocol),
  `src/t0_agents/compose/builtins.py:124` (`sequential`),
  `src/t0_agents/compose/builtins.py:189-217` (`parallel(reducer, error_policy)`),
  `src/t0_agents/compose/builtins.py:319` (`consensus`),
  `examples/14_multi_server/main.py:29-32` (multi-MCP mount pattern),
  `src/t0_agents/core/middleware/` (`GuardrailsMiddleware`).

## Dev-log

`logs/milestone-4-dev-log.md` at end.
