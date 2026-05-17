# M2 — `core-decorator`

> **Status:** Not started.
> **Branch:** `experimental/glow`.
> **Dependencies:** M1 (needs `output_inference` and `FluentRun`).
> **Blocks:** M3, M4, M5 (every later milestone uses `App` and
> `@app.agent`).

## Purpose

Land the headline DX: `App` + `@app.agent` + 4-line hello-world
works end-to-end. This is the **first milestone that needs a real
TensorZero gateway** because it actually drives `Agent.invoke()`
and shapes the response. Event selectors (`Thought`, `ToolCall`,
`Text`) also land here so streaming demos work from M2 onwards.

After M2, a user can write the entire hello-world example from the
plan (Example 1) and structured output (Example 2) and have it
work. They cannot yet use `Depends`, tools, composition, or MCP
typing — those are M3/M4.

## Scope

### In

- `src/t0_agents/glow/app.py` — `App` class:
  - `App(t0: AsyncTensorZeroGateway, *, servers: list[type[MCPServer]] | None = None)`
    constructor (servers param wired in M4; in M2 only `t0=` matters).
  - `App._agent_registry: dict[str, _AgentSpec]`.
  - `App.dependency_overrides: dict[Callable, Callable]` (wired
    in M3; stubbed empty here).
  - `async def build(self) -> None` — eagerly constructs every
    registered agent (catches config errors early).
  - `async def lifespan(self) -> AsyncContextManager` — manages
    gateway + MCP lifecycle (MCP wiring is M4; M2 just yields).
- `src/t0_agents/glow/decorators.py` — `@app.agent("name", *, function_name="agent")`:
  - Walks `inspect.signature` + `get_type_hints`.
  - Reads `fn.__doc__` (via `inspect.cleandoc`) as the system
    prompt.
  - Calls `output_mode_from_return_type(fn)` to pick `OutputMode`.
  - Registers an `_AgentSpec` in `App._agent_registry`.
  - Returns a callable wrapper: `wrapper(input: str) -> FluentRun[T]`.
- `src/t0_agents/glow/selectors.py` — `Selector` generic over
  event type:
  - `Thought = Selector(ThoughtEvent)`, `ToolCall = Selector(ToolCallEvent)`,
    `Text = Selector(TextDeltaEvent)`, `ToolResult = Selector(ToolResultEvent)`,
    `TurnEnd = Selector(TurnEndEvent)`, `Done = Selector(DoneEvent)`,
    `Error = Selector(ErrorEvent)`.
  - `Selector.where(predicate)` — returns a new `Selector` with
    chained predicate.
  - `Selector.map(transform)` — returns a new `Selector` with
    chained transform.
- `src/t0_agents/glow/handle.py` updates:
  - `FluentRun.events(*selectors)` now actually filters/transforms
    using the selectors. Empty `selectors` yields every event.
- `examples/glow/01_hello.py` — hello-world (Example 1 in the plan).
- `examples/glow/02_structured.py` — structured output (Example 2).
- `tests/test_glow/test_app.py` — unit tests for `App` and the
  decorator (mock `Agent`).
- `tests/test_glow/test_selectors.py` — unit tests for selectors
  over a fake event stream.
- `tests/test_glow/test_integration.py` — first integration test;
  marked `@pytest.mark.integration`. Verifies hello-world end-to-end
  against a live TZ gateway.

### Out

- `@app.tool` decorator — M3.
- `Depends` injection — M3.
- `compose()` and native-type composition — M4.
- Typed MCP integration — M4.
- `Annotated[T, Guardrail]` — M4.

## Checklist

- [ ] **Commit 4: feat(glow): `App` + `@app.agent` decorator (chat + structured)**
  - [ ] `glow/app.py`:
        - [ ] `App.__init__(self, t0, *, servers=None)` stores
              `t0`, allocates the registry.
        - [ ] `App.agent(name, *, function_name="agent")` returns
              a decorator factory. The decorator captures the
              function, extracts `fn.__doc__`, calls
              `output_mode_from_return_type`, builds an
              `_AgentSpec`, and returns a wrapper callable.
        - [ ] `App._lookup_or_build(spec)` constructs the
              underlying `Agent` (lazily — first call). Stash the
              `Agent` on the spec for reuse.
        - [ ] `App.build()` iterates the registry and force-builds
              every agent. Surface config errors.
        - [ ] `App.lifespan()` is an `@asynccontextmanager` that
              currently just yields (M4 will add MCP teardown).
        - [ ] Add `wrapper._agent` and `wrapper._spec` as public
              attributes so users can degrade to the base API.
  - [ ] `glow/decorators.py`:
        - [ ] `agent` factory (used by `App.agent`); split out so
              it can be reused (potentially by `@app.tool` in M3).
  - [ ] `glow/handle.py`:
        - [ ] `FluentRun.events(*selectors)` builds the `Run`
              immediately, then iterates `run.events()` and yields
              `selector.transform(event)` for events matching
              `isinstance(event, selector.event_type) and selector.predicate(event)`.
        - [ ] Empty `selectors` → yield every event (existing
              behavior preserved).
  - [ ] Unit tests `tests/test_glow/test_app.py`:
        - [ ] Decorator registers an agent; `wrapper("hi")`
              returns a `FluentRun`.
        - [ ] Mock `Agent` returning `AgentResult(value="hello")`
              → `await wrapper("hi")` returns `"hello"`.
        - [ ] `wrapper._agent` is the underlying `Agent`.
        - [ ] `App.build()` constructs every registered agent
              once; subsequent invocations reuse.
        - [ ] `App.agent("name")` with conflicting names raises
              `ConfigurationError`.

- [ ] **Commit 5: feat(glow): event selectors + filtering**
  - [ ] `glow/selectors.py`:
        - [ ] `class Selector(Generic[E])`:
              `__init__(self, event_type: type[E], predicate=lambda _: True, transform=lambda e: e)`.
        - [ ] `.where(pred)` — returns new `Selector` with
              `predicate=lambda e: self.predicate(e) and pred(e)`.
        - [ ] `.map(fn)` — returns new `Selector` with
              `transform=fn`.
        - [ ] Module-level instances: `Thought`, `ToolCall`,
              `Text`, `ToolResult`, `TurnEnd`, `Done`, `Error`.
              `Text = Selector(TextEndEvent)` (whole-text variant
              is more useful as a selector default than the delta
              variant; users can pick `Text.where(...)` for delta
              filters).
        - [ ] Re-export from `glow/__init__.py`.
  - [ ] Wire `FluentRun.events(*selectors)` to use the selectors:
        - [ ] Multiple selectors → union (event passes if any
              selector matches).
        - [ ] Transform of the matching selector is applied to
              the yielded value.
  - [ ] Unit tests `tests/test_glow/test_selectors.py`:
        - [ ] Feed a synthetic event stream
              `[TextEndEvent, ToolCallEvent, ThoughtEvent]` and
              verify each selector picks the right one.
        - [ ] `Thought.where(lambda t: "foo" in t.text)` filters.
        - [ ] `ToolCall.map(lambda tc: f"calling {tc.name}")`
              yields the transformed string.
        - [ ] Empty selectors → every event yielded (smoke test).

- [ ] **Examples:**
  - [ ] `examples/glow/01_hello.py` — runnable end-to-end (uses
        the `agent` TZ function from existing examples'
        `tensorzero.toml`).
  - [ ] `examples/glow/02_structured.py` — defines `TripPlan`
        Pydantic model, declares `@app.agent("planner") async def
        plan(query: str) -> TripPlan`, runs against the structured
        TZ function from `examples/17_structured/`.

- [ ] **Integration tests `tests/test_glow/test_integration.py`:**
  - [ ] `pytestmark = pytest.mark.integration` at module top.
  - [ ] Hello-world test: spin up the gateway, run
        `examples/glow/01_hello.py`'s flow, assert the response is
        non-empty.
  - [ ] Structured-output test: same for `02_structured.py`.

- [ ] **Run validation:**
  ```bash
  uv run ruff check src/ tests/
  uv run mypy src/
  uv run pytest -m "not integration"

  # And integration:
  docker compose up -d
  uv run pytest tests/test_glow/test_integration.py -q
  docker compose down
  ```

## Acceptance criteria

- `examples/glow/01_hello.py` and `examples/glow/02_structured.py`
  both run cold against a live TZ gateway and print a success
  banner.
- `mypy --strict` clean on every new file.
- `await tutor("x")` returns the unwrapped value typed as the
  function's return annotation (verified by mypy reveal-type in a
  test).
- Event selectors filter and transform correctly in unit tests
  with a fake stream.
- `wrapper._agent` exposes the underlying `Agent` (escape-hatch
  contract honored).

## Verification commands

```bash
uv run ruff check src/t0_agents/glow/ tests/test_glow/
uv run mypy src/t0_agents/glow/
uv run pytest tests/test_glow/ -q -m "not integration"

# Integration (requires gateway):
docker compose up -d
uv run pytest tests/test_glow/test_integration.py -q
uv run python examples/glow/01_hello.py
uv run python examples/glow/02_structured.py
docker compose down
```

## Cross-references

- Plan §"What `glow` adds" items 1–3, 7.
- Plan §"API design — disruptive examples" 1, 2, 3, 4.
- Plan §"Technical architecture" → "Adapting to v1.0 primitives"
  and "Event selectors" at
  `~/.claude/plans/la-idea-es-tener-glowing-gem.md`.
- v1.0 source to re-use:
  `src/t0_agents/core/agent.py:43,227-293` (`Agent`, `invoke()`),
  `src/t0_agents/core/config.py:24` (`AgentConfig`),
  `src/t0_agents/stream/events.py` (every event type).

## Dev-log

When the milestone completes, drop `logs/milestone-2-dev-log.md`.
