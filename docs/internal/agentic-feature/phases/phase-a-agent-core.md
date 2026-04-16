# Phase A — `Agent` Core

> Status: **ready to start**. Blockers: none.
> Delivers: **Milestone M1 — Declarative Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md)
- [`../01-implementation-plan.md`](../01-implementation-plan.md)
- [`../02-open-questions.md`](../02-open-questions.md)
- [`../problemas.md`](../problemas.md) — Problem 1
- Source of truth today: `src/t0_agents/mcp/loop.py` (`run_agent`)

---

## 1. Goal

Ship a declarative `Agent` class that encapsulates the configuration
that developers currently pass to `run_agent` and exposes a clean
`.run()` method. The class is a composition wrapper: it delegates
to `run_agent` today, and becomes the extension point for every later
phase (memory, guardrails, streaming, serving, …).

After Phase A:

```python
from t0_agents import Agent, AgentConfig

agent = Agent(
    AgentConfig(name="assistant", function_name="agent"),
    t0=t0,
    tool_provider=client,
)
result = await agent.run("What is the weather in Tokyo?")
```

works identically to the equivalent `run_agent(t0, client, "...", "agent")`
call today — no change in turns, IDs, response text, error paths, or
ClickHouse writes.

---

## 2. Scope

### 2.1 In scope

- New file: `src/t0_agents/core/agent.py` — `class Agent`.
- New file: `src/t0_agents/core/config.py` — `AgentConfig` dataclass.
- Extension of `src/t0_agents/core/types.py` — `ToolProvider` Protocol.
- Extension of `src/t0_agents/core/exceptions.py` — `T0AgentsError`
  base class; `MaxTurnsExceededError` becomes a subclass;
  `ConfigurationError` added.
- Re-exports in `src/t0_agents/core/__init__.py` and
  `src/t0_agents/__init__.py`.
- Parity tests under `tests/test_core/test_agent.py`.
- Example `examples/08_agent_class/`.
- Public doc page `docs/agent.md`.
- Implementation log
  `docs/internal/agentic-feature/logs/phase-a-implementation-log.md`.

### 2.2 Out of scope (reserved for later phases)

- Memory integration — Phase B.
- Guardrail integration — Phase C.
- Streaming — Phase E.
- Serving — Phase D.
- Multi-agent composition — Phase G.
- Sync wrapper (`Agent.run_sync`) — deferred; add only when a real
  user calls for it.
- Any modification to `mcp/` beyond imports; `mcp/loop.py:run_agent`
  signature is frozen (see contract §6.1).

### 2.3 Non-goals

- Do not move logic out of `run_agent` into `Agent`. The delegation
  direction is **Agent → run_agent**. Later phases may invert this if
  necessary, but not as part of Phase A.
- Do not introduce additional lifecycle responsibilities on the Agent
  (opening/closing `t0`, managing loops, installing handlers).
- Do not add `Agent.feedback(...)` in this phase; feedback remains a
  direct `t0.feedback(...)` call from the caller. Captured IDs on
  `AgentResult` are enough.

---

## 3. Constraints (restated from the contract)

1. `mcp/` stays standalone. `run_agent` signature frozen.
2. `core/` is the single shared import target.
3. Extras are opt-in. Phase A adds **no new deps**.
4. TensorZero-first. `AgentConfig` is a pass-through for T0 knobs;
   nothing is invented.
5. Async-first.
6. Backwards compatibility: every existing test under
   `tests/test_mcp/` passes unchanged.

See [`../00-integration-contract.md`](../00-integration-contract.md) for
rationale.

---

## 4. Module additions

### 4.1 `src/t0_agents/core/config.py` (new)

```python
# pseudo-signature — final types match tensorzero SDK
from dataclasses import dataclass

from tensorzero import ToolChoice


@dataclass(slots=True)
class AgentConfig:
    """Declarative configuration for an Agent.

    Every field maps onto a TensorZero primitive; nothing here is
    invented by t0-agents. See 00-integration-contract.md §3.2 for the
    forward-compatibility rules on this dataclass.
    """

    # Identity
    name: str
    function_name: str

    # Loop budget
    max_turns: int = 20
    tool_timeout: float | None = 30.0
    parallel_tools: bool = False

    # TensorZero knobs forwarded verbatim to `t0.inference(...)`
    dryrun: bool = False
    tool_choice: ToolChoice | None = None
    tags: dict[str, str] | None = None          # observability
    namespace: str | None = None                # experimentation (see T0 Q3)

    def __post_init__(self) -> None:
        # Validate name/function_name non-empty; max_turns positive; etc.
        ...
```

Notes:

- `slots=True` keeps the dataclass lean and catches typos.
- `tags` and `namespace` are **added to `run_agent`'s forwarded kwargs**
  — see §5.2. Those kwargs are new optional parameters on
  `run_agent`, defaulting to `None`, so the contract's signature-freeze
  rule (§6.1) is honoured.
- Every default matches today's defaults in `run_agent`.

### 4.2 `src/t0_agents/core/types.py` (extend)

Add:

```python
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    """Structural type used by Agent for tool discovery and execution.

    Matches the slice of `fastmcp.Client` that run_agent consumes:
    - `list_tools() -> list[<MCPTool>]`
    - `call_tool(name, arguments, timeout=None) -> <CallToolResult>`

    A `fastmcp.Client` satisfies this Protocol structurally — no adapter
    needed. Third parties can implement it to plug in a non-FastMCP
    tool source (e.g. a direct tensorzero `[tools.*]` list).
    """

    async def list_tools(self) -> list[Any]: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any: ...
```

Notes:

- `Any` is used in return annotations so the Protocol remains upstream
  compatible with `fastmcp`'s concrete types without forcing a
  `fastmcp` import in `core/types.py` (which is also imported by
  consumers that don't want fastmcp yet).
- The Protocol is `runtime_checkable` so mocks in later phases can
  assert conformance.

### 4.3 `src/t0_agents/core/exceptions.py` (extend)

```python
class T0AgentsError(Exception):
    """Base class for every exception raised by t0-agents."""


class MaxTurnsExceededError(T0AgentsError):
    """Raised when the agent loop exceeds `max_turns` without producing
    a tool-call-free response.
    """
    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Agent loop exceeded {max_turns} turns without terminating")
        self.max_turns = max_turns


class ConfigurationError(T0AgentsError):
    """Raised when AgentConfig is self-inconsistent or an Agent is
    constructed with incompatible collaborators.
    """
```

Today `MaxTurnsExceededError` inherits directly from `Exception`. This
change is a widening — every callers' `except MaxTurnsExceededError:`
still matches. Callers who catch `T0AgentsError` gain broader coverage.

### 4.4 `src/t0_agents/core/agent.py` (new)

```python
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.mcp.loop import run_agent


class Agent:
    """Declarative wrapper around run_agent.

    See 00-integration-contract.md §3 for the full contract and §4 for
    the composition sketch.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider,
    ) -> None:
        if not isinstance(tool_provider, ToolProvider):
            raise ConfigurationError(
                f"tool_provider={tool_provider!r} does not implement ToolProvider "
                f"(missing list_tools / call_tool)"
            )
        self._config = config
        self._t0 = t0
        self._tool_provider = tool_provider

    @property
    def config(self) -> AgentConfig:
        return self._config

    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
    ) -> AgentResult:
        """One-shot agent invocation. Returns when the loop terminates.

        No side effects on `self` — safe to call multiple times in
        sequence (concurrent use depends on the underlying `t0` and
        `tool_provider`; see contract §3.5).
        """
        return await run_agent(
            self._t0,
            self._tool_provider,
            user_message,
            self._config.function_name,
            episode_id=episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=self._config.tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._config.tags,              # new kwarg on run_agent (§5.2)
            namespace=self._config.namespace,    # new kwarg on run_agent (§5.2)
        )
```

### 4.5 `src/t0_agents/core/__init__.py` (new content)

```python
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.core.types import AgentResult, ToolProvider

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ConfigurationError",
    "MaxTurnsExceededError",
    "T0AgentsError",
    "ToolProvider",
]
```

### 4.6 `src/t0_agents/__init__.py` (extend)

Current content re-exports `run_agent`, `AgentResult`,
`MaxTurnsExceededError`, `ToolChoice`. Add:

```python
from t0_agents.core import (
    Agent,
    AgentConfig,
    ConfigurationError,
    T0AgentsError,
    ToolProvider,
)
```

Append each symbol to `__all__`. Do **not** remove the existing
exports.

---

## 5. Implementation strategy

### 5.1 Agent delegates to `run_agent`

The simplest, safest direction. Any logic encoded in `run_agent` —
recoverable errors, inference-id sentinels, elicitation timeouts —
carries through unchanged because `Agent.run` is a thin wrapper. No
regression in existing `mcp/` semantics is possible.

Why not invert (make `run_agent` call `Agent.run`)? Three reasons:

1. **Risk.** Inverting means rewriting the function that 100% of
   today's users depend on. Every existing test would need re-validation.
2. **Semantics.** `run_agent` takes no collaborators (no memory, no
   guardrails); `Agent` does. The wrapping direction should encode
   that — the thing with more responsibilities wraps the thing with
   fewer.
3. **Standalone constraint.** Contract §2.1 requires `mcp/` to work
   without `core/`. If `run_agent` imported `Agent`, that constraint
   breaks.

### 5.2 Widening `run_agent` for `tags` and `namespace`

Two new optional kwargs on `run_agent`:

```python
async def run_agent(
    ...,
    tags: dict[str, str] | None = None,
    namespace: str | None = None,
) -> AgentResult: ...
```

Forwarded to every `t0.inference(...)` call inside the loop. Default
`None` → not passed → today's behaviour. Per contract §6.1, widening
with None-default kwargs is allowed.

Update `mcp/loop.py` test suite:

- Add one test confirming `tags=None` / `namespace=None` produces the
  same inference calls as today.
- Add one test confirming non-None values are forwarded verbatim.

### 5.3 Nothing else changes in `mcp/`

No adapter changes, no elicitation changes, no new executor paths.
Phase A is additive to `core/` and adds two optional kwargs to
`run_agent`.

---

## 6. Tests

Location: `tests/test_core/test_agent.py` (new file).

### 6.1 Unit tests (offline, no gateway)

Reuse the existing `FakeT0` / `FakeMCPClient` duck-typed stubs from
`tests/test_mcp/test_loop.py` (or extract them to a shared helper).

Required test cases:

- [ ] `test_agent_defaults` — construct `Agent` with a minimal
      `AgentConfig` and confirm it delegates to `run_agent` with the
      expected kwargs.
- [ ] `test_agent_forwards_every_config_field` — parametrised over
      every `AgentConfig` field; each one reaches the `run_agent`
      mock.
- [ ] `test_agent_tool_provider_protocol_check` — passing an object
      that doesn't satisfy `ToolProvider` raises `ConfigurationError`.
- [ ] `test_agent_episode_id_plumbing` — `Agent.run(..., episode_id="x")`
      forwards `episode_id`; omitted `episode_id` forwards `None`.
- [ ] `test_agent_max_turns_exceeded_propagates` — the error bubbles
      out unchanged.
- [ ] `test_config_validation` — negative `max_turns`, empty
      `function_name` → `ConfigurationError` at construction.
- [ ] `test_agent_concurrent_runs_serial` — two awaited-sequentially
      `run()` calls share the same config but produce independent
      results.

### 6.2 Parity tests (offline, no gateway)

Location: `tests/test_core/test_agent_parity.py`.

Goal: for every test currently in `tests/test_mcp/test_loop.py` that
exercises `run_agent` directly, add a parity test that exercises
`Agent.run` with the same inputs and asserts the same output.

- [ ] Happy path with text response.
- [ ] Happy path with thought-only response (response_source="thought").
- [ ] Tool-call → tool-result → text response.
- [ ] `MaxTurnsExceededError`.
- [ ] Recoverable `TensorZeroError`.
- [ ] `parallel_tools=True`.

### 6.3 Integration test (live gateway)

Location: `tests/test_core/test_agent_integration.py`, module-level
`pytestmark = pytest.mark.integration`.

- [ ] `test_agent_end_to_end` — boot docker compose, instantiate
      `Agent`, call `.run()` against the `agent` function, assert
      a non-empty `response` and a valid UUID `episode_id`.

### 6.4 Backwards compat

- [ ] `uv run pytest -m "not integration"` green with **no** changes
      to any existing test file.
- [ ] `tests/test_mcp/` still runs byte-identically — no imports of
      `Agent` appear there.

---

## 7. Example

Location: `examples/08_agent_class/`.

Files:

- `server.py` — reuse the pattern from `examples/02_basic_agent/server.py`.
  One `@mcp.tool` function is enough.
- `main.py` — canonical agent usage:

  ```python
  import asyncio
  from collections.abc import Awaitable
  from typing import Any, cast

  from fastmcp import Client
  from tensorzero import AsyncTensorZeroGateway

  from t0_agents import Agent, AgentConfig

  from server import mcp


  async def main() -> None:
      gateway = cast(
          Awaitable[AsyncTensorZeroGateway],
          AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
      )
      mcp_client: Client[Any] = Client(mcp)

      async with await gateway as t0, mcp_client:
          agent = Agent(
              AgentConfig(name="weather-assistant", function_name="agent"),
              t0=t0,
              tool_provider=mcp_client,
          )

          result = await agent.run("What is the weather in Tokyo?")
          print(result.response)
          print(result.episode_id, result.turns)


  if __name__ == "__main__":
      asyncio.run(main())
  ```
- `README.md` — one-paragraph intro + a section titled
  "When to use `Agent` vs `run_agent`".
- Reuse existing `docker-compose.yml` and `config/tensorzero.toml` at
  the repo root. No new gateway config.

---

## 8. Documentation

### 8.1 New: `docs/agent.md`

Structure:

- Introduction: why `Agent` exists (quote Problem 1 verbatim).
- Minimal example (copy from §7).
- `AgentConfig` field table — every field, its type, default, and the
  T0 primitive it maps to.
- When to use `Agent` vs `run_agent` directly (the framework is
  honest: `run_agent` remains the simplest option for one-shot calls
  or tight integrations).
- Forward references to future phases — a placeholder section
  "Coming next: memory, guardrails, serving" that links to the phase
  docs once they ship.

### 8.2 Update: `docs/internal/agentic-feature/developer-guide.md`

- §4 (Public API surface) — add rows for `Agent`, `AgentConfig`,
  `ToolProvider`, `T0AgentsError`, `ConfigurationError`.
- §12 (Customization knobs) — add `tags` and `namespace` entries with
  the same format as today's bullets.

### 8.3 Update: `CHANGELOG.md`

Version bump to `0.2.0`. Summary line:

> Add declarative `Agent` class with `AgentConfig`. `run_agent` remains
> available and unchanged. Introduces `T0AgentsError` hierarchy; new
> `tags` and `namespace` kwargs on `run_agent`.

---

## 9. Checklist

Copy-paste into the PR description and tick off as you go.

### 9.1 Code

- [ ] `src/t0_agents/core/config.py` — `AgentConfig` dataclass with
      `__post_init__` validation.
- [ ] `src/t0_agents/core/types.py` — `ToolProvider` Protocol added
      (existing `AgentResult` unchanged).
- [ ] `src/t0_agents/core/exceptions.py` — `T0AgentsError` base;
      `MaxTurnsExceededError` inherits from it; `ConfigurationError`
      added.
- [ ] `src/t0_agents/core/agent.py` — `Agent` class delegating to
      `run_agent`.
- [ ] `src/t0_agents/core/__init__.py` — re-exports.
- [ ] `src/t0_agents/__init__.py` — re-exports; `__all__` updated.
- [ ] `src/t0_agents/mcp/loop.py:run_agent` — add `tags` and
      `namespace` kwargs; forward to `t0.inference(...)`.
- [ ] `uv run ruff check src/` green.
- [ ] `uv run mypy src/` green in strict mode.

### 9.2 Tests

- [ ] `tests/test_core/__init__.py` added (empty).
- [ ] `tests/test_core/test_agent.py` — every unit test from §6.1.
- [ ] `tests/test_core/test_agent_parity.py` — every parity test from
      §6.2.
- [ ] `tests/test_core/test_agent_integration.py` — the integration
      test from §6.3.
- [ ] `tests/test_mcp/test_loop.py` — new tests for the
      `tags`/`namespace` forwarding.
- [ ] `tests/test_mcp/test_adapter.py` — **unchanged**.
- [ ] `uv run pytest -m "not integration"` green.
- [ ] `docker compose up -d && uv run pytest -m integration` green.

### 9.3 Example

- [ ] `examples/08_agent_class/server.py`.
- [ ] `examples/08_agent_class/main.py`.
- [ ] `examples/08_agent_class/README.md`.
- [ ] Example runs against `docker compose up -d` with the existing
      `config/tensorzero.toml`.

### 9.4 Docs

- [ ] `docs/agent.md` created (public).
- [ ] `docs/internal/agentic-feature/developer-guide.md` §4 and §12
      updated.
- [ ] `CHANGELOG.md` entry for `0.2.0`.

### 9.5 Implementation log

- [ ] `docs/internal/agentic-feature/logs/phase-a-implementation-log.md`
      created, following the template used by
      `approval-and-elicitation-integration-plan/logs/phase-*-implementation-log.md`.
      Records: divergences from this spec, any contract amendments,
      the M1 release-notes snippet.

### 9.6 Review

- [ ] One reviewer confirms `mcp/` public API is byte-identical.
- [ ] One reviewer runs the new example locally.
- [ ] One reviewer confirms parity tests match the `run_agent` tests
      one-to-one.

---

## 10. Acceptance criteria (Milestone M1)

The phase is done and M1 ships when **all** of the following hold:

1. A user following `docs/agent.md` can import `Agent`, instantiate it
   with a gateway + client they own, call `.run()`, and get an
   `AgentResult`.
2. Every test that was green before the phase is still green.
3. `mcp/` public API — the set of symbols re-exported by
   `src/t0_agents/mcp/__init__.py` and their signatures — is
   byte-identical.
4. The phase implementation log ends with:

   ```
   ## Release notes (for CHANGELOG / M1 announcement)
   <2-3 sentences summarising the Agent class, anchored with a
   pointer to examples/08_agent_class/>
   ```

5. No PR touches `docs/internal/approval-and-elicitation-integration-plan/`
   or `src/t0_agents/mcp/elicitation.py` (keeps Phase A clearly
   scoped).

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Parity tests duplicate the `mcp/` test maintenance burden | medium | Keep parity tests minimal — one per major path. Extract shared fakes into `tests/helpers/fakes.py`. |
| `Protocol` runtime check misses subtle `Client` API drift | low | Both methods in the protocol are stable in `fastmcp >= 3.2`. Pinned in `pyproject.toml`. |
| `tags` / `namespace` kwargs introduce regressions for existing callers | low | Defaults `= None`; test that `None` → no kwarg is passed to `t0.inference`. |
| `core/__init__.py` side-effect imports slow down `from t0_agents.mcp import ...` paths | very low | Keep `core/__init__.py` to simple re-exports; no I/O at import time. Measure import time in CI. |

---

## 12. Estimated surface

- **New files:** 6 (`core/agent.py`, `core/config.py`,
  `tests/test_core/__init__.py`, `tests/test_core/test_agent.py`,
  `tests/test_core/test_agent_parity.py`,
  `tests/test_core/test_agent_integration.py`). Plus
  `examples/08_agent_class/*` (3 files) and `docs/agent.md`.
- **Modified files:** 6 (`core/types.py`, `core/exceptions.py`,
  `core/__init__.py`, `t0_agents/__init__.py`, `mcp/loop.py`,
  `tests/test_mcp/test_loop.py`, plus `CHANGELOG.md` and
  `developer-guide.md`).
- **Net new public symbols:** 5 (`Agent`, `AgentConfig`,
  `ToolProvider`, `T0AgentsError`, `ConfigurationError`).
- **New deps:** 0.

Intentional smallness: Phase A is the narrowest useful shipment. Every
later phase builds on exactly this surface.
