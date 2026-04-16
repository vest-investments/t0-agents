# Phase A Implementation Log

Phase: A (Agent Core)
Milestone: M1 — Declarative Agent
Date: 2026-04-15

## Divergences from Spec

### D1: tool_timeout None resolution in Agent.run()

**Spec (§4.4)** shows `tool_timeout=self._config.tool_timeout` passed directly
to `run_agent`. However, `run_agent` declares `tool_timeout: float = 30.0`
(not `float | None`), and spec §5.3 says "nothing else changes in `mcp/`"
beyond adding `tags`/`namespace`.

**Resolution:** `Agent.run()` resolves `None` to `30.0` (matching `run_agent`'s
default) before passing:

```python
tool_timeout = (
    self._config.tool_timeout
    if self._config.tool_timeout is not None
    else _DEFAULT_TOOL_TIMEOUT  # 30.0
)
```

`AgentConfig.tool_timeout` keeps the contract type `float | None = 30.0`.
Semantics: `None` = "use framework default" in Phase A. True "no timeout"
deferred to a future phase that widens `run_agent`.

**Impact:** None on callers. `AgentConfig(tool_timeout=None)` behaves
identically to `AgentConfig(tool_timeout=30.0)` in Phase A.

### D2: FakeMCPClient extraction

**Spec (§11, Risks)** suggests `tests/helpers/fakes.py` for shared fakes.
**Actual:** Extracted to `tests/_fake_mcp.py` following the existing
`tests/_fake_t0.py` convention. The existing pattern was cleaner and
consistent with the codebase.

## Contract Amendments

None. The integration contract (`00-integration-contract.md`) is unchanged.

## Release Notes Snippet

> **v0.2.0 — Declarative Agent (M1)**
>
> New `Agent` class wraps `run_agent` with `AgentConfig` for one-liner
> invocations. New `ToolProvider` Protocol, `T0AgentsError` base exception,
> and `tags`/`namespace` support on `run_agent`. See `docs/agent.md`.
