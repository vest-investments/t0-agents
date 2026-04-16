# Phase C — Guardrail Abstractions

> Status: **ready to start**. Phase dep: A complete. Open questions
> resolved: T0 Q5 (no auto-emit; manual feedback path documented).
> Delivers: **Milestone M3 — Trusted Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md) §2.5
  (elicitation as the only HITL primitive)
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q5
- [`../problemas.md`](../problemas.md) — Problem 6
- Existing elicitation work:
  `docs/internal/approval-and-elicitation-integration-plan/`
- [`phase-a-agent-core.md`](phase-a-agent-core.md)

---

## 1. Goal

Expose a declarative guardrail API on top of FastMCP elicitation and
the `Agent` lifecycle. Per the user's note on Problem 6, no separate
Approval system is built — guardrails are *better abstractions* over
existing primitives.

After Phase C:

```python
from t0_agents import Agent, AgentConfig
from t0_agents.guardrails import require_approval, max_cost, max_tool_calls, block_tools

agent = Agent(
    AgentConfig(
        name="executive_assistant",
        function_name="agent",
        guardrails=[
            require_approval(tools=["send_email", "make_purchase"]),
            max_cost(per_episode=0.50),       # USD
            max_tool_calls(per_run=15),
            block_tools(["delete_database"]),
        ],
    ),
    t0=t0,
    tool_provider=client,
)
```

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/guardrails/` with:
  - `base.py` — `Guardrail` Protocol(s), `GuardrailContext` dataclass.
  - `approval.py` — `require_approval()` factory, wraps elicitation.
  - `limits.py` — `max_cost`, `max_tool_calls`, `max_turns_alias`.
  - `validators.py` — `block_tools`, simple input/output validators.
- Extension of `AgentConfig`: `guardrails: list[Guardrail] = []`.
- Integration via a `GuardedToolProvider` wrapper installed by
  `Agent.__init__` when guardrails are configured.
- New exception: `GuardrailBlockedError` (T0AgentsError subclass) for
  fatal blocks; soft blocks return error strings to the LLM.
- Tests under `tests/test_guardrails/`.
- Example `examples/10_guardrails/`.
- Public doc page `docs/guardrails.md`.
- Implementation log.

### 2.2 Out of scope

- Per-call cost prediction. Cost tracking is post-hoc via `Usage`
  data on `ChatInferenceResponse`, not a budget-aware planner.
- Content moderation models (PII scrubbing, safety classifiers). The
  framework provides hook points; bring-your-own classifier.
- Auto-emitting `t0.feedback()` for guardrail outcomes (T0 Q5
  resolution: opt-in, manual).
- Guardrails on streaming partial events (Phase E may add this; Phase
  C only defines around `run()`).

### 2.3 Non-goals

- Do not replace the FastMCP Client's `elicitation_handler`.
  Guardrails wrap the tool provider, not the elicitation channel.
- Do not auto-declare TensorZero metrics. Manual config required.

---

## 3. Constraints (from the contract)

1. `mcp/` stays standalone. `run_agent` unchanged by this phase.
2. `guardrails/` depends only on `core/` (and indirectly on a
   `ToolProvider` Protocol that ships in Phase A).
3. Elicitation is the only HITL primitive (contract §2.5). The
   `require_approval` guardrail forwards prompts via the developer's
   existing `elicitation_handler` on the Client; it does not invent a
   parallel channel.
4. Async-first.

---

## 4. Module additions

### 4.1 `src/t0_agents/guardrails/base.py` (new)

```python
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tensorzero import ToolCall


@dataclass(slots=True)
class GuardrailContext:
    """State threaded through guardrail hooks during one Agent.run().

    Mutable: counters, accumulated cost, etc. Reset per-run.
    """
    agent_name: str
    function_name: str
    user_message: str
    turn: int = 0
    tool_calls_made: int = 0
    cost_so_far: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Guardrail(Protocol):
    """Hooks fired by Agent.run() and the GuardedToolProvider wrapper.

    All hooks default to no-op; implementers override only what they
    need. Returning None = pass; returning a string = surface that
    string to the LLM as a ToolResult; raising GuardrailBlockedError =
    abort the run.
    """

    name: str

    async def before_run(self, ctx: GuardrailContext) -> None: ...

    async def before_tool(
        self,
        ctx: GuardrailContext,
        tool_call: ToolCall,
    ) -> str | None: ...
    """Return a string to short-circuit (becomes the ToolResult), None to pass."""

    async def after_tool(
        self,
        ctx: GuardrailContext,
        tool_call: ToolCall,
        result_str: str,
    ) -> None: ...

    async def after_run(
        self,
        ctx: GuardrailContext,
        result: "AgentResult",
    ) -> None: ...
```

### 4.2 `src/t0_agents/guardrails/approval.py` (new)

```python
from typing import Any
from fastmcp.client.elicitation import ElicitResult


def require_approval(
    *,
    tools: list[str],
    prompt_template: str = "Approve calling {tool_name} with {arguments}?",
) -> Guardrail:
    """Returns a Guardrail that pauses on listed tools and asks the user
    for binary approval via FastMCP elicitation.

    Implementation note: this guardrail does NOT install its own
    elicitation handler. It calls into the tool_provider's existing
    handler indirectly by triggering an elicitation request from a
    framework-internal context.

    Caveat (FastMCP Q3 reminder): in the serving phase, the elicitation
    handler is per-Client and per-session. require_approval works
    correctly only when each Agent.run() targets a Client whose
    elicitation handler can answer (TUI session, WS-piped browser, etc.).
    """

    class RequireApproval:
        name = f"require_approval(tools={tools})"

        async def before_tool(self, ctx, tool_call):
            if tool_call.name not in tools:
                return None
            # Bridge to FastMCP elicitation via the framework's internal
            # asker (see core/elicitation_forwarder.py for the cross-
            # boundary path used by serve/).
            approved = await _ask_via_elicitation(
                ctx,
                message=prompt_template.format(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                ),
            )
            if not approved:
                return f"ERROR: user declined to run {tool_call.name}"
            return None

        async def before_run(self, ctx): ...
        async def after_tool(self, ctx, tool_call, result_str): ...
        async def after_run(self, ctx, result): ...

    return RequireApproval()
```

### 4.3 `src/t0_agents/guardrails/limits.py` (new)

```python
from t0_agents.core.exceptions import GuardrailBlockedError


def max_cost(*, per_episode: float, currency: str = "USD") -> Guardrail:
    """Track cumulative cost across turns; abort if exceeded.

    Cost source: ChatInferenceResponse.usage.cost (Usage dataclass).
    Updated in after_tool — actually updated in Agent.run after each
    inference response (handled by the wrapper, see §4.6).
    """

    class MaxCost:
        name = f"max_cost({per_episode} {currency})"

        async def before_tool(self, ctx, tool_call):
            if ctx.cost_so_far >= per_episode:
                raise GuardrailBlockedError(
                    f"Cost limit {per_episode} {currency} exceeded "
                    f"(spent {ctx.cost_so_far:.4f})"
                )
            return None

        # ... other hooks no-op

    return MaxCost()


def max_tool_calls(*, per_run: int) -> Guardrail:
    class MaxToolCalls:
        name = f"max_tool_calls({per_run})"

        async def before_tool(self, ctx, tool_call):
            if ctx.tool_calls_made >= per_run:
                raise GuardrailBlockedError(
                    f"Tool-call cap {per_run} hit in a single run"
                )
            return None

        async def after_tool(self, ctx, tool_call, result_str):
            ctx.tool_calls_made += 1

        # ... other hooks no-op

    return MaxToolCalls()
```

### 4.4 `src/t0_agents/guardrails/validators.py` (new)

```python
def block_tools(names: list[str]) -> Guardrail:
    """Hard-block specific tools. Useful as a defense-in-depth layer
    even if the variant doesn't advertise the tool.
    """

    class BlockTools:
        name = f"block_tools({names})"

        async def before_tool(self, ctx, tool_call):
            if tool_call.name in names:
                return f"ERROR: tool {tool_call.name!r} is blocked by policy"
            return None

        # ... other hooks no-op

    return BlockTools()
```

### 4.5 `GuardedToolProvider` wrapper (in `core/agent.py` or `guardrails/runtime.py`)

```python
class GuardedToolProvider:
    """Wraps a ToolProvider, calling guardrail hooks around every
    call_tool invocation. Constructed inside Agent.__init__ when
    AgentConfig.guardrails is non-empty.
    """

    def __init__(self, inner: ToolProvider, guardrails: list[Guardrail], ctx: GuardrailContext):
        self._inner = inner
        self._guardrails = guardrails
        self._ctx = ctx

    async def list_tools(self):
        return await self._inner.list_tools()

    async def call_tool(self, name, arguments, *, timeout=None):
        # Synthesise a ToolCall-like for the hooks (we only have name/args here)
        tool_call = _SyntheticToolCall(name=name, arguments=arguments)
        for g in self._guardrails:
            short_circuit = await g.before_tool(self._ctx, tool_call)
            if short_circuit is not None:
                return _short_circuit_result(short_circuit)
        result = await self._inner.call_tool(name, arguments, timeout=timeout)
        result_str = _stringify(result)
        for g in self._guardrails:
            await g.after_tool(self._ctx, tool_call, result_str)
        return result
```

`_short_circuit_result` returns a `CallToolResult` shaped so the
existing `serialize_tool_result` / `build_tool_result` adapters
produce the right `ToolResult.result` string for the LLM. No changes
to `mcp/adapter.py`.

### 4.6 `Agent.run()` integration (extend)

```python
async def run(self, user_message, *, episode_id=None):
    ctx = GuardrailContext(
        agent_name=self._config.name,
        function_name=self._config.function_name,
        user_message=user_message,
    )
    for g in self._config.guardrails:
        await g.before_run(ctx)

    wrapped_provider = (
        GuardedToolProvider(self._tool_provider, self._config.guardrails, ctx)
        if self._config.guardrails
        else self._tool_provider
    )

    try:
        result = await run_agent(
            self._t0,
            wrapped_provider,
            user_message,
            self._config.function_name,
            episode_id=episode_id,
            # ... other kwargs ...
        )
    except GuardrailBlockedError:
        raise
    finally:
        for g in self._config.guardrails:
            await g.after_run(ctx, result if 'result' in locals() else None)

    return result
```

### 4.7 Feedback emission (T0 Q5 resolution — manual, documented)

**The framework does NOT auto-emit `t0.feedback(...)` for guardrail
outcomes.** Per T0 Q5 — metrics must be pre-declared in
`tensorzero.toml`, and forcing a `[metrics.action_approved]` block on
every deployer is over-reach.

The guardrail returns a `result.guardrail_log` field on `AgentResult`
(see §4.8) so callers can introspect what fired and emit feedback if
they want:

```python
# Caller code (in a doc example):
result = await agent.run("...")

if any(entry.kind == "rejection" for entry in result.guardrail_log):
    await t0.feedback(
        metric_name="action_approved",   # MUST be declared in tensorzero.toml
        inference_id=result.inference_ids[-1],
        value=False,
    )
```

### 4.8 `AgentResult` extension (contract §6.2-compliant)

Adds one optional field (default empty list, preserves consumers):

```python
@dataclass
class AgentResult:
    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
    guardrail_log: list[GuardrailLogEntry] = field(default_factory=list)  # NEW
```

```python
@dataclass(slots=True)
class GuardrailLogEntry:
    guardrail_name: str
    hook: Literal["before_run", "before_tool", "after_tool", "after_run"]
    kind: Literal["pass", "rejection", "block", "modification"]
    tool_name: str | None = None
    detail: str = ""
```

---

## 5. Tests

Location: `tests/test_guardrails/`.

### 5.1 Unit (offline)

- [ ] `test_guardrail_protocol_compliance` — built-ins satisfy the
      Protocol (runtime_checkable assertion).
- [ ] `test_block_tools_short_circuits` — `block_tools(["foo"])`
      returns the error string, `_inner.call_tool` is not invoked.
- [ ] `test_max_cost_raises_when_exceeded` — `GuardrailBlockedError`.
- [ ] `test_max_tool_calls_increments_on_after_tool` — counter logic.
- [ ] `test_require_approval_calls_elicitation` — mock elicitation
      handler returns `accept` / `decline`; verify both branches.
- [ ] `test_guardrail_log_populated` — every hook fires the right
      log entry on `AgentResult.guardrail_log`.

### 5.2 Integration (live gateway, marked `@pytest.mark.integration`)

- [ ] `test_require_approval_e2e` — CLI elicitation handler that
      auto-declines; agent run with `delete_*` tool produces a
      "user declined" response from the LLM.
- [ ] `test_max_cost_e2e` — short-budget ceiling fires; assert
      `GuardrailBlockedError` propagates.

### 5.3 Regression

- [ ] `tests/test_mcp/`, `tests/test_core/`, `tests/test_memory/`
      unchanged.

---

## 6. Example

Location: `examples/10_guardrails/`.

- `server.py` — tools: `send_email(to, subject, body)`,
  `delete_task(task_id)`, `read_calendar()`. Mix of "always safe" and
  "needs approval".
- `main.py` — set up `Agent` with all four built-in guardrails,
  drive a CLI elicitation handler, demonstrate one approval, one
  decline, and a budget breach.
- `README.md` — explains where each guardrail intercepts and how to
  emit guardrail feedback to T0 manually.

---

## 7. Documentation

- New: `docs/guardrails.md` — public-facing guide.
- Sections:
  - "What is a `Guardrail`?" — Protocol + composition model.
  - "Built-in guardrails" — `require_approval`, `max_cost`,
    `max_tool_calls`, `block_tools`.
  - "Writing your own" — three small samples (regex validator,
    PII scrubber stub, schedule-bound execution gate).
  - "Emitting feedback for guardrail outcomes" — explicit code
    snippet for the manual `t0.feedback(...)` path; warning that
    `[metrics.action_approved]` (or whatever name) must be
    pre-declared in `tensorzero.toml`.
- Update `developer-guide.md` §4 (public API), §12 (knobs).
- `CHANGELOG.md` — `0.4.0`.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/guardrails/__init__.py`.
- [ ] `src/t0_agents/guardrails/base.py` — Protocol + `GuardrailContext`
      + `GuardrailLogEntry`.
- [ ] `src/t0_agents/guardrails/approval.py`.
- [ ] `src/t0_agents/guardrails/limits.py`.
- [ ] `src/t0_agents/guardrails/validators.py`.
- [ ] `src/t0_agents/core/exceptions.py` — add `GuardrailBlockedError`.
- [ ] `src/t0_agents/core/types.py` — extend `AgentResult` with
      `guardrail_log` field.
- [ ] `src/t0_agents/core/config.py` — `guardrails` field.
- [ ] `src/t0_agents/core/agent.py` — wrap `tool_provider` with
      `GuardedToolProvider` when guardrails are present.
- [ ] `uv run ruff check src/` green.
- [ ] `uv run mypy src/` green in strict mode.

### 8.2 Tests

- [ ] All §5.1 unit tests.
- [ ] §5.2 integration tests (live gateway).
- [ ] No regression in earlier phases' tests.

### 8.3 Example

- [ ] `examples/10_guardrails/` complete.

### 8.4 Docs

- [ ] `docs/guardrails.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `docs/internal/agentic-feature/logs/phase-c-implementation-log.md`
      with T0 Q5 reaffirmation, M3 release-notes snippet.

---

## 9. Acceptance criteria (Milestone M3)

1. `from t0_agents.guardrails import require_approval, max_cost,
   max_tool_calls, block_tools` works.
2. `examples/10_guardrails/` shows all four guardrails firing
   correctly.
3. The doc page documents the manual feedback path with an example
   `[metrics.action_approved]` config snippet.
4. Existing phases (A, B) tests untouched and green.
5. `Agent` without guardrails behaves byte-identically to Phase B's
   acceptance criteria.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| `require_approval` triggers infinitely if elicitation handler always re-prompts | low | Bound by `max_turns`; document the pattern in the example. |
| Cost tracking lags behind reality (T0 returns cost post-call) | medium | Acceptable — guardrail checks pre-call cost vs limit, not predictive. Document the semantic. |
| Guardrail wrapper hides bugs in the underlying ToolProvider | low | Keep `_inner.call_tool` invocation transparent; surface upstream exceptions verbatim. |
| Elicitation handler isn't wired (no human at the other end) | medium | `require_approval` falls through to a soft "no handler" error string, surfaced to the LLM (consistent with existing tool-error contract). |

---

## 11. Estimated surface

- **New files:** 5 src + 3 test + 3 example + 1 doc.
- **Modified files:** 4 (`core/agent.py`, `core/config.py`,
  `core/types.py`, `core/exceptions.py`) + `CHANGELOG.md` +
  `developer-guide.md`.
- **Net new public symbols:** 7 (`Guardrail`, `GuardrailContext`,
  `GuardrailLogEntry`, `GuardrailBlockedError`, `require_approval`,
  `max_cost`, `max_tool_calls`, `block_tools`).
- **New deps:** 0.
