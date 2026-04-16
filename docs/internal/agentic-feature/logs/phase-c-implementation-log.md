# Phase C Implementation Log

Phase: C (Guardrails)
Milestone: M3 — Trusted Agent
Date: 2026-04-16

## Divergences from Spec

### D1: `require_approval(handler=...)` kwarg instead of implicit elicitation bridging

**Spec (§4.2)** sketched a framework-internal helper `_ask_via_elicitation`
that would "bridge to FastMCP elicitation via the framework's internal
asker" by reading the Client's existing elicitation handler.

**Actual:** `require_approval` takes an explicit
`handler: ApprovalHandler | None = None` kwarg whose signature is an alias
for `fastmcp.client.elicitation.ElicitationHandler`
(`async (message, response_type, params, context) -> ElicitResult`).

**Reason:** Source validation of the installed FastMCP SDK shows that
`Client.elicitation_handler` is wrapped into
`self._session_kwargs["elicitation_callback"]` at construction time
(`.venv/lib/python3.12/site-packages/fastmcp/client/client.py:254, 334-337`)
with **no public getter**. A setter exists (`set_elicitation_callback`)
but the handler can only fire through the MCP transport when the server
calls `ctx.elicit()`. `require_approval` runs client-side — there is no
supported public API to invoke the Client's elicitation handler directly.

The alternative — synthesising an in-process MCP round trip — would couple
the guardrail to transport internals and fail if the developer's handler
depends on a live `RequestContext`/`ClientSession`. A handler kwarg is the
cleanest path that also honours contract §2.5 ("elicitation is the only
HITL primitive"): the developer writes **one** handler and passes it to
both `Client(elicitation_handler=...)` (for server-side `ctx.elicit()`)
and `require_approval(handler=...)` (for client-side gates). The two
surfaces share the same signature and the same mental model.

**Impact:**

- Soft-block when `handler is None` → `"ERROR: approval required for
  <tool> but no handler is configured"` + a single WARNING log per run.
- Non-fatal exceptions from the handler → `"ERROR: approval handler
  failed for <tool>: <exc>"`, treated as a policy decline for that call.
- Contract unchanged. Phase D will build `core/elicitation_forwarder.py`
  for the cross-process case; `require_approval` on the serving surface
  will pass that forwarder as its handler — no API break.

### D2: New `on_inference` kwarg on `run_agent`

**Spec (§4.3)** said `max_cost` updates `ctx.cost_so_far` "actually updated
in `Agent.run` after each inference response (handled by the wrapper, see
§4.6)" but §4.6's sketch did not show how `Agent.run()` would see per-turn
cost — `run_agent` owns the inference loop.

**Actual:** Added an optional keyword argument to `run_agent`:

```python
on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None
```

Invoked inside the loop immediately after `response = cast(ChatInferenceResponse,
raw)`, BEFORE any tool call on that turn. `Agent.run()` passes a closure
(`_make_cost_tracker`) that updates `ctx.cost_so_far +=
response.usage.cost or 0.0` and increments `ctx.turn`.

**Reason:** Contract §6.1 freezes `run_agent`'s signature but explicitly
allows new optional kwargs with `None` defaults that preserve current
behaviour byte-for-byte. Post-hoc cost lookup via `get_inferences()` does
not return `usage` on `StoredInference` (source-validated at
`.venv/lib/python3.12/site-packages/tensorzero/generated_types.py:1920-1962`),
so in-loop capture is the only reliable path. The callback-based design
keeps the dependency direction correct (`Agent` → `run_agent`, not the
other way) and generalises to future observability hooks beyond cost.

**Impact:** Preserves Phase-B regression invariant
(`uv run pytest -m "not integration"` on `tests/test_mcp`, `tests/test_core`,
`tests/test_memory` → 239 tests, all green). Documented in
`developer-guide.md` §4 (signature) and §12 (knobs).

### D3: `T0AgentsFatalError` marker base class in `core/exceptions.py`

**Spec (§4.5)** described `GuardrailBlockedError` inheriting from
`T0AgentsError` and propagating out of `GuardedToolProvider`. It did not
address the interaction with `_execute_single_tool_call`'s belt-and-braces
`except Exception` clause, which would otherwise stringify any fatal
guardrail error into a tool-result error string and swallow the abort.

**Actual:** Added `T0AgentsFatalError(T0AgentsError)` as a new marker
class. `GuardrailBlockedError` inherits from `T0AgentsFatalError`.
`_execute_single_tool_call` gained a new FIRST except clause:

```python
try:
    mcp_result = await mcp_client.call_tool(
        block.name, block.arguments, timeout=timeout
    )
except T0AgentsFatalError:
    raise
except ToolError as e:
    ...
```

**Reason:** Without the marker + re-raise, the broad `except Exception` on
`_execute_single_tool_call:132` would catch `GuardrailBlockedError` and
return a ToolResult with `"ERROR: unexpected GuardrailBlockedError: ..."` —
exactly the opposite of what a hard block should do.

The marker is defined in `core/exceptions.py` (which `mcp/` already
depends on via `MaxTurnsExceededError`), so no new module-direction edges
are introduced. `mcp/` does NOT import from `guardrails/`; it only knows
the abstract fatal marker.

**Impact:** Exposed at the top of the library (`from t0_agents import
T0AgentsFatalError, GuardrailBlockedError`) for third-party guardrail
authors who want their own fatal errors to propagate the same way.

### D4: Inline guardrail wiring in `Agent.run()`; private helper `_make_cost_tracker`

**Spec (§4.6)** sketched guardrail setup as a multi-line block inside
`Agent.run()`.

**Actual:** The cost-tracking closure was factored out to a module-level
helper `_make_cost_tracker(ctx)` for readability; the rest of the setup
(ctx construction, wrapper install, `before_run` loop, `try/finally` for
`after_run`) is inline to keep the run method legible at one glance.

**Impact:** None on callers. Matches the Phase B pattern
(`_prepare_memory_injection` as an instance helper, `render_memory_summary`
as a module helper).

### D5 (post-merge): `RECOVERABLE_INFERENCE_STATUS_CODES` includes 502

**Observed during manual validation (2026-04-16).** The `require_approval`
integration test failed because Groq emitted a `<|channel|>`-artifact
tool call, TensorZero v2026.4.0 wrapped the provider's 400 as a
gateway-level 502 Bad Gateway, and the existing recovery code in
`_execute_single_tool_call`'s caller checked `e.status_code == 400` —
missing the recovery path entirely.

**Fix.** Added a module-level frozen set
`RECOVERABLE_INFERENCE_STATUS_CODES = frozenset({400, 502})` in
`src/t0_agents/mcp/loop.py` and replaced the `status_code == 400` guard
with membership in the set.  `503` remains excluded because Service
Unavailable is a rate-limit / outage signal, not a format issue — the
pre-existing `test_503_with_matching_text_re_raised` test was renamed
from `test_non_400_error_re_raised` to make the intent explicit and a
new `test_502_bad_gateway_recovered` test pins the new behaviour.

**Impact on the contract.** None. `run_agent`'s public signature and
documented semantics are unchanged.  Callers who were already handling
`TensorZeroError` propagation continue to see the same exceptions for
non-recoverable failures.  This is a latent-regression fix, not a new
behaviour.

### D6 (post-merge): max_cost integration test replaced with max_tool_calls

**Observed during manual validation (2026-04-16).**
`test_max_cost_tiny_budget_aborts` failed with `DID NOT RAISE`.  Probe
against the live gateway (`tests/server.py` + default `tensorzero.toml`)
confirmed `Usage(input_tokens=77, output_tokens=70, cost=None)` — the
Groq-hosted `gpt_oss_20` model has no `pricing_per_million_tokens_*`
block, so T0 cannot compute a cost.  The cost tracker treats `None` as
`0.0` (correct behaviour: missing cost data must not crash the run), so
`ctx.cost_so_far` stays at `0.0` regardless of traffic volume.

**Fix.** Rewrote the fatal-propagation integration test to use
`max_tool_calls(per_run=0)`, which fires deterministically on the first
tool dispatch without depending on provider pricing.  The unit test
`tests/test_guardrails/test_max_cost.py` already covers the full
accumulation/abort path with a seeded `FakeUsage(cost=...)`, so no
coverage is lost.

**Impact on callers.** None.  Documented the cost-reporting dependency
as a caveat in `docs/guardrails.md` → "Cost reporting depends on TOML
pricing" and in `examples/10_guardrails/README.md`.  Users who want the
`max_cost` cap to fire must declare pricing in `tensorzero.toml` or pick
a provider whose built-in pricing T0 ships with.

## Resolved Open Questions

### T0 Q5 — reaffirmed: no auto-emit of `t0.feedback(...)` for guardrail outcomes

The resolution in `02-open-questions.md` is reaffirmed verbatim by Phase C.
Metrics must be pre-declared in `tensorzero.toml`; forcing an
`[metrics.action_approved]` block on every deployer is over-reach.
`docs/guardrails.md` documents the manual feedback path with a concrete
`tensorzero.toml` snippet and code example. `AgentResult.guardrail_log`
surfaces the data callers need to emit feedback themselves.

## Contract Amendments

None. `00-integration-contract.md` is unchanged:

- §2.1 (`mcp/` standalone): `mcp/__init__.py` exports unchanged;
  `mcp/loop.py` only gained the new `except T0AgentsFatalError` clause
  (dependency on `core/exceptions`, already present).
- §2.5 (elicitation-only HITL): `require_approval` rides the exact
  FastMCP `ElicitationHandler` signature; no parallel HITL channel.
- §6.1 (frozen `run_agent`): new `on_inference` kwarg defaults to `None`;
  regression suite confirms byte-identical behaviour.
- §6.2 (frozen `AgentResult`): `guardrail_log` is a new field with an
  empty-list default — preserves every existing consumer.
- §6.3 (frozen `mcp/` exports): unchanged.

## Release Notes Snippet

> **v0.4.0 — Trusted Agent (M3)**
>
> Declarative guardrails for `Agent`: `from t0_agents.guardrails import
> require_approval, max_cost, max_tool_calls, block_tools`. `require_approval`
> rides FastMCP's `elicitation_handler` signature so you plug in the same
> callable you already pass to `Client(...)`. `max_cost` aborts the run
> via `GuardrailBlockedError` when cumulative `usage.cost` crosses your
> ceiling; `max_tool_calls` caps per-run invocations; `block_tools` keeps
> dangerous names off the LLM's execution path. Bring your own guardrail
> by implementing the `Guardrail` Protocol — a `runtime_checkable` four-
> hook surface. Every activation is recorded on `AgentResult.guardrail_log`
> for audit and manual `t0.feedback(...)` emission. No new base deps.
> See `docs/guardrails.md` and `examples/10_guardrails/`.
