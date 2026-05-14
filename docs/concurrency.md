# Concurrency

Three places in the stack can run things in parallel. Picking the right one is the whole game — the framework will not pick for you, and it intentionally does not ship a planner, a DAG DSL, or a `parallel()` helper that wraps `asyncio.gather`.

This page maps the three layers, says which knob applies to which, and lists the patterns t0-agents deliberately does **not** offer.

## Decision table

| Scenario | Layer | Knob |
|---|---|---|
| The LLM emits several tool calls in one assistant turn (e.g. three independent lookups) and you want them to run at the same time | [Intra-turn](#l1-intra-turn-parallel-tool-execution) | `AgentConfig(parallel_tools=True)` |
| The LLM keeps emitting one tool per turn and you want to compress that into a single round trip | [Inter-turn](#l2-inter-turn-dependencies-and-the-batch-tool-pattern) | A "batch tool" on your MCP server |
| You have an orchestrator with sub-agents and you want them to work concurrently when the orchestrator delegates to several at once | [Multi-agent](#l3-multi-agent-parallel-composition) | Same `parallel_tools=True` on the orchestrator |
| You want to drive several agents from your own Python code (no orchestrator LLM) | [Multi-agent](#l3-multi-agent-parallel-composition) | `asyncio.gather(a.run(...), b.run(...))` |

If your problem does not fit any of the above, t0-agents is the wrong layer for it — see [Anti-patterns](#anti-patterns-that-t0-agents-deliberately-omits).

## L1: Intra-turn parallel tool execution

When a model decides several tool calls in one assistant turn are independent, TensorZero returns them in the same `ChatInferenceResponse.content` list. The agent loop can either await each one in sequence or fan them out with `asyncio.gather`.

```python
agent = Agent(
    AgentConfig(
        name="researcher",
        function_name="research",
        parallel_tools=True,        # <- the flag
    ),
    t0=t0,
    tool_provider=mcp_client,
)
result = await agent.run("Fetch the abstracts of arXiv 2401.0001, 2401.0002, 2401.0003")
```

With `parallel_tools=False` (the default) the three abstract fetches run one after the other and the run takes ~3 × the per-call latency. With `parallel_tools=True` they overlap and the run takes ~1 × the per-call latency. Result order is preserved regardless of completion order — `asyncio.gather` returns in argument order, not completion order.

The same flag is accepted by the low-level [`run_agent`](api-reference.md#run_agent) entry point, the [`Agent`](agent.md) class, and the [`Agent.stream`](streaming.md) variant.

### When to leave it `False`

The default is intentionally safe. Flip it to `True` only when **all** of these hold:

- Every tool the agent might call is **idempotent** or otherwise safe to run alongside its siblings.
- No tool mutates shared per-session state that another tool reads in the same turn.
- The configured guardrails are themselves race-safe under `asyncio.gather`. The built-in `max_cost` and `max_tool_calls` are; **`require_approval` is not** (it would fan out N approval prompts to one human in the same turn — see [Guardrails / Production considerations](guardrails.md#parallel_toolstrue)).
- Provider and tool rate limits can absorb the burst — `asyncio.gather` collapses N latency budgets into one but multiplies your peak QPS.

When in doubt, leave it off and turn it on per-agent once you have read the [Production checklist](production.md#parallel-vs-sequential-tool-execution).

## L2: Inter-turn dependencies and the "batch tool" pattern

If tool B reads the output of tool A, the LLM will emit A on turn N and B on turn N+1. That is **not a problem to solve with parallelism** — by construction the two are dependent, and the agent loop is already the dependency graph: each turn is a level. `parallel_tools=True` will not help here, and adding a planner that "discovers" the dependency is the kind of parallel system that duplicates what the model already does turn by turn.

The interesting case is the opposite one: the LLM is emitting **one tool per turn** even though the calls are independent. That is a prompt / capability artifact (small models do this; some providers do not emit multiple tool calls per turn at all). The L1 flag will not help — there is only ever one tool call per turn to begin with.

The fix lives in your MCP server, not in the agent loop. Expose a single tool that accepts a list and parallelises internally:

```python
@mcp.tool
async def fetch_papers(arxiv_ids: list[str]) -> list[dict]:
    """Fetch metadata for several arXiv papers at once."""
    return await asyncio.gather(*[_fetch_one(i) for i in arxiv_ids])
```

The model now sees `fetch_papers(arxiv_ids=...)` and naturally batches its three lookups into one turn. The parallelism happens inside your tool, outside the agent loop, and the dependency-free portion of the workflow collapses to a single inference round trip. This pattern composes cleanly with `parallel_tools=True`: independent **batch tools** in the same turn still dispatch concurrently.

## L3: Multi-agent parallel composition

Sub-agents are exposed as MCP tools on the orchestrator's in-process FastMCP server (see [Multi-Agent Composition](multi-agent.md)). That means **L3 reduces to L1**: if the orchestrator's `AgentConfig` carries `parallel_tools=True` and the orchestrator LLM decides to delegate to several sub-agents in one turn, those delegations run concurrently — no extra primitive needed.

```python
orchestrator = Agent(
    AgentConfig(
        name="report_generator",
        function_name="orchestrator",
        sub_agents=[researcher, writer],
        parallel_tools=True,        # researcher_ask and writer_ask can overlap
    ),
    t0=t0,
    tool_provider=orchestrator_client,
)
```

ContextVar-based episode correlation is preserved across the concurrent calls — every sub-agent inference still carries the `t0_agents.parent_episode_id` tag pointing back at the orchestrator. The `_OrchestrationToolProvider` and `agent_as_tool` handler are designed for this.

If you want **deterministic** fan-out — "always run these two sub-agents in parallel, then feed both results to the third" — write it as ordinary Python, not as framework configuration:

```python
research, draft = await asyncio.gather(
    researcher.run("Find arXiv 2401.* RLHF papers"),
    writer.run("Draft a markdown outline for a 2025 RLHF report"),
)
final = await editor.run(f"Combine:\n\n{research.response}\n\n{draft.response}")
```

This is the t0-agents-blessed equivalent of a `chain`/`parallel` block in other frameworks. `Agent.run` is an `async def` returning `AgentResult`; that is the only contract you need.

## Anti-patterns that t0-agents deliberately omits

- **A `parallel(*coros)` helper.** It would wrap `asyncio.gather` and add no value over the stdlib primitive that any async-Python developer already imports.
- **A DAG / graph builder for tool dependencies.** The LLM is the planner. Hand-rolling a parallel dependency graph in framework configuration would duplicate that planning and force users to maintain two ways of describing the same workflow.
- **Flipping `parallel_tools=True` by default.** Existing agents with `require_approval` or any await-yielding guardrail would silently change semantics. The opt-in posture is on purpose — see [Production considerations](guardrails.md#parallel_toolstrue).
- **Per-tool `serial=True` metadata.** Mixed read/write MCP servers would benefit, but it requires an MCP-extension contract that does not exist today. Tracked as a follow-up; not part of the current API.

If you find yourself reaching for any of the above, the answer is almost always L1 (turn the flag on) or L2 (write a batch tool) or L3 (let the orchestrator LLM delegate, with the flag on).

## References

- `parallel_tools` field — [`AgentConfig`](agent.md#agentconfig) and the low-level [`run_agent`](api-reference.md#run_agent) signature
- Sequential and parallel executors — `src/t0_agents/mcp/loop.py` (`execute_tool_calls`, `execute_tool_calls_parallel`)
- Sub-agent dispatch — [`Multi-Agent Composition`](multi-agent.md), `src/t0_agents/compose/`
- Guardrail safety under parallel dispatch — [`Guardrails`](guardrails.md#parallel_toolstrue)
- Provider-side caveats (which models even emit multiple tool calls per turn) — [`Production`](production.md#parallel-vs-sequential-tool-execution)
