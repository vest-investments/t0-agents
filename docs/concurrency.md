# Concurrency

Three places in the stack can run things in parallel. Picking the right one is the whole game — the framework will not pick for you, and it intentionally does not ship a planner, a DAG DSL, or a `parallel()` helper that wraps `asyncio.gather`.

This page maps the three layers, says which knob applies to which, and lists the patterns t0-agents deliberately does **not** offer.

## Decision table

| Scenario | Layer | Knob |
|---|---|---|
| The LLM emits several tool calls in one assistant turn (e.g. three independent lookups) and you want them to run at the same time | [Intra-turn](#l1-intra-turn-parallel-tool-execution) | `AgentConfig(parallel_tools=True)` |
| The LLM keeps emitting one tool per turn and you want to compress that into a single round trip | [Inter-turn](#l2-inter-turn-dependencies-and-the-batch-tool-pattern) | A "batch tool" on your MCP server |
| You have an orchestrator with sub-agents and you want them to work concurrently when the orchestrator delegates to several at once | [Multi-agent](#l3-multi-agent-parallel-composition) | Same `parallel_tools=True` on the orchestrator |
| You want to drive several agents from your own Python code (no orchestrator LLM) | [Multi-agent](#l3-multi-agent-parallel-composition) | `asyncio.gather(a.invoke(...), b.invoke(...))` |

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
result = await agent.invoke("Fetch the abstracts of arXiv 2401.0001, 2401.0002, 2401.0003")
```

With `parallel_tools=False` (the default) the three abstract fetches run one after the other and the run takes ~3 × the per-call latency. With `parallel_tools=True` they overlap and the run takes ~1 × the per-call latency. Result order is preserved regardless of completion order — `asyncio.gather` returns in argument order, not completion order.

The same flag is accepted by the low-level [`run_agent`](api-reference.md#run_agent) entry point, the [`Agent`](agent.md) class, and the [`Run.events`](streaming.md) variant.

### When to leave it `False`

The default is intentionally safe. Flip it to `True` only when **all** of these hold:

- Every tool the agent might call is **idempotent** or otherwise safe to run alongside its siblings.
- No tool mutates shared per-session state that another tool reads in the same turn.
- The configured guardrails are themselves race-safe under `asyncio.gather`. The built-in `max_cost` and `max_tool_calls` are; **`require_approval` is not** (it would fan out N approval prompts to one human in the same turn — see [Guardrails / Production considerations](guardrails.md#parallel_toolstrue)).
- Provider and tool rate limits can absorb the burst — `asyncio.gather` collapses N latency budgets into one but multiplies your peak QPS.

When in doubt, leave it off and turn it on per-agent once you have read the [Production checklist](production.md#parallel-vs-sequential-tool-execution).

### Model and provider support

`parallel_tools=True` only helps when the LLM actually emits several tool calls in the same assistant turn. That decision lives at the *provider*, not at t0-agents, and three independent things have to line up:

1. **Native model capability.** Some models routinely emit multi-tool turns; others always emit exactly one tool call per turn no matter how you prompt them. In a model card, grep for:
   - "parallel function calling" (OpenAI's term)
   - "parallel tool use" / "parallel tool calls" (Anthropic, Mistral, Cohere)
   - explicit "supports" / "does not support" statements
   - mentions of the BFCL parallel-category benchmark
   - whether the provider's API exposes a `parallel_tool_calls: bool` parameter — if it does, the model is probably capable

   Known-capable families at the time of writing: OpenAI GPT-4-turbo and GPT-4o, Anthropic Claude 3.x and 4.x, Mistral Large 2, Cohere Command R+. Models that ship with parallel calling **disabled by default** or simply not supported: many small open-source variants such as `openai/gpt-oss-20b` (observed in `examples/12_concurrency/`), older `gpt-3.5-turbo-0613`, and minor providers without a documented parallel mode.

2. **Provider-gateway support inside TensorZero.** Even when the upstream model is capable, the gateway has to forward the parallel-emission hint. TensorZero maintains an allowlist; at the time of writing **Groq is not on it** (`tests/test_mcp/test_integration.py:388` documents the constraint), while OpenAI and Fireworks AI are. Check TensorZero's release notes and your provider block in `tensorzero.toml`.

3. **The inference-level hint.** TensorZero's `t0.inference()` accepts a `parallel_tool_calls: bool` argument that, when forwarded to a capable provider, instructs the model to prefer parallel emissions. **t0-agents does not currently plumb this through `AgentConfig` or `run_agent`** — until that is added, you rely on the model's spontaneous behavior or call `t0.inference()` directly when you need the hint. This is a parked follow-up; see the bottom of this page.

If any of the three fails, the loop sees one `ToolCall` block per turn and `parallel_tools=True` has nothing to parallelise. The mitigation is the L2 "batch tool" pattern below — it sidesteps the LLM's emission style entirely.

### Verifying support without committing to a model

A 5-line probe is enough to know whether a given model + provider + TensorZero function actually delivers multi-tool turns. Send one real inference with a prompt that should produce parallel calls and count the `ToolCall` blocks in the response:

```python
from tensorzero import ToolCall

response = await t0.inference(
    function_name="agent",
    input={"messages": [{"role": "user",
                          "content": "Fetch /a, /b, /c at the same time."}]},
    additional_tools=t0_tools,
    dryrun=True,
)
n_tool_calls = sum(1 for b in response.content if isinstance(b, ToolCall))
# n_tool_calls >= 2  =>  parallel_tools=True will speed this model up
# n_tool_calls == 1  =>  use the L2 batch-tool pattern instead
```

`dryrun=True` keeps the probe out of ClickHouse and off your billing while still going through the real provider. Run it once per model upgrade in CI to detect regressions.

### Testing the executor selection without an LLM

For deterministic unit tests of the parallel dispatch itself — independent of which model you ultimately ship — bypass the LLM with [`MockGateway`](testing.md):

```python
from typing import cast
from t0_agents.testing import MockGateway
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

gateway = MockGateway.script(
    [
        # Turn 1: three independent tool calls in one assistant turn
        [
            ToolCall(id=f"c{i}", raw_name="fetch", name="fetch",
                     raw_arguments=f'{{"url": "{u}"}}', arguments={"url": u})
            for i, u in enumerate(["a", "b", "c"])
        ],
        # Turn 2: terminating text
        [Text(text="Done")],
    ]
)
agent = Agent(config, t0=cast(AsyncTensorZeroGateway, gateway), tool_provider=...)
result = await agent.invoke("...")
```

Pair the probe above with `parallel_tools=True` in your own integration tests to cover both the executor selection (deterministic) and the model emission style (model-dependent) in CI.

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
    researcher.invoke("Find arXiv 2401.* RLHF papers"),
    writer.invoke("Draft a markdown outline for a 2025 RLHF report"),
)
final = await editor.invoke(f"Combine:\n\n{research.response}\n\n{draft.response}")
```

This is the t0-agents-blessed equivalent of a `chain`/`parallel` block in other frameworks. `Agent.invoke` is an `async def` returning `AgentResult`; that is the only contract you need.

## Anti-patterns that t0-agents deliberately omits

- **A `parallel(*coros)` helper.** It would wrap `asyncio.gather` and add no value over the stdlib primitive that any async-Python developer already imports.
- **A DAG / graph builder for tool dependencies.** The LLM is the planner. Hand-rolling a parallel dependency graph in framework configuration would duplicate that planning and force users to maintain two ways of describing the same workflow.
- **Flipping `parallel_tools=True` by default.** Existing agents with `require_approval` or any await-yielding guardrail would silently change semantics. The opt-in posture is on purpose — see [Production considerations](guardrails.md#parallel_toolstrue).
- **Per-tool `serial=True` metadata.** Mixed read/write MCP servers would benefit, but it requires an MCP-extension contract that does not exist today. Tracked as a follow-up; not part of the current API.
- **Forwarding `parallel_tool_calls` from `AgentConfig` to `t0.inference()`.** TensorZero exposes the inference-level hint, but `AgentConfig` does not plumb it through today. Tracked as a follow-up; until then, callers that need to instruct the model rather than rely on its spontaneous behavior must call `t0.inference()` directly.

If you find yourself reaching for any of the above, the answer is almost always L1 (turn the flag on) or L2 (write a batch tool) or L3 (let the orchestrator LLM delegate, with the flag on).

## References

- `parallel_tools` field — [`AgentConfig`](agent.md#agentconfig) and the low-level [`run_agent`](api-reference.md#run_agent) signature
- Sequential and parallel executors — `src/t0_agents/mcp/loop.py` (`execute_tool_calls`, `execute_tool_calls_parallel`)
- Sub-agent dispatch — [`Multi-Agent Composition`](multi-agent.md), `src/t0_agents/compose/`
- Guardrail safety under parallel dispatch — [`Guardrails`](guardrails.md#parallel_toolstrue)
- Provider-side caveats (which models even emit multiple tool calls per turn) — [`Production`](production.md#parallel-vs-sequential-tool-execution)
