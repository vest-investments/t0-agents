# Multi-Agent Composition

Delegation between three `Agent` instances through the classic
**"agent-as-tool"** pattern.

- `researcher` — wraps a tiny arXiv-flavoured FastMCP server; its job
  is to find relevant papers for a query.
- `writer` — wraps a markdown-formatting FastMCP server; its job is
  to format text into a final deliverable.
- `report_generator` — the orchestrator.  Sees `researcher_ask` and
  `writer_ask` as MCP tools (installed by the
  `AgentConfig.sub_agents=[...]` hook at construction time) and
  decides when to delegate.

Each sub-agent runs its own TensorZero function (`research`,
`writing`) so the ClickHouse trail shows one inference sequence per
agent.  The orchestrator's inferences are separately tagged
`role=orchestrator`; sub-agent inferences carry a
`t0_agents.parent_episode_id` tag pointing back at the orchestrator
for correlation.

## Prerequisites

```bash
# From examples/14_multi_agent/:
export GROQ_API_KEY=gsk_...                # or the provider your variant uses
docker compose up -d                       # boots the per-example gateway on :3000
uv sync                                    # no extras needed — compose has zero deps
```

## Run the demo

```bash
# Python entrypoint:
uv run python examples/14_multi_agent/main.py

# CLI factory (same thing, through t0-agents run):
uv run t0-agents run examples.14_multi_agent.agents:orchestrator_factory

# Advanced variant — nested orchestrators (project_lead → report_drafter
# → researcher + writer). Same prerequisites as main.py.
uv run python examples/14_multi_agent/advanced.py
```

Sample run:

```
=== Orchestrator response ===

# Report

## Finding

RLHF (Reinforcement Learning from Human Feedback) is the dominant …

=== Run metadata ===
Episode ID    : 018f3b4a-…-orchestrator
Turns         : 4
Inference IDs : 4 total
```

## What the orchestrator sees

When the orchestrator's first inference starts, its `additional_tools`
list contains every tool on its own FastMCP server.  In this example,
the orchestrator's server starts **empty** — the only tools the
orchestrator sees are the two installed by `install_sub_agents`:

```
researcher_ask(message: str) -> str
writer_ask(message: str) -> str
```

The orchestrator LLM decides when to call each one, and the return
value is whatever the sub-agent produced via `sub_agent.invoke(message)`.

## Layout

```
examples/14_multi_agent/
├── README.md                   ← this file
├── agents.py                   ← @asynccontextmanager factory for the CLI
├── main.py                     ← Python entrypoint (Intermediate tier)
├── advanced.py                 ← nested orchestrators (Advanced tier)
├── arxiv_server.py             ← FastMCP server with search_papers
├── markdown_server.py          ← FastMCP server with format_markdown
├── docker-compose.yml          ← per-example gateway + clickhouse
└── config/
    └── tensorzero.toml         ← research, writing, orchestrator functions
```

## Inspect composed agents

`t0-agents info` only shows the orchestrator's config — sub-agent
metadata is not surfaced via `info` (the field is a `list[Agent]`
under `AgentConfig.sub_agents`, not a scalar knob).  To see the
sub-agent tools on the orchestrator's FastMCP server, run the
example and watch the tool calls scroll past.

## Caveats

- **Tool-schema stability per run.** Sub-agents are registered at
  Agent construction time — dynamic mid-run addition is out of scope.
  The gateway permits per-call tool lists, but the LLM-side cost of
  seeing the tool list change between turns isn't worth it.
- **In-process FastMCP required.** `sub_agents` only works with a
  `fastmcp.Client` backed by an in-process `FastMCP` (the transport
  must be `FastMCPTransport`).  Remote MCP transports cannot host
  runtime-registered tools from this framework.
- **Errors are hybrid-fatal.**
  [`GuardrailBlockedError`](../../docs/guardrails.md) raised inside a
  sub-agent propagates through the orchestrator unchanged; every
  other exception is wrapped in `SubAgentError` so the orchestrator
  LLM can decide whether to retry.
