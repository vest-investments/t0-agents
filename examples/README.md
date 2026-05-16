# t0-agents examples

A runnable, progressive tour of t0-agents.  Numbered to match the
[Reading path](../docs/index.md) — start at `01_first_agent`, work
forward chapter by chapter, then dip into the appendices for the
cross-cutting concerns.

## Prerequisites

All examples require:

- Python 3.11+
- `uv sync` (creates the venv and installs the package).

Examples that boot a TensorZero gateway additionally need:

- Docker + Docker Compose
- A provider API key in `.env` at the repo root.  The default
  configurations expect `GROQ_API_KEY`; swap to `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` by editing the per-example `config/tensorzero.toml`.

## Reading path

| # | Folder | Feature (docs chapter) | Docker? |
|---|---|---|---|
| 01 | [`01_first_agent`](01_first_agent/)         | The agent loop, top to bottom — one user message, two tool calls, one final answer. | yes |
| 02 | [`02_agent_class`](02_agent_class/)         | The declarative `Agent` class.  Three tiers: bare config, baked-in system prompt, per-call system override. | yes |
| 03 | [`03_output_modes`](03_output_modes/)       | `OutputMode[T]` — extract typed JSON, run the recovery ladder offline, hot-swap a schema mid-run, decorate with Pydantic. | yes / no |
| 04 | [`04_elicitation`](04_elicitation/)         | Tools that pause to ask a human — approval gates, structured input, the elicitation handler. | yes |
| 05 | [`05_memory`](05_memory/)                   | Three tiers of `MemoryProvider`: an ephemeral dict, the SQLite backend, and a TZ-native history strategy across process restarts. | yes |
| 06 | [`06_guardrails`](06_guardrails/)           | `require_approval`, `max_cost`, `max_tool_calls`, `block_tools`, plus a custom `Guardrail` with an audit log. | yes |
| 07 | [`07_streaming`](07_streaming/)             | The live event stream — `Run.events()` over SSE and WebSocket, with elicitation piped to a browser UI. | yes |
| 08 | [`08_multi_agent`](08_multi_agent/)         | `AgentConfig.sub_agents=[...]` — orchestrator + research/writer sub-agents, plus nested orchestrators. | yes |
| 09 | [`09_serving`](09_serving/)                 | `create_app(agent)` — drop the whole stack behind `/chat`, `/feedback`, `/health`. | yes |
| 10 | [`10_cli`](10_cli/)                         | The terminal TUI — `t0-agents run`, `serve`, `info` with a Rich renderer and elicitation-as-prompt. | yes |
| 11 | [`11_testing`](11_testing/)                 | `MockGateway` for deterministic unit tests, `EmbeddedGatewayHelper` for real-inference integration tests without Docker. | mixed |

## Appendices

| # | Folder | Feature (docs chapter) | Docker? |
|---|---|---|---|
| 12 | [`12_concurrency`](12_concurrency/)         | Parallel tool dispatch and parallel sub-agent fan-out — quantified wall-clock speedup. | no / yes |
| 13 | [`13_observability`](13_observability/)     | Episode + inference correlation, `t0.feedback(...)`, ClickHouse-ready dashboards. | yes |
| 14 | [`14_multi_server`](14_multi_server/)       | Mount two FastMCP servers behind one agent with namespace prefixes. | yes |
| 15 | [`15_adapter_low_level`](15_adapter_low_level/) | The `mcp/` submodule on its own — tool conversion, no Agent class, no gateway.  The plumbing the rest of the framework is built on. | no |

## How to read this

Each folder has:

- A short `README.md` with the "Aha!" moment, prerequisites, and what
  to observe when you run it.
- One or more runnable `.py` entry points.  Where it makes sense, three
  tiers (`basic.py`, `intermediate.py`, `advanced.py`).
- A self-contained `docker-compose.yml` and `config/tensorzero.toml`
  for the examples that need them — no global gateway required.

If you are new to t0-agents, run `01_first_agent` first; then read
`docs/index.md` and follow the Reading path with the corresponding
example open in another tab.
