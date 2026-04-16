# Phase G — Multi-Agent Composition

> Status: **ready to start**. Phase deps: A–F complete. Open
> questions resolved: T0 Q2 (`additional_tools` per-call is
> gateway-safe; LLM-side schema-change risk on us → ship stable list
> per run), FastMCP Q1 (`Tool.from_function(...)` + `add_tool(tool)`
> at runtime; **NO** `name=` kwarg on `add_tool`).
> Delivers: **Milestone M7 — Composed Agents**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md) §3.2
  (`sub_agents` field), §7 (resolved Qs)
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q2,
  FastMCP Q1
- [`../problemas.md`](../problemas.md) — Problem 4 (deferred until A–F
  shipped)
- [`phase-a-agent-core.md`](phase-a-agent-core.md)

---

## 1. Goal

Let one `Agent` delegate to other `Agent`s. Each sub-agent exposes
itself as an MCP tool to the orchestrator; the "tool body" is a full
`sub_agent.run(args["message"])` call.

After Phase G:

```python
researcher = Agent(
    AgentConfig(name="researcher", function_name="research"),
    t0=t0,
    tool_provider=arxiv_client,
)
writer = Agent(
    AgentConfig(name="writer", function_name="writing"),
    t0=t0,
    tool_provider=markdown_client,
)

orchestrator = Agent(
    AgentConfig(
        name="report_generator",
        function_name="orchestrator",
        sub_agents=[researcher, writer],
    ),
    t0=t0,
    tool_provider=orchestrator_client,
)

result = await orchestrator.run("Write a report on RLHF advances in 2025")
```

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/compose/` with:
  - `delegation.py` — `agent_as_tool(sub_agent)` returns a callable
    suitable for `Tool.from_function`.
  - `orchestrator.py` — wires sub-agents onto a parent `Agent`'s
    `tool_provider` at construction time.
  - `routing.py` — sub-agent name → tool name conversion (default
    `f"{sub_agent.config.name}_ask"`); collision detection.
- Extension on `AgentConfig`: `sub_agents: list[Agent] = []`.
- Extension on `Agent.__init__`: when `sub_agents` is non-empty,
  detect that `tool_provider` is a `fastmcp.Client` (otherwise
  raise `ConfigurationError`), retrieve the underlying `FastMCP`
  instance via `client.local_provider` (or equivalent), and call
  `add_tool(Tool.from_function(handler, name="..."))` once per
  sub-agent.
- New exception: `SubAgentError` (T0AgentsError subclass).
- Tests under `tests/test_compose/`.
- Example `examples/14_multi_agent/`.
- Public doc page `docs/multi-agent.md`.

### 2.2 Out of scope

- Dynamic mid-run sub-agent addition. T0 Q2 says it's gateway-safe,
  but the LLM-side cost (tool schema changes mid-conversation)
  isn't worth it for now. All sub-agents are declared at
  construction.
- Sub-agent-to-sub-agent direct delegation (peer-to-peer mesh).
  Sub-agents are children of the orchestrator only; deeper trees are
  achievable by nesting orchestrators.
- Per-sub-agent memory isolation — Phase B's `MemoryProvider` is
  per-Agent, so sub-agents naturally have their own.
- Cross-sub-agent shared state. Each sub-agent has its own episode.

### 2.3 Non-goals

- Do not invent a workflow DSL. The orchestrator's own LLM decides
  what to delegate; that's the whole point of agent-as-tool.
- Do not auto-spawn FastMCP servers (Phase G's earlier skeleton
  considered `mount` per sub-agent; FastMCP Q1 made that unnecessary).

---

## 3. Constraints (from the contract)

1. `mcp/` stays standalone. `compose/` lives above `mcp/` and `core/`.
2. `compose/` depends on `core/` and `mcp/` only.
3. Sub-agents are real `Agent` instances; they go through the same
   `run()` path as standalone agents (memory, guardrails, logging
   all carry over).
4. `Tool.from_function` + `add_tool(tool)` is the registration path
   (FastMCP Q1, source-validated). No `mount()` per sub-agent.
5. Tool-name collisions raise `ConfigurationError` at
   `Agent.__init__` time, not at runtime.

---

## 4. Module additions

### 4.1 `src/t0_agents/compose/delegation.py` (new)

```python
import inspect
from typing import Any

from fastmcp.tools import Tool

from t0_agents.core.agent import Agent


def agent_as_tool(
    sub_agent: Agent,
    *,
    tool_name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap a sub-agent as an MCP Tool for runtime registration on a
    parent FastMCP server.

    Generated handler signature:
        async def <tool_name>(message: str) -> str

    The handler invokes `sub_agent.run(message)` and returns the
    response text.

    `Tool.from_function` extracts the schema from the function's
    type annotations + docstring; we craft both deliberately so the
    LLM sees a clear "ask the {sub_agent.name} agent a question" tool.
    """
    name = tool_name or f"{sub_agent.config.name}_ask"
    desc = description or (
        f"Delegate a question or sub-task to the {sub_agent.config.name} agent. "
        f"Use when the task fits {sub_agent.config.name}'s specialisation."
    )

    async def _handler(message: str) -> str:
        try:
            result = await sub_agent.run(message)
            return result.response
        except Exception as e:
            from t0_agents.core.exceptions import SubAgentError
            raise SubAgentError(
                f"Sub-agent {sub_agent.config.name!r} failed: {e}"
            ) from e

    _handler.__name__ = name
    _handler.__doc__ = desc

    return Tool.from_function(_handler, name=name, description=desc)
```

### 4.2 `src/t0_agents/compose/routing.py` (new)

```python
from collections import Counter

from t0_agents.core.exceptions import ConfigurationError


def assert_unique_names(sub_agents):
    """Reject sub-agent name collisions at construction time."""
    counts = Counter(sa.config.name for sa in sub_agents)
    duplicates = [n for n, c in counts.items() if c > 1]
    if duplicates:
        raise ConfigurationError(
            f"Duplicate sub-agent names: {duplicates}. "
            f"Each sub-agent's config.name must be unique."
        )
```

### 4.3 `src/t0_agents/compose/orchestrator.py` (new)

```python
from fastmcp import FastMCP, Client

from t0_agents.core.agent import Agent
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.compose.delegation import agent_as_tool
from t0_agents.compose.routing import assert_unique_names


def install_sub_agents(parent: Agent) -> None:
    """Register each sub-agent as a tool on the parent's FastMCP server.

    Called by Agent.__init__ when AgentConfig.sub_agents is non-empty.
    Requires that parent.tool_provider is a fastmcp.Client backed by a
    FastMCP instance — otherwise raises ConfigurationError.
    """
    sub_agents = parent.config.sub_agents
    if not sub_agents:
        return

    assert_unique_names(sub_agents)

    client = parent._tool_provider
    if not isinstance(client, Client):
        raise ConfigurationError(
            "sub_agents only supported when tool_provider is a fastmcp.Client. "
            f"Got {type(client).__name__}."
        )
    server = client.local_provider  # FastMCP instance
    if server is None:
        raise ConfigurationError(
            "tool_provider must be backed by a local FastMCP server "
            "(in-process). Remote MCP transports cannot host runtime tools."
        )

    for sa in sub_agents:
        tool = agent_as_tool(sa)
        # Detect collisions with existing tools on the parent server
        # (e.g. a tool with the same name was already declared via @mcp.tool)
        existing = {t.name for t in server.list_tools_sync()}
        if tool.name in existing:
            raise ConfigurationError(
                f"Sub-agent tool {tool.name!r} collides with an existing tool "
                f"on the parent FastMCP server."
            )
        server.add_tool(tool)
```

### 4.4 `core/agent.py` extension

```python
class Agent:
    def __init__(self, config, *, t0, tool_provider):
        ...
        self._config = config
        self._t0 = t0
        self._tool_provider = tool_provider

        # Sub-agent installation (Phase G)
        if config.sub_agents:
            from t0_agents.compose.orchestrator import install_sub_agents
            install_sub_agents(self)
```

### 4.5 `core/exceptions.py` extension

```python
class SubAgentError(T0AgentsError):
    """Wraps an error from a delegated sub-agent. The original
    exception is preserved as `__cause__`.
    """
```

### 4.6 `core/config.py` extension

```python
@dataclass(slots=True)
class AgentConfig:
    ...
    sub_agents: list["Agent"] = field(default_factory=list)
```

### 4.7 Episode propagation strategy

Sub-agents start their own episodes by default. This is the cleanest
path for ClickHouse analytics — each sub-agent's chain of inferences
forms its own episode, and the orchestrator's episode references them
via tag:

```python
async def _handler(message: str) -> str:
    parent_episode_id = current_orchestrator_episode_id  # captured via contextvar
    sub_agent.config.tags = {
        **(sub_agent.config.tags or {}),
        "t0_agents.parent_episode_id": parent_episode_id or "unknown",
    }
    result = await sub_agent.run(message)
    return result.response
```

Future enhancement: opt-in shared episodes via
`agent_as_tool(sub_agent, share_episode=True)`.

---

## 5. Tests

Location: `tests/test_compose/`.

### 5.1 Unit (offline)

- [ ] `test_agent_as_tool_returns_tool` — return type + name +
      description shape correct.
- [ ] `test_agent_as_tool_handler_invokes_sub_agent` — mock sub-agent
      `run` returns "hello"; tool handler returns "hello".
- [ ] `test_agent_as_tool_wraps_subagent_errors` — sub-agent raises;
      tool handler raises `SubAgentError` with `__cause__` set.
- [ ] `test_install_sub_agents_collision_with_existing_tool` —
      orchestrator already has a tool named `researcher_ask`; install
      raises `ConfigurationError`.
- [ ] `test_install_sub_agents_duplicate_subagent_names` —
      `ConfigurationError`.
- [ ] `test_install_sub_agents_non_fastmcp_provider` —
      `ConfigurationError`.
- [ ] `test_orchestrator_run_dispatches_correctly` — mock sub-agents;
      orchestrator with two sub-agents calls each; assert message
      delegation.

### 5.2 Integration (live gateway)

- [ ] `test_two_agent_orchestration_e2e` — small researcher + writer
      pair; orchestrator delegates and produces a final report.
- [ ] `test_subagent_episode_isolation` — sub-agent's `episode_id`
      differs from orchestrator's; assert via `AgentResult` inspection.
- [ ] `test_parent_episode_tag_propagation` — sub-agent's inference
      rows in ClickHouse carry `t0_agents.parent_episode_id` tag
      (verified via `t0.list_inferences` query).

### 5.3 Regression

- [ ] All earlier phases' tests still green.
- [ ] `mcp/` public API unchanged.

---

## 6. Example

Location: `examples/14_multi_agent/`.

- `arxiv_server.py` — minimal MCP server with `search_papers(query)`.
- `markdown_server.py` — `format_markdown(text, structure)`.
- `agents.py` — defines `researcher`, `writer`, `orchestrator` at
  module level (for use with `t0-agents run examples.14_multi_agent.agents:orchestrator`).
- `main.py` — Python entrypoint mirroring §1's snippet.
- `tensorzero.toml` — three function declarations: `research`,
  `writing`, `orchestrator`.
- `README.md` — "what each sub-agent does" + delegation walkthrough.

---

## 7. Documentation

- New: `docs/multi-agent.md`.
- Sections:
  - "Why agent-as-tool?" — design rationale.
  - "Defining sub-agents" — per-sub-agent `AgentConfig`.
  - "Wiring an orchestrator" — `sub_agents=[...]`.
  - "Naming and collisions" — default `_ask` suffix; how to override.
  - "Episode propagation" — separate episodes by default; tag for
    correlation; opt-in shared episodes (future).
  - "Caveats" — LLM-side schema risk if you ever do mid-run
    additions; T0 Q2 reminder.
- Update `developer-guide.md` §4 (public API) for `agent_as_tool`,
  `SubAgentError`, `AgentConfig.sub_agents`.
- `CHANGELOG.md` — `0.8.0`.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/compose/__init__.py`.
- [ ] `src/t0_agents/compose/delegation.py`.
- [ ] `src/t0_agents/compose/orchestrator.py`.
- [ ] `src/t0_agents/compose/routing.py`.
- [ ] `src/t0_agents/core/agent.py` — `install_sub_agents` call.
- [ ] `src/t0_agents/core/config.py` — `sub_agents` field.
- [ ] `src/t0_agents/core/exceptions.py` — `SubAgentError`.
- [ ] `pyproject.toml` — no new deps.
- [ ] `uv run ruff check src/`, `uv run mypy src/` green.

### 8.2 Tests

- [ ] All §5.1 unit tests.
- [ ] All §5.2 integration tests.

### 8.3 Example

- [ ] `examples/14_multi_agent/` complete and runnable.

### 8.4 Docs

- [ ] `docs/multi-agent.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `phase-g-implementation-log.md` with the FastMCP Q1 caveat
      (no `name=` kwarg) and how `Tool.from_function` is used.

---

## 9. Acceptance criteria (Milestone M7)

1. `from t0_agents.compose import agent_as_tool` works.
2. `examples/14_multi_agent/` runs end-to-end against the live
   gateway.
3. ClickHouse shows separate episodes per sub-agent with the
   `t0_agents.parent_episode_id` correlation tag.
4. Tool name collisions are caught at construction time.
5. Earlier phases' tests still green.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| `client.local_provider` API differs across FastMCP versions | medium | Pin `fastmcp>=3.2.3`; gracefully error with a pointer if attribute missing. |
| Sub-agent tool descriptions confuse the orchestrator LLM | medium | Provide a sensible default `description`; let users override per sub-agent. |
| Long-running sub-agent blocks orchestrator | medium | Inherit `tool_timeout` from orchestrator's config; document. |
| Memory writes from sub-agents pollute orchestrator's memory | low | Sub-agents have their own `MemoryProvider`; no cross-contamination by default. |
| LLM hallucinates a sub-agent name not declared | low | Existing `mcp/loop.py` error path handles unknown tool names. |

---

## 11. Estimated surface

- **New files:** 4 src + 3 test + 5 example + 1 doc.
- **Modified files:** 3 (`core/agent.py`, `core/config.py`,
  `core/exceptions.py`) + `CHANGELOG.md` + `developer-guide.md`.
- **Net new public symbols:** 3 (`agent_as_tool`, `SubAgentError`,
  `AgentConfig.sub_agents`).
- **New deps:** 0.
