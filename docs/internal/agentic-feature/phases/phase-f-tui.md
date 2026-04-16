# Phase F — TUI Demo (Typer + Rich)

> Status: **ready to start**. Phase deps: A, C, E complete. Open
> question resolved: FastMCP Q5 (strict elicitation schema:
> flat `type:object`, primitives + `enum` + `const`, inlined enums,
> `description` and `default` preserved, scalar wrapping in `{value:
> ...}`).
> Delivers: **Milestone M6 — TUI Agent**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md)
- [`../02-open-questions.md`](../02-open-questions.md) — FastMCP Q5
- [`../problemas.md`](../problemas.md) — Problem 8
- [`phase-c-guardrails.md`](phase-c-guardrails.md),
  [`phase-e-streaming.md`](phase-e-streaming.md)

---

## 1. Goal

Ship a Typer + Rich CLI that demonstrates how easy it is to attach
`t0-agents` to a TUI. Validates streaming + elicitation outside a
browser. Per the user's hint: *"como es de fácil integrar t0-agents a
un TUI usando Typer [la lib de Tiangolo]"*.

After Phase F:

```bash
$ uv run t0-agents run myapp.agents:assistant

╭─ assistant ──────────────────────────────────────────╮
│ You: Find recent papers about RLHF                   │
│                                                       │
│ 🔧 Calling search_papers(query="RLHF 2025")…        │
│ ✓ Found 3 results                                    │
│                                                       │
│ Assistant: Here are the most relevant papers…        │
╰───────────────────────────────────────────────────────╯
```

Subcommands:

- `t0-agents run <agent>` — interactive REPL with streaming + Rich
  rendering + elicitation prompts.
- `t0-agents serve <agent>` — wraps `serve.create_app` + `uvicorn`
  for a one-shot dev server.
- `t0-agents info <agent>` — print `AgentConfig` summary.

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/cli/` with:
  - `app.py` — Typer entrypoint.
  - `run.py` — `run` subcommand (REPL).
  - `serve_cmd.py` — `serve` subcommand.
  - `info_cmd.py` — `info` subcommand.
  - `tui.py` — Rich live renderer for `AgentEvent` stream.
  - `elicit_tui.py` — Rich-based elicitation handler (built-in).
  - `loader.py` — load an agent from `module:attr` syntax.
- `[cli]` extra: `typer`, `rich`.
- Entry-point in `pyproject.toml`:
  `t0-agents = "t0_agents.cli.app:main"`.
- Tests under `tests/test_cli/` (Typer's `CliRunner` + a
  capsys-style transcript snapshot).
- Example `examples/13_tui/` with a sample agent module.
- Public doc page `docs/cli.md`.

### 2.2 Out of scope

- Multiple agents in one TUI session (one agent per `run` invocation).
- Mouse interaction; arrow-key shortcuts beyond what Rich provides
  natively. Keyboard-only.
- TUI-side memory inspection. The agent's memory is opaque to the
  TUI; what you see is the response stream.

### 2.3 Non-goals

- Do not depend on Textual. Rich alone is sufficient for the spec
  and keeps install footprint smaller.
- Do not invent a TUI dashboard for production observability.

---

## 3. Constraints (from the contract)

1. `cli/` depends on `core/`, `stream/`, `serve/` (for the `serve`
   subcommand). No new sibling deps.
2. `[cli]` extra is opt-in.
3. **Elicitation schema guarantees** (FastMCP Q5): renderer relies on
   the strict subset — flat `type:object` with primitives + `enum` +
   `const`, inlined enums, preserved `description`/`default`, scalar
   wrapping in `{value: ...}`. Fail-soft fallback to `Prompt.ask(str)`
   only for non-FastMCP-server schemas (in case a developer points
   the CLI at an external MCP server with looser semantics).
4. Async-first, but the Typer entrypoint wraps `asyncio.run(...)` so
   users see a normal `t0-agents` command.

---

## 4. Module additions

### 4.1 `src/t0_agents/cli/loader.py` (new)

```python
import importlib
from t0_agents.core.agent import Agent


def load_agent(spec: str) -> Agent:
    """Load an Agent instance from a 'module:attr' spec.

    Examples:
        myapp.agents:assistant
        examples.13_tui.agent:assistant
    """
    if ":" not in spec:
        raise typer.BadParameter("Spec must be 'module:attr'")
    module_name, attr = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    if not isinstance(obj, Agent):
        raise typer.BadParameter(f"{spec} is not an Agent instance")
    return obj
```

### 4.2 `src/t0_agents/cli/tui.py` (new)

Rich live renderer for `AgentEvent`s. Uses `rich.live.Live` +
`rich.console.Console` to update an interactive panel.

```python
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from t0_agents.stream.events import (
    TextDeltaEvent, TextEndEvent, ToolCallEvent, ToolResultEvent,
    ThoughtEvent, DoneEvent, ErrorEvent,
)


class TUIRenderer:
    def __init__(self, agent_name: str, console: Console | None = None):
        self.agent_name = agent_name
        self.console = console or Console()
        self.text_buffers: dict[str, str] = {}
        self.tool_status: dict[str, str] = {}

    async def consume(self, event_stream):
        with Live(self._render(), refresh_per_second=20) as live:
            async for event in event_stream:
                self._update(event)
                live.update(self._render())

    def _update(self, event):
        match event:
            case TextDeltaEvent(id=tid, delta=d):
                self.text_buffers[tid] = self.text_buffers.get(tid, "") + d
            case ToolCallEvent(id=tid, name=n, arguments=a):
                self.tool_status[tid] = f"🔧 Calling {n}({a})…"
            case ToolResultEvent(id=tid, name=n, is_error=err):
                self.tool_status[tid] = f"{'❌' if err else '✓'} {n}"
            case ThoughtEvent(text=t):
                # Show only if --verbose
                ...
            case ErrorEvent(message=m):
                self.console.log(f"[red]error: {m}[/red]")
            case DoneEvent():
                ...

    def _render(self):
        body = Text()
        for status in self.tool_status.values():
            body.append(status + "\n", style="dim")
        for buf in self.text_buffers.values():
            body.append(buf + "\n")
        return Panel(body, title=self.agent_name)
```

### 4.3 `src/t0_agents/cli/elicit_tui.py` (new)

```python
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt
from fastmcp.client.elicitation import ElicitResult


async def rich_elicitation_handler(message, response_type, params, context):
    """Renders elicitation prompts using Rich.

    Relies on FastMCP Q5 schema guarantees:
    - flat type:object
    - primitives + enum + const
    - description + default preserved
    - scalar wrapping in {value: ...}
    """
    if response_type is None:
        # Binary approval
        ok = Confirm.ask(message, default=False)
        return ElicitResult(action="accept" if ok else "decline", content={})

    schema = params.requestedSchema or {}
    properties = schema.get("properties", {})
    is_scalar_wrap = list(properties.keys()) == ["value"]

    content = {}
    for name, spec in properties.items():
        label = spec.get("description") or name
        ftype = spec.get("type", "string")
        default = spec.get("default")
        enum = spec.get("enum")

        if enum is not None:
            choices = [str(v) for v in enum]
            answer = Prompt.ask(label, choices=choices, default=str(default) if default else None)
            # Coerce back to the original enum type
            for v in enum:
                if str(v) == answer:
                    content[name] = v
                    break
        elif ftype == "boolean":
            content[name] = Confirm.ask(label, default=bool(default))
        elif ftype == "integer":
            content[name] = IntPrompt.ask(label, default=default)
        elif ftype == "number":
            content[name] = FloatPrompt.ask(label, default=default)
        else:
            content[name] = Prompt.ask(label, default=default if isinstance(default, str) else None)

    return ElicitResult(action="accept", content=content)
```

### 4.4 `src/t0_agents/cli/run.py` (new)

```python
import asyncio
import typer

from rich.console import Console
from rich.prompt import Prompt


app = typer.Typer()


@app.command()
def run(
    agent_spec: str = typer.Argument(..., help="module:attr path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run an agent in an interactive REPL with streaming + Rich UI."""
    asyncio.run(_run(agent_spec, verbose))


async def _run(agent_spec: str, verbose: bool):
    agent = load_agent(agent_spec)
    console = Console()
    renderer = TUIRenderer(agent.config.name, console=console)
    episode_id: str | None = None

    while True:
        try:
            user = Prompt.ask("[bold]You[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]bye[/dim]")
            return
        if not user.strip():
            continue

        try:
            stream = agent.stream(user, episode_id=episode_id)
            await renderer.consume(stream)
            # Capture episode_id from the last DoneEvent for continuity
            episode_id = renderer.last_episode_id
        except Exception as e:
            console.log(f"[red]error: {e}[/red]")
```

### 4.5 `src/t0_agents/cli/serve_cmd.py` (new)

```python
@app.command(name="serve")
def serve_cmd(
    agent_spec: str = typer.Argument(...),
    host: str = "0.0.0.0",
    port: int = 8080,
    deployment: str | None = None,
):
    """Wrap an agent in create_app + uvicorn for a quick dev server."""
    import uvicorn
    from t0_agents.serve import create_app

    agent = load_agent(agent_spec)
    app = create_app(agent, deployment_label=deployment)
    uvicorn.run(app, host=host, port=port)
```

### 4.6 `src/t0_agents/cli/info_cmd.py` (new)

```python
@app.command(name="info")
def info_cmd(agent_spec: str = typer.Argument(...)):
    """Pretty-print AgentConfig summary."""
    agent = load_agent(agent_spec)
    config = agent.config
    table = Table(title=f"AgentConfig: {config.name}")
    table.add_column("Field"); table.add_column("Value")
    for f in fields(config):
        v = getattr(config, f.name)
        if f.name == "guardrails":
            v = ", ".join(g.name for g in v) or "—"
        table.add_row(f.name, repr(v))
    Console().print(table)
```

### 4.7 `pyproject.toml` extras + scripts

```toml
[project.optional-dependencies.cli]
typer = ">=0.12.0"
rich = ">=13.0.0"

[project.scripts]
t0-agents = "t0_agents.cli.app:main"
```

---

## 5. Tests

Location: `tests/test_cli/`.

### 5.1 Unit (offline)

- [ ] `test_loader_valid_spec` — `module:attr` resolves.
- [ ] `test_loader_invalid_spec` — bad format → `BadParameter`.
- [ ] `test_loader_not_an_agent` — non-Agent attribute → error.
- [ ] `test_tui_renderer_text_deltas` — deltas accumulate.
- [ ] `test_tui_renderer_tool_calls` — status line rendered.
- [ ] `test_rich_elicitation_handler_binary_approval` — uses `Confirm`
      mock; both branches.
- [ ] `test_rich_elicitation_handler_typed_input` — for each type
      (string/integer/number/boolean/enum), correct prompt + content.
- [ ] `test_rich_elicitation_handler_scalar_wrap` — single `value`
      property unwrapped (handled server-side by FastMCP, but the
      handler wraps correctly).
- [ ] `test_info_command_output` — snapshot.

### 5.2 Integration (live gateway)

- [ ] `test_run_subcommand_e2e` — Typer's CliRunner invokes
      `t0-agents run examples.13_tui.agent:assistant`, scripted with
      input lines; assert final transcript.
- [ ] `test_serve_subcommand_smoke` — boots uvicorn in subprocess,
      hits `/health`, terminates.

---

## 6. Example

Location: `examples/13_tui/`.

- `server.py` — small MCP server with one elicitation tool
  (e.g. `set_reminder(when, what)` requiring user-confirmed scheduling).
- `agent.py` — defines `assistant: Agent` at module level.
- `README.md` — `uv run t0-agents run examples.13_tui.agent:assistant`
  walkthrough; demo screenshot description; pointer to elicitation
  flow.

---

## 7. Documentation

- New: `docs/cli.md`.
- Sections:
  - "Install" — `pip install 't0-agents[cli]'`.
  - "Three commands" — run, serve, info.
  - "REPL behaviour" — episode continuity across user turns.
  - "Elicitation in the TUI" — schema-driven prompts via Rich.
  - "Custom renderers" — point at `TUIRenderer` for users wanting to
    swap out the look.
- Update `developer-guide.md` §4 (public API): `cli.app:main`,
  `TUIRenderer`, `rich_elicitation_handler`.
- `CHANGELOG.md` — `0.7.0`.

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/cli/__init__.py`.
- [ ] `src/t0_agents/cli/app.py` (Typer entry).
- [ ] `src/t0_agents/cli/run.py`, `serve_cmd.py`, `info_cmd.py`.
- [ ] `src/t0_agents/cli/tui.py`.
- [ ] `src/t0_agents/cli/elicit_tui.py`.
- [ ] `src/t0_agents/cli/loader.py`.
- [ ] `pyproject.toml` — `[cli]` extra, `t0-agents` entry-point.

### 8.2 Tests

- [ ] All §5.1 unit tests with snapshots for the renderer.
- [ ] §5.2 integration tests.

### 8.3 Example

- [ ] `examples/13_tui/` complete; demo script tested manually.

### 8.4 Docs

- [ ] `docs/cli.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `phase-f-implementation-log.md` with M6 release-notes snippet.

---

## 9. Acceptance criteria (Milestone M6)

1. `pip install 't0-agents[cli]'` provides the `t0-agents` command.
2. `t0-agents run <agent>` shows live deltas, tool-call status, and
   elicitation prompts via Rich.
3. `t0-agents serve <agent>` boots a working dev server.
4. `t0-agents info <agent>` prints a coherent `AgentConfig` summary.
5. Earlier phases unchanged.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Rich + Live conflicts with stdin prompts during elicitation | medium | Pause `Live` while a `Prompt` is pending; resume after. |
| Schema surprises from non-FastMCP MCP servers | medium | Fail-soft to `Prompt.ask(str)` with a logged warning. |
| Long-running tools freeze the TUI | low | `Live` continues to render `ToolCallEvent` + spinner; only the model output is delayed. |
| User wants Textual instead of Rich | low | Document `TUIRenderer` extension point; future PR can add a Textual flavour without changing the CLI surface. |

---

## 11. Estimated surface

- **New files:** 7 src + 4 test + 3 example + 1 doc.
- **Modified files:** 2 (`pyproject.toml`, `CHANGELOG.md`).
- **Net new public symbols:** ~5 (`cli.app:main`, `TUIRenderer`,
  `rich_elicitation_handler`, `load_agent`, the Typer commands).
- **New deps:** 2 under `[cli]` (`typer`, `rich`).
