"""``t0-agents run <spec>`` — interactive streaming REPL."""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.prompt import Prompt

from t0_agents.cli.loader import load_agent_factory
from t0_agents.cli.tui import TUIRenderer
from t0_agents.core.agent import Agent
from t0_agents.core.exceptions import T0AgentsFatalError

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    """Attach the ``run`` subcommand to ``app``."""

    @app.command("run")
    def run_cmd(
        agent_spec: str = typer.Argument(
            ...,
            help="Factory spec in 'module:attr' form, e.g. "
            "'examples.13_tui.agent:assistant'.",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Render reasoning (ThoughtEvent) content in the panel.",
        ),
        debug_log: str | None = typer.Option(
            None,
            "--debug-log",
            help=(
                "Write DEBUG-level logs from t0_agents.* to this file path. "
                "Useful for diagnosing elicitation / streaming bugs without "
                "mixing logs into the Live panel."
            ),
        ),
    ) -> None:
        """Start an interactive streaming REPL against an agent factory.

        The factory must be a zero-argument callable returning an async
        context manager yielding an :class:`Agent` (typically decorated
        with :func:`contextlib.asynccontextmanager`).
        """
        if debug_log is not None:
            handler = logging.FileHandler(debug_log, mode="w")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s",
                ),
            )
            root = logging.getLogger("t0_agents")
            root.addHandler(handler)
            root.setLevel(logging.DEBUG)
        asyncio.run(_run(agent_spec, verbose))


async def _run(spec: str, verbose: bool) -> None:
    factory = load_agent_factory(spec)
    console = Console()
    try:
        async with factory() as agent:
            await agent.warmup()
            renderer = TUIRenderer(
                agent.config.name,
                console=console,
                verbose=verbose,
            )
            console.print(
                f"[bold green]t0-agents[/bold green] · "
                f"[cyan]{agent.config.name}[/cyan] "
                f"(function [italic]{agent.config.function_name}[/italic])",
            )
            console.print("[dim]Ctrl-D or Ctrl-C to exit.[/dim]")
            await _repl(agent, renderer, console)
    except typer.BadParameter:
        raise
    except Exception:
        logger.exception("t0-agents run: unexpected fatal error")
        raise


async def _repl(agent: Agent, renderer: TUIRenderer, console: Console) -> None:
    episode_id: str | None = None
    while True:
        try:
            user_message = Prompt.ask("[bold]You[/bold]", console=console)
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]bye[/dim]")
            return
        if not user_message.strip():
            continue
        try:
            run = agent.invoke(
                user_message,
                episode_id=episode_id,
            )
            result = await renderer.consume(run.events())
        except T0AgentsFatalError as exc:
            console.print(f"[bold red]{type(exc).__name__}: {exc}[/bold red]")
            return
        except Exception as exc:
            console.print(f"[red]error: {exc}[/red]")
            logger.exception("Streaming turn failed")
            continue
        if result is not None and result.episode_id:
            episode_id = result.episode_id
