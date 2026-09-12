"""``t0-agents info <spec>`` — print the AgentConfig."""

from __future__ import annotations

import asyncio
import dataclasses

import typer
from rich.console import Console
from rich.table import Table

from t0_agents.cli.loader import load_agent_factory
from t0_agents.core.config import AgentConfig


def register(app: typer.Typer) -> None:
    """Attach the ``info`` subcommand to ``app``."""

    @app.command("info")
    def info_cmd(
        agent_spec: str = typer.Argument(
            ...,
            help="Factory spec in 'module:attr' form, or a glow App "
            "instance (e.g. 'mymodule:app').",
        ),
        agent: str | None = typer.Option(
            None,
            "--agent",
            help="Agent to inspect when the spec is a glow App with several "
            "registered agents (defaults to the sole registered one).",
        ),
    ) -> None:
        """Print the ``AgentConfig`` that the factory produces.

        Enters the factory context just long enough to read
        ``agent.config`` and exits.  If your factory opens a real
        gateway / MCP client, expect a few seconds of setup.
        """
        asyncio.run(_info(agent_spec, agent_name=agent))


async def _info(spec: str, *, agent_name: str | None = None) -> None:
    factory = load_agent_factory(spec, agent_name=agent_name)
    async with factory() as agent:
        config = agent.config
    _print_config(config, spec)


def _print_config(config: AgentConfig, spec: str) -> None:
    table = Table(
        title=f"AgentConfig · {config.name}",
        caption=f"loaded from {spec!r}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field")
    table.add_column("Value")
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        table.add_row(field.name, _format_value(field.name, value))
    Console().print(table)


def _format_value(name: str, value: object) -> str:
    if value is None:
        return "—"
    if name == "guardrails":
        if not value:
            return "—"
        return ", ".join(getattr(g, "name", repr(g)) for g in value)  # type: ignore[attr-defined]
    if name == "memory":
        return type(value).__name__
    if isinstance(value, dict):
        if not value:
            return "—"
        return ", ".join(f"{k}={v!r}" for k, v in value.items())
    return repr(value)
