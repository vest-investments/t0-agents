"""``t0-agents serve <spec>`` — uvicorn dev server wrapper."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from t0_agents.cli.loader import load_agent_factory


def register(app: typer.Typer) -> None:
    """Attach the ``serve`` subcommand to ``app``."""

    @app.command("serve")
    def serve_cmd(
        agent_spec: str = typer.Argument(
            ...,
            help="Factory spec in 'module:attr' form.",
        ),
        host: str = typer.Option("0.0.0.0", help="Host to bind."),
        port: int = typer.Option(8080, help="Port to listen on."),
        streaming: bool = typer.Option(
            False,
            "--streaming/--no-streaming",
            help=(
                "Enable SSE + WebSocket streaming routes (default: off). "
                "WS elicitation requires a per-session Client, which the "
                "CLI's shared-Client mode cannot provide; opting in will "
                "boot the SSE route fine but every WS connection that "
                "triggers ctx.elicit() will close with a ConfigurationError."
            ),
        ),
        deployment: str | None = typer.Option(
            None,
            "--deployment",
            help="Value merged into tags as 't0_agents.deployment=<label>'.",
        ),
    ) -> None:
        """Wrap an agent in :func:`t0_agents.serve.create_app` + uvicorn.

        The CLI uses ``share_client=True``: the agent's tool provider is
        reused across sessions.  This is fine for a single-user dev
        server but not for production multi-tenant elicitation — use
        :func:`t0_agents.serve.create_app` directly with a
        ``client_factory`` / ``session_provider`` for that (see
        ``examples/12_streaming_http/agent_app.py``).

        ``--streaming`` defaults to ``False`` because the WS elicitation
        path is structurally incompatible with shared-Client mode
        (FastMCP Q3 needs one elicitation handler per Client).  Use
        ``--streaming`` only when the served agent's tools never call
        ``ctx.elicit`` — then SSE works and WS is only an unused
        endpoint.
        """
        asyncio.run(_serve(agent_spec, host, port, streaming, deployment))


async def _serve(
    spec: str,
    host: str,
    port: int,
    streaming: bool,
    deployment: str | None,
) -> None:
    import uvicorn

    from t0_agents.serve import create_app

    factory = load_agent_factory(spec)
    console = Console()
    async with factory() as agent:
        await agent.warmup()
        app = create_app(
            agent,
            share_client=True,
            streaming=streaming,
            deployment_label=deployment,
        )
        mode = "streaming (SSE + WS)" if streaming else "non-streaming"
        console.print(
            f"[bold green]t0-agents serve[/bold green] · "
            f"[cyan]{agent.config.name}[/cyan] — {mode} on "
            f"http://{host}:{port}",
        )
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        await server.serve()
