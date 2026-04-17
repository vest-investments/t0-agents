"""Typer entry point for the ``t0-agents`` CLI.

Wired to the ``t0-agents`` script entry in ``pyproject.toml``:

.. code-block:: toml

    [project.scripts]
    t0-agents = "t0_agents.cli.app:main"

The app exposes three subcommands, each in its own module:

- :mod:`t0_agents.cli.run_cmd` → ``t0-agents run``
- :mod:`t0_agents.cli.serve_cmd` → ``t0-agents serve``
- :mod:`t0_agents.cli.info_cmd` → ``t0-agents info``
"""

from __future__ import annotations

import typer

from t0_agents.cli import info_cmd, run_cmd, serve_cmd

app: typer.Typer = typer.Typer(
    name="t0-agents",
    help=(
        "TensorZero-first Agent CLI — streaming REPL, dev HTTP server, "
        "and config introspection."
    ),
    no_args_is_help=True,
    add_completion=False,
)

run_cmd.register(app)
serve_cmd.register(app)
info_cmd.register(app)


def main() -> None:
    """Script entry point used by the ``t0-agents`` console script."""
    app()


if __name__ == "__main__":
    main()
