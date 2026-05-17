# `glow` examples (experimental)

This directory will host runnable examples for the `t0_agents.glow`
sub-package — a Pythonic high-level surface over t0-agents v1.0.

The layer is experimental. Public API is unstable until further notice.

Examples will be added as the implementation lands on the
`experimental/glow` branch. Each example will be self-contained,
runnable as `uv run python examples/glow/<name>.py`, and demonstrate
one capability:

- `01_hello.py` — agent in 4 lines (`@app.agent`, docstring system prompt)
- `02_structured.py` — return type → `StructuredOutput`
- `03_streaming.py` — `FluentRun.events()` with selectors
- `04_composition.py` — `compose([...])`, `compose({...})`, `compose({a, b})`
- `05_di_tools.py` — `@app.tool` + `Depends`
- `06_mcp_typing.py` — `MCPServer` class + `Annotated` MCP injection
