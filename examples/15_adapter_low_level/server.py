"""FastMCP server with two tools for demonstrating the adapter layer."""
from fastmcp import FastMCP

mcp = FastMCP("DemoServer")


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data)."""
    return f"The weather in {city} is 22°C and sunny."
