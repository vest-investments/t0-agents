"""Example FastMCP server with basic tools for the system prompt demo."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("SystemPromptDemo")


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22C."


@mcp.tool
def convert_temperature(celsius: float) -> str:
    """Convert a temperature from Celsius to Fahrenheit."""
    fahrenheit = celsius * 9 / 5 + 32
    return f"{celsius}C = {fahrenheit}F"
