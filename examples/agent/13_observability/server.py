"""MCP server for the feedback loop example."""
from fastmcp import FastMCP

mcp = FastMCP("FeedbackDemoServer")


@mcp.tool
def search_docs(query: str) -> str:
    """Search the documentation for a topic."""
    return f"Found 3 results for '{query}': [Getting Started, API Reference, FAQ]"


@mcp.tool
def get_doc_content(doc_name: str) -> str:
    """Get the content of a specific document."""
    docs = {
        "Getting Started": "Install with pip install t0-agents. See docs for quickstart.",
        "API Reference": "await agent.invoke(message) -> AgentResult",
        "FAQ": "Q: Does it require Docker? A: Only for the TensorZero gateway.",
    }
    return docs.get(doc_name, f"Document '{doc_name}' not found.")
