"""Toy arXiv-flavoured MCP server for the researcher sub-agent.

Ships a single ``search_papers`` tool that returns canned paper titles
keyed off the query string.  Real-world researchers would wire this to
the arXiv API; the example focuses on the delegation mechanics.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("arxiv-research")


@mcp.tool
def search_papers(query: str) -> list[str]:
    """Return a handful of paper titles that mention ``query``.

    The canned mapping is deliberately shallow so the LLM can see the
    signal (query → papers) without the example depending on network
    access.  Unknown queries fall back to a generic placeholder list.
    """
    corpus: dict[str, list[str]] = {
        "rlhf": [
            "Constitutional AI: Harmlessness from AI Feedback (2022)",
            "Training Language Models to Follow Instructions with Human "
            "Feedback (2022)",
            "Fine-Tuning Language Models from Human Preferences (2019)",
        ],
        "mixture of experts": [
            "Mixtral of Experts (2024)",
            "Switch Transformers: Scaling to Trillion Parameter Models (2021)",
        ],
        "reasoning": [
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language "
            "Models (2022)",
            "Self-Consistency Improves Chain of Thought Reasoning (2022)",
        ],
    }
    key = query.lower().strip()
    for keyword, papers in corpus.items():
        if keyword in key:
            return papers
    return [f"No arXiv results matched {query!r}. Try 'RLHF' or 'reasoning'."]
