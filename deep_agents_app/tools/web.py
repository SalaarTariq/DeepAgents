"""Tavily-backed web search, exposed as a LangChain tool."""

from __future__ import annotations

import os
from typing import Any, Literal

from tavily import TavilyClient

Topic = Literal["general", "news", "medical", "finance", "technology"]

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ["TAVILY_API_KEY"]
        _client = TavilyClient(api_key=api_key)
    return _client


def web_query(
    query: str,
    max_results: int = 3,
    topic: Topic = "general",
) -> dict[str, Any]:
    """Search the web for information on a given topic.

    Args:
        query: The natural-language search query.
        max_results: How many results to return (default 3).
        topic: Tavily topic vertical.
    """
    return _get_client().search(
        query=query,
        max_results=max_results,
        include_raw_content=True,
        topic=topic,
    )
