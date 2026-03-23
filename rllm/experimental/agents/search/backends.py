"""Search tool backends for web search agent.

Provides search tool implementations (Serper, Brave) with a common interface:
- `.json` property: OpenAI function-calling tool schema
- `.run(query)` method: synchronous search returning formatted string
- `.name` attribute: tool function name

Use `resolve_search_backend()` to auto-detect which backend to use.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information. Returns relevant snippets from search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information",
                },
            },
            "required": ["query"],
        },
    },
}


def _format_results(results: list[dict]) -> str:
    """Format search results as numbered snippets for LLM consumption."""
    if not results:
        return "No relevant results found."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        lines = [f"[{i}] {title}"]
        if snippet:
            lines.append(snippet)
        if link:
            lines.append(f"URL: {link}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


class SerperSearchTool:
    """Web search via Serper (Google Search API).

    Endpoint: POST https://google.serper.dev/search
    Auth: X-API-KEY header from SERPER_API_KEY env var.
    """

    name = "web_search"

    def __init__(self, api_key: str | None = None, max_results: int = 10, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        self.max_results = max_results
        self.timeout = timeout

    @property
    def json(self) -> dict:
        return _SEARCH_TOOL_SCHEMA

    def run(self, query: str) -> str:
        if not self.api_key:
            return "Error: SERPER_API_KEY not set."
        if not query:
            return "Error: Empty search query."
        try:
            response = httpx.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "num": self.max_results},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            organic = data.get("organic", [])
            return _format_results(organic)
        except httpx.TimeoutException:
            return "Error: Search request timed out."
        except httpx.HTTPStatusError as e:
            return f"Error: Search API returned {e.response.status_code}."
        except Exception as e:
            return f"Error: Search failed: {e}"


class BraveSearchTool:
    """Web search via Brave Search API.

    Endpoint: GET https://api.search.brave.com/res/v1/web/search
    Auth: X-Subscription-Token header from BRAVE_API_KEY env var.
    """

    name = "web_search"

    def __init__(self, api_key: str | None = None, max_results: int = 10, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results
        self.timeout = timeout

    @property
    def json(self) -> dict:
        return _SEARCH_TOOL_SCHEMA

    def run(self, query: str) -> str:
        if not self.api_key:
            return "Error: BRAVE_API_KEY not set."
        if not query:
            return "Error: Empty search query."
        try:
            response = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                params={"q": query, "count": self.max_results},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("web", {}).get("results", [])
            results = [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("description", ""),
                    "link": r.get("url", ""),
                }
                for r in raw_results
            ]
            return _format_results(results)
        except httpx.TimeoutException:
            return "Error: Search request timed out."
        except httpx.HTTPStatusError as e:
            return f"Error: Search API returned {e.response.status_code}."
        except Exception as e:
            return f"Error: Search failed: {e}"


_BACKENDS = {
    "serper": SerperSearchTool,
    "brave": BraveSearchTool,
}


def resolve_search_backend(name: str | None = None) -> SerperSearchTool | BraveSearchTool:
    """Resolve which search backend to use.

    Priority: explicit name -> RLLM_SEARCH_BACKEND env -> auto-detect from API keys.
    Detection order: SERPER_API_KEY -> BRAVE_API_KEY.

    Raises:
        ValueError: If no backend can be resolved.
    """
    # Explicit name
    backend_name = name or os.environ.get("RLLM_SEARCH_BACKEND")
    if backend_name:
        backend_name = backend_name.lower()
        if backend_name not in _BACKENDS:
            raise ValueError(f"Unknown search backend '{backend_name}'. Available: {', '.join(_BACKENDS)}")
        return _BACKENDS[backend_name]()

    # Auto-detect from API keys
    if os.environ.get("SERPER_API_KEY"):
        return SerperSearchTool()
    if os.environ.get("BRAVE_API_KEY"):
        return BraveSearchTool()

    raise ValueError("No search backend configured. Set SERPER_API_KEY or BRAVE_API_KEY, or pass --search-backend explicitly.")
