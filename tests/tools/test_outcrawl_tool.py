import json

import httpx
import pytest

from rllm.tools import tool_registry
from rllm.tools.web_tools.outcrawl_tool import OUTCRAWL_SCRAPE_ENDPOINT, OUTCRAWL_SEARCH_ENDPOINT, OutcrawlScrapeTool, OutcrawlSearchTool


def _mock(tool, status: int, body: dict, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    tool.client = httpx.Client(transport=httpx.MockTransport(handler))
    return tool


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OUTCRAWL_API_KEY", "oc-test")


def test_tools_are_registered():
    assert isinstance(tool_registry.instantiate("outcrawl-scrape"), OutcrawlScrapeTool)
    assert isinstance(tool_registry.instantiate("outcrawl-search"), OutcrawlSearchTool)


def test_scrape_posts_url_and_returns_markdown(api_key):
    seen: list[httpx.Request] = []
    tool = _mock(
        OutcrawlScrapeTool(),
        200,
        {"url": "https://example.com/", "statusCode": 200, "markdown": "# Example", "metadata": {"title": "Example Domain"}},
        seen,
    )

    result = tool(url="https://example.com")

    assert result.error is None
    assert result.output == {"url": "https://example.com/", "title": "Example Domain", "markdown": "# Example"}
    request = seen[0]
    assert str(request.url) == OUTCRAWL_SCRAPE_ENDPOINT
    assert request.headers["Authorization"] == "Bearer oc-test"
    assert json.loads(request.content) == {"url": "https://example.com", "formats": ["markdown"]}


def test_search_posts_query_and_returns_hits(api_key):
    seen: list[httpx.Request] = []
    hits = [{"title": "rLLM", "url": "https://github.com/rllm-org/rllm", "snippet": "RL for LLMs", "rank": 1}]
    tool = _mock(OutcrawlSearchTool(), 200, {"hits": hits, "usage": {"credits": "5"}}, seen)

    result = tool(query="rllm", max_results=3)

    assert result.error is None
    assert result.output == {"results": hits}
    request = seen[0]
    assert str(request.url) == OUTCRAWL_SEARCH_ENDPOINT
    assert json.loads(request.content) == {"query": "rllm", "limit": 3}


def test_search_with_scrape_asks_for_markdown(api_key):
    seen: list[httpx.Request] = []
    tool = _mock(OutcrawlSearchTool(), 200, {"hits": []}, seen)

    tool(query="rllm", scrape=True)

    assert json.loads(seen[0].content) == {"query": "rllm", "limit": 5, "scrape": True, "formats": ["markdown"]}


def test_api_error_message_is_returned(api_key):
    tool = _mock(OutcrawlScrapeTool(), 402, {"code": "insufficient_credits", "message": "not enough credits"}, [])

    result = tool(url="https://example.com")

    assert result.output is None
    assert result.error == "Error: 402 - insufficient_credits: not enough credits"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OUTCRAWL_API_KEY"):
        OutcrawlSearchTool().forward(query="rllm")
