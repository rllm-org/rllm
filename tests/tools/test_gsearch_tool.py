"""Unit tests for GoogleSearchTool's request construction."""

from rllm.tools.web_tools.gsearch_tool import GoogleSearchTool


class _StubResponse:
    is_success = True

    def json(self):
        return {"items": [{"link": f"https://example.com/{i}", "snippet": f"snippet {i}"} for i in range(8)]}


def test_google_search_honors_constructor_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_SEARCH_SECRET_KEY", "secret")
    monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "engine")

    tool = GoogleSearchTool(timeout=30.0, reference_count=2)

    seen: dict = {}

    def _get(url, params=None, timeout=None):  # noqa: ARG001
        seen["params"] = params
        seen["timeout"] = timeout
        return _StubResponse()

    monkeypatch.setattr(tool.client, "get", _get)

    output = tool.forward(query="what is rllm")

    assert output.error is None
    assert seen["timeout"] == 30.0
    assert seen["params"]["num"] == 2
    assert len(output.output) == 2
