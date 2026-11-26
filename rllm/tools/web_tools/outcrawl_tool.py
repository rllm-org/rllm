import os

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

OUTCRAWL_SCRAPE_ENDPOINT = "https://api.outcrawl.ai/v1/scrape"
OUTCRAWL_SEARCH_ENDPOINT = "https://api.outcrawl.ai/v1/search"
TIMEOUT = 60


def _api_key() -> str:
    api_key = os.getenv("OUTCRAWL_API_KEY")
    if not api_key:
        raise ValueError("OUTCRAWL_API_KEY is not set")
    return api_key


def _error_text(response: httpx.Response) -> str:
    """Outcrawl explains failures in a {code, message} body; fall back to the raw text."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("message"):
        return f"Error: {response.status_code} - {body.get('code', '')}: {body['message']}"
    return f"Error: {response.status_code} - {response.text}"


class OutcrawlScrapeTool(Tool):
    """A tool for reading a web page as markdown using the Outcrawl API."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout
        self._init_client()
        super().__init__(name="outcrawl-scrape", description="Outcrawl reads a web page and returns its content as a markdown document along with its title.")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Web URL to read."},
                    },
                    "required": ["url"],
                },
            },
        }

    def _init_client(self):
        self.client: httpx.Client | None = httpx.Client(timeout=self.timeout)

    def _close_client(self):
        if self.client:
            self.client.close()
        self.client = None

    def forward(self, url: str) -> ToolOutput:
        """
        Read a web page using the Outcrawl API.

        Args:
            url (str): The URL to read.

        Returns:
            ToolOutput: An object containing either {url, title, markdown} or an error message.
        """
        api_key = _api_key()

        if self.client is None:
            raise RuntimeError("HTTP client is not initialized")

        try:
            params = {"url": url, "formats": ["markdown"]}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            response = self.client.post(url=OUTCRAWL_SCRAPE_ENDPOINT, json=params, headers=headers)

            if not response.is_success:
                return ToolOutput(name=self.name or "outcrawl-scrape", error=_error_text(response))

            result = response.json()
            metadata = result.get("metadata") or {}
            output = {"url": result.get("url", url), "title": metadata.get("title"), "markdown": result.get("markdown", "")}
            return ToolOutput(name=self.name or "outcrawl-scrape", output=output)
        except Exception as e:
            return ToolOutput(name=self.name or "outcrawl-scrape", error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        """Clean up resources when the tool is garbage collected."""
        self._close_client()


class OutcrawlSearchTool(Tool):
    """A tool for searching the web using the Outcrawl API."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout
        self._init_client()
        super().__init__(name="outcrawl-search", description="Search the web for information on a specific query, optionally returning the markdown of each result page")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {"type": "integer", "description": "Maximum number of search results to return"},
                        "scrape": {"type": "boolean", "description": "Also return the markdown content of each result page"},
                    },
                    "required": ["query"],
                },
            },
        }

    def _init_client(self):
        self.client: httpx.Client | None = httpx.Client(timeout=self.timeout)

    def _close_client(self):
        if self.client:
            self.client.close()
        self.client = None

    def forward(self, query: str, max_results: int = 5, scrape: bool = False) -> ToolOutput:
        """
        Search the web using the Outcrawl API.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of search results to return. Defaults to 5.
            scrape (bool, optional): Also return the markdown of each result page. Defaults to False.

        Returns:
            ToolOutput: An object containing either the search hits ({title, url, snippet, rank}, plus markdown when scraped) or an error message.
        """
        api_key = _api_key()

        if self.client is None:
            raise RuntimeError("HTTP client is not initialized")

        try:
            params: dict = {"query": query, "limit": max_results}
            if scrape:
                params["scrape"] = True
                params["formats"] = ["markdown"]

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            response = self.client.post(url=OUTCRAWL_SEARCH_ENDPOINT, json=params, headers=headers)

            if not response.is_success:
                return ToolOutput(name=self.name or "outcrawl-search", error=_error_text(response))

            result = response.json()
            return ToolOutput(name=self.name or "outcrawl-search", output={"results": result.get("hits", [])})
        except Exception as e:
            return ToolOutput(name=self.name or "outcrawl-search", error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        """Clean up resources when the tool is garbage collected."""
        self._close_client()


if __name__ == "__main__":
    scrape_tool = OutcrawlScrapeTool()
    print("Scrape Tool Result:")
    print(scrape_tool(url="https://agentica-project.com/"))

    search_tool = OutcrawlSearchTool()
    print("\nSearch Tool Result:")
    print(search_tool(query="Latest developments in AI research"))
