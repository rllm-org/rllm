import os

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"


class TavilyExtractTool(Tool):
    """A tool for extracting data from websites."""

    def __init__(self):
        self._init_client()
        super().__init__(name="tavily-extract", description="Extract web page content from one or more specified URLs")

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
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of URLs to extract content from",
                        }
                    },
                    "required": ["urls"],
                },
            },
        }

    def _init_client(self):
        self.client: httpx.Client | None = httpx.Client()

    def _close_client(self):
        if self.client:
            self.client.close()
        self.client = None

    def forward(self, urls: list[str]) -> ToolOutput:
        """
        Extract content from provided URLs using Tavily API.

        Args:
            urls (List[str]): List of URLs to extract content from.

        Returns:
            ToolOutput: An object containing either the extracted content or an error message.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set")

        if self.client is None:
            raise RuntimeError("HTTP client is not initialized")

        try:
            params = {"urls": urls, "include_images": False, "extract_depth": "basic"}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            response = self.client.post(url=TAVILY_EXTRACT_ENDPOINT, json=params, headers=headers)

            if not response.is_success:
                return ToolOutput(name=self.name or "tavily-extract", error=f"Error: {response.status_code} - {response.text}")

            output = response.json()
            return ToolOutput(name=self.name or "tavily-extract", output=output)
        except Exception as e:
            return ToolOutput(name=self.name or "tavily-extract", error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        """Clean up resources when the tool is garbage collected."""
        self._close_client()


class TavilySearchTool(Tool):
    """A tool for searching the web using Tavily API."""

    def __init__(self):
        self._init_client()
        super().__init__(name="tavily-search", description="Search the web for information on a specific query")

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
                        "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "The depth of search (basic or advanced)"},
                        "include_domains": {"type": "array", "items": {"type": "string"}, "description": "List of domains to include in the search"},
                        "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "List of domains to exclude from the search"},
                        "max_results": {"type": "integer", "description": "Maximum number of search results to return"},
                    },
                    "required": ["query"],
                },
            },
        }

    def _init_client(self):
        self.client: httpx.Client | None = httpx.Client()

    def _close_client(self):
        if self.client:
            self.client.close()
        self.client = None

    def forward(self, query: str, search_depth: str = "basic", include_domains: list[str] | None = None, exclude_domains: list[str] | None = None, max_results: int = 5) -> ToolOutput:
        """
        Search the web using Tavily API.

        Args:
            query (str): The search query.
            search_depth (str, optional): The depth of search. Defaults to "basic".
            include_domains (List[str], optional): List of domains to include in the search.
            exclude_domains (List[str], optional): List of domains to exclude from the search.
            max_results (int, optional): Maximum number of search results to return. Defaults to 5.

        Returns:
            ToolOutput: An object containing either the search results or an error message.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set")

        if self.client is None:
            raise RuntimeError("HTTP client is not initialized")

        try:
            params = {"query": query, "search_depth": search_depth, "max_results": max_results}

            if include_domains:
                params["include_domains"] = include_domains
            if exclude_domains:
                params["exclude_domains"] = exclude_domains

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            response = self.client.post(url=TAVILY_SEARCH_ENDPOINT, json=params, headers=headers)

            if not response.is_success:
                return ToolOutput(name=self.name or "tavily-search", error=f"Error: {response.status_code} - {response.text}")

            result = response.json()
            return ToolOutput(name=self.name or "tavily-search", output=result)
        except Exception as e:
            return ToolOutput(name=self.name or "tavily-search", error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        """Clean up resources when the tool is garbage collected."""
        self._close_client()


if __name__ == "__main__":
    # Test extract tool
    extract_tool = TavilyExtractTool()
    extract_result = extract_tool(urls=["https://agentica-project.com/"])
    print("Extract Tool Result:")
    print(extract_result)

    # Test search tool
    search_tool = TavilySearchTool()
    search_result = search_tool(query="Latest developments in AI research")
    print("\nSearch Tool Result:")
    print(search_result)

    import asyncio

    async def test_async():
        print("\nStarting async requests...")

        # Extract async
        extract_coro = extract_tool(urls=["https://agentica-project.com/"], use_async=True)
        extract_result = await extract_coro
        print("Async extract completed!")
        print(extract_result)

        # Search async
        search_coro = search_tool(query="Python programming best practices", use_async=True)
        search_result = await search_coro
        print("Async search completed!")
        print(search_result)

    asyncio.run(test_async())
