import httpx
from typing import List, Dict

from rllm.tools.tool_base import Tool, ToolOutput

TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"
# https://docs.tavily.com/api-reference/endpoint/extract#body-extract-depth
API_KEY = ""

class TavilyTool(Tool):
    """A tool for extracting data from websites."""

    def __init__(self):
        self._init_client()
        super().__init__(
            name="tavily",
            description="Extract web page content from one or more specified URLs"
        )
        
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
                            "items": {
                                "type": "string"
                            },
                            "description": "Array of URLs to extract content from"
                        }
                    },
                    "required": ["urls"]
                }
            }
        }

    def _init_client(self):
        self.client = httpx.Client()

    def _close_client(self):
        if self.client:
            self.client.close()
        self.client = None

    def forward(self, urls: List[str]) -> ToolOutput:
        """
        Extract content from provided URLs using Tavily API.
        
        Args:
            urls (List[str]): List of URLs to extract content from.
            
        Returns:
            ToolOutput: An object containing either the extracted content or an error message.
        """
        try:
            params = {
                "urls": urls,
                "include_images": False,
                "extract_depth": "basic"
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = self.client.post(
                url=TAVILY_EXTRACT_ENDPOINT, 
                json=params, 
                headers=headers
            )
            
            if not response.is_success:
                return ToolOutput(name=self.name, error=f"Error: {response.status_code} - {response.text}")
            
            results = response.json()['results']
            output = {
                res['url']: res['raw_content'] for res in results
            }
            return ToolOutput(name=self.name, output=output)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        """Clean up resources when the tool is garbage collected."""
        self._close_client()


if __name__ == '__main__':
    tavily_tool = TavilyTool()
    result = tavily_tool(urls=["https://agentica-project.com/", "https://michaelzhiluo.github.io/"])
    print(result)
    
    # Try async
    import asyncio
    
    async def test_async():
        print("Starting async request...")
        # Get coroutine without executing it
        coro = tavily_tool(urls=["https://agentica-project.com/", "https://michaelzhiluo.github.io/"], use_async=True)
        print(f"Coroutine created: {coro}")
        print("Executing in background...")
        
        # Execute the coroutine
        result = await coro
        print("Async request completed!")
        print(result)
    asyncio.run(test_async())
