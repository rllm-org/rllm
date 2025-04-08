import asyncio
import time
from typing import Dict

try:
    from firecrawl import FirecrawlApp
except ImportError as e:
    print(e)
    FirecrawlApp = None

from rllm.tools.tool_base import Tool, ToolOutput

FIRECRAWL_API = ""
TIMEOUT = 10

class FirecrawlTool(Tool):
    """A tool for extracting data from websites using the FireCrawl service."""

    def __init__(self, timeout: int = TIMEOUT, api_key: str = FIRECRAWL_API, api_url: str = None):
        """
        Initialize the Firecrawl tool.
        
        Args:
            timeout (int): Maximum time in seconds to wait for scraping results.
            api_key (str): API key for FireCrawl service.
            api_url (str, optional): Custom API URL endpoint.
        """
        if FirecrawlApp is None:
            raise ImportError("Firecrawl is not installed. Please install it using 'pip install firecrawl'.")
        self.timeout = timeout
        self.api_key = api_key
        self.api_url = api_url
        self._init_app()
        super().__init__(
            name="firecrawl",
            description="FireCrawl is a tool that scrapes a url link and returns content as a markdown document along with any links."
        )

    def _init_app(self):
        """Initialize the FirecrawlApp instance with appropriate configuration."""
        assert self.api_key is not None or self.api_url is not None, "Either api_key or api_url must be provided."
        if self.api_url is None:
            self.app = FirecrawlApp(api_key=self.api_key)
        else:
            self.app = FirecrawlApp(api_url=self.api_url)

    def _start_firecrawl_job(self, url):
        """
        Start a job with firecrawl async API and return job ID.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            dict: Response from the FireCrawl API containing job information.
        """
        # crawl has many scrape options, potentially can let the agent choose
        return self.app.async_batch_scrape_urls(
            [url], 
            params={
                'formats': ['markdown', 'links'],
                'onlyMainContent': True
            }
        )

    @property
    def json(self):
        """Return the tool's information in a standardized format for tool registration."""
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Web URL to scrape content from."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    def forward(self, url: str) -> ToolOutput:
        """
        Run firecrawl job asynchronously.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            ToolOutput: An object containing either the scraped content or an error message.
        """
        try:
            job = self._start_firecrawl_job(url)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Firecrawl job could not start: {e}")
        
        if not job['success']:
            return ToolOutput(name=self.name, error="Firecrawl job failed to start")
        
        job_id = job['id']
        start_time = time.monotonic()
        while True:
            status = self.app.check_batch_scrape_status(job_id)
            if status['completed']:
                break
            time.sleep(1)
            if time.monotonic() - start_time > self.timeout:
                return ToolOutput(name=self.name, error="Firecrawl request timed out")

        if status['success']:
            results = {
                page['metadata']['url']: page['markdown'] for page in status['data']
            }
            return ToolOutput(name=self.name, output=results)
        return ToolOutput(name=self.name, error=f"Firecrawl request errored: {status['error']}")

    async def async_forward(self, url: str) -> ToolOutput:
        """
        Asynchronous version of the forward method.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            ToolOutput: An object containing either the scraped content or an error message.
        """
        # For now, just call the synchronous version
        # This could be optimized later to use async I/O properly
        return self.forward(url=url)

if __name__ == '__main__':
    search = FirecrawlTool()
    
    start_time = time.monotonic()
    print(search(url="https://agentica-project.com/"))
    end_time = time.monotonic()
    print(f"Time taken for sync: {end_time - start_time} seconds")
    
    # Test Async
    import asyncio
    async def test_async():
        coro = search(url="https://agentica-project.com/", use_async=True)
        
        start_time = time.monotonic()
        result = await coro
        end_time = time.monotonic()
        print("Async result:", result)
        print(f"Time taken for async: {end_time - start_time} seconds")
    # Run the async test
    asyncio.run(test_async())