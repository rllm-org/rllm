import httpx
from typing import Dict

from rllm.tools.tool_base import Tool, ToolOutput

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"

# Must enter secret key and engine id https://programmablesearchengine.google.com/controlpanel/all
SECRET_KEY = ""
ENGINE_ID = ""

class GoogleSearchTool(Tool):
    """A tool for searching google."""
    NAME = "google_search"
    DESCRIPTION = f"Search a query using the Google search engine, returning the top {REFERENCE_COUNT} results along with a short snippet about their contents"

    def __init__(self, name: str=NAME, description: str=DESCRIPTION, timeout: float=DEFAULT_SEARCH_ENGINE_TIMEOUT, reference_count: int=REFERENCE_COUNT):
        """
        Initialize the GoogleSearch tool.
        
        Args:
            name (str): The name of the tool, defaults to GoogleSearch.NAME.
            description (str): A description of the tool's purpose, defaults to GoogleSearch.DESCRIPTION.
            timeout (float): Maximum time in seconds to wait for search results, defaults to DEFAULT_SEARCH_ENGINE_TIMEOUT.
            reference_count (int): Number of results to return, defaults to REFERENCE_COUNT.
        """
        self.timeout = timeout
        self.reference_count = reference_count
        self._init_client()
        super().__init__(name=name, description=description)
    
    def _init_client(self):
        """
        Initialize the HTTP client for making asynchronous requests.
        
        Creates an instance of httpx.AsyncClient for the current instance.
        """
        self.client = httpx.Client()

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
                        "query": {
                            "type": "string",
                            "description": "Query to be submitted to Google search engine."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def _search_with_google(self, query: str):
        """
        Search with google and return the contexts.
        """
        params = {
            "key": SECRET_KEY,
            "cx": ENGINE_ID,
            "q": query,
            "num": REFERENCE_COUNT,
        }
        
        response = self.client.get(url=GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT)
        if not response.is_success:
            print(f"{response.status_code} {response.text}")
        json_content = response.json()
        try:
            contexts = json_content["items"][:REFERENCE_COUNT]
        except KeyError:
            print(f"Error encountered: {json_content}")
            return []
        return contexts

    def forward(self, query: str) -> ToolOutput:
        """
        Execute a Google search with the given query.
        
        Args:
            query (str): Query to be submitted to Google search engine.
            
        Returns:
            ToolOutput: An object containing either the search results or an error message.
        """
        try:
            assert self.client is not None, "Google Search Client not initialized"
            contexts = self._search_with_google(query)
            results = {
                c['link']: c['snippet'] for c in contexts
            }
            return ToolOutput(name=self.name, output=results)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")

    def __del__(self):
        self.client.close()


if __name__ == '__main__':
    search = GoogleSearchTool()
    print(search(query='Give me current time right now in PST'))